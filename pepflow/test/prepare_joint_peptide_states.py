#!/usr/bin/env python3
"""Prepare an aligned two-peptide/two-receptor-state joint-flow case."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from rdkit import Chem
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "joint-peptide-states"
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


def require_clean_commit():
    status = git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return
    print("ERROR: preparation must run from a committed worktree.", file=sys.stderr)
    print("\n--- git status --short ---", file=sys.stderr)
    print(status, file=sys.stderr)
    for title, args in (
        ("git diff", ("diff", "--no-ext-diff")),
        ("git diff --cached", ("diff", "--cached", "--no-ext-diff")),
    ):
        diff = git(*args, check=False)
        if diff:
            print(f"\n--- {title} ---", file=sys.stderr)
            print(diff, file=sys.stderr)
    raise SystemExit(2)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fasta(raw):
    return "".join(
        line.strip().upper() for line in raw.decode("utf-8", errors="replace").splitlines()
        if not line.startswith(">")
    )


def coordinate(line):
    return np.array([
        float(line[30:38]), float(line[38:46]), float(line[46:54])
    ], dtype=np.float64)


def replace_coordinate(line, xyz):
    return f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"


def atom_lines(raw):
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return [line for line in lines if line.startswith(("ATOM  ", "HETATM")) and line[16] in (" ", "A")]


def receptor_residues(lines):
    residues = []
    current_key = None
    for line in lines:
        if not line.startswith("ATOM  "):
            continue
        key = (line[21], line[22:26], line[26])
        if key != current_key:
            residues.append({"key": key, "name": line[17:20].strip(), "atoms": {}})
            current_key = key
        residues[-1]["atoms"].setdefault(line[12:16].strip(), line)
    return residues


def kabsch(mobile, reference):
    mc, rc = mobile.mean(0), reference.mean(0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (reference - rc))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, rc - mc @ rotation


def transformed_pdb(lines, rotation, translation):
    return "\n".join(replace_coordinate(line, coordinate(line) @ rotation + translation) for line in lines) + "\nEND\n"


def midpoint_anchor(res_a, res_b, rotation, translation):
    output = []
    for a, b in zip(res_a, res_b):
        for atom_name, line_a in a["atoms"].items():
            xyz_a = coordinate(line_a)
            if atom_name in b["atoms"]:
                xyz_b = coordinate(b["atoms"][atom_name]) @ rotation + translation
                xyz = (xyz_a + xyz_b) / 2.0
            else:
                xyz = xyz_a
            output.append(replace_coordinate(line_a, xyz))
    return "\n".join(output) + "\nEND\n"


def pdb_to_sdf(pdb_text, path):
    molecule = Chem.MolFromPDBBlock(
        pdb_text, sanitize=False, removeHs=False, proximityBonding=True
    )
    if molecule is None:
        raise ValueError("RDKit could not parse peptide PDB")
    Chem.SanitizeMol(molecule)
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()
    check = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    if check is None or check.GetNumAtoms() < 10:
        raise ValueError(f"invalid peptide SDF: {path}")
    return check.GetNumAtoms()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--entry-a", default="1eb1_A")
    parser.add_argument("--entry-b", default="3vxf_J")
    args = parser.parse_args()
    require_clean_commit()
    archive_path = Path(args.archive).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    started = datetime.now().astimezone()
    result_dir = RESULTS_ROOT / started.strftime("%Y%m%d-%H%M%S%z")
    result_dir.mkdir(parents=True)
    commit = git("rev-parse", "HEAD")
    message = git("log", "-1", "--format=%B")
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")

    with zipfile.ZipFile(archive_path) as archive:
        data = {}
        for label, entry in (("a", args.entry_a), ("b", args.entry_b)):
            data[label] = {
                "entry": entry,
                "receptor_fasta": fasta(archive.read(f"{entry}/receptor.fasta")),
                "peptide_fasta": fasta(archive.read(f"{entry}/peptide.fasta")),
                "receptor_lines": atom_lines(archive.read(f"{entry}/receptor.pdb")),
                "peptide_lines": atom_lines(archive.read(f"{entry}/peptide.pdb")),
            }
    if data["a"]["receptor_fasta"] != data["b"]["receptor_fasta"]:
        raise ValueError("selected entries do not have identical receptor FASTA")
    res_a = receptor_residues(data["a"]["receptor_lines"])
    res_b = receptor_residues(data["b"]["receptor_lines"])
    seq_a = "".join(AA3.get(residue["name"], "X") for residue in res_a)
    seq_b = "".join(AA3.get(residue["name"], "X") for residue in res_b)
    if seq_a != seq_b or len(res_a) < 20:
        raise ValueError("selected receptor coordinate sequences do not match")
    ca_a = np.asarray([coordinate(residue["atoms"]["CA"]) for residue in res_a])
    ca_b = np.asarray([coordinate(residue["atoms"]["CA"]) for residue in res_b])
    rotation, translation = kabsch(ca_b, ca_a)
    aligned_ca_b = ca_b @ rotation + translation
    receptor_rmsd = float(np.sqrt(np.mean(np.sum((ca_a - aligned_ca_b) ** 2, axis=1))))

    common_anchor = midpoint_anchor(res_a, res_b, rotation, translation)
    holo_a = "\n".join(data["a"]["receptor_lines"]) + "\nEND\n"
    holo_b = transformed_pdb(data["b"]["receptor_lines"], rotation, translation)
    peptide_a = "\n".join(data["a"]["peptide_lines"]) + "\nEND\n"
    peptide_b = transformed_pdb(data["b"]["peptide_lines"], rotation, translation)
    files = {
        "common_anchor.pdb": common_anchor, "holo_a.pdb": holo_a, "holo_b.pdb": holo_b,
        "peptide_a.pdb": peptide_a, "peptide_b.pdb": peptide_b,
    }
    for name, content in files.items():
        (result_dir / name).write_text(content, encoding="utf-8")
    atoms_a = pdb_to_sdf(peptide_a, result_dir / "peptide_a.sdf")
    atoms_b = pdb_to_sdf(peptide_b, result_dir / "peptide_b.sdf")

    esm_dir = result_dir / "esm2_output"
    esm_dir.mkdir()
    torch.save(torch.zeros((len(res_a), 1280), dtype=torch.float32), esm_dir / "common_anchor.pdb_chain_0.pt")
    with (result_dir / "inference_inputs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["protein_path", "ligand", "complex_name"])
        writer.writeheader()
        for label in ("a", "b"):
            writer.writerow({
                "protein_path": str(result_dir / "common_anchor.pdb"),
                "ligand": str(result_dir / f"peptide_{label}.sdf"),
                "complex_name": f"joint_state_{label}",
            })
    for label in ("a", "b"):
        case = result_dir / f"pepflow_{label}"
        case.mkdir()
        (case / "pocket.pdb").write_text(common_anchor, encoding="utf-8")
        (case / "receptor.pdb").write_text(common_anchor, encoding="utf-8")
        (case / "peptide.pdb").write_text(peptide_a if label == "a" else peptide_b, encoding="utf-8")

    finished = datetime.now().astimezone()
    output_hashes = {
        str(path.relative_to(result_dir)): sha256(path)
        for path in result_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "status": "passed", "exit_code": 0, "branch": git("branch", "--show-current"),
        "commit": commit, "commit_message": message.strip(), "command": " ".join(sys.argv),
        "python": sys.executable, "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "archive": str(archive_path), "archive_sha256": sha256(archive_path),
        "entries": [args.entry_a, args.entry_b],
        "peptides": [data["a"]["peptide_fasta"], data["b"]["peptide_fasta"]],
        "receptor_coordinate_residues": len(res_a), "receptor_alignment_ca_rmsd": receptor_rmsd,
        "common_anchor_definition": "per-atom midpoint after aligning state B receptor to state A",
        "zero_esm_control": True, "peptide_atoms": [atoms_a, atoms_b],
        "output_hashes": output_hashes, "elapsed_seconds": (finished - started).total_seconds(),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Joint peptide states</title></head><body>"
        f"<h1>{args.entry_a} / {args.entry_b}</h1><p>Peptides: {manifest['peptides']}</p>"
        f"<p>Mapped receptor residues: {len(res_a)}; aligned CA RMSD: {receptor_rmsd:.4f} A</p>"
        "<p>The common anchor is the aligned per-atom midpoint; native peptide coordinates are retained.</p>"
        "</body></html>\n", encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
