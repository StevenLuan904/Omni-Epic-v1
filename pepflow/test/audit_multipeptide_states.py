#!/usr/bin/env python3
"""Find same-receptor, distinct-peptide holo pairs in PepMerge_release.zip."""

import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
from itertools import combinations
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "multipeptide-state-audit"
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
    print("ERROR: audit must run from a committed worktree.", file=sys.stderr)
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


def fasta_sequence(raw):
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return "".join(line.strip().upper() for line in lines if not line.startswith(">"))


def pdb_records(raw):
    residues = []
    residue_index = {}
    heavy = []
    for line in io.StringIO(raw.decode("utf-8", errors="replace")):
        if not line.startswith(("ATOM  ", "HETATM")) or line[16] not in (" ", "A"):
            continue
        element = line[76:78].strip().upper()
        if element == "H":
            continue
        try:
            coord = np.array([
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            ], dtype=np.float64)
        except ValueError:
            continue
        heavy.append(coord)
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        key = (line[21], line[22:26].strip(), line[26])
        if key in residue_index:
            continue
        residue_index[key] = len(residues)
        residues.append((AA3.get(line[17:20].strip(), "X"), coord))
    return residues, np.asarray(heavy, dtype=np.float64)


def kabsch(mobile, reference):
    mobile_center = mobile.mean(0)
    reference_center = reference.mean(0)
    covariance = (mobile - mobile_center).T @ (reference - reference_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = reference_center - mobile_center @ rotation
    return rotation, translation


def pair_metrics(archive, a, b, pocket_cutoff):
    rec_a, _ = pdb_records(archive.read(f"{a['entry']}/receptor.pdb"))
    rec_b, _ = pdb_records(archive.read(f"{b['entry']}/receptor.pdb"))
    _, pep_a = pdb_records(archive.read(f"{a['entry']}/peptide.pdb"))
    _, pep_b = pdb_records(archive.read(f"{b['entry']}/peptide.pdb"))
    seq_a = "".join(item[0] for item in rec_a)
    seq_b = "".join(item[0] for item in rec_b)
    if len(rec_a) < 20 or seq_a != seq_b or not len(pep_a) or not len(pep_b):
        return None
    xyz_a = np.asarray([item[1] for item in rec_a])
    xyz_b = np.asarray([item[1] for item in rec_b])
    rotation, translation = kabsch(xyz_b, xyz_a)
    aligned_b = xyz_b @ rotation + translation
    aligned_pep_b = pep_b @ rotation + translation
    all_rmsd = float(np.sqrt(np.mean(np.sum((xyz_a - aligned_b) ** 2, axis=1))))
    near_a = np.min(np.linalg.norm(xyz_a[:, None] - pep_a[None], axis=2), axis=1) <= pocket_cutoff
    near_b = np.min(np.linalg.norm(aligned_b[:, None] - aligned_pep_b[None], axis=2), axis=1) <= pocket_cutoff
    pocket = near_a | near_b
    if int(pocket.sum()) < 3:
        return None
    pocket_rmsd = float(np.sqrt(np.mean(np.sum((xyz_a[pocket] - aligned_b[pocket]) ** 2, axis=1))))
    return {
        "mapped_ca": len(rec_a),
        "pocket_ca": int(pocket.sum()),
        "receptor_ca_rmsd": all_rmsd,
        "pocket_ca_rmsd": pocket_rmsd,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--pocket-cutoff", type=float, default=10.0)
    parser.add_argument("--minimum-pocket-rmsd", type=float, default=0.25)
    args = parser.parse_args()
    require_clean_commit()
    archive_path = Path(args.archive).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / stamp
    result_dir.mkdir(parents=True)
    commit = git("rev-parse", "HEAD")
    message = git("log", "-1", "--format=%B")
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")

    rows = []
    entries = []
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        roots = sorted({name.split("/", 1)[0] for name in names if "/" in name})
        for root in roots:
            required = [f"{root}/{name}" for name in (
                "receptor.fasta", "peptide.fasta", "receptor.pdb", "peptide.pdb"
            )]
            if not all(name in names for name in required):
                continue
            rec_seq = fasta_sequence(archive.read(required[0]))
            pep_seq = fasta_sequence(archive.read(required[1]))
            if rec_seq and 3 <= len(pep_seq) <= 25:
                entries.append({"entry": root, "receptor_sequence": rec_seq, "peptide_sequence": pep_seq})
        groups = {}
        for entry in entries:
            groups.setdefault(entry["receptor_sequence"], []).append(entry)
        for group in groups.values():
            if len({entry["peptide_sequence"] for entry in group}) < 2:
                continue
            for a, b in combinations(group, 2):
                if a["peptide_sequence"] == b["peptide_sequence"]:
                    continue
                metrics = pair_metrics(archive, a, b, args.pocket_cutoff)
                if metrics is None:
                    continue
                rows.append({
                    "entry_a": a["entry"], "entry_b": b["entry"],
                    "peptide_a": a["peptide_sequence"], "peptide_b": b["peptide_sequence"],
                    "peptide_length_a": len(a["peptide_sequence"]),
                    "peptide_length_b": len(b["peptide_sequence"]),
                    **metrics,
                    "passes_minimum": metrics["pocket_ca_rmsd"] >= args.minimum_pocket_rmsd,
                })
    rows.sort(key=lambda row: (-row["pocket_ca_rmsd"], row["entry_a"], row["entry_b"]))
    fieldnames = list(rows[0]) if rows else [
        "entry_a", "entry_b", "peptide_a", "peptide_b", "peptide_length_a",
        "peptide_length_b", "mapped_ca", "pocket_ca", "receptor_ca_rmsd",
        "pocket_ca_rmsd", "passes_minimum",
    ]
    csv_path = result_dir / "candidate_pairs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finished = datetime.now().astimezone()
    metadata = {
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "status": "passed", "exit_code": 0, "branch": git("branch", "--show-current"),
        "commit": commit, "commit_message": message.strip(), "command": " ".join(sys.argv),
        "python": sys.executable, "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "archive": str(archive_path), "archive_sha256": sha256(archive_path),
        "pocket_cutoff": args.pocket_cutoff, "minimum_pocket_rmsd": args.minimum_pocket_rmsd,
        "complete_entries": len(entries), "exact_receptor_groups": len(groups),
        "candidate_pairs": len(rows), "passing_pairs": sum(row["passes_minimum"] for row in rows),
        "candidate_csv": str(csv_path), "candidate_csv_sha256": sha256(csv_path),
        "elapsed_seconds": (finished - started).total_seconds(),
    }
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    top = rows[:20]
    html_rows = "".join(
        "<tr>" + "".join(f"<td>{row[key]}</td>" for key in fieldnames) + "</tr>" for row in top
    )
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Multi-peptide state audit</title>"
        "<style>table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:4px}</style>"
        "</head><body><h1>Same-receptor distinct-peptide candidates</h1><table><tr>"
        + "".join(f"<th>{key}</th>" for key in fieldnames) + "</tr>" + html_rows
        + "</table></body></html>\n", encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
