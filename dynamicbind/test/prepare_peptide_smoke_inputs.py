#!/usr/bin/env python3
"""Export auditable native short-peptide inputs from the DynamicBind dataset."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "day3-peptide-inputs"
)
DEFAULT_ENTRIES = "1jet,6m8w,1oai"


def git(*args, check=True):
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def require_clean_commit():
    status = git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return
    print("ERROR: input export must run from a committed worktree.", file=sys.stderr)
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


def build_member_index(archive):
    index = {}
    for name in archive.namelist():
        parts = name.rstrip("/").split("/")
        if len(parts) >= 2 and not name.endswith("/"):
            index[(parts[-2], parts[-1])] = name
    return index


def archive_member(index, uid, source_path):
    basename = Path(str(source_path).replace(chr(92), "/")).name
    key = (str(uid), basename)
    if key not in index:
        raise FileNotFoundError(f"archive member ending in /{uid}/{basename}")
    return index[key]


def copy_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def peptide_length(label):
    text = str(label)
    if not text.endswith("-mer"):
        raise ValueError(f"not an n-mer ligand label: {label}")
    return int(text[:-4])


def ligand_audit(path, expected_residues):
    molecule = Chem.MolFromMolFile(str(path), removeHs=False, sanitize=True)
    if molecule is None:
        raise ValueError(f"RDKit failed to parse {path}")
    amide_bonds = 0
    for bond in molecule.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        carbon, nitrogen = bond.GetBeginAtom(), bond.GetEndAtom()
        if carbon.GetAtomicNum() == 7:
            carbon, nitrogen = nitrogen, carbon
        if carbon.GetAtomicNum() != 6 or nitrogen.GetAtomicNum() != 7:
            continue
        carbonyl_oxygen = any(
            neighbor.GetAtomicNum() == 8
            and molecule.GetBondBetweenAtoms(
                carbon.GetIdx(), neighbor.GetIdx()
            ).GetBondType() == Chem.BondType.DOUBLE
            for neighbor in carbon.GetNeighbors()
        )
        amide_bonds += int(carbonyl_oxygen)
    record = {
        "expected_residues": expected_residues,
        "atoms": molecule.GetNumAtoms(),
        "heavy_atoms": sum(
            atom.GetAtomicNum() > 1 for atom in molecule.GetAtoms()
        ),
        "bonds": molecule.GetNumBonds(),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
        "rings": rdMolDescriptors.CalcNumRings(molecule),
        "fragments": len(Chem.GetMolFrags(molecule)),
        "amide_bonds": amide_bonds,
        "formal_charge": Chem.GetFormalCharge(molecule),
        "smiles": Chem.MolToSmiles(molecule),
    }
    record["peptide_like"] = (
        record["fragments"] == 1 and amide_bonds >= expected_residues - 1
    )
    if not record["peptide_like"]:
        raise ValueError(
            f"{path} failed peptide topology check: "
            f"{amide_bonds} amides for {expected_residues}-mer"
        )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--structure-zip", required=True)
    parser.add_argument("--entries", default=DEFAULT_ENTRIES)
    args = parser.parse_args()

    require_clean_commit()
    metadata_path = Path(args.metadata_csv).resolve()
    archive_path = Path(args.structure_zip).resolve()
    for path in (metadata_path, archive_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    entries = [value.strip().lower() for value in args.entries.split(",") if value.strip()]
    if not entries:
        raise ValueError("at least one PDB entry is required")
    if len(entries) != len(set(entries)):
        raise ValueError("PDB entries must be unique")

    metadata = pd.read_csv(metadata_path)
    selected = []
    for entry in entries:
        rows = metadata[metadata["pdb"].astype(str).str.lower() == entry]
        if len(rows) != 1:
            raise ValueError(f"expected one metadata row for {entry}, found {len(rows)}")
        selected.append(rows.iloc[0])

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    assets_dir = result_dir / "assets"
    assets_dir.mkdir(parents=True)
    inference_rows = []
    systems = []
    with zipfile.ZipFile(archive_path) as archive:
        member_index = build_member_index(archive)
        for row in selected:
            uid = str(row["uid"])
            entry = str(row["pdb"]).lower()
            length = peptide_length(row["ligand"])
            extracted = {}
            for label, column, filename in (
                ("anchor", "af2File", f"anchor_{uid}_{entry}.pdb"),
                ("holo", "pdbFile", f"holo_{uid}_{entry}.pdb"),
                ("ligand", "ligandFile", f"ligand_{uid}_{entry}.sdf"),
            ):
                member = archive_member(member_index, uid, row[column])
                destination = assets_dir / filename
                copy_member(archive, member, destination)
                extracted[label] = {
                    "archive_member": member,
                    "path": str(destination.resolve()),
                    "sha256": sha256(destination),
                }
            audit = ligand_audit(extracted["ligand"]["path"], length)
            inference_rows.append({
                "name": f"{uid}_{entry}_{length}mer",
                "protein_path": extracted["anchor"]["path"],
                "ligand": extracted["ligand"]["path"],
                "condition": f"native_{length}mer",
                "target_uid": uid,
                "holo_endpoint": extracted["holo"]["path"],
            })
            systems.append({
                "uid": uid,
                "entry": entry,
                "ligand_label": str(row["ligand"]),
                "resolution": str(row["resolution"]),
                "protein_rmsd": float(row["protein_rmsd"]),
                "pocket_rmsd": float(row["pocket_rmsd"]),
                "clash_score": float(row["clashScore"]),
                "extracted": extracted,
                "ligand_audit": audit,
            })

    inference_path = result_dir / "inference_inputs.csv"
    pd.DataFrame(inference_rows).to_csv(inference_path, index=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "selection": {
            "entries": entries,
            "criteria": (
                "DynamicBind native 3-9-mer; X-ray resolution <=1.2 A; "
                "filled_num=0; low clash; one connected peptide-like ligand"
            ),
        },
        "inputs": {
            "metadata_csv": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "structure_zip": {"path": str(archive_path), "sha256": sha256(archive_path)},
        },
        "inference_csv": {
            "path": str(inference_path.resolve()),
            "sha256": sha256(inference_path),
            "rows": len(inference_rows),
            "unique_protein_paths": len(
                {row["protein_path"] for row in inference_rows}
            ),
        },
        "systems": systems,
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (result_dir / "commit.txt").write_text(
        f"{manifest['commit']}\n{manifest['commit_message'].strip()}\n",
        encoding="utf-8",
    )
    print(f"Peptide smoke inputs ready: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
