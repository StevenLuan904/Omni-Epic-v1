#!/usr/bin/env python3
"""Export the exact c-Met examples shown in DynamicBind Figure 3."""

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


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "figure3-cmet-inputs"
)
UID = "P08581"
CASES = (
    {
        "entry": "6ubw_84S_A",
        "pdb": "6UBW",
        "ligand": "84S",
        "state": "DFG-in",
        "paper_ligand_rmsd": 0.49,
        "paper_predicted_pocket_rmsd": 1.97,
        "paper_af_pocket_rmsd": 9.44,
    },
    {
        "entry": "7v3s_5I9_A",
        "pdb": "7V3S",
        "ligand": "5I9",
        "state": "DFG-out",
        "paper_ligand_rmsd": 0.51,
        "paper_predicted_pocket_rmsd": 1.19,
        "paper_af_pocket_rmsd": 6.02,
    },
)


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


def basename(value):
    return Path(str(value).replace("\\", "/")).name


def copy_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--structure-zip", required=True)
    args = parser.parse_args()

    require_clean_commit()
    metadata_path = Path(args.metadata_csv).resolve()
    archive_path = Path(args.structure_zip).resolve()
    for path in (metadata_path, archive_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = pd.read_csv(metadata_path)
    rows = {}
    for case in CASES:
        selected = metadata[
            (metadata["uid"].astype(str) == UID)
            & (metadata["entryName"].astype(str) == case["entry"])
        ]
        if len(selected) != 1:
            raise ValueError(
                f"expected one metadata row for {UID}/{case['entry']}, "
                f"found {len(selected)}"
            )
        rows[case["entry"]] = selected.iloc[0]

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    assets_dir = result_dir / "assets"
    assets_dir.mkdir(parents=True)

    extracted = {}
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for case in CASES:
            row = rows[case["entry"]]
            for kind, column, suffix in (
                ("holo", "pdbFile", ".pdb"),
                ("ligand", "ligandFile", ".sdf"),
                ("af2", "af2File", ".pdb"),
            ):
                source_name = basename(row[column])
                member = f"MDT/{UID}/{source_name}"
                if member not in members:
                    raise FileNotFoundError(member)
                destination = (
                    assets_dir
                    / f"{case['entry']}_{kind}{suffix}"
                )
                copy_member(archive, member, destination)
                extracted[f"{case['entry']}_{kind}"] = {
                    "archive_member": member,
                    "path": str(destination.resolve()),
                    "sha256": sha256(destination),
                }

    # Figure 3 must test ligand-conditioned state selection from one receptor.
    # Use the 6UBW-aligned AF2 coordinates for both ligands and retain the
    # second aligned AF2 file above for coordinate-frame auditing.
    common_anchor = extracted[f"{CASES[0]['entry']}_af2"]["path"]
    inference_rows = []
    for case in CASES:
        inference_rows.append({
            "name": case["entry"],
            "protein_path": common_anchor,
            "ligand": extracted[f"{case['entry']}_ligand"]["path"],
            "condition": case["state"],
            "target_uid": UID,
            "holo_endpoint": case["entry"],
        })
    inference_table = pd.DataFrame(inference_rows)
    if inference_table["protein_path"].nunique() != 1:
        raise AssertionError("Figure 3 conditions do not share one AF2 anchor")
    inference_table.to_csv(result_dir / "inference_inputs.csv", index=False)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "figure": "DynamicBind Figure 3",
        "target_uid": UID,
        "common_anchor": common_anchor,
        "common_anchor_entry": CASES[0]["entry"],
        "cases": list(CASES),
        "paper_reference": {
            "doi": "10.1038/s41467-024-45461-2",
            "note": (
                "Paper values are acceptance references, not values computed "
                "by this exporter."
            ),
        },
        "inputs": {
            "metadata_csv": {
                "path": str(metadata_path),
                "sha256": sha256(metadata_path),
            },
            "structure_zip": {
                "path": str(archive_path),
                "sha256": sha256(archive_path),
            },
        },
        "extracted": extracted,
        "same_anchor_check": {
            "unique_protein_paths": int(
                inference_table["protein_path"].nunique()
            ),
            "conditions": inference_table["condition"].tolist(),
        },
    }
    (result_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (result_dir / "commit.txt").write_text(
        f"{manifest['commit']}\n{manifest['commit_message'].strip()}\n",
        encoding="utf-8",
    )
    print(f"Figure 3 c-Met inputs ready: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
