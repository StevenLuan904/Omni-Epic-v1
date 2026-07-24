#!/usr/bin/env python3
"""Export a reproducible same-receptor-anchor DynamicBind inference case."""

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
    REPO_ROOT / "dynamicbind" / "test" / "results" / "day2-same-anchor-inputs"
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


def metadata_row(metadata, uid, entry):
    rows = metadata[
        (metadata["uid"].astype(str) == str(uid))
        & (metadata["entryName"].astype(str) == str(entry))
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one metadata row for {uid}/{entry}, found {len(rows)}")
    return rows.iloc[0]


def choose_control(candidates, target_row):
    target_size = (
        float(target_row["heavy_atoms_a"]) + float(target_row["heavy_atoms_b"])
    ) / 2.0
    options = []
    for _, row in candidates.iterrows():
        if str(row["uid"]) == str(target_row["uid"]):
            continue
        for suffix in ("a", "b"):
            options.append((
                abs(float(row[f"heavy_atoms_{suffix}"]) - target_size),
                str(row["uid"]),
                str(row[f"entry_{suffix}"]),
                str(row[f"ligand_{suffix}"]),
            ))
    if not options:
        raise ValueError("candidate table does not contain an alternate-UID control ligand")
    return min(options)


def copy_member(archive, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--structure-zip", required=True)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--candidate-rank", type=int, default=1)
    args = parser.parse_args()

    require_clean_commit()
    metadata_path = Path(args.metadata_csv).resolve()
    archive_path = Path(args.structure_zip).resolve()
    candidates_path = Path(args.candidates_csv).resolve()
    for path in (metadata_path, archive_path, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates = pd.read_csv(candidates_path)
    if not 1 <= args.candidate_rank <= len(candidates):
        raise ValueError(
            f"candidate rank must be between 1 and {len(candidates)}, "
            f"got {args.candidate_rank}"
        )
    target = candidates.iloc[args.candidate_rank - 1]
    _, control_uid, control_entry, control_ligand = choose_control(candidates, target)

    metadata = pd.read_csv(metadata_path)
    row_a = metadata_row(metadata, target["uid"], target["entry_a"])
    row_b = metadata_row(metadata, target["uid"], target["entry_b"])
    row_control = metadata_row(metadata, control_uid, control_entry)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    assets_dir = result_dir / "assets"
    assets_dir.mkdir(parents=True)

    extracted = {}
    with zipfile.ZipFile(archive_path) as archive:
        member_index = build_member_index(archive)
        specifications = {
            "common_anchor": (row_a, "af2File", f"common_anchor_{target['uid']}.pdb"),
            "holo_a": (row_a, "pdbFile", f"holo_A_{target['entry_a']}.pdb"),
            "holo_b": (row_b, "pdbFile", f"holo_B_{target['entry_b']}.pdb"),
            "ligand_a": (row_a, "ligandFile", f"ligand_A_{target['ligand_a']}.sdf"),
            "ligand_b": (row_b, "ligandFile", f"ligand_B_{target['ligand_b']}.sdf"),
            "control_ligand": (
                row_control,
                "ligandFile",
                f"ligand_control_{control_uid}_{control_ligand}.sdf",
            ),
        }
        for label, (row, column, filename) in specifications.items():
            member = archive_member(member_index, row["uid"], row[column])
            destination = assets_dir / filename
            copy_member(archive, member, destination)
            extracted[label] = {
                "archive_member": member,
                "path": str(destination.resolve()),
                "sha256": sha256(destination),
            }

    anchor = extracted["common_anchor"]["path"]
    inference_rows = [
        {
            "name": "ligand_A",
            "protein_path": anchor,
            "ligand": extracted["ligand_a"]["path"],
            "condition": "ligand_A",
            "target_uid": target["uid"],
            "holo_endpoint": "A",
        },
        {
            "name": "ligand_B",
            "protein_path": anchor,
            "ligand": extracted["ligand_b"]["path"],
            "condition": "ligand_B",
            "target_uid": target["uid"],
            "holo_endpoint": "B",
        },
        {
            "name": "shuffled_control",
            "protein_path": anchor,
            "ligand": extracted["control_ligand"]["path"],
            "condition": "shuffled_control",
            "target_uid": target["uid"],
            "holo_endpoint": "none",
        },
    ]
    inference_table = pd.DataFrame(inference_rows)
    if inference_table["protein_path"].nunique() != 1:
        raise AssertionError("all conditions must share exactly one receptor anchor")
    inference_table.to_csv(result_dir / "inference_inputs.csv", index=False)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "candidate_rank": args.candidate_rank,
        "target_uid": str(target["uid"]),
        "holo_a": str(target["entry_a"]),
        "ligand_a": str(target["ligand_a"]),
        "holo_b": str(target["entry_b"]),
        "ligand_b": str(target["ligand_b"]),
        "control_uid": control_uid,
        "control_entry": control_entry,
        "control_ligand": control_ligand,
        "inputs": {
            "metadata_csv": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
            "structure_zip": {"path": str(archive_path), "sha256": sha256(archive_path)},
            "candidates_csv": {
                "path": str(candidates_path),
                "sha256": sha256(candidates_path),
            },
        },
        "extracted": extracted,
        "same_anchor_check": {
            "unique_protein_paths": int(inference_table["protein_path"].nunique()),
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
    print(f"Same-anchor inputs ready: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
