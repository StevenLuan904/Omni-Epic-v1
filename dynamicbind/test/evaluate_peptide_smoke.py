#!/usr/bin/env python3
"""Evaluate native-peptide DynamicBind inference and preserve auditable metrics."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_multistate_dataset import (
    kabsch,
    parse_protein,
    parse_sdf,
    pocket_residues,
    transform,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "day3-peptide-evaluation"
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
    print("ERROR: evaluation must run from a committed worktree.", file=sys.stderr)
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


def read_protein(path):
    path = Path(path)
    return parse_protein(
        path.read_text(encoding="utf-8", errors="replace"),
        path.suffix,
        path.stem,
    )


def read_ligand(path):
    path = Path(path)
    return parse_sdf(path.read_text(encoding="utf-8", errors="replace"))


def shared_keys(reference, mobile):
    return sorted(
        key
        for key in set(reference["residues"]) & set(mobile["residues"])
        if reference["residues"][key]["resname"]
        == mobile["residues"][key]["resname"]
    )


def align_to_anchor(anchor, mobile):
    keys = shared_keys(anchor, mobile)
    if len(keys) < 20:
        raise ValueError(f"only {len(keys)} shared same-name residues")
    reference_coords = np.asarray(
        [anchor["residues"][key]["atoms"]["CA"] for key in keys]
    )
    mobile_coords = np.asarray(
        [mobile["residues"][key]["atoms"]["CA"] for key in keys]
    )
    alignment = kabsch(reference_coords, mobile_coords)
    aligned = {
        key: transform(
            np.asarray([residue["atoms"]["CA"]]), alignment
        )[0]
        for key, residue in mobile["residues"].items()
    }
    rmsd = float(np.sqrt(np.mean(np.sum(
        (
            reference_coords
            - np.asarray([aligned[key] for key in keys])
        ) ** 2,
        axis=1,
    ))))
    return aligned, alignment, rmsd


def rmsd(first, second):
    if first.shape != second.shape:
        raise ValueError(f"coordinate shape mismatch: {first.shape} vs {second.shape}")
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def plot_metrics(table, path):
    labels = [f"{row.entry}\n{row.peptide_length}-mer" for row in table.itertuples()]
    x = np.arange(len(labels))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(
        x - width / 2,
        table["ligand_pose_rmsd"],
        width,
        label="Ligand pose RMSD",
        color="#d1495b",
    )
    axis.bar(
        x + width / 2,
        table["receptor_pocket_ca_rmsd_to_holo"],
        width,
        label="Receptor pocket Cα RMSD",
        color="#2b6cb0",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel("RMSD (Å)")
    axis.set_title("DynamicBind native short-peptide smoke")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--inference-dir", required=True)
    args = parser.parse_args()

    require_clean_commit()
    started_at = datetime.now().astimezone().isoformat()
    case_dir = Path(args.case_dir).resolve()
    inference_dir = Path(args.inference_dir).resolve()
    case_manifest_path = case_dir / "manifest.json"
    inference_metadata_path = inference_dir / "metadata.json"
    case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    inference_metadata = json.loads(
        inference_metadata_path.read_text(encoding="utf-8")
    )
    if inference_metadata["status"] != "passed":
        raise ValueError("inference metadata is not passed")
    systems = case_manifest["systems"]
    records = []
    for run in inference_metadata["runs"]:
        for output in run["validation"]["conditions"]:
            index = int(output["index"])
            system = systems[index]
            anchor = read_protein(system["extracted"]["anchor"]["path"])
            holo = read_protein(system["extracted"]["holo"]["path"])
            prediction = read_protein(output["receptor"])
            native_ligand = read_ligand(system["extracted"]["ligand"]["path"])
            predicted_ligand = read_ligand(output["ligand"])
            if native_ligand["elements"] != predicted_ligand["elements"]:
                raise ValueError(
                    f"ligand atom ordering changed for {system['uid']}/{system['entry']}"
                )

            anchor_ca = {
                key: residue["atoms"]["CA"]
                for key, residue in anchor["residues"].items()
            }
            holo_ca, holo_alignment, holo_global = align_to_anchor(anchor, holo)
            pred_ca, pred_alignment, pred_global = align_to_anchor(anchor, prediction)
            pocket = pocket_residues(holo, native_ligand["coords"], 5.0)
            keys = sorted(
                pocket
                & set(anchor_ca)
                & set(holo_ca)
                & set(pred_ca)
            )
            if len(keys) < 3:
                raise ValueError(
                    f"only {len(keys)} shared pocket residues for "
                    f"{system['uid']}/{system['entry']}"
                )
            native_coords = transform(native_ligand["coords"], holo_alignment)
            predicted_coords = transform(
                predicted_ligand["coords"], pred_alignment
            )
            conformer_alignment = kabsch(native_coords, predicted_coords)
            records.append({
                "seed": int(run["seed"]),
                "uid": system["uid"],
                "entry": system["entry"],
                "peptide_length": system["ligand_audit"]["expected_residues"],
                "heavy_atoms": system["ligand_audit"]["heavy_atoms"],
                "rotatable_bonds": system["ligand_audit"]["rotatable_bonds"],
                "shared_pocket_residues": len(keys),
                "ligand_pose_rmsd": rmsd(predicted_coords, native_coords),
                "ligand_conformer_rmsd": rmsd(
                    transform(predicted_coords, conformer_alignment),
                    native_coords,
                ),
                "receptor_global_ca_rmsd_to_anchor": pred_global,
                "holo_global_ca_rmsd_to_anchor": holo_global,
                "receptor_pocket_ca_rmsd_to_anchor": rmsd(
                    np.asarray([pred_ca[key] for key in keys]),
                    np.asarray([anchor_ca[key] for key in keys]),
                ),
                "receptor_pocket_ca_rmsd_to_holo": rmsd(
                    np.asarray([pred_ca[key] for key in keys]),
                    np.asarray([holo_ca[key] for key in keys]),
                ),
                "receptor_sha256": sha256(output["receptor"]),
                "ligand_sha256": sha256(output["ligand"]),
            })

    table = pd.DataFrame(records)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    table.to_csv(result_dir / "peptide_metrics.csv", index=False)
    plot_metrics(table, result_dir / "peptide_metrics.png")
    summary = {
        "started_at": started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "exit_code": 0,
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python": sys.executable,
        "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "source_inference_commit": inference_metadata["commit"],
        "case_manifest_sha256": sha256(case_manifest_path),
        "inference_metadata_sha256": sha256(inference_metadata_path),
        "systems": len(table),
        "graph_success_fraction": float(len(table) / len(systems)),
        "inference_success_fraction": float(len(table) / len(systems)),
        "mean_ligand_pose_rmsd": float(table["ligand_pose_rmsd"].mean()),
        "mean_receptor_pocket_ca_rmsd_to_holo": float(
            table["receptor_pocket_ca_rmsd_to_holo"].mean()
        ),
        "per_system": records,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (result_dir / "commit.txt").write_text(
        f"{summary['commit']}\n{summary['commit_message'].strip()}\n",
        encoding="utf-8",
    )
    print(f"Peptide evaluation passed: {result_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
