#!/usr/bin/env python3
"""Evaluate DynamicBind predictions against two holo endpoints in one anchor frame."""

import argparse
from datetime import datetime
import hashlib
import json
import math
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
    REPO_ROOT / "dynamicbind" / "test" / "results" / "day2-state-evaluation"
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


def same_residue_keys(first, second):
    shared = set(first["residues"]) & set(second["residues"])
    return sorted(
        key for key in shared
        if first["residues"][key]["resname"] == second["residues"][key]["resname"]
    )


def align_protein(reference, mobile):
    common = same_residue_keys(reference, mobile)
    if len(common) < 20:
        raise ValueError(f"only {len(common)} common same-name Cα residues")
    reference_ca = np.asarray(
        [reference["residues"][key]["atoms"]["CA"] for key in common]
    )
    mobile_ca = np.asarray(
        [mobile["residues"][key]["atoms"]["CA"] for key in common]
    )
    alignment = kabsch(reference_ca, mobile_ca)
    aligned = {
        key: transform(
            np.asarray([residue["atoms"]["CA"]]), alignment
        )[0]
        for key, residue in mobile["residues"].items()
    }
    global_rmsd = float(np.sqrt(np.mean(np.sum(
        (
            reference_ca
            - np.asarray([aligned[key] for key in common])
        ) ** 2,
        axis=1,
    ))))
    return aligned, alignment, global_rmsd


def ca_map(protein):
    return {
        key: residue["atoms"]["CA"]
        for key, residue in protein["residues"].items()
    }


def ca_rmsd(first, second, keys):
    first_coords = np.asarray([first[key] for key in keys])
    second_coords = np.asarray([second[key] for key in keys])
    return float(np.sqrt(np.mean(np.sum((first_coords - second_coords) ** 2, axis=1))))


def shared_pocket_keys(anchor, holo_a, holo_b, prediction, pocket_union):
    keys = (
        set(anchor["residues"])
        & set(holo_a["residues"])
        & set(holo_b["residues"])
        & set(prediction["residues"])
        & pocket_union
    )
    return sorted(
        key for key in keys
        if len({
            anchor["residues"][key]["resname"],
            holo_a["residues"][key]["resname"],
            holo_b["residues"][key]["resname"],
            prediction["residues"][key]["resname"],
        }) == 1
    )


def plot_results(table, path):
    colors = {
        "ligand_A": "#d1495b",
        "ligand_B": "#2b6cb0",
        "shuffled_control": "#777777",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for condition, group in table.groupby("condition", sort=False):
        axes[0].scatter(
            group["rmsd_to_holo_a"],
            group["rmsd_to_holo_b"],
            label=condition,
            s=70,
            alpha=0.85,
            color=colors.get(condition),
        )
    limit = max(
        float(table["rmsd_to_holo_a"].max()),
        float(table["rmsd_to_holo_b"].max()),
    ) * 1.08
    axes[0].plot([0, limit], [0, limit], "--", color="#555555", linewidth=1)
    axes[0].set_xlim(0, limit)
    axes[0].set_ylim(0, limit)
    axes[0].set_xlabel("Pocket Cα RMSD to holo A (Å)")
    axes[0].set_ylabel("Pocket Cα RMSD to holo B (Å)")
    axes[0].set_title("Endpoint proximity in common anchor frame")
    axes[0].legend(frameon=False)

    directional = table[table["condition"].isin(["ligand_A", "ligand_B"])]
    labels, values = [], []
    for condition, group in directional.groupby("condition", sort=False):
        labels.append(condition)
        values.append(group["correct_state_margin"].to_numpy())
    axes[1].boxplot(values, labels=labels, showmeans=True)
    axes[1].axhline(0, linestyle="--", color="#555555", linewidth=1)
    axes[1].set_ylabel("Correct-state margin (Å)")
    axes[1].set_title("Positive values favor the matched holo")
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

    extracted = case_manifest["extracted"]
    anchor_path = Path(extracted["common_anchor"]["path"])
    holo_a_path = Path(extracted["holo_a"]["path"])
    holo_b_path = Path(extracted["holo_b"]["path"])
    ligand_a_path = Path(extracted["ligand_a"]["path"])
    ligand_b_path = Path(extracted["ligand_b"]["path"])
    anchor = read_protein(anchor_path)
    holo_a = read_protein(holo_a_path)
    holo_b = read_protein(holo_b_path)
    ligand_a = parse_sdf(ligand_a_path.read_text(encoding="utf-8"))
    ligand_b = parse_sdf(ligand_b_path.read_text(encoding="utf-8"))
    pocket_union = (
        pocket_residues(holo_a, ligand_a["coords"], 5.0)
        | pocket_residues(holo_b, ligand_b["coords"], 5.0)
    )

    anchor_ca = ca_map(anchor)
    holo_a_ca, _, holo_a_global_rmsd = align_protein(anchor, holo_a)
    holo_b_ca, _, holo_b_global_rmsd = align_protein(anchor, holo_b)
    conditions = inference_metadata["inputs"]["conditions"]
    records = []
    endpoint_rmsds = []
    for run in inference_metadata["runs"]:
        seed = int(run["seed"])
        for output in run["validation"]["conditions"]:
            index = int(output["index"])
            condition = conditions[index]
            prediction_path = Path(output["receptor"])
            prediction = read_protein(prediction_path)
            prediction_ca, _, prediction_global_rmsd = align_protein(
                anchor, prediction
            )
            keys = shared_pocket_keys(
                anchor, holo_a, holo_b, prediction, pocket_union
            )
            if len(keys) < 3:
                raise ValueError(
                    f"only {len(keys)} shared pocket residues for {prediction_path}"
                )
            rmsd_to_a = ca_rmsd(prediction_ca, holo_a_ca, keys)
            rmsd_to_b = ca_rmsd(prediction_ca, holo_b_ca, keys)
            rmsd_to_anchor = ca_rmsd(prediction_ca, anchor_ca, keys)
            endpoint_rmsds.append(ca_rmsd(holo_a_ca, holo_b_ca, keys))
            if condition == "ligand_A":
                correct_margin = rmsd_to_b - rmsd_to_a
                selected_correct = rmsd_to_a < rmsd_to_b
            elif condition == "ligand_B":
                correct_margin = rmsd_to_a - rmsd_to_b
                selected_correct = rmsd_to_b < rmsd_to_a
            else:
                correct_margin = math.nan
                selected_correct = False
            records.append({
                "seed": seed,
                "condition": condition,
                "prediction": str(prediction_path),
                "prediction_sha256": sha256(prediction_path),
                "shared_pocket_residues": len(keys),
                "prediction_global_ca_rmsd_to_anchor": prediction_global_rmsd,
                "pocket_ca_rmsd_to_anchor": rmsd_to_anchor,
                "rmsd_to_holo_a": rmsd_to_a,
                "rmsd_to_holo_b": rmsd_to_b,
                "correct_state_margin": correct_margin,
                "selected_correct_endpoint": bool(selected_correct),
            })

    table = pd.DataFrame(records)
    directional = table[table["condition"].isin(["ligand_A", "ligand_B"])]
    condition_summary = {}
    for condition, group in table.groupby("condition", sort=False):
        condition_summary[condition] = {
            "n": int(len(group)),
            "mean_rmsd_to_holo_a": float(group["rmsd_to_holo_a"].mean()),
            "mean_rmsd_to_holo_b": float(group["rmsd_to_holo_b"].mean()),
            "mean_pocket_rmsd_to_anchor": float(
                group["pocket_ca_rmsd_to_anchor"].mean()
            ),
            "mean_correct_state_margin": (
                float(group["correct_state_margin"].mean())
                if condition in ("ligand_A", "ligand_B")
                else None
            ),
            "correct_endpoint_fraction": (
                float(group["selected_correct_endpoint"].mean())
                if condition in ("ligand_A", "ligand_B")
                else None
            ),
        }
    directional_mean = float(directional["correct_state_margin"].mean())
    directional_fraction = float(directional["selected_correct_endpoint"].mean())
    if directional_mean > 0 and directional_fraction >= 0.75:
        decision = "positive"
    elif directional_mean > 0 and directional_fraction > 0.5:
        decision = "mixed-positive"
    else:
        decision = "negative-or-inconclusive"

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    table.to_csv(result_dir / "state_metrics.csv", index=False)
    plot_results(table, result_dir / "state_margin.png")
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
        "target_uid": case_manifest["target_uid"],
        "holo_a": case_manifest["holo_a"],
        "holo_b": case_manifest["holo_b"],
        "pocket_union_residues": len(pocket_union),
        "mean_holo_endpoint_pocket_rmsd": float(np.mean(endpoint_rmsds)),
        "holo_a_global_ca_rmsd_to_anchor": holo_a_global_rmsd,
        "holo_b_global_ca_rmsd_to_anchor": holo_b_global_rmsd,
        "directional_mean_correct_state_margin": directional_mean,
        "directional_correct_endpoint_fraction": directional_fraction,
        "condition_summary": condition_summary,
        "decision": decision,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (result_dir / "commit.txt").write_text(
        f"{summary['commit']}\n{summary['commit_message'].strip()}\n",
        encoding="utf-8",
    )
    report = [
        f"# DynamicBind same-anchor state evaluation: {case_manifest['target_uid']}",
        "",
        f"- Holo A/B: {case_manifest['holo_a']} / {case_manifest['holo_b']}",
        f"- Predictions: {len(table)}",
        f"- Mean holo endpoint pocket RMSD: {summary['mean_holo_endpoint_pocket_rmsd']:.3f} Å",
        f"- Directional mean correct-state margin: {directional_mean:.3f} Å",
        f"- Correct endpoint fraction: {directional_fraction:.3f}",
        f"- Decision: {decision}",
        "",
        "## Per-condition summary",
        "",
    ]
    for condition, values in condition_summary.items():
        report.append(
            f"- {condition}: RMSD(A)={values['mean_rmsd_to_holo_a']:.3f} Å, "
            f"RMSD(B)={values['mean_rmsd_to_holo_b']:.3f} Å, "
            f"margin={values['mean_correct_state_margin']}, "
            f"correct fraction={values['correct_endpoint_fraction']}"
        )
    (result_dir / "report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"State evaluation passed: {result_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
