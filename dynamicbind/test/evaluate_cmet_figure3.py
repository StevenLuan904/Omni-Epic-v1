#!/usr/bin/env python3
"""Evaluate the exact DynamicBind Figure 3 c-Met reproduction."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

import numpy as np
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "figure3-cmet-evaluation"
)
RANK_PATTERN = re.compile(r"rank(\d+)_receptor_")


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
    atoms = {}
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            key = (
                line[21],
                line[22:26].strip(),
                line[26],
                line[17:20].strip(),
                line[12:16].strip(),
            )
            atoms[key] = np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ]
            )
    if not atoms:
        raise ValueError(f"no protein atoms in {path}")
    return atoms


def read_molecule(path):
    molecule = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=True)
    if molecule is None:
        raise ValueError(f"RDKit could not read {path}")
    return molecule


def molecule_coords(molecule):
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=float)


def rmsd(first, second):
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def fit_transform(target, mobile):
    target_center = target.mean(axis=0)
    mobile_center = mobile.mean(axis=0)
    covariance = (mobile - mobile_center).T @ (target - target_center)
    left, _, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = target_center - mobile_center @ rotation
    return rotation, translation


def transform(coords, rotation, translation):
    return coords @ rotation + translation


def pocket_alignment(target, prediction, native_ligand_coords):
    centroid = native_ligand_coords.mean(axis=0)
    keys = []
    for key, coord in target.items():
        if key[-1] != "CA" or key not in prediction:
            continue
        if np.linalg.norm(coord - centroid) < 15.0:
            keys.append(key)
    if len(keys) < 3:
        raise ValueError(f"only {len(keys)} pocket CA atoms available for alignment")
    target_coords = np.stack([target[key] for key in keys])
    prediction_coords = np.stack([prediction[key] for key in keys])
    rotation, translation = fit_transform(target_coords, prediction_coords)
    return rotation, translation, rmsd(
        target_coords,
        transform(prediction_coords, rotation, translation),
    )


def protein_metrics(target, prediction, native_ligand_coords):
    keys = [key for key in target if key in prediction]
    if not keys:
        raise ValueError("target and prediction have no mapped atoms")
    target_coords = np.stack([target[key] for key in keys])
    prediction_coords = np.stack([prediction[key] for key in keys])
    distances = np.sqrt(
        np.sum(
            (target_coords[:, None, :] - native_ligand_coords[None, :, :]) ** 2,
            axis=2,
        )
    )
    pocket_mask = distances.min(axis=1) < 5.0
    if not pocket_mask.any():
        raise ValueError("native ligand has no mapped protein atom within 5 A")
    return {
        "mapped_atom_count": len(keys),
        "pocket_atom_count": int(pocket_mask.sum()),
        "protein_rmsd": rmsd(target_coords, prediction_coords),
        "pocket_rmsd": rmsd(
            target_coords[pocket_mask], prediction_coords[pocket_mask]
        ),
    }


def symmetry_rmsd(reference, prediction, prediction_coords):
    reference_coords = molecule_coords(reference)
    matches = prediction.GetSubstructMatches(
        reference, uniquify=False, maxMatches=10000
    )
    if not matches:
        raise ValueError("predicted ligand does not match reference graph")
    return min(
        rmsd(reference_coords, prediction_coords[np.asarray(match)])
        for match in matches
    )


def baseline(case, extracted):
    entry = case["entry"]
    target_path = Path(extracted[f"{entry}_holo"]["path"])
    anchor_path = Path(extracted[f"{entry}_af2"]["path"])
    ligand_path = Path(extracted[f"{entry}_ligand"]["path"])
    target = read_protein(target_path)
    anchor = read_protein(anchor_path)
    ligand_coords = molecule_coords(read_molecule(ligand_path))
    values = protein_metrics(target, anchor, ligand_coords)
    expected = float(case["paper_af_pocket_rmsd"])
    values.update({
        "entry": entry,
        "target_path": str(target_path),
        "anchor_path": str(anchor_path),
        "computed_pocket_rmsd": values["pocket_rmsd"],
        "paper_pocket_rmsd": expected,
        "absolute_error": abs(values["pocket_rmsd"] - expected),
    })
    return values


def prediction_pairs(condition_output):
    output_dir = Path(condition_output["receptor"]).parent
    pairs = []
    for receptor in sorted(output_dir.glob("rank*_receptor_*.pdb")):
        match = RANK_PATTERN.search(receptor.name)
        if match is None:
            continue
        rank = int(match.group(1))
        ligands = sorted(output_dir.glob(f"rank{rank}_ligand_*.sdf"))
        if len(ligands) != 1:
            raise ValueError(
                f"expected one ligand for rank {rank} in {output_dir}, "
                f"found {len(ligands)}"
            )
        pairs.append((rank, receptor, ligands[0]))
    if not pairs:
        raise ValueError(f"no ranked receptor-ligand pairs in {output_dir}")
    return pairs


def evaluate_prediction(case, other_case, extracted, receptor_path, ligand_path):
    entry = case["entry"]
    target = read_protein(Path(extracted[f"{entry}_holo"]["path"]))
    other_target = read_protein(
        Path(extracted[f"{other_case['entry']}_holo"]["path"])
    )
    native_ligand = read_molecule(Path(extracted[f"{entry}_ligand"]["path"]))
    native_coords = molecule_coords(native_ligand)
    prediction = read_protein(receptor_path)
    predicted_ligand = read_molecule(ligand_path)
    predicted_ligand_coords = molecule_coords(predicted_ligand)

    rotation, translation, alignment_rmsd = pocket_alignment(
        target, prediction, native_coords
    )
    aligned_prediction = {
        key: transform(coord[None, :], rotation, translation)[0]
        for key, coord in prediction.items()
    }
    aligned_ligand_coords = transform(
        predicted_ligand_coords, rotation, translation
    )
    correct = protein_metrics(target, aligned_prediction, native_coords)

    other_rotation, other_translation, _ = pocket_alignment(
        other_target, prediction, native_coords
    )
    other_aligned_prediction = {
        key: transform(coord[None, :], other_rotation, other_translation)[0]
        for key, coord in prediction.items()
    }
    wrong = protein_metrics(
        other_target, other_aligned_prediction, native_coords
    )
    return {
        "receptor": str(receptor_path),
        "receptor_sha256": sha256(receptor_path),
        "ligand": str(ligand_path),
        "ligand_sha256": sha256(ligand_path),
        "pocket_alignment_ca_rmsd": alignment_rmsd,
        "ligand_symmetry_rmsd": symmetry_rmsd(
            native_ligand, predicted_ligand, aligned_ligand_coords
        ),
        "correct_pocket_rmsd": correct["pocket_rmsd"],
        "wrong_pocket_rmsd": wrong["pocket_rmsd"],
        "correct_state_margin": (
            wrong["pocket_rmsd"] - correct["pocket_rmsd"]
        ),
        "selected_correct_state": (
            correct["pocket_rmsd"] < wrong["pocket_rmsd"]
        ),
        "mapped_atom_count": correct["mapped_atom_count"],
        "pocket_atom_count": correct["pocket_atom_count"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--inference-dir", required=True)
    parser.add_argument("--baseline-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    require_clean_commit()
    case_dir = Path(args.case_dir).resolve()
    inference_dir = Path(args.inference_dir).resolve()
    case_manifest_path = case_dir / "manifest.json"
    inference_metadata_path = inference_dir / "metadata.json"
    case_manifest = json.loads(case_manifest_path.read_text(encoding="utf-8"))
    inference_metadata = json.loads(
        inference_metadata_path.read_text(encoding="utf-8")
    )
    if inference_metadata["status"] != "passed":
        raise ValueError("inference metadata status is not passed")

    cases = case_manifest["cases"]
    if len(cases) != 2:
        raise ValueError(f"expected two c-Met cases, found {len(cases)}")
    extracted = case_manifest["extracted"]
    baselines = [baseline(case, extracted) for case in cases]
    baseline_passed = all(
        item["absolute_error"] <= args.baseline_tolerance for item in baselines
    )
    if not baseline_passed:
        raise ValueError(f"paper AF baseline check failed: {baselines}")

    records = []
    for run in inference_metadata["runs"]:
        seed = int(run["seed"])
        conditions = run["validation"]["conditions"]
        if len(conditions) != 2:
            raise ValueError(
                f"seed {seed} has {len(conditions)} conditions, expected two"
            )
        for index, output in enumerate(conditions):
            case = cases[index]
            other_case = cases[1 - index]
            for rank, receptor, ligand in prediction_pairs(output):
                record = evaluate_prediction(
                    case, other_case, extracted, receptor, ligand
                )
                record.update({
                    "seed": seed,
                    "rank": rank,
                    "entry": case["entry"],
                    "state": case["state"],
                    "paper_ligand_rmsd": case["paper_ligand_rmsd"],
                    "paper_predicted_pocket_rmsd": (
                        case["paper_predicted_pocket_rmsd"]
                    ),
                })
                records.append(record)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    table_path = result_dir / "metrics.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    correct = sum(bool(record["selected_correct_state"]) for record in records)
    summary = {
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "passed",
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "case_manifest": {
            "path": str(case_manifest_path),
            "sha256": sha256(case_manifest_path),
        },
        "inference_metadata": {
            "path": str(inference_metadata_path),
            "sha256": sha256(inference_metadata_path),
        },
        "baseline_tolerance": args.baseline_tolerance,
        "baseline_passed": baseline_passed,
        "baselines": baselines,
        "prediction_count": len(records),
        "correct_state_count": correct,
        "state_selection_accuracy": correct / len(records),
        "metrics_csv": str(table_path),
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Figure 3 evaluation ready: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
