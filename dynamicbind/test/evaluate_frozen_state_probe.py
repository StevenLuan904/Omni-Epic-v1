#!/usr/bin/env python3
"""Train and evaluate tiny frozen-feature state probes."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "frozen-state-probe"
)
HEAD_SEEDS = (20260726, 20260727, 20260728)


def git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


def require_clean_commit():
    status = git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return
    print("ERROR: probe evaluation requires a committed worktree.", file=sys.stderr)
    print("\n--- git status --short ---", file=sys.stderr)
    print(status, file=sys.stderr)
    for title, args in (
        ("git diff", ("diff", "--no-ext-diff")),
        ("git diff --cached", ("diff", "--cached", "--no-ext-diff")),
    ):
        print(f"\n--- {title} ---", file=sys.stderr)
        diff = git(*args, check=False)
        if diff:
            print(diff, file=sys.stderr)
    raise SystemExit(2)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.log = Path(path).open("a", encoding="utf-8", buffering=1)

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()


def auc_score(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean())


def signed_pairs(features, labels, placement_seeds):
    seeds = np.unique(placement_seeds)
    deltas = []
    for seed in seeds:
        first = features[(placement_seeds == seed) & (labels == 0)]
        second = features[(placement_seeds == seed) & (labels == 1)]
        if len(first) != 1 or len(second) != 1:
            raise ValueError(f"seed {seed} does not have exactly one A/B pair")
        deltas.append(second[0] - first[0])
    return seeds, np.asarray(deltas, dtype=np.float32)


def split_seed_groups(seeds):
    if len(seeds) != 24:
        raise ValueError(f"exactly 24 paired placement seeds are required, got {len(seeds)}")
    return {
        "train": seeds[:12], "validation": seeds[12:18], "test": seeds[18:24]
    }


def symmetric_examples(deltas):
    x = np.concatenate([deltas, -deltas], axis=0)
    y = np.concatenate([
        np.ones(len(deltas), dtype=np.float32),
        np.zeros(len(deltas), dtype=np.float32),
    ])
    return x, y


def fit_head(train_delta, seed, steps=500, learning_rate=0.03):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x, y = symmetric_examples(train_delta)
    scale = x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_tensor = torch.from_numpy(x / scale).float()
    y_tensor = torch.from_numpy(y).float()
    head = torch.nn.Linear(x.shape[1], 1)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=learning_rate, weight_decay=1e-2
    )
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = head(x_tensor).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y_tensor
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return head.eval(), scale, losses


def evaluate_head(head, scale, deltas):
    x, y = symmetric_examples(deltas)
    with torch.no_grad():
        logits = head(torch.from_numpy(x / scale).float()).squeeze(1).numpy()
    predictions = (logits >= 0).astype(int)
    signed = np.where(y == 1, logits, -logits)
    zero_x = torch.zeros((2, x.shape[1]), dtype=torch.float32)
    with torch.no_grad():
        zero_logits = head(zero_x).squeeze(1).numpy()
    zero_labels = np.asarray([1, 0])
    zero_accuracy = float(((zero_logits >= 0).astype(int) == zero_labels).mean())
    return {
        "accuracy": float((predictions == y).mean()),
        "pair_accuracy": float((logits[:len(deltas)] >= 0).mean()),
        "pair_correct": int((logits[:len(deltas)] >= 0).sum()),
        "pair_total": len(deltas),
        "auc": auc_score(y, logits),
        "mean_signed_margin": float(signed.mean()),
        "minimum_signed_margin": float(signed.min()),
        "synthetic_zero_accuracy": zero_accuracy,
        "zero_feature_logit": float(zero_logits[0]),
        "logits": logits.tolist(),
        "labels": y.astype(int).tolist(),
    }


def ligand_descriptors(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"invalid canonical SMILES: {smiles}")
    return np.asarray([
        Descriptors.MolWt(molecule), Descriptors.MolLogP(molecule),
        Descriptors.TPSA(molecule), Descriptors.NumHDonors(molecule),
        Descriptors.NumHAcceptors(molecule), Descriptors.NumRotatableBonds(molecule),
        Descriptors.RingCount(molecule), Descriptors.HeavyAtomCount(molecule),
    ], dtype=np.float32)


def permutation_null(train_delta, test_delta, observed, repetitions=99):
    accuracies = []
    for index in range(repetitions):
        generator = np.random.RandomState(91000 + index)
        signs = generator.choice([-1.0, 1.0], size=(len(train_delta), 1))
        head, scale, _ = fit_head(train_delta * signs, 80000 + index, steps=150)
        accuracies.append(evaluate_head(head, scale, test_delta)["pair_accuracy"])
    return {
        "repetitions": repetitions,
        "mean_accuracy": float(np.mean(accuracies)),
        "maximum_accuracy": float(np.max(accuracies)),
        "empirical_p_value": float(
            (1 + np.count_nonzero(np.asarray(accuracies) >= observed)) /
            (repetitions + 1)
        ),
    }


def wilson_interval(correct, total, z=1.96):
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [float(center - radius), float(center + radius)]


def evaluate_system(system, feature_path):
    feature_path = Path(feature_path).resolve()
    metadata_path = feature_path.with_name("metadata.json")
    for path in (feature_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    extraction = json.loads(metadata_path.read_text(encoding="utf-8"))
    if extraction.get("status") != "passed":
        raise ValueError(f"feature extraction was not passed: {metadata_path}")
    if extraction.get("system") != system:
        raise ValueError(
            f"system mismatch: CLI={system}, extraction={extraction.get('system')}"
        )
    if extraction.get("outputs", {}).get("features_sha256") != sha256(feature_path):
        raise ValueError("feature SHA-256 does not match extraction metadata")
    data = np.load(feature_path)
    features = data["features"].astype(np.float32)
    swap_features = data["input_swap_features"].astype(np.float32)
    dummy_features = data["dummy_ligand_features"].astype(np.float32)
    labels = data["labels"].astype(int)
    placement_seeds = data["placement_seeds"].astype(int)
    expected_rows = sum(item["samples"] for item in extraction["conditions"])
    if features.shape[0] != expected_rows or features.shape != swap_features.shape \
            or features.shape != dummy_features.shape:
        raise ValueError("feature/control shapes do not match extraction metadata")
    if not all(np.isfinite(matrix).all() for matrix in (
        features, swap_features, dummy_features
    )) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("invalid feature values or labels")
    seeds, deltas = signed_pairs(features, labels, placement_seeds)
    swap_seeds, swap_deltas = signed_pairs(
        swap_features, labels, placement_seeds
    )
    dummy_seeds, dummy_deltas = signed_pairs(
        dummy_features, labels, placement_seeds
    )
    if not np.array_equal(seeds, swap_seeds) or not np.array_equal(seeds, dummy_seeds):
        raise ValueError("control placement seeds do not match primary features")
    groups = split_seed_groups(seeds)
    delta_by_seed = {seed: delta for seed, delta in zip(seeds, deltas)}
    swap_by_seed = {seed: delta for seed, delta in zip(seeds, swap_deltas)}
    dummy_by_seed = {seed: delta for seed, delta in zip(seeds, dummy_deltas)}
    split_delta = {
        name: np.asarray([delta_by_seed[seed] for seed in split_seeds])
        for name, split_seeds in groups.items()
    }
    split_swap = {
        name: np.asarray([swap_by_seed[seed] for seed in split_seeds])
        for name, split_seeds in groups.items()
    }
    split_dummy = {
        name: np.asarray([dummy_by_seed[seed] for seed in split_seeds])
        for name, split_seeds in groups.items()
    }

    runs = []
    for head_seed in HEAD_SEEDS:
        head, scale, losses = fit_head(split_delta["train"], head_seed)
        test_metrics = evaluate_head(head, scale, split_delta["test"])
        swap_metrics = evaluate_head(head, scale, split_swap["test"])
        dummy_metrics = evaluate_head(head, scale, split_dummy["test"])
        dummy_logits = np.asarray(dummy_metrics["logits"])
        runs.append({
            "head_seed": head_seed,
            "initial_loss": losses[0], "final_loss": losses[-1],
            "train": evaluate_head(head, scale, split_delta["train"]),
            "validation": evaluate_head(head, scale, split_delta["validation"]),
            "test": test_metrics,
            "input_row_swap": {
                **swap_metrics,
                "endpoint_exchange_accuracy": 1.0 - swap_metrics["pair_accuracy"],
            },
            "dummy_null_ligand": {
                **dummy_metrics,
                "maximum_absolute_logit": float(np.max(np.abs(dummy_logits))),
            },
        })
    test_accuracy = [run["test"]["pair_accuracy"] for run in runs]
    test_auc = [run["test"]["auc"] for run in runs]
    dummy_logit = [
        run["dummy_null_ligand"]["maximum_absolute_logit"] for run in runs
    ]
    swap_accuracy = [
        run["input_row_swap"]["endpoint_exchange_accuracy"] for run in runs
    ]
    fixed_pair_gate = (
        np.mean(test_accuracy) >= 0.80 and np.mean(test_auc) >= 0.80
        and np.min(test_accuracy) >= 0.75 and np.min(swap_accuracy) >= 0.75
        and np.max(dummy_logit) <= 0.405
    )

    smiles = [item["canonical_isomeric_smiles"] for item in extraction["conditions"]]
    descriptor_delta = ligand_descriptors(smiles[1]) - ligand_descriptors(smiles[0])
    descriptor_train = np.repeat(descriptor_delta[None, :], 12, axis=0)
    descriptor_test = np.repeat(descriptor_delta[None, :], 6, axis=0)
    baseline_head, baseline_scale, _ = fit_head(descriptor_train, HEAD_SEEDS[0])
    ligand_only = evaluate_head(baseline_head, baseline_scale, descriptor_test)
    observed_for_null = float(np.mean(test_accuracy))
    representative_correct = runs[0]["test"]["pair_correct"]
    representative_total = runs[0]["test"]["pair_total"]
    return {
        "system": system, "feature_path": str(feature_path),
        "feature_sha256": sha256(feature_path),
        "extraction_metadata": str(metadata_path),
        "extraction_metadata_sha256": sha256(metadata_path),
        "extraction_commit": extraction["commit"],
        "feature_shape": list(features.shape),
        "split_seed_groups": {key: value.tolist() for key, value in groups.items()},
        "runs": runs,
        "summary": {
            "mean_test_accuracy": float(np.mean(test_accuracy)),
            "minimum_test_accuracy": float(np.min(test_accuracy)),
            "mean_test_auc": float(np.mean(test_auc)),
            "minimum_swap_exchange_accuracy": float(np.min(swap_accuracy)),
            "maximum_dummy_absolute_logit": float(np.max(dummy_logit)),
            "fixed_pair_readout_gate": bool(fixed_pair_gate),
            "ligand_only_accuracy": ligand_only["accuracy"],
            "test_pair_wilson_95_interval": wilson_interval(
                representative_correct, representative_total
            ),
            "state_compatibility_identifiable": False,
        },
        "ligand_only_baseline": ligand_only,
        "label_permutation_null": permutation_null(
            split_delta["train"], split_delta["test"], observed_for_null
        ),
    }


def write_report(path, metadata):
    rows = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in (
            result["system"],
            f"{result['summary']['mean_test_accuracy']:.3f}",
            f"{result['summary']['mean_test_auc']:.3f}",
            f"{result['summary']['ligand_only_accuracy']:.3f}",
            result["summary"]["fixed_pair_readout_gate"],
            result["summary"]["state_compatibility_identifiable"],
        )) + "</tr>" for result in metadata.get("systems", [])
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Frozen state probe</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:2rem auto}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:.45rem}"
        "code{overflow-wrap:anywhere}</style><h1>Frozen DynamicBind state probe</h1>"
        f"<p>Status: <b>{metadata['status']}</b>; commit: "
        f"<code>{metadata['commit']}</code></p>"
        "<table><tr><th>System</th><th>Test accuracy</th><th>AUC</th>"
        "<th>Ligand-only</th><th>Fixed-pair readable</th>"
        f"<th>Compatibility identifiable</th></tr>{rows}</table>"
        "<p>The held-out units are placement seeds, not unseen ligands. "
        "With one ligand per state, compatibility is not identifiable separately "
        "from ligand identity; the ligand-only baseline exposes this ceiling.</p>",
        encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", nargs=2, action="append", metavar=("SYSTEM", "NPZ"),
        required=True,
    )
    args = parser.parse_args()
    require_clean_commit()
    started = time.monotonic()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    stdout_tee = Tee(sys.stdout, result_dir / "stdout.log")
    stderr_tee = Tee(sys.stderr, result_dir / "stderr.log")
    sys.stdout, sys.stderr = stdout_tee, stderr_tee
    metadata_path = result_dir / "metrics.json"
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None, "elapsed_seconds": None,
        "status": "running", "exit_code": None,
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "protocol": {
            "head": "single linear layer", "head_seeds": HEAD_SEEDS,
            "split": "12 train / 6 validation / 6 final test placement seeds",
            "fixed_pair_gate": "mean accuracy/AUC >=0.80; min accuracy and swap >=0.75; dummy |logit| <=0.405",
            "scientific_limit": "one ligand per state cannot identify compatibility apart from ligand identity",
        },
    }
    (result_dir / "commit.txt").write_text(
        f"{metadata['commit']}\n{metadata['commit_message'].strip()}\n",
        encoding="utf-8",
    )
    exit_code = 0
    try:
        metadata["systems"] = [
            evaluate_system(system, path) for system, path in args.features
        ]
        metadata["status"] = "passed"
        rows = []
        for result in metadata["systems"]:
            for run in result["runs"]:
                rows.append({
                    "system": result["system"], "head_seed": run["head_seed"],
                    "test_accuracy": run["test"]["pair_accuracy"],
                    "test_auc": run["test"]["auc"],
                    "test_margin": run["test"]["mean_signed_margin"],
                    "swap_exchange_accuracy": run["input_row_swap"]["endpoint_exchange_accuracy"],
                    "dummy_max_abs_logit": run["dummy_null_ligand"]["maximum_absolute_logit"],
                })
        with (result_dir / "run_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        exit_code = 1
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["exit_code"] = exit_code
        write_report(result_dir / "report.html", metadata)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Frozen state probe {metadata['status']}: {result_dir}")
    if exit_code:
        print(metadata["error"], file=sys.stderr)
    stdout_tee.flush()
    stderr_tee.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
