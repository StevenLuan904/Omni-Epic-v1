#!/usr/bin/env python3
"""Minimally fine-tune ligand-conditioned c-Met receptor translation scores."""

import argparse
import copy
import csv
from datetime import datetime
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from argparse import Namespace
from rdkit import Chem
from torch_geometric.data import Batch

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import set_time, t_to_sigma as t_to_sigma_compl
from utils.utils import get_model


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "figure3-cmet-finetune"
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
    print("ERROR: fine-tuning must run from a committed worktree.", file=sys.stderr)
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


def gpu_snapshot():
    queries = {
        "devices": [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.free,"
            "utilization.gpu",
            "--format=csv,noheader",
        ],
        "processes": [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader",
        ],
    }
    snapshot = {}
    for label, command in queries.items():
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        snapshot[label] = {
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    return snapshot


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def ca_records(path):
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            key = (
                line[21],
                line[22:26].strip(),
                line[26],
                line[17:20].strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            coord = np.asarray(
                [
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ],
                dtype=np.float32,
            )
            records.append((key, coord))
    if not records:
        raise ValueError(f"no CA records in {path}")
    return records


def molecule_coords(path):
    molecule = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    if molecule is None:
        raise ValueError(f"RDKit could not read {path}")
    return np.asarray(molecule.GetConformer().GetPositions(), dtype=np.float32)


def target_from_paths(anchor_path, target_path, ligand_path):
    anchor = ca_records(anchor_path)
    target = dict(ca_records(target_path))
    ligand = molecule_coords(ligand_path)
    displacement = np.zeros((len(anchor), 3), dtype=np.float32)
    valid = np.zeros(len(anchor), dtype=bool)
    pocket_weight = np.zeros(len(anchor), dtype=np.float32)
    for index, (key, anchor_coord) in enumerate(anchor):
        if key not in target:
            continue
        valid[index] = True
        displacement[index] = target[key] - anchor_coord
        distance = np.linalg.norm(ligand - target[key][None, :], axis=1).min()
        pocket_weight[index] = np.exp(-max(float(distance) - 6.0, 0.0) / 10.0)
    pocket_weight *= valid
    if valid.sum() < 20:
        raise ValueError(
            f"only {int(valid.sum())} target CA atoms map to {anchor_path}"
        )
    return {
        "displacement": torch.from_numpy(displacement),
        "valid": torch.from_numpy(valid),
        "weight": torch.from_numpy(pocket_weight),
        "mapped_residues": int(valid.sum()),
        "pocket_residues_12a": int(
            np.logical_and(valid, pocket_weight >= np.exp(-0.6)).sum()
        ),
    }


def restore_native_ligand_pose(graph):
    coords = np.asarray(
        graph.mol.GetConformer().GetPositions(), dtype=np.float32
    )
    if coords.shape[0] != graph["ligand"].num_nodes:
        raise ValueError(
            f"ligand graph has {graph['ligand'].num_nodes} nodes but "
            f"molecule has {coords.shape[0]} atoms"
        )
    graph["ligand"].pos = (
        torch.from_numpy(coords) - graph.original_center.float()
    )
    return graph


def graph_for_model(graph, device):
    graph = copy.deepcopy(graph)
    set_time(
        graph, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        batchsize=1, all_atoms=False, device=None,
    )
    return Batch.from_data_list([graph]).to(device)


def weighted_error(prediction, target, mask, weight):
    per_residue = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction="none"
    ).mean(dim=1)
    selected_weight = weight * mask.float()
    return (per_residue * selected_weight).sum() / (
        selected_weight.sum() + 1e-8
    )


def state_errors(prediction, correct, wrong):
    common = correct["valid"] & wrong["valid"]
    weight = correct["weight"]
    correct_error = weighted_error(
        prediction, correct["displacement"], common, weight
    )
    wrong_error = weighted_error(
        prediction, wrong["displacement"], common, weight
    )
    return correct_error, wrong_error


def trainable_parameters(model, last_layer):
    prefixes = (
        "cross_edge_embedding.",
        f"lig_to_rec_conv_layers.{last_layer}.",
        "res_tr_final_layer.",
    )
    selected = {}
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(prefixes)
        if parameter.requires_grad:
            selected[name] = parameter
    if not selected:
        raise ValueError("fine-tune parameter selection is empty")
    return selected


def evaluate_graphs(model, graphs, targets, device):
    output = []
    model.eval()
    with torch.no_grad():
        for index, graph in enumerate(graphs):
            prediction = model(graph_for_model(graph, device))[5]
            correct_error, wrong_error = state_errors(
                prediction, targets[index], targets[1 - index]
            )
            output.append({
                "condition_index": index,
                "correct_error": float(correct_error),
                "wrong_error": float(wrong_error),
                "margin": float(wrong_error - correct_error),
                "selected_correct": bool(correct_error < wrong_error),
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--inputs-csv", required=True)
    parser.add_argument("--esm-embeddings", required=True)
    parser.add_argument("--inference-cache", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", default="ema_inference_epoch314_model.pt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--preservation-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    require_clean_commit()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this fine-tune")
    device = torch.device("cuda")

    case_dir = Path(args.case_dir).resolve()
    inputs_csv = Path(args.inputs_csv).resolve()
    esm_embeddings = Path(args.esm_embeddings).resolve()
    inference_cache = Path(args.inference_cache).resolve()
    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / args.ckpt
    manifest_path = case_dir / "manifest.json"
    for path in (
        case_dir, inputs_csv, esm_embeddings, inference_cache,
        model_dir, checkpoint, manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    metrics_path = result_dir / "training_metrics.csv"
    metadata_path = result_dir / "metadata.json"
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "exit_code": None,
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "ranking_margin": args.ranking_margin,
        "ranking_weight": args.ranking_weight,
        "preservation_weight": args.preservation_weight,
        "inputs": {
            "case_manifest": str(manifest_path),
            "case_manifest_sha256": sha256(manifest_path),
            "inputs_csv": str(inputs_csv),
            "inputs_csv_sha256": sha256(inputs_csv),
            "esm_embeddings": str(esm_embeddings),
            "inference_cache": str(inference_cache),
            "model_dir": str(model_dir),
            "model_parameters_sha256": sha256(
                model_dir / "model_parameters.yml"
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "gpu_before": gpu_snapshot(),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (result_dir / "commit.txt").write_text(
        f"{metadata['commit']}\n{metadata['commit_message'].strip()}\n",
        encoding="utf-8",
    )

    exit_code = 0
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest["cases"]
        extracted = manifest["extracted"]
        if len(cases) != 2:
            raise ValueError(f"expected two cases, found {len(cases)}")
        anchor_path = Path(manifest["common_anchor"])
        targets = []
        for case in cases:
            entry = case["entry"]
            targets.append(target_from_paths(
                anchor_path,
                Path(extracted[f"{entry}_holo"]["path"]),
                Path(extracted[f"{entry}_ligand"]["path"]),
            ))

        with (model_dir / "model_parameters.yml").open() as handle:
            model_args = Namespace(**yaml.full_load(handle))
        frame = pd.read_csv(inputs_csv)
        dataset = PDBBind(
            transform=None,
            root="",
            name_list=[f"idx_{index}" for index in range(len(frame))],
            protein_path_list=frame["protein_path"].tolist(),
            ligand_descriptions=frame["ligand"].tolist(),
            receptor_radius=model_args.receptor_radius,
            cache_path=str(inference_cache),
            remove_hs=model_args.remove_hs,
            max_lig_size=None,
            c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
            matching=False,
            keep_original=False,
            popsize=model_args.matching_popsize,
            maxiter=model_args.matching_maxiter,
            center_ligand=True,
            all_atoms=model_args.all_atoms,
            atom_radius=model_args.atom_radius,
            atom_max_neighbors=model_args.atom_max_neighbors,
            esm_embeddings_path=(
                str(esm_embeddings)
                if model_args.esm_embeddings_path is not None else None
            ),
            require_ligand=True,
            require_receptor=False,
            num_workers=1,
            keep_local_structures=True,
            use_existing_cache=True,
        )
        graphs = [
            restore_native_ligand_pose(dataset.get(index))
            for index in range(len(dataset))
        ]
        if len(graphs) != 2:
            raise ValueError(f"expected two graphs, found {len(graphs)}")
        for graph, target in zip(graphs, targets):
            if graph["receptor"].num_nodes != len(target["displacement"]):
                raise ValueError(
                    f"graph has {graph['receptor'].num_nodes} residues but "
                    f"target has {len(target['displacement'])}"
                )
            for key in ("displacement", "valid", "weight"):
                target[key] = target[key].to(device)

        sigma = partial(t_to_sigma_compl, args=model_args)
        model = get_model(
            model_args, device, t_to_sigma=sigma, no_parallel=True
        )
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        selected = trainable_parameters(
            model, int(model_args.num_conv_layers) - 1
        )
        parameter_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in selected.values())
        initial_selected = {
            name: parameter.detach().clone()
            for name, parameter in selected.items()
        }
        metadata["parameters"] = {
            "total": parameter_count,
            "trainable": trainable_count,
            "trainable_fraction": trainable_count / parameter_count,
            "names": list(selected),
            "frozen": parameter_count - trainable_count,
        }

        before = evaluate_graphs(model, graphs, targets, device)
        original_predictions = []
        with torch.no_grad():
            for graph in graphs:
                original_predictions.append(
                    model(graph_for_model(graph, device))[5].detach()
                )

        optimizer = torch.optim.AdamW(
            selected.values(), lr=args.learning_rate, weight_decay=1e-4
        )
        metric_rows = []
        model.eval()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            correct_losses = []
            ranking_losses = []
            preservation_losses = []
            margins = []
            for index, graph in enumerate(graphs):
                prediction = model(graph_for_model(graph, device))[5]
                correct_error, wrong_error = state_errors(
                    prediction, targets[index], targets[1 - index]
                )
                ranking = torch.relu(
                    args.ranking_margin + correct_error - wrong_error
                )
                distal = (
                    targets[index]["valid"]
                    & (targets[index]["weight"] < np.exp(-0.6))
                )
                preservation = weighted_error(
                    prediction,
                    original_predictions[index],
                    distal,
                    torch.ones_like(targets[index]["weight"]),
                )
                correct_losses.append(correct_error)
                ranking_losses.append(ranking)
                preservation_losses.append(preservation)
                margins.append(wrong_error - correct_error)

            correct_loss = torch.stack(correct_losses).mean()
            ranking_loss = torch.stack(ranking_losses).mean()
            preservation_loss = torch.stack(preservation_losses).mean()
            regularization = torch.stack([
                (parameter - initial_selected[name]).pow(2).mean()
                for name, parameter in selected.items()
            ]).mean()
            loss = (
                correct_loss
                + args.ranking_weight * ranking_loss
                + args.preservation_weight * preservation_loss
                + 1e-4 * regularization
            )
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                selected.values(), max_norm=1.0
            )
            optimizer.step()
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "correct_loss": float(correct_loss.detach()),
                "ranking_loss": float(ranking_loss.detach()),
                "preservation_loss": float(preservation_loss.detach()),
                "margin_a": float(margins[0].detach()),
                "margin_b": float(margins[1].detach()),
                "gradient_norm": float(gradient_norm),
            }
            metric_rows.append(row)
            print(json.dumps(row))

        after = evaluate_graphs(model, graphs, targets, device)
        full_checkpoint = result_dir / "finetuned_model.pt"
        delta_checkpoint = result_dir / "trainable_delta.pt"
        torch.save(model.state_dict(), full_checkpoint)
        torch.save(
            {
                "base_checkpoint_sha256": sha256(checkpoint),
                "trainable_state_dict": {
                    name: parameter.detach().cpu()
                    for name, parameter in selected.items()
                },
            },
            delta_checkpoint,
        )
        shutil.copy2(
            model_dir / "model_parameters.yml",
            result_dir / "model_parameters.yml",
        )
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(metric_rows[0])
            )
            writer.writeheader()
            writer.writerows(metric_rows)

        metadata.update({
            "status": "passed",
            "before": before,
            "after": after,
            "targets": [{
                "entry": case["entry"],
                "mapped_residues": target["mapped_residues"],
                "pocket_residues_12a": target["pocket_residues_12a"],
            } for case, target in zip(cases, targets)],
            "outputs": {
                "full_checkpoint": str(full_checkpoint),
                "full_checkpoint_sha256": sha256(full_checkpoint),
                "delta_checkpoint": str(delta_checkpoint),
                "delta_checkpoint_sha256": sha256(delta_checkpoint),
                "training_metrics": str(metrics_path),
                "training_metrics_sha256": sha256(metrics_path),
            },
        })
    except Exception as exc:
        exit_code = 1
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata["exit_code"] = exit_code
        metadata["gpu_after"] = gpu_snapshot()
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"c-Met fine-tune passed: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
