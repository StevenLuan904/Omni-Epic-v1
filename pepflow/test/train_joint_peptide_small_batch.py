#!/usr/bin/env python3
"""Train the retained joint flow on a split-audited peptide small batch."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from train_joint_peptide_receptor_overfit import (
    git,
    gpu_snapshot,
    require_clean_commit,
    sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "joint-small-batch-train"


def worker(args):
    import copy
    from argparse import Namespace
    from functools import partial
    import random

    import numpy as np
    from scipy.spatial.transform import Rotation
    import torch
    from torch_geometric.data import Batch
    import yaml

    pepflow_root, dynamicbind_root = REPO_ROOT / "pepflow", REPO_ROOT / "dynamicbind"
    sys.path.insert(0, str(dynamicbind_root))
    sys.path.insert(0, str(pepflow_root))
    from datasets.pdbbind import PDBBind
    from models_con.flow_model import FlowModel
    from models_con.joint_flow import JointPeptideReceptorFlow
    from models_con.pep_dataloader import preprocess_structure
    from models_con.utils import process_dic
    from pepflow.utils.data import PaddingCollate
    from pepflow.utils.misc import load_config
    from pepflow.utils.train import recursive_to, sum_weighted_losses
    from utils.diffusion_utils import set_time, t_to_sigma as t_to_sigma_compl
    from utils.utils import get_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    data_dir = Path(args.data_dir).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    split = manifest["splits"][args.split]
    if "train" not in split:
        raise ValueError(f"split {args.split} is not executable")
    case_records = {record["case_id"]: record for record in manifest["cases"]}
    train_ids = split["train"][:args.max_train_cases or None]
    evaluation_ids = {}
    for split_name in ("random_complex", "peptide_cluster", "receptor_family_proxy"):
        values = manifest["splits"][split_name]["test"]
        evaluation_ids[split_name] = values[:args.max_eval_cases or None]
    if len(train_ids) < 2 or any(not values for values in evaluation_ids.values()):
        raise RuntimeError("small-batch split has insufficient train/evaluation cases")

    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / args.dynamicbind_checkpoint
    with (model_dir / "model_parameters.yml").open() as handle:
        model_args = Namespace(**yaml.full_load(handle))
    config, _ = load_config(args.pepflow_config)
    peptide_flow = FlowModel(config.model)
    pep_checkpoint = torch.load(args.pepflow_checkpoint, map_location="cpu")
    peptide_flow.load_state_dict(process_dic(pep_checkpoint.get("model", pep_checkpoint)), strict=True)
    sigma = partial(t_to_sigma_compl, args=model_args)
    receptor_flow = get_model(model_args, device, t_to_sigma=sigma, no_parallel=True)
    receptor_flow.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    model = JointPeptideReceptorFlow(
        peptide_flow, receptor_flow, config.model.encoder.node_embed_size, model_args.ns
    ).to(device)

    for parameter in model.parameters():
        parameter.requires_grad = False
    selected = {}
    last = model_args.num_conv_layers - 1
    peptide_prefixes = ("peptide_flow.ga_encoder.", "peptide_flow.node_embedder.aatype_embed.")
    receptor_prefixes = (
        f"receptor_flow.lig_conv_layers.{last}.",
        f"receptor_flow.rec_conv_layers.{last}.",
        f"receptor_flow.lig_to_rec_conv_layers.{last}.",
        f"receptor_flow.rec_to_lig_conv_layers.{last}.",
        "receptor_flow.res_tr_final_layer.",
        "receptor_flow.res_rot_final_layer.",
        "receptor_flow.res_chi_final_layer.",
    )
    for name, parameter in model.named_parameters():
        if name.startswith("condition_adapter.") or name.startswith(peptide_prefixes) or name.startswith(receptor_prefixes):
            parameter.requires_grad = True
            selected[name] = parameter
    if not selected:
        raise RuntimeError("trainable parameter selection is empty")

    collate = PaddingCollate(eight=False)
    graph_cache_root = Path(args.graph_cache)
    graph_cache_root.mkdir(parents=True, exist_ok=True)
    loaded_cases, preprocessing_failures = {}, {}

    def basis(frame):
        n, c = frame[:, 0] - frame[:, 1], frame[:, 2] - frame[:, 1]
        x = c / (torch.linalg.vector_norm(c, dim=1, keepdim=True) + 1e-8)
        z = torch.linalg.cross(x, n, dim=1)
        z = z / (torch.linalg.vector_norm(z, dim=1, keepdim=True) + 1e-8)
        y = torch.linalg.cross(z, x, dim=1)
        return torch.stack([x, y, z], dim=-1)

    def load_case(case_id):
        if case_id in loaded_cases:
            return loaded_cases[case_id]
        case_dir = data_dir / "cases" / case_id
        peptide_items = []
        for label in ("a", "b"):
            item = preprocess_structure({"id": f"{case_id}_{label}", "pdb_path": str(case_dir / f"pepflow_{label}")})
            if item is None:
                raise RuntimeError(f"PepFlow preprocessing failed for {case_id}/{label}")
            peptide_items.append(item)
        protein_paths = [
            case_dir / "anchor_a.pdb", case_dir / "anchor_b.pdb",
            case_dir / "holo_a.pdb", case_dir / "holo_b.pdb",
        ]
        ligands = [
            case_dir / "peptide_a.sdf", case_dir / "peptide_b.sdf",
            case_dir / "peptide_a.sdf", case_dir / "peptide_b.sdf",
        ]
        cache_path = str(graph_cache_root / case_id)
        dataset = PDBBind(
            transform=None, root="", name_list=[f"{case_id}_{index}" for index in range(4)],
            protein_path_list=[str(path) for path in protein_paths],
            ligand_descriptions=[str(path) for path in ligands],
            receptor_radius=model_args.receptor_radius, cache_path=cache_path,
            remove_hs=model_args.remove_hs, max_lig_size=None,
            c_alpha_max_neighbors=model_args.c_alpha_max_neighbors, matching=False,
            keep_original=False, popsize=model_args.matching_popsize,
            maxiter=model_args.matching_maxiter, center_ligand=False,
            all_atoms=model_args.all_atoms, atom_radius=model_args.atom_radius,
            atom_max_neighbors=model_args.atom_max_neighbors,
            esm_embeddings_path=str(case_dir / "esm2_output"),
            require_ligand=True, require_receptor=False, num_workers=1,
            keep_local_structures=True, use_existing_cache=Path(cache_path + "_torsion").exists(),
        )
        graphs = [dataset.get(index) for index in range(4)]
        targets = []
        for index in range(2):
            anchor, target = graphs[index], graphs[index + 2]
            if anchor["receptor"].num_nodes != target["receptor"].num_nodes:
                raise RuntimeError(f"receptor graph shape mismatch for {case_id}/{index}")
            translation = target["receptor"].pos - anchor["receptor"].pos
            rotation_matrix = basis(target["receptor"].lf_3pts) @ basis(anchor["receptor"].lf_3pts).transpose(1, 2)
            rotation = torch.from_numpy(Rotation.from_matrix(rotation_matrix.numpy()).as_rotvec()).float()
            chi_columns = [0, 2, 4, 5, 6]
            chi_delta = target["receptor"].chis[:, chi_columns] - anchor["receptor"].chis[:, chi_columns]
            chi_delta = torch.atan2(torch.sin(chi_delta), torch.cos(chi_delta))
            chi_mask = target["receptor"].chi_masks[:, chi_columns] * anchor["receptor"].chi_masks[:, chi_columns]
            distances = torch.cdist(target["receptor"].pos, target["ligand"].pos)
            pocket = distances.min(1).values <= 12.0
            targets.append({
                "translation": translation, "rotation": rotation, "chi": chi_delta,
                "chi_mask": chi_mask, "weight": 0.1 + 0.9 * pocket.float(), "pocket": pocket,
                "anchor_pos": anchor["receptor"].pos, "target_pos": target["receptor"].pos,
            })
        loaded_cases[case_id] = {"peptide_items": peptide_items, "graphs": graphs, "targets": targets}
        return loaded_cases[case_id]

    required_ids = sorted(set(train_ids).union(*evaluation_ids.values()))
    for case_id in required_ids:
        try:
            load_case(case_id)
        except Exception as error:
            preprocessing_failures[case_id] = repr(error)
            print(json.dumps({"preprocessing_failure": case_id, "error": repr(error)}), flush=True)
    train_ids = [case_id for case_id in train_ids if case_id not in preprocessing_failures]
    evaluation_ids = {
        name: [case_id for case_id in values if case_id not in preprocessing_failures]
        for name, values in evaluation_ids.items()
    }
    if len(train_ids) < args.minimum_train_cases or any(not values for values in evaluation_ids.values()):
        raise RuntimeError(f"too many preprocessing failures: {preprocessing_failures}")

    def peptide_batch(item):
        return recursive_to(collate([item]), device)

    def target_on_device(target):
        return {name: value.to(device) for name, value in target.items()}

    def graph_batch(graph, noise=0.0):
        copied = copy.deepcopy(graph)
        if noise:
            copied["ligand"].pos = copied["ligand"].pos + noise * torch.randn_like(copied["ligand"].pos)
        set_time(copied, args.time, args.time, args.time, args.time, args.time, args.time,
                 batchsize=1, all_atoms=False, device=None)
        return Batch.from_data_list([copied]).to(device)

    def component_error(prediction, target):
        tr, rot, chi = prediction[5], prediction[6], prediction[7]
        weight = target["weight"]
        tr_loss = (torch.nn.functional.smooth_l1_loss(tr, target["translation"], reduction="none").mean(1) * weight).sum() / weight.sum()
        rot_loss = (torch.nn.functional.smooth_l1_loss(rot, target["rotation"], reduction="none").mean(1) * weight).sum() / weight.sum()
        chi_raw = torch.nn.functional.smooth_l1_loss(chi, target["chi"], reduction="none")
        chi_weight = target["chi_mask"] * weight[:, None]
        chi_loss = (chi_raw * chi_weight).sum() / (chi_weight.sum() + 1e-8)
        total = tr_loss + args.rotation_weight * rot_loss + args.chi_weight * chi_loss
        return total, tr_loss, rot_loss, chi_loss

    def relaxation_loss(prediction, target):
        tr, rot, chi = prediction[5], prediction[6], prediction[7]
        weight = target["weight"]
        tr_term = (tr.square().mean(1) * weight).sum() / weight.sum()
        rot_term = (rot.square().mean(1) * weight).sum() / weight.sum()
        chi_weight = target["chi_mask"] * weight[:, None]
        chi_term = (chi.square() * chi_weight).sum() / (chi_weight.sum() + 1e-8)
        return tr_term + args.rotation_weight * rot_term + args.chi_weight * chi_term

    def peptide_denoising(batch, fixed_seed):
        target_trans = model.peptide_flow.encode(batch)[1]
        mask, captured = batch["generate_mask"].bool(), {}

        def capture_prediction(_module, inputs, output):
            captured["noisy"] = inputs[2].detach()
            captured["predicted"] = output[1].detach()

        handle = model.peptide_flow.ga_encoder.register_forward_hook(capture_prediction)
        try:
            with torch.random.fork_rng(devices=[device]):
                torch.manual_seed(fixed_seed)
                torch.cuda.manual_seed_all(fixed_seed)
                losses = model.peptide_flow(batch)
        finally:
            handle.remove()

        def rmsd(trans):
            return float(torch.sqrt(((trans - target_trans).square().sum(-1))[mask].mean()))

        return {
            "noisy_ca_rmsd": rmsd(captured["noisy"]),
            "peptide_pose_rmsd": rmsd(captured["predicted"]),
            "peptide_flow_loss": float(sum_weighted_losses(losses, config.train.loss_weights)),
        }

    def evaluate_case(case_id, split_name, phase):
        data = load_case(case_id)
        batches = [peptide_batch(item) for item in data["peptide_items"]]
        targets = [target_on_device(target) for target in data["targets"]]
        rows = []
        model.eval()
        with torch.no_grad():
            for index in range(2):
                outputs = {}
                for control in ("normal", "shuffle", "zero"):
                    condition = batches[1 - index] if control == "shuffle" else None
                    outputs[control] = model.receptor_forward(
                        batches[index], graph_batch(data["graphs"][index]),
                        condition_batch=condition, zero_condition=(control == "zero"),
                    )
                normal = outputs["normal"]
                correct = component_error(normal, targets[index])[0]
                wrong = component_error(normal, targets[1 - index])[0]
                shuffle_error = component_error(outputs["shuffle"], targets[index])[0]
                zero_error = component_error(outputs["zero"], targets[index])[0]
                tr = normal[5]
                pocket = targets[index]["pocket"]
                predicted_pos = targets[index]["anchor_pos"] + tr
                pocket_rmsd = torch.sqrt(
                    (predicted_pos[pocket] - targets[index]["target_pos"][pocket]).square().sum(-1).mean()
                )
                anchor_rmsd = torch.sqrt(
                    (targets[index]["anchor_pos"][pocket] - targets[index]["target_pos"][pocket]).square().sum(-1).mean()
                )
                denoising = peptide_denoising(
                    batches[index], args.seed + 100000 + 1000 * list(case_records).index(case_id) + index
                )
                rows.append({
                    "phase": phase, "split": split_name, "case_id": case_id, "state": index,
                    **denoising, "endpoint_error": float(correct), "wrong_endpoint_error": float(wrong),
                    "correct_state_margin": float(wrong - correct),
                    "selected_correct": bool(correct < wrong),
                    "shuffle_endpoint_error": float(shuffle_error),
                    "shuffle_degradation": float(shuffle_error - correct),
                    "zero_endpoint_error": float(zero_error),
                    "anchor_pocket_rmsd": float(anchor_rmsd), "pocket_rmsd": float(pocket_rmsd),
                    "pocket_rmsd_improvement": float(anchor_rmsd - pocket_rmsd),
                    "receptor_movement_magnitude": float(torch.sqrt(tr.square().sum(-1).mean())),
                    "sequence_recovery": None,
                })
        del batches, targets
        torch.cuda.empty_cache()
        return rows

    evaluation_rows = []
    for split_name, case_ids in evaluation_ids.items():
        for case_id in case_ids:
            evaluation_rows.extend(evaluate_case(case_id, split_name, "before"))

    optimizer = torch.optim.AdamW(selected.values(), lr=args.learning_rate, weight_decay=1e-4)
    training_rows, order, cursor = [], [], 0
    model.eval()
    for step in range(1, args.steps + 1):
        if cursor + args.cases_per_step > len(order):
            order = list(train_ids)
            random.Random(args.seed + step).shuffle(order)
            cursor = 0
        case_batch = order[cursor:cursor + args.cases_per_step]
        cursor += args.cases_per_step
        optimizer.zero_grad(set_to_none=True)
        accumulators = {name: [] for name in (
            "loss", "peptide_loss", "endpoint_loss", "ranking_loss", "relaxation_loss",
            "translation_loss", "rotation_loss", "chi_loss",
        )}
        for case_id in case_batch:
            data = load_case(case_id)
            batches = [peptide_batch(item) for item in data["peptide_items"]]
            targets = [target_on_device(target) for target in data["targets"]]
            per_state = []
            for index in range(2):
                peptide_losses, output = model(
                    batches[index], graph_batch(data["graphs"][index], args.peptide_noise)
                )
                peptide_value = sum_weighted_losses(peptide_losses, config.train.loss_weights)
                correct, tr_loss, rot_loss, chi_loss = component_error(output, targets[index])
                wrong = component_error(output, targets[1 - index])[0]
                ranking = torch.relu(args.ranking_margin + correct - wrong)
                relaxation = relaxation_loss(output, targets[index])
                total = (
                    args.peptide_weight * peptide_value + correct
                    + args.ranking_weight * ranking + args.relaxation_weight * relaxation
                )
                per_state.append(total)
                for name, value in (
                    ("peptide_loss", peptide_value), ("endpoint_loss", correct),
                    ("ranking_loss", ranking), ("relaxation_loss", relaxation),
                    ("translation_loss", tr_loss), ("rotation_loss", rot_loss), ("chi_loss", chi_loss),
                ):
                    accumulators[name].append(value.detach())
            case_loss = torch.stack(per_state).mean()
            (case_loss / len(case_batch)).backward()
            accumulators["loss"].append(case_loss.detach())
            del batches, targets, per_state
        gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), args.max_grad_norm)
        optimizer.step()
        row = {"step": step, "cases": ";".join(case_batch), "gradient_norm": float(gradient_norm)}
        row.update({name: float(torch.stack(values).mean()) for name, values in accumulators.items()})
        training_rows.append(row)
        print(json.dumps(row), flush=True)

    for split_name, case_ids in evaluation_ids.items():
        for case_id in case_ids:
            evaluation_rows.extend(evaluate_case(case_id, split_name, "after"))

    def aggregate(split_name, phase):
        rows = [row for row in evaluation_rows if row["split"] == split_name and row["phase"] == phase]
        numeric = (
            "peptide_pose_rmsd", "peptide_flow_loss", "endpoint_error", "correct_state_margin",
            "shuffle_degradation", "anchor_pocket_rmsd", "pocket_rmsd",
            "pocket_rmsd_improvement", "receptor_movement_magnitude",
        )
        return {
            "states": len(rows),
            **{name: sum(row[name] for row in rows) / len(rows) for name in numeric},
            "correct_state_accuracy": sum(row["selected_correct"] for row in rows) / len(rows),
            "sequence_recovery": None,
            "sequence_recovery_status": "not_enabled_until_pose_and_selectivity_gate_passes",
        }

    result_dir = Path(args.result_dir)
    with (result_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(training_rows[0]))
        writer.writeheader(); writer.writerows(training_rows)
    with (result_dir / "evaluation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evaluation_rows[0]))
        writer.writeheader(); writer.writerows(evaluation_rows)
    torch.save({
        "base_dynamicbind_sha256": sha256(checkpoint),
        "base_pepflow_sha256": sha256(args.pepflow_checkpoint),
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "trainable_state_dict": {name: parameter.detach().cpu() for name, parameter in selected.items()},
    }, result_dir / "trainable_delta.pt")
    aggregates = {
        split_name: {phase: aggregate(split_name, phase) for phase in ("before", "after")}
        for split_name in evaluation_ids
    }
    summary = {
        "parameters": {"total": sum(p.numel() for p in model.parameters()),
                       "trainable": sum(p.numel() for p in selected.values()), "names": list(selected)},
        "pepflow_initialization": "checkpoint",
        "pepflow_checkpoint_sha256": sha256(args.pepflow_checkpoint),
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
        "split": args.split, "train_cases": train_ids, "evaluation_cases": evaluation_ids,
        "preprocessing_failures": preprocessing_failures, "aggregates": aggregates,
        "initial_loss": training_rows[0]["loss"], "final_loss": training_rows[-1]["loss"],
        "small_batch_gate": bool(
            aggregates["peptide_cluster"]["after"]["peptide_pose_rmsd"]
            < aggregates["peptide_cluster"]["before"]["peptide_pose_rmsd"]
            and aggregates["receptor_family_proxy"]["after"]["correct_state_accuracy"] > 0.5
            and aggregates["receptor_family_proxy"]["after"]["shuffle_degradation"] > 0
        ),
    }
    (result_dir / "worker_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--graph-cache", required=True)
    parser.add_argument("--pepflow-config", default="pepflow/configs/learn_angle.yaml")
    parser.add_argument("--pepflow-checkpoint", required=True)
    parser.add_argument("--dynamicbind-checkpoint", default="ema_inference_epoch314_model.pt")
    parser.add_argument("--split", default="random_complex", choices=("random_complex", "peptide_cluster", "receptor_family_proxy"))
    parser.add_argument("--visible-gpus", default="5")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cases-per-step", type=int, default=2)
    parser.add_argument("--max-train-cases", type=int, default=0)
    parser.add_argument("--max-eval-cases", type=int, default=0)
    parser.add_argument("--minimum-train-cases", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--peptide-noise", type=float, default=0.5)
    parser.add_argument("--time", type=float, default=0.6)
    parser.add_argument("--peptide-weight", type=float, default=0.1)
    parser.add_argument("--rotation-weight", type=float, default=0.33)
    parser.add_argument("--chi-weight", type=float, default=0.33)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--relaxation-weight", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0

    require_clean_commit()
    paths = {
        "dataset_manifest": Path(args.data_dir).resolve() / "manifest.json",
        "model_parameters": Path(args.model_dir).resolve() / "model_parameters.yml",
        "dynamicbind_checkpoint": Path(args.model_dir).resolve() / args.dynamicbind_checkpoint,
        "pepflow_config": (REPO_ROOT / args.pepflow_config).resolve(),
        "pepflow_checkpoint": Path(args.pepflow_checkpoint).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    started, start_clock = datetime.now().astimezone(), time.monotonic()
    result_dir = RESULTS_ROOT / started.strftime("%Y%m%d-%H%M%S%z")
    result_dir.mkdir(parents=True)
    commit, message = git("rev-parse", "HEAD"), git("log", "-1", "--format=%B")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    for name, value in vars(args).items():
        if name in ("worker", "result_dir"):
            continue
        option = "--" + name.replace("_", "-")
        command.extend([option, str(value)])
    command.extend(["--result-dir", str(result_dir)])
    metadata = {
        "started_at": started.isoformat(), "finished_at": None, "status": "running", "exit_code": None,
        "branch": git("branch", "--show-current"), "commit": commit, "commit_message": message.strip(),
        "command": command, "python": sys.executable, "cuda_visible_devices": args.visible_gpus,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "gpu_before": gpu_snapshot(), "gpu_after": None,
    }
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    environment["WANDB_MODE"] = "disabled"
    process = subprocess.run(
        command, cwd=REPO_ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    (result_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (result_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    print(process.stdout, end="")
    if process.stderr:
        print(process.stderr, end="", file=sys.stderr)
    summary_path = result_dir / "worker_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    finished = datetime.now().astimezone()
    metadata.update({
        "finished_at": finished.isoformat(), "status": "passed" if process.returncode == 0 else "failed",
        "exit_code": process.returncode, "elapsed_seconds": time.monotonic() - start_clock,
        "summary": summary, "gpu_after": gpu_snapshot(),
    })
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Joint small-batch train</title></head><body>"
        f"<h1>{metadata['status']}</h1><p>Commit: {commit}</p><pre>{json.dumps(summary, indent=2)}</pre>"
        "</body></html>\n", encoding="utf-8",
    )
    print(f"Joint small-batch training {metadata['status']}: {result_dir}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
