#!/usr/bin/env python3
"""Overfit the retained PepFlow + DynamicBind joint architecture on two states."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "joint-overfit"


def git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


def require_clean_commit():
    status = git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return
    print("ERROR: joint training must run from a committed worktree.", file=sys.stderr)
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
    snapshot = {}
    for label, command in {
        "devices": [
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader",
        ],
        "processes": [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
    }.items():
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        snapshot[label] = {"exit_code": result.returncode, "output": result.stdout.strip()}
    return snapshot


def worker(args):
    import copy
    import csv
    from argparse import Namespace
    from functools import partial
    import random

    import numpy as np
    from scipy.spatial.transform import Rotation
    import torch
    from torch_geometric.data import Batch
    import yaml

    pepflow_root = REPO_ROOT / "pepflow"
    dynamicbind_root = REPO_ROOT / "dynamicbind"
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
    from utils.diffusion_utils import modify_conformer, set_time, t_to_sigma as t_to_sigma_compl
    from utils.utils import get_model

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    case_dir = Path(args.case_dir).resolve()
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / args.dynamicbind_checkpoint
    with (model_dir / "model_parameters.yml").open() as handle:
        model_args = Namespace(**yaml.full_load(handle))

    config, _ = load_config(args.pepflow_config)
    peptide_flow = FlowModel(config.model)
    pepflow_initialization = "random"
    if args.pepflow_checkpoint:
        pep_checkpoint = torch.load(args.pepflow_checkpoint, map_location="cpu")
        peptide_flow.load_state_dict(process_dic(pep_checkpoint.get("model", pep_checkpoint)), strict=True)
        pepflow_initialization = "checkpoint"
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
    pep_prefixes = ("peptide_flow.ga_encoder.", "peptide_flow.node_embedder.aatype_embed.")
    rec_prefixes = (
        f"receptor_flow.lig_conv_layers.{last}.",
        f"receptor_flow.rec_conv_layers.{last}.",
        f"receptor_flow.lig_to_rec_conv_layers.{last}.",
        f"receptor_flow.rec_to_lig_conv_layers.{last}.",
        "receptor_flow.res_tr_final_layer.",
        "receptor_flow.res_rot_final_layer.",
        "receptor_flow.res_chi_final_layer.",
    )
    for name, parameter in model.named_parameters():
        if name.startswith("condition_adapter.") or name.startswith(pep_prefixes) or name.startswith(rec_prefixes):
            parameter.requires_grad = True
            selected[name] = parameter
    if not selected:
        raise RuntimeError("trainable parameter selection is empty")

    collate = PaddingCollate(eight=False)
    peptide_batches = []
    for label in ("a", "b"):
        item = preprocess_structure({"id": f"joint_{label}", "pdb_path": str(case_dir / f"pepflow_{label}")})
        if item is None:
            raise RuntimeError(f"PepFlow preprocessing failed for state {label}")
        peptide_batches.append(recursive_to(collate([item]), device))

    protein_paths = [
        case_dir / "common_anchor.pdb", case_dir / "common_anchor.pdb",
        case_dir / "holo_a.pdb", case_dir / "holo_b.pdb",
    ]
    ligands = [
        case_dir / "peptide_a.sdf", case_dir / "peptide_b.sdf",
        case_dir / "peptide_a.sdf", case_dir / "peptide_b.sdf",
    ]
    dataset = PDBBind(
        transform=None, root="", name_list=[f"joint_{i}" for i in range(4)],
        protein_path_list=[str(path) for path in protein_paths],
        ligand_descriptions=[str(path) for path in ligands],
        receptor_radius=model_args.receptor_radius, cache_path=args.graph_cache,
        remove_hs=model_args.remove_hs, max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors, matching=False,
        keep_original=False, popsize=model_args.matching_popsize,
        maxiter=model_args.matching_maxiter, center_ligand=False,
        all_atoms=model_args.all_atoms, atom_radius=model_args.atom_radius,
        atom_max_neighbors=model_args.atom_max_neighbors,
        esm_embeddings_path=str(case_dir / "esm2_output"),
        require_ligand=True, require_receptor=False, num_workers=1,
        keep_local_structures=True, use_existing_cache=Path(args.graph_cache + "_torsion").exists(),
    )
    graphs = [dataset.get(index) for index in range(4)]
    if len(graphs) != 4 or any(graph["receptor"].num_nodes != 257 for graph in graphs):
        raise RuntimeError("joint receptor graph shape contract failed")

    def basis(frame):
        n = frame[:, 0] - frame[:, 1]
        c = frame[:, 2] - frame[:, 1]
        x = c / (torch.linalg.vector_norm(c, dim=1, keepdim=True) + 1e-8)
        z = torch.linalg.cross(x, n, dim=1)
        z = z / (torch.linalg.vector_norm(z, dim=1, keepdim=True) + 1e-8)
        y = torch.linalg.cross(z, x, dim=1)
        return torch.stack([x, y, z], dim=-1)

    targets = []
    for index in range(2):
        anchor, target = graphs[index], graphs[index + 2]
        translation = target["receptor"].pos - anchor["receptor"].pos
        rotation_matrix = basis(target["receptor"].lf_3pts) @ basis(anchor["receptor"].lf_3pts).transpose(1, 2)
        rotation = torch.from_numpy(Rotation.from_matrix(rotation_matrix.numpy()).as_rotvec()).float()
        chi_columns = [0, 2, 4, 5, 6]
        chi_delta = target["receptor"].chis[:, chi_columns] - anchor["receptor"].chis[:, chi_columns]
        chi_delta = torch.atan2(torch.sin(chi_delta), torch.cos(chi_delta))
        chi_mask = target["receptor"].chi_masks[:, chi_columns] * anchor["receptor"].chi_masks[:, chi_columns]
        distances = torch.cdist(target["receptor"].pos, target["ligand"].pos)
        pocket = (distances.min(1).values <= 12.0).float()
        targets.append({
            "translation": translation.to(device), "rotation": rotation.to(device),
            "chi": chi_delta.to(device), "chi_mask": chi_mask.to(device),
            "weight": (0.1 + 0.9 * pocket).to(device),
        })

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

    def evaluate(control="normal"):
        model.eval()
        rows = []
        with torch.no_grad():
            for index in range(2):
                condition = peptide_batches[1 - index] if control == "shuffle" else None
                output = model.receptor_forward(
                    peptide_batches[index], graph_batch(graphs[index]),
                    condition_batch=condition, zero_condition=(control == "zero"),
                )
                correct = component_error(output, targets[index])[0]
                wrong = component_error(output, targets[1 - index])[0]
                rows.append({
                    "state": index, "correct_error": float(correct), "wrong_error": float(wrong),
                    "margin": float(wrong - correct), "selected_correct": bool(correct < wrong),
                })
        return rows

    def evaluate_peptide_denoising():
        model.peptide_flow.eval()
        rows = []
        with torch.no_grad():
            for index, batch in enumerate(peptide_batches):
                captured = {}

                def capture_prediction(module, inputs, output):
                    captured["noisy_trans"] = inputs[2].detach()
                    captured["pred_trans"] = output[1].detach()

                handle = model.peptide_flow.ga_encoder.register_forward_hook(capture_prediction)
                try:
                    with torch.random.fork_rng(devices=[device]):
                        torch.manual_seed(args.seed + 1000 + index)
                        torch.cuda.manual_seed_all(args.seed + 1000 + index)
                        losses = model.peptide_flow(batch)
                finally:
                    handle.remove()
                target_trans = model.peptide_flow.encode(batch)[1]
                mask = batch["generate_mask"].bool()

                def ca_rmsd(trans):
                    squared = ((trans - target_trans) ** 2).sum(-1)
                    return float(torch.sqrt(squared[mask].mean()))

                rows.append({
                    "state": index,
                    "noisy_ca_rmsd": ca_rmsd(captured["noisy_trans"]),
                    "denoised_ca_rmsd": ca_rmsd(captured["pred_trans"]),
                    "weighted_flow_loss": float(sum_weighted_losses(losses, config.train.loss_weights)),
                })
        return rows

    before = {name: evaluate(name) for name in ("normal", "shuffle", "zero")}
    peptide_denoising_before = evaluate_peptide_denoising()
    optimizer = torch.optim.AdamW(selected.values(), lr=args.learning_rate, weight_decay=1e-4)
    rows = []
    model.eval()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        peptide_loss_values, endpoint_values, rank_values = [], [], []
        component_values = []
        for index in range(2):
            peptide_losses, output = model(peptide_batches[index], graph_batch(graphs[index], args.peptide_noise))
            peptide_loss = sum_weighted_losses(peptide_losses, config.train.loss_weights)
            correct, tr_loss, rot_loss, chi_loss = component_error(output, targets[index])
            wrong = component_error(output, targets[1 - index])[0]
            rank = torch.relu(args.ranking_margin + correct - wrong)
            peptide_loss_values.append(peptide_loss)
            endpoint_values.append(correct)
            rank_values.append(rank)
            component_values.append((tr_loss, rot_loss, chi_loss))
        peptide_loss = torch.stack(peptide_loss_values).mean()
        endpoint_loss = torch.stack(endpoint_values).mean()
        ranking_loss = torch.stack(rank_values).mean()
        loss = args.peptide_weight * peptide_loss + endpoint_loss + args.ranking_weight * ranking_loss
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(selected.values(), args.max_grad_norm)
        optimizer.step()
        row = {
            "step": step, "loss": float(loss.detach()),
            "peptide_loss": float(peptide_loss.detach()), "endpoint_loss": float(endpoint_loss.detach()),
            "ranking_loss": float(ranking_loss.detach()), "translation_loss": float(torch.stack([x[0] for x in component_values]).mean().detach()),
            "rotation_loss": float(torch.stack([x[1] for x in component_values]).mean().detach()),
            "chi_loss": float(torch.stack([x[2] for x in component_values]).mean().detach()),
            "gradient_norm": float(gradient_norm),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    after = {name: evaluate(name) for name in ("normal", "shuffle", "zero")}
    peptide_denoising_after = evaluate_peptide_denoising()
    peptide_samples = []
    model.peptide_flow.eval()
    with torch.no_grad():
        for index, batch in enumerate(peptide_batches):
            trajectory = model.peptide_flow.sample(
                batch, num_steps=args.sample_steps, sample_bb=True, sample_ang=True, sample_seq=False
            )
            final = trajectory[-1]
            mask = batch["generate_mask"].cpu().bool()
            squared = ((final["trans"] - final["trans_1"]) ** 2).sum(-1)
            peptide_samples.append({
                "state": index, "ca_rmsd": float(torch.sqrt(squared[mask].mean())),
            })

    result_dir = Path(args.result_dir)
    with (result_dir / "training_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    torch.save({
        "base_dynamicbind_sha256": sha256(checkpoint),
        "pepflow_initialization": pepflow_initialization,
        "trainable_state_dict": {name: parameter.detach().cpu() for name, parameter in selected.items()},
    }, result_dir / "trainable_delta.pt")
    normal_after = after["normal"]
    initial_loss, final_loss = rows[0]["loss"], rows[-1]["loss"]
    summary = {
        "pepflow_initialization": pepflow_initialization,
        "parameters": {
            "total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in selected.values()),
            "names": list(selected),
        },
        "before": before, "after": after,
        "peptide_denoising_before": peptide_denoising_before,
        "peptide_denoising_after": peptide_denoising_after,
        "peptide_samples": peptide_samples,
        "initial_loss": initial_loss, "final_loss": final_loss,
        "loss_reduction_fraction": (initial_loss - final_loss) / max(abs(initial_loss), 1e-8),
        "joint_overfit_gate": bool(
            final_loss < initial_loss
            and all(row["selected_correct"] for row in normal_after)
            and all(
                after_row["denoised_ca_rmsd"] < before_row["denoised_ca_rmsd"]
                and after_row["denoised_ca_rmsd"] < after_row["noisy_ca_rmsd"]
                for before_row, after_row in zip(peptide_denoising_before, peptide_denoising_after)
            )
        ),
    }
    (result_dir / "worker_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--graph-cache", required=True)
    parser.add_argument("--pepflow-config", default="pepflow/configs/learn_angle.smoke.yaml")
    parser.add_argument("--pepflow-checkpoint")
    parser.add_argument("--dynamicbind-checkpoint", default="ema_inference_epoch314_model.pt")
    parser.add_argument("--visible-gpus", default="5")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--peptide-noise", type=float, default=0.5)
    parser.add_argument("--time", type=float, default=0.6)
    parser.add_argument("--peptide-weight", type=float, default=0.1)
    parser.add_argument("--rotation-weight", type=float, default=0.33)
    parser.add_argument("--chi-weight", type=float, default=0.33)
    parser.add_argument("--ranking-weight", type=float, default=0.5)
    parser.add_argument("--ranking-margin", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0

    require_clean_commit()
    paths = {
        "case_manifest": Path(args.case_dir).resolve() / "manifest.json",
        "model_parameters": Path(args.model_dir).resolve() / "model_parameters.yml",
        "dynamicbind_checkpoint": Path(args.model_dir).resolve() / args.dynamicbind_checkpoint,
        "pepflow_config": (REPO_ROOT / args.pepflow_config).resolve(),
    }
    if args.pepflow_checkpoint:
        paths["pepflow_checkpoint"] = Path(args.pepflow_checkpoint).resolve()
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    started = datetime.now().astimezone()
    start_clock = time.monotonic()
    result_dir = RESULTS_ROOT / started.strftime("%Y%m%d-%H%M%S%z")
    result_dir.mkdir(parents=True)
    commit = git("rev-parse", "HEAD")
    message = git("log", "-1", "--format=%B")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
        "--case-dir", args.case_dir, "--model-dir", args.model_dir,
        "--graph-cache", args.graph_cache, "--pepflow-config", args.pepflow_config,
        "--dynamicbind-checkpoint", args.dynamicbind_checkpoint,
        "--visible-gpus", args.visible_gpus, "--steps", str(args.steps),
        "--sample-steps", str(args.sample_steps), "--learning-rate", str(args.learning_rate),
        "--peptide-noise", str(args.peptide_noise), "--time", str(args.time),
        "--peptide-weight", str(args.peptide_weight), "--rotation-weight", str(args.rotation_weight),
        "--chi-weight", str(args.chi_weight), "--ranking-weight", str(args.ranking_weight),
        "--ranking-margin", str(args.ranking_margin), "--max-grad-norm", str(args.max_grad_norm),
        "--seed", str(args.seed), "--result-dir", str(result_dir)]
    if args.pepflow_checkpoint:
        command.extend(["--pepflow-checkpoint", args.pepflow_checkpoint])
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
    process = subprocess.run(command, cwd=REPO_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        "<html><head><meta charset='utf-8'><title>Joint peptide receptor overfit</title></head><body>"
        f"<h1>Joint overfit: {metadata['status']}</h1><p>Commit: {commit}</p>"
        f"<pre>{json.dumps(summary, indent=2)}</pre></body></html>\n", encoding="utf-8",
    )
    print(f"Joint overfit {metadata['status']}: {result_dir}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
