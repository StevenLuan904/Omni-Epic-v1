#!/usr/bin/env python3
"""Extract frozen DynamicBind ligand-to-receptor interaction features."""

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime
from functools import partial
from glob import glob
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time
import traceback

DYNAMICBIND_ROOT = Path(__file__).resolve().parents[1]
if str(DYNAMICBIND_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAMICBIND_ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from argparse import Namespace
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation
from torch_geometric.data import Batch

from datasets.pdbbind import PDBBind
from utils.diffusion_utils import set_time, t_to_sigma as t_to_sigma_compl
from utils.utils import get_model


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" /
    "frozen-interaction-features"
)


def git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


def require_clean_commit():
    status = git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return
    print("ERROR: feature extraction requires a committed worktree.", file=sys.stderr)
    print("\n--- git status --short ---", file=sys.stderr)
    print(status, file=sys.stderr)
    for title, args in (
        ("git diff", ("diff", "--no-ext-diff")),
        ("git diff --cached", ("diff", "--cached", "--no-ext-diff")),
    ):
        diff = git(*args, check=False)
        print(f"\n--- {title} ---", file=sys.stderr)
        if diff:
            print(diff, file=sys.stderr)
    raise SystemExit(2)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_parameter_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Tee:
    def __init__(self, stream, log_path):
        self.stream = stream
        self.log = Path(log_path).open("a", encoding="utf-8", buffering=1)

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def gpu_snapshot():
    snapshot = {}
    for label, command in {
        "devices": [
            "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,"
            "memory.free,utilization.gpu", "--format=csv,noheader",
        ],
        "processes": [
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader",
        ],
    }.items():
        completed = subprocess.run(
            command, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False,
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


def canonical_smiles(path):
    molecule = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    if molecule is None:
        raise ValueError(f"RDKit could not read {path}")
    return Chem.MolToSmiles(Chem.RemoveHs(molecule), isomericSmiles=True)


def rebuild_ligand(source_path, destination, seed):
    smiles = canonical_smiles(source_path)
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    parameters = AllChem.ETKDGv2()
    parameters.randomSeed = int(seed % (2**31 - 1))
    if AllChem.EmbedMolecule(molecule, parameters) != 0:
        raise RuntimeError(f"seeded ETKDG failed for {source_path}")
    AllChem.MMFFOptimizeMolecule(
        molecule, mmffVariant="MMFF94s", maxIters=500
    )
    writer = Chem.SDWriter(str(destination))
    writer.write(molecule)
    writer.close()
    return smiles


def build_dataset(frame, cache_path, esm_embeddings, model_args, seed):
    protein_paths = [str(Path(path).resolve()) for path in frame["protein_path"]]
    ligand_paths = [Path(path).resolve() for path in frame["ligand"]]
    for path in [*map(Path, protein_paths), *ligand_paths]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(set(protein_paths)) != 1:
        raise ValueError("both ligand conditions must use one receptor anchor")
    if len(ligand_paths) != 2:
        raise ValueError("exactly two ligand conditions are required")
    rebuilt_dir = cache_path.parent / "rebuilt_ligands"
    rebuilt_dir.mkdir(parents=True)
    rebuilt_paths = [rebuilt_dir / f"state_{label}.sdf" for label in range(2)]
    smiles = [
        rebuild_ligand(source, rebuilt, seed + label)
        for label, (source, rebuilt) in enumerate(zip(ligand_paths, rebuilt_paths))
    ]
    dataset = PDBBind(
        transform=None, root="", name_list=["state_A", "state_B"],
        protein_path_list=protein_paths,
        ligand_descriptions=[str(path) for path in rebuilt_paths],
        receptor_radius=model_args.receptor_radius,
        cache_path=str(cache_path), remove_hs=model_args.remove_hs,
        max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors,
        matching=False, keep_original=False,
        popsize=model_args.matching_popsize,
        maxiter=model_args.matching_maxiter, center_ligand=True,
        all_atoms=model_args.all_atoms, atom_radius=model_args.atom_radius,
        atom_max_neighbors=model_args.atom_max_neighbors,
        esm_embeddings_path=(
            str(esm_embeddings)
            if model_args.esm_embeddings_path is not None else None
        ),
        require_ligand=True, require_receptor=False, num_workers=1,
        keep_local_structures=True, use_existing_cache=False,
    )
    graphs = [dataset.get(index) for index in range(len(dataset))]
    if len(graphs) != 2:
        raise ValueError(f"expected two graphs, found {len(graphs)}")
    return (
        graphs, protein_paths[0], ligand_paths, rebuilt_paths, smiles,
        dataset.full_cache_path,
    )


def positioned_graph(graph, placement_seed, time_value, device):
    graph = copy.deepcopy(graph)
    generator = np.random.RandomState(placement_seed)
    rotation = Rotation.random(random_state=generator).as_matrix().astype(np.float32)
    ligand_pos = graph["ligand"].pos.float()
    ligand_pos = ligand_pos - ligand_pos.mean(dim=0, keepdim=True)
    ligand_pos = ligand_pos @ torch.from_numpy(rotation).T
    receptor_index = placement_seed % graph["receptor"].num_nodes
    offset = torch.from_numpy(generator.normal(0.0, 0.35, size=3).astype(np.float32))
    graph["ligand"].pos = ligand_pos + graph["receptor"].pos[receptor_index] + offset
    set_time(
        graph, time_value, time_value, time_value, time_value, time_value,
        time_value, batchsize=1, all_atoms=False, device=None,
    )
    return Batch.from_data_list([graph]).to(device)


def pooled_last_cross_feature(model, graph):
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach())

    handle = model.lig_to_rec_conv_layers[-1].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            model(graph)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(f"expected one final cross update, captured {len(captured)}")
    scalar = captured[0][:, :model.ns]
    feature = torch.cat(
        [scalar.mean(0), scalar.std(0, unbiased=False), scalar.amax(0)], dim=0
    ).cpu().numpy().astype(np.float32)
    if feature.shape != (3 * model.ns,) or not np.isfinite(feature).all():
        raise RuntimeError(
            f"invalid cross feature shape/values: {feature.shape}, "
            f"finite={bool(np.isfinite(feature).all())}"
        )
    return feature


def write_report(path, metadata):
    rows = "".join(
        f"<tr><td>{item['condition']}</td><td>{item['label']}</td>"
        f"<td>{item['samples']}</td><td>{item['ligand_sha256']}</td></tr>"
        for item in metadata.get("conditions", [])
    )
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Frozen interaction "
        "features</title><style>body{font-family:system-ui;max-width:960px;"
        "margin:2rem auto}table{border-collapse:collapse}td,th{border:1px "
        "solid #bbb;padding:.4rem}code{overflow-wrap:anywhere}</style>"
        f"<h1>{metadata['system']} frozen interaction features</h1>"
        f"<p>Status: <b>{metadata['status']}</b></p>"
        f"<p>Commit: <code>{metadata['commit']}</code></p>"
        "<p>Feature: final ligand-to-receptor cross update; invariant scalar "
        "channels pooled by mean, standard deviation, and maximum.</p>"
        "<p>Native SDF coordinates were discarded. Ligands were rebuilt from "
        "canonical isomeric SMILES and placed with paired deterministic seeds.</p>"
        "<table><tr><th>Condition</th><th>Label</th><th>Samples</th>"
        f"<th>Source ligand SHA-256</th></tr>{rows}</table>",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--inputs-csv", required=True)
    parser.add_argument("--esm-embeddings", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", default="ema_inference_epoch314_model.pt")
    parser.add_argument("--samples-per-condition", type=int, default=24)
    parser.add_argument("--time", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    require_clean_commit()
    if args.samples_per_condition < 12:
        raise ValueError("--samples-per-condition must be at least 12")
    if not 0.0 <= args.time <= 1.0:
        raise ValueError("--time must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for feature extraction")
    seed_everything(args.seed)
    started_monotonic = time.monotonic()

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / f"{timestamp}-{args.system}"
    result_dir.mkdir(parents=True)
    stdout_tee = Tee(sys.stdout, result_dir / "stdout.log")
    stderr_tee = Tee(sys.stderr, result_dir / "stderr.log")
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee
    metadata_path = result_dir / "metadata.json"
    inputs_csv = Path(args.inputs_csv).resolve()
    esm_embeddings = Path(args.esm_embeddings).resolve()
    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / args.ckpt
    for path in (inputs_csv, esm_embeddings, model_dir, checkpoint,
                 model_dir / "model_parameters.yml"):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None, "status": "running", "exit_code": None,
        "system": args.system,
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "command": shlex.join([sys.executable, *sys.argv]),
        "python": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(torch.cuda.get_device_name(0)),
        "logical_cuda_device": 0,
        "samples_per_condition": args.samples_per_condition,
        "time": args.time, "seed": args.seed,
        "inputs": {
            "csv": str(inputs_csv), "csv_sha256": sha256(inputs_csv),
            "esm_embeddings": str(esm_embeddings),
            "model_parameters_sha256": sha256(model_dir / "model_parameters.yml"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "feature_definition": {
            "module": "lig_to_rec_conv_layers[-1]",
            "channels": "first ns invariant 0e scalar channels",
            "pooling": ["mean", "standard_deviation", "maximum"],
            "native_ligand_coordinates_used": False,
            "receptor_pooling_uses_holo_pocket": False,
        },
        "logs": {
            "stdout": str(result_dir / "stdout.log"),
            "stderr": str(result_dir / "stderr.log"),
        },
        "gpu_before": gpu_snapshot(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (result_dir / "commit.txt").write_text(
        f"{metadata['commit']}\n{metadata['commit_message'].strip()}\n",
        encoding="utf-8",
    )

    exit_code = 0
    try:
        frame = pd.read_csv(inputs_csv)
        required = {"protein_path", "ligand"}
        if not required.issubset(frame.columns):
            raise ValueError(f"input CSV requires columns {sorted(required)}")
        if "condition" in frame:
            frame = frame[frame["condition"] != "shuffled_control"]
        if len(frame) != 2 or frame.get("condition", pd.Series()).nunique() != 2:
            raise ValueError("input CSV must contain exactly two unique state rows")
        frame = frame.copy()
        with (model_dir / "model_parameters.yml").open() as handle:
            model_args = Namespace(**yaml.full_load(handle))
        if model_args.all_atoms:
            raise ValueError("this probe supports the coarse-grained v1 model only")
        cache_path = result_dir / "cache"
        (
            graphs, anchor, ligand_paths, rebuilt_paths, smiles, full_cache,
        ) = build_dataset(
            frame, cache_path, esm_embeddings, model_args, args.seed
        )
        device = torch.device("cuda:0")
        sigma = partial(t_to_sigma_compl, args=model_args)
        model = get_model(model_args, device, t_to_sigma=sigma, no_parallel=True)
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        before_hash = model_parameter_sha256(model)

        features, labels, placement_seeds = [], [], []
        conditions = []
        for label, (graph, ligand_path, rebuilt_path, smiles_value) in enumerate(
            zip(graphs, ligand_paths, rebuilt_paths, smiles)
        ):
            for sample_index in range(args.samples_per_condition):
                placement_seed = args.seed + sample_index
                batch = positioned_graph(
                    graph, placement_seed, args.time, device
                )
                features.append(pooled_last_cross_feature(model, batch))
                labels.append(label)
                placement_seeds.append(placement_seed)
            conditions.append({
                "condition": str(frame.iloc[label].get("condition", f"state_{label}")),
                "label": label, "samples": args.samples_per_condition,
                "ligand_path": str(ligand_path),
                "ligand_sha256": sha256(ligand_path),
                "rebuilt_ligand_path": str(rebuilt_path),
                "rebuilt_ligand_sha256": sha256(rebuilt_path),
                "canonical_isomeric_smiles": smiles_value,
            })
        after_hash = model_parameter_sha256(model)
        if before_hash != after_hash:
            raise RuntimeError("frozen model parameter hash changed")
        feature_matrix = np.asarray(features)
        if not np.isfinite(feature_matrix).all():
            raise RuntimeError("feature matrix contains NaN or Inf")

        feature_path = result_dir / "features.npz"
        np.savez_compressed(
            feature_path, features=feature_matrix, labels=np.asarray(labels),
            placement_seeds=np.asarray(placement_seeds),
            conditions=np.asarray([item["condition"] for item in conditions]),
        )
        metadata.update({
            "status": "passed", "anchor": anchor,
            "anchor_sha256": sha256(anchor),
            "esm_embedding_files": [{
                "path": path,
                "sha256": sha256(path),
            } for path in sorted(glob(
                str(esm_embeddings / f"{Path(anchor).name}*")
            ))],
            "conditions": conditions,
            "materialized_cache": str(full_cache),
            "model_parameter_hash_before": before_hash,
            "model_parameter_hash_after": after_hash,
            "feature_shape": list(feature_matrix.shape),
            "feature_summary": {
                "variance": float(feature_matrix.var()),
                "paired_l2_mean": float(np.linalg.norm(
                    feature_matrix[:args.samples_per_condition] -
                    feature_matrix[args.samples_per_condition:], axis=1
                ).mean()),
            },
            "outputs": {
                "features": str(feature_path),
                "features_sha256": sha256(feature_path),
                "report": str(result_dir / "report.html"),
            },
        })
    except Exception as exc:
        exit_code = 1
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        metadata["traceback"] = traceback.format_exc()
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata["elapsed_seconds"] = time.monotonic() - started_monotonic
        metadata["exit_code"] = exit_code
        metadata["gpu_after"] = gpu_snapshot()
        write_report(result_dir / "report.html", metadata)
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Frozen feature extraction {metadata['status']}: {result_dir}")
    if exit_code:
        print(metadata["error"], file=sys.stderr)
    stdout_tee.flush()
    stderr_tee.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
