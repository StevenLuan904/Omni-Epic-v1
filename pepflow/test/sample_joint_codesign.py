#!/usr/bin/env python3
"""Generate peptide sequence/structure and an adapted receptor pocket."""

import argparse
import copy
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
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "joint-codesign-sample"


def merge_complex(receptor_path, peptide_complex_path, output_path, peptide_chain):
    receptor_lines = [
        line for line in receptor_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    peptide_lines = []
    for line in peptide_complex_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            original_chain = line[21]
            if original_chain in peptide_chain["source"]:
                peptide_lines.append(line[:21] + peptide_chain["output"] + line[22:])
    if not peptide_lines:
        raise ValueError(f"no generated peptide atoms found in {peptide_complex_path}")
    output_path.write_text(
        "\n".join(receptor_lines + ["TER"] + peptide_lines + ["TER", "END"]) + "\n",
        encoding="utf-8",
    )


def worker(args):
    from argparse import Namespace
    from functools import partial

    import numpy as np
    import torch
    from Bio.PDB import PDBParser
    from torch_geometric.data import Batch
    import yaml

    pepflow_root, dynamicbind_root = REPO_ROOT / "pepflow", REPO_ROOT / "dynamicbind"
    sys.path.insert(0, str(dynamicbind_root))
    sys.path.insert(0, str(pepflow_root))
    from datasets.pdbbind import PDBBind
    from models_con.flow_model import FlowModel
    from models_con.joint_flow import JointPeptideReceptorFlow
    from models_con.pep_dataloader import preprocess_structure
    from models_con.sample import save_samples_sc
    from models_con.utils import process_dic
    from pepflow.utils.data import PaddingCollate
    from pepflow.utils.misc import load_config
    from pepflow.utils.train import recursive_to
    from utils.diffusion_utils import modify_conformer, set_time, t_to_sigma as t_to_sigma_compl
    from utils.utils import get_model
    from utils.visualise import modify_pdb, save_protein

    experiment = yaml.safe_load(Path(args.experiment_config).read_text(encoding="utf-8"))
    generation = experiment["generation"]
    preset = experiment["sampler"]["presets"][experiment["sampler"]["active_preset"]]
    if experiment["clocks"]["semantics"] != "denoising_progress_0_noise_1_data":
        raise ValueError("sampling requires normalized denoising-progress clocks")
    torch.manual_seed(experiment["experiment"]["seed"])
    np.random.seed(experiment["experiment"]["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    data_dir, case_id, state = Path(args.data_dir), generation["case_id"], generation["state"]
    case_dir = data_dir / "cases" / case_id
    config, _ = load_config(args.pepflow_config)
    peptide_flow = FlowModel(config.model)
    pep_checkpoint = torch.load(args.pepflow_checkpoint, map_location="cpu")
    peptide_flow.load_state_dict(process_dic(pep_checkpoint.get("model", pep_checkpoint)), strict=True)
    model_dir = Path(args.model_dir)
    with (model_dir / "model_parameters.yml").open() as handle:
        model_args = Namespace(**yaml.full_load(handle))
    receptor_flow = get_model(
        model_args, device, t_to_sigma=partial(t_to_sigma_compl, args=model_args),
        no_parallel=True,
    )
    receptor_checkpoint = model_dir / args.dynamicbind_checkpoint
    receptor_flow.load_state_dict(torch.load(receptor_checkpoint, map_location="cpu"), strict=True)
    model = JointPeptideReceptorFlow(
        peptide_flow,
        receptor_flow,
        config.model.encoder.node_embed_size,
        model_args.ns,
        clock_embedding=experiment["clocks"]["embedding"],
    ).to(device)
    delta = torch.load(args.trained_delta, map_location="cpu")
    expected_inputs = {
        "base_dynamicbind_sha256": sha256(receptor_checkpoint),
        "base_pepflow_sha256": sha256(args.pepflow_checkpoint),
        "dataset_manifest_sha256": sha256(data_dir / "manifest.json"),
    }
    mismatches = {
        name: {"delta": delta.get(name), "input": expected}
        for name, expected in expected_inputs.items()
        if delta.get(name) != expected
    }
    if mismatches:
        raise ValueError(f"trained delta input hashes do not match: {mismatches}")
    named_parameters = dict(model.named_parameters())
    missing = set(delta["trainable_state_dict"]) - set(named_parameters)
    if missing:
        raise ValueError(f"delta contains unknown parameters: {sorted(missing)}")
    with torch.no_grad():
        for name, value in delta["trainable_state_dict"].items():
            named_parameters[name].copy_(value)
    model.eval()

    item = preprocess_structure({
        "id": f"{case_id}_{state}", "pdb_path": str(case_dir / f"pepflow_{state}")
    })
    if item is None:
        raise RuntimeError("PepFlow preprocessing failed")
    batch = recursive_to(
        PaddingCollate(eight=False)([
            copy.deepcopy(item) for _ in range(generation["num_samples"])
        ]),
        device,
    )
    batch["sequence_generate_mask"] = batch["generate_mask"].bool().clone()
    receptor_steps = preset["receptor_steps"]

    def clock_context(peptide_progress):
        pocket_progress = torch.floor(peptide_progress * receptor_steps) / receptor_steps
        clocks = {
            "peptide_seq": peptide_progress,
            "peptide_struct": peptide_progress,
            "pocket_struct": pocket_progress,
        }
        return model.clock_conditioner(clocks)

    with torch.no_grad():
        trajectory = model.peptide_flow.sample(
            batch,
            num_steps=preset["peptide_steps"],
            sample_bb=generation["sample_backbone"],
            sample_ang=generation["sample_torsion"],
            sample_seq=generation["sample_sequence"],
            clock_context_fn=clock_context,
        )
    final = trajectory[-1]
    final["batch"] = batch
    result_dir = Path(args.result_dir)
    anchor_complex_dir = result_dir / "anchor_pocket_complexes"
    adapted_receptor_dir = result_dir / "adapted_receptors"
    complex_dir = result_dir / "complexes"
    for directory in (anchor_complex_dir, adapted_receptor_dir, complex_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_samples_sc(final, anchor_complex_dir)

    protein_path = case_dir / f"anchor_{state}.pdb"
    ligand_path = case_dir / f"peptide_{state}.sdf"
    graph_dataset = PDBBind(
        transform=None, root="", name_list=[f"{case_id}_{state}"],
        protein_path_list=[str(protein_path)], ligand_descriptions=[str(ligand_path)],
        receptor_radius=model_args.receptor_radius, cache_path=args.graph_cache,
        remove_hs=model_args.remove_hs, max_lig_size=None,
        c_alpha_max_neighbors=model_args.c_alpha_max_neighbors, matching=False,
        keep_original=False, popsize=model_args.matching_popsize,
        maxiter=model_args.matching_maxiter, center_ligand=False,
        all_atoms=model_args.all_atoms, atom_radius=model_args.atom_radius,
        atom_max_neighbors=model_args.atom_max_neighbors,
        esm_embeddings_path=str(case_dir / "esm2_output"),
        require_ligand=True, require_receptor=False, num_workers=1,
        keep_local_structures=True,
        use_existing_cache=Path(args.graph_cache + "_torsion").exists(),
    )
    base_graph = graph_dataset.get(0)
    generated_source_chains = sorted({
        item["chain_id"][index]
        for index, generated in enumerate(item["generate_mask"])
        if bool(generated)
    })
    endpoint_progress = torch.ones(1, device=device)
    endpoint_clocks = {
        "peptide_seq": endpoint_progress,
        "peptide_struct": endpoint_progress,
        "pocket_struct": endpoint_progress,
    }
    endpoint_context = model.clock_conditioner(endpoint_clocks)
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    rows = []
    for sample_index in range(generation["num_samples"]):
        graph = copy.deepcopy(base_graph)
        set_time(graph, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 batchsize=1, all_atoms=False, device=None)
        graph_batch = Batch.from_data_list([graph]).to(device)
        peptide_condition = {
            "aa": final["seqs"][sample_index:sample_index + 1].to(device),
            "generate_mask": batch["generate_mask"][sample_index:sample_index + 1],
            "pos_heavyatom": batch["pos_heavyatom"][sample_index:sample_index + 1],
        }
        with torch.no_grad():
            output = model.receptor_forward(
                peptide_condition,
                graph_batch,
                clock_context=endpoint_context,
                peptide_translation=final["trans"][sample_index:sample_index + 1].to(device),
            )
        receptor_translation = output[5].detach().cpu()
        receptor_rotation = output[6].detach().cpu()
        receptor_chi = output[7].detach().cpu()
        updated_graph = modify_conformer(
            copy.deepcopy(base_graph),
            torch.zeros((1, 3)), torch.zeros(3), None,
            receptor_translation, receptor_rotation, receptor_chi,
        )
        receptor_structure = PDBParser(QUIET=True).get_structure(
            f"{case_id}_{state}_{sample_index}", str(protein_path)
        )
        modify_pdb(receptor_structure, updated_graph)
        receptor_path = adapted_receptor_dir / f"receptor_{sample_index:03d}.pdb"
        save_protein(receptor_structure, str(receptor_path))
        complex_path = complex_dir / f"complex_{sample_index:03d}.pdb"
        merge_complex(
            receptor_path,
            anchor_complex_dir / f"sample_{sample_index}.pdb",
            complex_path,
            {"source": generated_source_chains, "output": generation["peptide_chain"]},
        )
        generated_sequence = "".join(
            amino_acids[int(index)]
            for index in final["seqs"][sample_index][batch["generate_mask"][sample_index].cpu()]
        )
        rows.append({
            "sample": sample_index,
            "sequence": generated_sequence,
            "complex_pdb": str(complex_path),
            "mean_pocket_translation": float(
                torch.linalg.vector_norm(receptor_translation, dim=-1).mean()
            ),
            "mean_pocket_rotation": float(
                torch.linalg.vector_norm(receptor_rotation, dim=-1).mean()
            ),
        })
    unique_sequences = len({row["sequence"] for row in rows})
    summary = {
        "case_id": case_id,
        "state": state,
        "samples": len(rows),
        "unique_sequences": unique_sequences,
        "sequence_diversity_fraction": unique_sequences / len(rows),
        "sampler_preset": experiment["sampler"]["active_preset"],
        "clock_slot_order": experiment["clocks"]["embedding"]["slot_order"],
        "trained_delta_sha256": sha256(args.trained_delta),
        "samples_detail": rows,
    }
    (result_dir / "worker_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--graph-cache", required=True)
    parser.add_argument("--pepflow-config", default="pepflow/configs/learn_angle.yaml")
    parser.add_argument("--pepflow-checkpoint", required=True)
    parser.add_argument("--dynamicbind-checkpoint", default="ema_inference_epoch314_model.pt")
    parser.add_argument("--trained-delta", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--visible-gpus", default="5")
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
        "trained_delta": Path(args.trained_delta).resolve(),
        "experiment_config": Path(args.experiment_config).resolve(),
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
        command.extend(["--" + name.replace("_", "-"), str(value)])
    command.extend(["--result-dir", str(result_dir)])
    metadata = {
        "started_at": started.isoformat(), "finished_at": None,
        "status": "running", "exit_code": None,
        "branch": git("branch", "--show-current"), "commit": commit,
        "commit_message": message.strip(), "command": command,
        "python": sys.executable, "cuda_visible_devices": args.visible_gpus,
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
        "gpu_before": gpu_snapshot(), "gpu_after": None,
    }
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")
    (result_dir / "experiment_config.yaml").write_text(
        Path(args.experiment_config).read_text(encoding="utf-8"), encoding="utf-8"
    )
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
    summary_path = result_dir / "worker_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    metadata.update({
        "finished_at": datetime.now().astimezone().isoformat(),
        "status": "passed" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - start_clock,
        "summary": summary, "gpu_after": gpu_snapshot(),
    })
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Joint codesign sample</title></head><body>"
        f"<h1>{metadata['status']}</h1><p>Commit: {commit}</p>"
        f"<pre>{json.dumps(summary, indent=2)}</pre></body></html>\n",
        encoding="utf-8",
    )
    print(process.stdout, end="")
    print(process.stderr, end="", file=sys.stderr)
    print(f"Joint codesign sampling {metadata['status']}: {result_dir}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
