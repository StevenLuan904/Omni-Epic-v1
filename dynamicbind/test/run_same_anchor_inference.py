#!/usr/bin/env python3
"""Run auditable DynamicBind same-anchor inference across deterministic seeds."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = (
    REPO_ROOT / "dynamicbind" / "test" / "results" / "day2-same-anchor-inference"
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
    print("ERROR: inference must run from a committed worktree.", file=sys.stderr)
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
            "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
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


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_metadata(path, metadata):
    path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_logged(command, environment, log_path):
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT / "dynamicbind",
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return_code = process.wait()
    return return_code, round(time.time() - started, 3)


def validate_seed_output(seed_dir, expected_conditions):
    affinity_path = seed_dir / "affinity_prediction.csv"
    complete_path = seed_dir / "complete_affinity_prediction.csv"
    if not affinity_path.is_file() or not complete_path.is_file():
        raise RuntimeError(f"missing affinity output in {seed_dir}")
    affinity_rows = read_rows(affinity_path)
    complete_rows = read_rows(complete_path)
    if len(affinity_rows) != expected_conditions:
        raise RuntimeError(
            f"expected {expected_conditions} affinity rows, got {len(affinity_rows)}"
        )
    conditions = []
    for index in range(expected_conditions):
        output_dir = seed_dir / f"index{index}_idx_{index}"
        receptors = sorted(output_dir.glob("rank1_receptor_*.pdb"))
        ligands = sorted(output_dir.glob("rank1_ligand_*.sdf"))
        if len(receptors) != 1 or len(ligands) != 1:
            raise RuntimeError(
                f"expected one rank1 receptor and ligand for index {index}, "
                f"found {len(receptors)} and {len(ligands)}"
            )
        conditions.append({
            "index": index,
            "receptor": str(receptors[0]),
            "receptor_sha256": sha256(receptors[0]),
            "ligand": str(ligands[0]),
            "ligand_sha256": sha256(ligands[0]),
        })
    return {
        "affinity_rows": affinity_rows,
        "complete_affinity_rows": len(complete_rows),
        "conditions": conditions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--inputs-csv", required=True)
    parser.add_argument("--esm-embeddings", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--ckpt", default="ema_inference_epoch314_model.pt")
    parser.add_argument("--visible-gpu", required=True, type=int)
    parser.add_argument("--seeds", default="11,23,37,51")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--samples-per-complex", type=int, default=1)
    args = parser.parse_args()

    require_clean_commit()
    python = Path(os.path.abspath(args.python))
    inputs_csv = Path(args.inputs_csv).resolve()
    esm_embeddings = Path(args.esm_embeddings).resolve()
    model_dir = Path(args.model_dir).resolve()
    checkpoint = model_dir / args.ckpt
    for path in (python, inputs_csv, esm_embeddings, model_dir, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.inference_steps < 1:
        raise ValueError("--inference-steps must be positive")
    if args.samples_per_complex < 1:
        raise ValueError("--samples-per-complex must be positive")

    input_rows = read_rows(inputs_csv)
    if len(input_rows) < 2:
        raise ValueError("same-anchor inference requires at least two conditions")
    protein_paths = {row["protein_path"] for row in input_rows}
    if len(protein_paths) != 1:
        raise ValueError(f"expected one receptor anchor, found {len(protein_paths)}")

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    commit = git("rev-parse", "HEAD")
    commit_message = git("log", "-1", "--format=%B")
    metadata_path = result_dir / "metadata.json"
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "status": "running",
        "exit_code": None,
        "branch": git("branch", "--show-current"),
        "commit": commit,
        "commit_message": commit_message,
        "python": str(python),
        "visible_gpu": str(args.visible_gpu),
        "seeds": seeds,
        "inference_steps": args.inference_steps,
        "samples_per_complex": args.samples_per_complex,
        "inputs": {
            "csv": str(inputs_csv),
            "csv_sha256": sha256(inputs_csv),
            "unique_protein_paths": len(protein_paths),
            "conditions": [row.get("condition", row.get("name")) for row in input_rows],
            "esm_embeddings": str(esm_embeddings),
            "esm_embedding_files": {
                str(path): sha256(path)
                for path in sorted(esm_embeddings.glob("*.pt"))
            },
            "model_dir": str(model_dir),
            "model_parameters_sha256": sha256(model_dir / "model_parameters.yml"),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
        },
        "gpu_before": gpu_snapshot(),
        "runs": [],
    }
    write_metadata(metadata_path, metadata)
    (result_dir / "commit.txt").write_text(
        f"{commit}\n{commit_message.strip()}\n", encoding="utf-8"
    )

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    environment["PYTHONUNBUFFERED"] = "1"
    cache_path = result_dir / "cache"
    overall_exit = 0
    try:
        for seed in seeds:
            seed_dir = result_dir / f"seed_{seed}"
            command = [
                str(python),
                "inference.py",
                "--protein_ligand_csv",
                str(inputs_csv),
                "--esm_embeddings_path",
                str(esm_embeddings),
                "--model_dir",
                str(model_dir),
                "--ckpt",
                args.ckpt,
                "--out_dir",
                str(seed_dir),
                "--cache_path",
                str(cache_path),
                "--seed",
                str(seed),
                "--inference_steps",
                str(args.inference_steps),
                "--actual_steps",
                str(args.inference_steps),
                "--samples_per_complex",
                str(args.samples_per_complex),
                "--savings_per_complex",
                str(args.samples_per_complex),
                "--batch_size",
                str(args.samples_per_complex),
                "--num_workers",
                "1",
                "--protein_dynamic",
                "--keep_local_structures",
                "--use_existing_cache",
                "--no_final_step_noise",
            ]
            return_code, elapsed = run_logged(
                command,
                environment,
                result_dir / f"seed_{seed}.log",
            )
            run_record = {
                "seed": seed,
                "command": shlex.join(command),
                "elapsed_seconds": elapsed,
                "exit_code": return_code,
            }
            if return_code != 0:
                metadata["runs"].append(run_record)
                raise RuntimeError(f"seed {seed} exited with code {return_code}")
            run_record["validation"] = validate_seed_output(
                seed_dir, len(input_rows)
            )
            metadata["runs"].append(run_record)
            write_metadata(metadata_path, metadata)
        metadata["status"] = "passed"
    except Exception as exc:
        overall_exit = 1
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        metadata["finished_at"] = datetime.now().astimezone().isoformat()
        metadata["exit_code"] = overall_exit
        metadata["gpu_after"] = gpu_snapshot()
        write_metadata(metadata_path, metadata)

    print(f"Same-anchor inference {metadata['status']}: {result_dir}")
    if overall_exit:
        print(metadata["error"], file=sys.stderr)
    return overall_exit


if __name__ == "__main__":
    raise SystemExit(main())
