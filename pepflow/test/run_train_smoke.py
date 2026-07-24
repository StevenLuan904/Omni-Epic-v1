#!/usr/bin/env python3
"""Run the one-step training smoke test and preserve reproducibility evidence."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "test" / "results" / "train-smoke"
DEFAULT_DATA_FILES = (
    "../datasets/PepMerge_lmdb.zip",
    "../datasets/PepMerge_release.zip",
    "../datasets/lmdb/pep_pocket_train_structure_cache.lmdb",
)


def git(*args, check=True):
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    ).stdout.strip()


def require_clean_commit():
    status = git("status", "--porcelain", "--untracked-files=all")
    if not status:
        return

    print("ERROR: smoke tests must run from a committed worktree.", file=sys.stderr)
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
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot():
    snapshot = {}
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
    for label, command in queries.items():
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        snapshot[label] = {
            "exit_code": completed.returncode,
            "output": completed.stdout.strip(),
        }
    return snapshot


def parse_metrics(log_text):
    lines = [line for line in log_text.splitlines() if "[train] Iter" in line]
    if not lines:
        return {}
    metrics = {}
    for name, detail, value in re.findall(
        r"\|\s*([A-Za-z_]+)(?:\(([^)]+)\))?\s+([-+0-9.eE]+)", lines[-1]
    ):
        metrics[f"{name}.{detail}" if detail else name] = float(value)
    return metrics


def write_visualization(path, metadata, metrics):
    losses = {key: value for key, value in metrics.items() if key.startswith("loss")}
    scale = max((abs(value) for value in losses.values()), default=1.0) or 1.0
    bars = []
    for key, value in losses.items():
        width = max(1.0, 100.0 * abs(value) / scale)
        bars.append(
            f'<tr><td>{html.escape(key)}</td><td>{value:.6g}</td>'
            f'<td class="chart"><i style="width:{width:.2f}%"></i></td></tr>'
        )
    body = "\n".join(bars) or '<tr><td colspan="3">No metrics parsed</td></tr>'
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>PepFlow smoke test</title>"
        "<style>body{font:15px system-ui;margin:2rem;max-width:1000px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border-bottom:1px solid #ddd}"
        ".chart{width:60%}.chart i{display:block;height:1rem;background:#3977d5}</style>"
        f"<h1>PepFlow train smoke test</h1><p>Commit: <code>{html.escape(metadata['commit'])}</code>"
        f"<br>Started: {html.escape(metadata['started_at'])}<br>Status: {html.escape(metadata['status'])}</p>"
        f"<table><thead><tr><th>Metric</th><th>Value</th><th>Relative magnitude</th></tr></thead>"
        f"<tbody>{body}</tbody></table>",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/learn_angle.smoke.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visible-gpus", default="0,1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-files", nargs="+", default=DEFAULT_DATA_FILES)
    args = parser.parse_args()

    require_clean_commit()
    data_files = [(ROOT / value).resolve() for value in args.data_files]
    for path in data_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_hashes = {str(path): sha256(path) for path in data_files}
    started = datetime.now().astimezone()
    started_monotonic = time.monotonic()
    stamp = started.strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / stamp
    result_dir.mkdir(parents=True, exist_ok=False)

    commit = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    commit_message = git("log", "-1", "--format=%B")
    command = [
        sys.executable, "train.py", "--config", args.config, "--debug",
        "--device", args.device, "--num_workers", str(args.num_workers),
    ]
    metadata = {
        "started_at": started.isoformat(),
        "finished_at": None,
        "status": "running",
        "exit_code": None,
        "commit": commit,
        "branch": branch,
        "commit_message": commit_message,
        "visible_gpus": args.visible_gpus,
        "command": command,
        "python": sys.executable,
        "input_sha256": input_hashes,
        "gpu_before": gpu_snapshot(),
        "gpu_after": None,
        "elapsed_seconds": None,
    }
    (result_dir / "commit.txt").write_text(
        f"{commit}\n{commit_message.rstrip()}\n", encoding="utf-8"
    )
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    env["WANDB_MODE"] = "disabled"
    log_path = result_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, text=True, errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        exit_code = process.wait()

    metrics = parse_metrics(log_path.read_text(encoding="utf-8"))
    finished = datetime.now().astimezone()
    metadata.update({
        "finished_at": finished.isoformat(),
        "status": "passed" if exit_code == 0 else "failed",
        "exit_code": exit_code,
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
        "metrics": metrics,
        "gpu_after": gpu_snapshot(),
    })
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_visualization(result_dir / "metrics.html", metadata, metrics)
    print(f"\nSmoke test {metadata['status']}: {result_dir}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
