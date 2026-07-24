#!/usr/bin/env python3
"""Generate DynamicBind's torus lookup cache in bounded parallel chunks."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "dynamicbind" / "test" / "results" / "torus-cache"
X_MIN = 1e-5
X_N = 5000
SIGMA_MIN = 3e-3
SIGMA_MAX = 2.0
SIGMA_N = 5000
IMAGE_COUNT = 100
SHAPE = (SIGMA_N + 1, X_N + 1)


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
    print("ERROR: cache generation must run from a committed worktree.", file=sys.stderr)
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


def grid():
    x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi
    sigma = (
        10
        ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1)
        * np.pi
    )
    return x, sigma


def compute_chunk(start, stop):
    x, sigma = grid()
    sigma_squared = sigma[start:stop, None] ** 2
    density = np.zeros((stop - start, X_N + 1), dtype=np.float64)
    gradient = np.zeros_like(density)
    for image in range(-IMAGE_COUNT, IMAGE_COUNT + 1):
        delta = x[None, :] + 2 * np.pi * image
        exponential = np.exp(-(delta**2) / (2 * sigma_squared))
        density += exponential
        gradient += delta / sigma_squared * exponential
    return start, density, gradient / density


def chunk_ranges(workers):
    chunk_size = math.ceil(SHAPE[0] / workers)
    return [
        (start, min(start + chunk_size, SHAPE[0]))
        for start in range(0, SHAPE[0], chunk_size)
    ]


def direct_values(row, column):
    x, sigma = grid()
    delta = x[column] + 2 * np.pi * np.arange(-IMAGE_COUNT, IMAGE_COUNT + 1)
    sigma_squared = sigma[row] ** 2
    exponential = np.exp(-(delta**2) / (2 * sigma_squared))
    density = exponential.sum()
    score = (delta / sigma_squared * exponential).sum() / density
    return density, score


def validate(p_path, score_path):
    density = np.load(p_path, mmap_mode="r")
    score = np.load(score_path, mmap_mode="r")
    if density.shape != SHAPE or score.shape != SHAPE:
        raise ValueError(
            f"unexpected cache shapes: density={density.shape}, score={score.shape}"
        )
    checks = []
    for row, column in ((0, 0), (137, 911), (2500, 2500), (4999, 4999)):
        expected_density, expected_score = direct_values(row, column)
        actual_density = float(density[row, column])
        actual_score = float(score[row, column])
        if not np.isclose(actual_density, expected_density, rtol=1e-12, atol=1e-14):
            raise ValueError(f"density validation failed at ({row}, {column})")
        if not np.isclose(actual_score, expected_score, rtol=1e-12, atol=1e-12):
            raise ValueError(f"score validation failed at ({row}, {column})")
        checks.append({
            "row": row,
            "column": column,
            "density": actual_density,
            "score": actual_score,
        })
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--cache-dir",
        default=str(REPO_ROOT / "dynamicbind"),
        help="Directory read by dynamicbind/utils/torus.py",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be between 1 and 16")

    require_clean_commit()
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    p_path = cache_dir / ".p.npy"
    score_path = cache_dir / ".score.npy"

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    result_dir = RESULTS_ROOT / timestamp
    result_dir.mkdir(parents=True)
    p_temp = result_dir / "p.npy.tmp"
    score_temp = result_dir / "score.npy.tmp"
    metadata = {
        "started_at": datetime.now().astimezone().isoformat(),
        "commit": git("rev-parse", "HEAD"),
        "commit_message": git("log", "-1", "--format=%B"),
        "workers": args.workers,
        "shape": list(SHAPE),
        "image_count_each_side": IMAGE_COUNT,
        "cache_dir": str(cache_dir),
        "status": "running",
    }
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    start_time = time.time()
    try:
        density_out = np.lib.format.open_memmap(
            p_temp, mode="w+", dtype=np.float64, shape=SHAPE
        )
        score_out = np.lib.format.open_memmap(
            score_temp, mode="w+", dtype=np.float64, shape=SHAPE
        )
        ranges = chunk_ranges(args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(compute_chunk, start, stop): (start, stop)
                for start, stop in ranges
            }
            for future in as_completed(futures):
                start, stop = futures[future]
                returned_start, density, score = future.result()
                if returned_start != start:
                    raise AssertionError(
                        f"chunk start mismatch: expected {start}, got {returned_start}"
                    )
                density_out[start:stop] = density
                score_out[start:stop] = score
                density_out.flush()
                score_out.flush()
                print(f"completed rows {start}:{stop}", flush=True)
        del density_out
        del score_out
        os.replace(p_temp, p_path)
        os.replace(score_temp, score_path)

        checks = validate(p_path, score_path)
        metadata.update({
            "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": round(time.time() - start_time, 3),
            "status": "passed",
            "density_sha256": sha256(p_path),
            "score_sha256": sha256(score_path),
            "validation": checks,
        })
    except Exception as exc:
        metadata.update({
            "finished_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": round(time.time() - start_time, 3),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    finally:
        (result_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Torus cache {metadata['status']}: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
