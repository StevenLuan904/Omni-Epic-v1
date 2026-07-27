#!/usr/bin/env python3
"""Validate generated complex geometry without estimating binding energy."""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from train_joint_peptide_receptor_overfit import git, require_clean_commit, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "codesign-geometry"


def distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def score_pdb(path, peptide_chain, contact_cutoff, clash_cutoff):
    peptide, receptor, peptide_ca = [], [], []
    seen_ca = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        chain, atom = line[21], line[12:16].strip()
        residue_key = (line[22:26], line[26])
        xyz = tuple(float(line[start:end]) for start, end in ((30, 38), (38, 46), (46, 54)))
        if chain == peptide_chain:
            peptide.append((residue_key, atom, xyz))
            if atom == "CA" and residue_key not in seen_ca:
                peptide_ca.append(xyz)
                seen_ca.add(residue_key)
        else:
            receptor.append((residue_key, atom, xyz))
    if not peptide or not receptor or len(peptide_ca) < 2:
        raise ValueError("missing peptide/receptor atoms or peptide CA trace")
    ca_distances = [
        distance(peptide_ca[index], peptide_ca[index + 1])
        for index in range(len(peptide_ca) - 1)
    ]
    contacts = clashes = 0
    minimum_interface_distance = float("inf")
    for _, _, peptide_xyz in peptide:
        for _, _, receptor_xyz in receptor:
            value = distance(peptide_xyz, receptor_xyz)
            minimum_interface_distance = min(minimum_interface_distance, value)
            contacts += value <= contact_cutoff
            clashes += value < clash_cutoff
    return {
        "sample": path.name,
        "peptide_residues": len(peptide_ca),
        "peptide_atoms": len(peptide),
        "receptor_atoms": len(receptor),
        "mean_adjacent_ca_distance": statistics.fmean(ca_distances),
        "min_adjacent_ca_distance": min(ca_distances),
        "max_adjacent_ca_distance": max(ca_distances),
        "ca_trace_valid": all(2.8 <= value <= 4.5 for value in ca_distances),
        "interface_heavy_atom_contacts": contacts,
        "interface_heavy_atom_clashes": clashes,
        "minimum_interface_distance": minimum_interface_distance,
        "interface_present": contacts > 0,
    }


def worker(args):
    import yaml

    settings = yaml.safe_load(
        Path(args.experiment_config).read_text(encoding="utf-8")
    )["rosetta"]
    rows, failures = [], []
    for path in sorted(Path(args.input_dir).glob("*.pdb")):
        try:
            rows.append(score_pdb(
                path,
                settings["peptide_chain"],
                settings["contact_cutoff_angstrom"],
                settings["clash_cutoff_angstrom"],
            ))
        except Exception as error:
            failures.append({"sample": path.name, "failure": repr(error)})
    if not rows and not failures:
        raise RuntimeError("no PDB files found")
    result_dir = Path(args.result_dir)
    if rows:
        with (result_dir / "geometry_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "metric_scope": "geometry only; no Rosetta energy and no binding-energy claim",
        "samples": len(rows) + len(failures),
        "parsed_samples": len(rows),
        "failures": failures,
        "ca_trace_valid_fraction": sum(row["ca_trace_valid"] for row in rows) / len(rows) if rows else 0,
        "interface_present_fraction": sum(row["interface_present"] for row in rows) / len(rows) if rows else 0,
        "mean_interface_contacts": statistics.fmean(row["interface_heavy_atom_contacts"] for row in rows) if rows else None,
        "mean_interface_clashes": statistics.fmean(row["interface_heavy_atom_clashes"] for row in rows) if rows else None,
        "minimum_interface_distance": min(row["minimum_interface_distance"] for row in rows) if rows else None,
    }
    (result_dir / "worker_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--result-dir")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return 0
    require_clean_commit()
    input_dir, config = Path(args.input_dir).resolve(), Path(args.experiment_config).resolve()
    if not input_dir.is_dir() or not config.is_file():
        raise FileNotFoundError(input_dir if not input_dir.is_dir() else config)
    started, start_clock = datetime.now().astimezone(), time.monotonic()
    result_dir = RESULTS_ROOT / started.strftime("%Y%m%d-%H%M%S%z")
    result_dir.mkdir(parents=True)
    commit, message = git("rev-parse", "HEAD"), git("log", "-1", "--format=%B")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
               "--input-dir", str(input_dir), "--experiment-config", str(config),
               "--result-dir", str(result_dir)]
    pdbs = sorted(input_dir.glob("*.pdb"))
    metadata = {
        "started_at": started.isoformat(), "finished_at": None, "status": "running",
        "exit_code": None, "branch": git("branch", "--show-current"), "commit": commit,
        "commit_message": message.strip(), "command": command, "python": sys.executable,
        "input_sha256": {"experiment_config": sha256(config), **{p.name: sha256(p) for p in pdbs}},
    }
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    process = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (result_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (result_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    summary_path = result_dir / "worker_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    metadata.update({"finished_at": datetime.now().astimezone().isoformat(),
                     "status": "passed" if process.returncode == 0 else "failed",
                     "exit_code": process.returncode,
                     "elapsed_seconds": time.monotonic() - start_clock, "summary": summary})
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Codesign geometry</title></head><body>"
        f"<h1>{metadata['status']}</h1><p>Commit: {commit}</p>"
        f"<pre>{json.dumps(summary, indent=2)}</pre></body></html>\n", encoding="utf-8")
    print(process.stdout, end="")
    print(process.stderr, end="", file=sys.stderr)
    print(f"Geometry validation {metadata['status']}: {result_dir}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
