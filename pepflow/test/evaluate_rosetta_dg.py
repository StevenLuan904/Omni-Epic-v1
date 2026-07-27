#!/usr/bin/env python3
"""Score generated peptide complexes with Rosetta InterfaceAnalyzer."""

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from train_joint_peptide_receptor_overfit import git, require_clean_commit, sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "rosetta-dg"


def percentile(values, probability):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def interface_geometry(pose, peptide_chain, contact_cutoff, clash_cutoff):
    peptide_atoms, receptor_atoms = [], []
    pdb_info = pose.pdb_info()
    for residue_index in range(1, pose.total_residue() + 1):
        residue = pose.residue(residue_index)
        target = peptide_atoms if pdb_info.chain(residue_index) == peptide_chain else receptor_atoms
        for atom_index in range(1, residue.nheavyatoms() + 1):
            xyz = residue.xyz(atom_index)
            target.append((xyz.x, xyz.y, xyz.z))
    if not peptide_atoms or not receptor_atoms:
        raise ValueError(f"peptide chain {peptide_chain!r} or receptor atoms are absent")
    contacts, clashes, minimum = 0, 0, float("inf")
    contact_sq, clash_sq = contact_cutoff ** 2, clash_cutoff ** 2
    for px, py, pz in peptide_atoms:
        for rx, ry, rz in receptor_atoms:
            distance_sq = (px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2
            minimum = min(minimum, distance_sq)
            contacts += distance_sq <= contact_sq
            clashes += distance_sq < clash_sq
    return contacts, clashes, math.sqrt(minimum)


def worker(args):
    import yaml
    import pyrosetta
    from pyrosetta.rosetta.core.kinematics import MoveMap
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    from pyrosetta.rosetta.protocols.minimization_packing import MinMover, PackRotamersMover

    config = yaml.safe_load(Path(args.experiment_config).read_text(encoding="utf-8"))
    settings = config["rosetta"]
    if settings["protocol"] != "InterfaceAnalyzer" or settings["score_field"] != "dG_separated":
        raise ValueError("Rosetta protocol must be InterfaceAnalyzer/dG_separated")
    pyrosetta.init("-mute all -ignore_unrecognized_res true -load_PDB_components false")
    score_function = pyrosetta.get_fa_scorefxn()
    result_dir = Path(args.result_dir)
    relaxed_dir = result_dir / "relaxed_pdb"
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pdb_path in sorted(Path(args.input_dir).glob("*.pdb")):
        row = {"sample": pdb_path.name, "status": "failed", "failure": None}
        try:
            pose = pyrosetta.pose_from_pdb(str(pdb_path))
            peptide_chain = settings["peptide_chain"]
            receptor_chains = sorted({
                pose.pdb_info().chain(index)
                for index in range(1, pose.total_residue() + 1)
                if pose.pdb_info().chain(index) != peptide_chain
            })
            if not receptor_chains:
                raise ValueError("no receptor chain remains after peptide-chain selection")
            interface = f"{peptide_chain}_{''.join(receptor_chains)}"
            if settings["repack"]:
                task = TaskFactory.create_packer_task(pose)
                task.restrict_to_repacking()
                packer = PackRotamersMover(score_function, task)
                packer.apply(pose)
            if settings["minimize"]:
                move_map = MoveMap()
                move_map.set_bb(False)
                move_map.set_chi(True)
                move_map.set_jump(True)
                minimizer = MinMover(
                    move_map,
                    score_function,
                    settings["minimizer"],
                    settings["minimizer_tolerance"],
                    True,
                )
                minimizer.apply(pose)
            total_score = float(score_function(pose))
            analyzer = InterfaceAnalyzerMover(interface)
            analyzer.set_pack_input(False)
            analyzer.set_pack_separated(True)
            analyzer.apply(pose)
            contacts, clashes, minimum_distance = interface_geometry(
                pose,
                peptide_chain,
                settings["contact_cutoff_angstrom"],
                settings["clash_cutoff_angstrom"],
            )
            relaxed_path = relaxed_dir / pdb_path.name
            pose.dump_pdb(str(relaxed_path))
            row.update({
                "status": "passed",
                "interface": interface,
                "dG_separated": float(analyzer.get_interface_dG()),
                "dSASA_int": float(analyzer.get_interface_delta_sasa()),
                "shape_complementarity": float(analyzer.get_sc_value()),
                "total_score": total_score,
                "interface_heavy_atom_contacts": contacts,
                "interface_heavy_atom_clashes": clashes,
                "minimum_interface_distance": minimum_distance,
                "relaxed_pdb": str(relaxed_path),
            })
        except Exception as error:
            row["failure"] = repr(error)
        rows.append(row)
        print(json.dumps(row), flush=True)
    if not rows:
        raise RuntimeError(f"no PDB files found in {args.input_dir}")
    fieldnames = sorted({key for row in rows for key in row})
    with (result_dir / "rosetta_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row["status"] == "passed"]
    energies = [row["dG_separated"] for row in valid]
    quantiles = {
        str(probability): percentile(energies, probability)
        for probability in settings["report_quantiles"]
    } if energies else {}
    summary = {
        "score_semantics": "absolute Rosetta interface dG proxy; lower is better; not ddG",
        "total_samples": len(rows),
        "valid_samples": len(valid),
        "failed_samples": len(rows) - len(valid),
        "failure_rate": (len(rows) - len(valid)) / len(rows),
        "dG_separated": ({
            "mean": statistics.fmean(energies),
            "median": statistics.median(energies),
            "minimum": min(energies),
            "quantiles": quantiles,
            "top_k": sorted(energies)[:min(5, len(energies))],
        } if energies else None),
        "failures": [
            {"sample": row["sample"], "failure": row["failure"]}
            for row in rows if row["status"] == "failed"
        ],
    }
    (result_dir / "worker_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


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
    input_dir = Path(args.input_dir).resolve()
    experiment_config = Path(args.experiment_config).resolve()
    if not input_dir.is_dir() or not experiment_config.is_file():
        raise FileNotFoundError(input_dir if not input_dir.is_dir() else experiment_config)
    pdb_files = sorted(input_dir.glob("*.pdb"))
    started, start_clock = datetime.now().astimezone(), time.monotonic()
    result_dir = RESULTS_ROOT / started.strftime("%Y%m%d-%H%M%S%z")
    result_dir.mkdir(parents=True)
    commit, message = git("rev-parse", "HEAD"), git("log", "-1", "--format=%B")
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--input-dir", str(input_dir), "--experiment-config", str(experiment_config),
        "--result-dir", str(result_dir),
    ]
    metadata = {
        "started_at": started.isoformat(), "finished_at": None,
        "status": "running", "exit_code": None,
        "branch": git("branch", "--show-current"), "commit": commit,
        "commit_message": message.strip(), "command": command,
        "python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "input_sha256": {
            "experiment_config": sha256(experiment_config),
            **{pdb.name: sha256(pdb) for pdb in pdb_files},
        },
    }
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")
    (result_dir / "experiment_config.yaml").write_text(
        experiment_config.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    process = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (result_dir / "stdout.log").write_text(process.stdout, encoding="utf-8")
    (result_dir / "stderr.log").write_text(process.stderr, encoding="utf-8")
    summary_path = result_dir / "worker_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    metadata.update({
        "finished_at": datetime.now().astimezone().isoformat(),
        "status": "passed" if process.returncode == 0 else "failed",
        "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - start_clock,
        "summary": summary,
    })
    (result_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Rosetta dG</title></head><body>"
        f"<h1>{metadata['status']}</h1><p>Commit: {commit}</p>"
        f"<pre>{json.dumps(summary, indent=2)}</pre></body></html>\n",
        encoding="utf-8",
    )
    print(process.stdout, end="")
    print(process.stderr, end="", file=sys.stderr)
    print(f"Rosetta dG evaluation {metadata['status']}: {result_dir}")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
