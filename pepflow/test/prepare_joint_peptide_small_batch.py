#!/usr/bin/env python3
"""Prepare a diverse, split-audited PepMerge joint-flow small batch."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import zipfile

import numpy as np
import torch

from prepare_joint_peptide_states import (
    AA3,
    atom_lines,
    coordinate,
    fasta,
    git,
    kabsch,
    pdb_to_sdf,
    receptor_residues,
    replace_coordinate,
    require_clean_commit,
    sha256,
    transformed_pdb,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "pepflow" / "test" / "results" / "joint-small-batch-data"


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def sequence_identity(left, right):
    """Global identity using an LCS alignment, normalized by longer length."""
    previous = [0] * (len(right) + 1)
    for aa_left in left:
        current = [0]
        for index, aa_right in enumerate(right, 1):
            current.append(
                previous[index - 1] + 1
                if aa_left == aa_right
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1] / max(len(left), len(right), 1)


def kmer_jaccard(left, right, size=3):
    left_kmers = {left[index:index + size] for index in range(max(1, len(left) - size + 1))}
    right_kmers = {right[index:index + size] for index in range(max(1, len(right) - size + 1))}
    return len(left_kmers & right_kmers) / max(len(left_kmers | right_kmers), 1)


def connected_groups(records, related):
    groups = UnionFind(len(records))
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if related(records[left], records[right]):
                groups.union(left, right)
    output = {}
    for index in range(len(records)):
        output.setdefault(groups.find(index), []).append(records[index]["case_id"])
    return sorted(output.values(), key=lambda group: (-len(group), group))


def grouped_split(groups, seed, test_fraction=0.2, validation_fraction=0.1):
    rng = random.Random(seed)
    groups = list(groups)
    rng.shuffle(groups)
    total = sum(len(group) for group in groups)
    targets = {"test": max(1, round(total * test_fraction)),
               "validation": max(1, round(total * validation_fraction))}
    split = {"train": [], "validation": [], "test": []}
    for name in ("test", "validation"):
        while groups and len(split[name]) < targets[name]:
            split[name].extend(groups.pop())
    for group in groups:
        split["train"].extend(group)
    return {name: sorted(values) for name, values in split.items()}


def prepare_case(archive, row, case_dir):
    data = {}
    for label in ("a", "b"):
        entry = row[f"entry_{label}"]
        data[label] = {
            "entry": entry,
            "receptor_fasta": fasta(archive.read(f"{entry}/receptor.fasta")),
            "peptide_fasta": fasta(archive.read(f"{entry}/peptide.fasta")),
            "receptor_lines": atom_lines(archive.read(f"{entry}/receptor.pdb")),
            "peptide_lines": atom_lines(archive.read(f"{entry}/peptide.pdb")),
        }
    if data["a"]["receptor_fasta"] != data["b"]["receptor_fasta"]:
        raise ValueError("receptor FASTA mismatch")
    residues = {label: receptor_residues(data[label]["receptor_lines"]) for label in ("a", "b")}
    sequences = {
        label: "".join(AA3.get(residue["name"], "X") for residue in residues[label])
        for label in ("a", "b")
    }
    if sequences["a"] != sequences["b"] or len(residues["a"]) < 20:
        raise ValueError("receptor coordinate sequence mismatch")
    ca_a = np.asarray([coordinate(residue["atoms"]["CA"]) for residue in residues["a"]])
    ca_b = np.asarray([coordinate(residue["atoms"]["CA"]) for residue in residues["b"]])
    rotation, translation = kabsch(ca_b, ca_a)
    aligned_ca_b = ca_b @ rotation + translation
    receptor_rmsd = float(np.sqrt(np.mean(np.sum((ca_a - aligned_ca_b) ** 2, axis=1))))

    holo_a = "\n".join(data["a"]["receptor_lines"]) + "\nEND\n"
    holo_b = transformed_pdb(data["b"]["receptor_lines"], rotation, translation)
    peptide_a = "\n".join(data["a"]["peptide_lines"]) + "\nEND\n"
    peptide_b = transformed_pdb(data["b"]["peptide_lines"], rotation, translation)
    case_dir.mkdir(parents=True)
    files = {
        "anchor_a.pdb": holo_b,
        "anchor_b.pdb": holo_a,
        "holo_a.pdb": holo_a,
        "holo_b.pdb": holo_b,
        "peptide_a.pdb": peptide_a,
        "peptide_b.pdb": peptide_b,
    }
    for name, content in files.items():
        (case_dir / name).write_text(content, encoding="utf-8")
    peptide_atoms = [
        pdb_to_sdf(peptide_a, case_dir / "peptide_a.sdf"),
        pdb_to_sdf(peptide_b, case_dir / "peptide_b.sdf"),
    ]
    esm_dir = case_dir / "esm2_output"
    esm_dir.mkdir()
    embedding_residues = {}
    for receptor_name in ("anchor_a.pdb", "anchor_b.pdb", "holo_a.pdb", "holo_b.pdb"):
        parsed = receptor_residues(atom_lines(files[receptor_name].encode("utf-8")))
        chain_counts = {}
        for residue in parsed:
            chain = residue["key"][0]
            chain_counts.setdefault(chain, 0)
            chain_counts[chain] += all(atom in residue["atoms"] for atom in ("N", "CA", "C"))
        embedding_residues[receptor_name] = list(chain_counts.values())
        for chain_index, count in enumerate(chain_counts.values()):
            zero_esm = torch.zeros((count, 1280), dtype=torch.float32)
            torch.save(
                {"representations": {33: zero_esm}},
                esm_dir / f"{receptor_name}_chain_{chain_index}.pt",
            )
    for label in ("a", "b"):
        state_dir = case_dir / f"pepflow_{label}"
        state_dir.mkdir()
        (state_dir / "pocket.pdb").write_text(files[f"anchor_{label}.pdb"], encoding="utf-8")
        (state_dir / "receptor.pdb").write_text(files[f"anchor_{label}.pdb"], encoding="utf-8")
        (state_dir / "peptide.pdb").write_text(files[f"peptide_{label}.pdb"], encoding="utf-8")
    return {
        "entries": [data["a"]["entry"], data["b"]["entry"]],
        "peptides": [data["a"]["peptide_fasta"], data["b"]["peptide_fasta"]],
        "receptor_sequence": data["a"]["receptor_fasta"],
        "receptor_sequence_sha256": hashlib.sha256(data["a"]["receptor_fasta"].encode()).hexdigest(),
        "receptor_residues": len(residues["a"]),
        "receptor_alignment_ca_rmsd": receptor_rmsd,
        "embedding_residues": embedding_residues,
        "pocket_ca_rmsd": float(row["pocket_ca_rmsd"]),
        "peptide_atoms": peptide_atoms,
        "anchor_type": "cross_holo",
        "anchor_definition": "state B holo anchors state A and state A holo anchors state B",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--pairs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--maximum-receptor-residues", type=int, default=500)
    parser.add_argument("--maximum-receptor-rmsd", type=float, default=3.0)
    parser.add_argument("--minimum-pocket-rmsd", type=float, default=0.25)
    parser.add_argument("--maximum-pocket-rmsd", type=float, default=5.0)
    args = parser.parse_args()
    require_clean_commit()
    archive_path, candidate_path = Path(args.archive).resolve(), Path(args.candidate_csv).resolve()
    for path in (archive_path, candidate_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    started = datetime.now().astimezone()
    result_dir = RESULTS_ROOT / started.strftime("%Y%m%d-%H%M%S%z")
    cases_dir = result_dir / "cases"
    cases_dir.mkdir(parents=True)
    commit, message = git("rev-parse", "HEAD"), git("log", "-1", "--format=%B")
    (result_dir / "commit.txt").write_text(f"{commit}\n{message.strip()}\n", encoding="utf-8")
    with candidate_path.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    candidates = [row for row in candidates if (
        5 <= int(row["peptide_length_a"]) <= 30
        and 5 <= int(row["peptide_length_b"]) <= 30
        and 40 <= int(row["mapped_ca"]) <= args.maximum_receptor_residues
        and float(row["receptor_ca_rmsd"]) <= args.maximum_receptor_rmsd
        and args.minimum_pocket_rmsd <= float(row["pocket_ca_rmsd"]) <= args.maximum_pocket_rmsd
    )]
    candidates.sort(key=lambda row: (
        abs(float(row["pocket_ca_rmsd"]) - 1.5),
        float(row["receptor_ca_rmsd"]), row["entry_a"], row["entry_b"],
    ))
    selected, failures, seen_receptors, seen_entries = [], [], set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        for row in candidates:
            receptor_sequence = fasta(archive.read(f"{row['entry_a']}/receptor.fasta"))
            receptor_hash = hashlib.sha256(receptor_sequence.encode()).hexdigest()
            if receptor_hash in seen_receptors or row["entry_a"] in seen_entries or row["entry_b"] in seen_entries:
                continue
            case_id = f"case-{len(selected):03d}-{row['entry_a']}-{row['entry_b']}"
            case_dir = cases_dir / case_id
            try:
                record = prepare_case(archive, row, case_dir)
            except Exception as error:
                shutil.rmtree(case_dir, ignore_errors=True)
                failures.append({"entries": [row["entry_a"], row["entry_b"]], "error": repr(error)})
                continue
            record["case_id"] = case_id
            selected.append(record)
            seen_receptors.add(receptor_hash)
            seen_entries.update(record["entries"])
            if len(selected) == args.pairs:
                break
    if len(selected) < args.pairs:
        raise RuntimeError(f"prepared only {len(selected)} of {args.pairs} requested pairs")

    random_groups = [[record["case_id"]] for record in selected]
    peptide_groups = connected_groups(
        selected,
        lambda left, right: any(
            sequence_identity(a, b) >= 0.5 for a in left["peptides"] for b in right["peptides"]
        ),
    )
    receptor_groups = connected_groups(
        selected,
        lambda left, right: kmer_jaccard(left["receptor_sequence"], right["receptor_sequence"]) >= 0.3,
    )
    splits = {
        "random_complex": grouped_split(random_groups, args.seed),
        "peptide_cluster": grouped_split(peptide_groups, args.seed + 1),
        "receptor_family_proxy": grouped_split(receptor_groups, args.seed + 2),
        "anchor_type": {
            "status": "unavailable",
            "reason": "this PepMerge subset contains cross-holo anchors only; apo/AF data are not fabricated",
            "available_types": ["cross_holo"],
        },
    }
    for record in selected:
        record.pop("receptor_sequence")
    finished = datetime.now().astimezone()
    output_hashes = {
        str(path.relative_to(result_dir)): sha256(path)
        for path in result_dir.rglob("*") if path.is_file() and path.name != "manifest.json"
    }
    manifest = {
        "started_at": started.isoformat(), "finished_at": finished.isoformat(),
        "status": "passed", "exit_code": 0, "branch": git("branch", "--show-current"),
        "commit": commit, "commit_message": message.strip(), "command": " ".join(sys.argv),
        "python": sys.executable, "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        "archive": str(archive_path), "archive_sha256": sha256(archive_path),
        "candidate_csv": str(candidate_path), "candidate_csv_sha256": sha256(candidate_path),
        "candidate_filter_count": len(candidates), "requested_pairs": args.pairs,
        "prepared_pairs": len(selected), "complex_states": 2 * len(selected),
        "failed_candidates": failures, "cases": selected, "splits": splits,
        "output_hashes": output_hashes, "elapsed_seconds": (finished - started).total_seconds(),
    }
    (result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    split_rows = "".join(
        f"<tr><td>{name}</td><td>{len(split.get('train', []))}</td>"
        f"<td>{len(split.get('validation', []))}</td><td>{len(split.get('test', []))}</td></tr>"
        for name, split in splits.items() if "train" in split
    )
    (result_dir / "report.html").write_text(
        "<html><head><meta charset='utf-8'><title>Joint small batch</title></head><body>"
        f"<h1>{len(selected)} cross-holo pairs / {2 * len(selected)} states</h1>"
        "<table><tr><th>split</th><th>train</th><th>validation</th><th>test</th></tr>"
        f"{split_rows}</table><p>Anchor-type holdout unavailable: only cross-holo is present.</p>"
        "</body></html>\n", encoding="utf-8",
    )
    print(json.dumps({
        "result_dir": str(result_dir), "prepared_pairs": len(selected),
        "complex_states": 2 * len(selected), "splits": splits,
        "failures": len(failures), "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
