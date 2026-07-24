#!/usr/bin/env python3
"""Audit ligand-associated conformational heterogeneity in a DynamicBind dataset."""

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser, PDBParser


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "dynamicbind" / "test" / "results" / "day1-multistate-audit"
ATOMIC_WEIGHTS = {
    "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45, "Br": 79.904,
    "I": 126.904,
}
TYPICAL_VALENCE = {
    "B": 3, "C": 4, "N": 3, "O": 2, "F": 1, "P": 3, "S": 2,
    "Cl": 1, "Br": 1, "I": 1,
}
CHI_ATOMS = {
    "ARG": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "NE"), ("CG", "CD", "NE", "CZ")),
    "ASN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "ASP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "OD1")),
    "CYS": (("N", "CA", "CB", "SG"),),
    "GLN": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
    "GLU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "OE1")),
    "HIS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "ND1")),
    "ILE": (("N", "CA", "CB", "CG1"), ("CA", "CB", "CG1", "CD1")),
    "LEU": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "LYS": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD"), ("CB", "CG", "CD", "CE"), ("CG", "CD", "CE", "NZ")),
    "MET": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "SD"), ("CB", "CG", "SD", "CE")),
    "PHE": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "PRO": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD")),
    "SER": (("N", "CA", "CB", "OG"),),
    "THR": (("N", "CA", "CB", "OG1"),),
    "TRP": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "TYR": (("N", "CA", "CB", "CG"), ("CA", "CB", "CG", "CD1")),
    "VAL": (("N", "CA", "CB", "CG1"),),
}


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
    print("ERROR: audit must run from a committed worktree.", file=sys.stderr)
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


def build_member_index(archive):
    index = {}
    for name in archive.namelist():
        parts = name.rstrip("/").split("/")
        if len(parts) >= 2 and not name.endswith("/"):
            index[(parts[-2], parts[-1])] = name
    return index


def member_name(member_index, uid, source_path):
    basename = Path(str(source_path).replace(chr(92), "/")).name
    key = (uid, basename)
    if key not in member_index:
        raise FileNotFoundError(f"archive member ending in /{uid}/{basename}")
    return member_index[key]


def parse_protein(text, suffix, structure_id):
    parser = MMCIFParser(QUIET=True) if suffix.lower() == ".cif" else PDBParser(QUIET=True)
    structure = parser.get_structure(structure_id, io.StringIO(text))
    model = next(structure.get_models())
    residues = {}
    order = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            # Files are aligned and renumbered to a common UniProt sequence, but
            # the crystallographic chain ID can differ between entries.
            key = (int(residue.id[1]), str(residue.id[2]).strip())
            atoms = {}
            for atom in residue:
                altloc = str(atom.get_altloc()).strip()
                if altloc not in ("", "A", "1"):
                    continue
                atoms[str(atom.get_name()).strip()] = np.asarray(atom.coord, dtype=float)
            if "CA" not in atoms:
                continue
            residues[key] = {"resname": residue.get_resname().strip(), "atoms": atoms}
            order.append(key)
    if not residues:
        raise ValueError("no standard residues with CA atoms")
    return {"residues": residues, "order": order}


def parse_sdf(text):
    block = text.split("$$$$", 1)[0]
    lines = block.splitlines()
    if len(lines) < 4:
        raise ValueError("invalid SDF")
    counts = lines[3]
    try:
        atom_count = int(counts[:3])
        bond_count = int(counts[3:6])
    except ValueError as exc:
        raise ValueError("only V2000 SDF is supported by the audit") from exc
    coords, elements = [], []
    for line in lines[4:4 + atom_count]:
        coords.append([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        elements.append(line[31:34].strip())
    bonds = []
    start = 4 + atom_count
    for line in lines[start:start + bond_count]:
        bonds.append((int(line[:3]) - 1, int(line[3:6]) - 1, int(line[6:9])))
    molecular_weight, amide_bonds = ligand_descriptors(elements, bonds)
    return {
        "coords": np.asarray(coords, dtype=float),
        "elements": elements,
        "bonds": bonds,
        "rotatable_bonds": approximate_rotatable_bonds(elements, bonds),
        "molecular_weight": molecular_weight,
        "amide_bonds": amide_bonds,
    }


def ligand_descriptors(elements, bonds):
    adjacency = defaultdict(list)
    bond_order_sum = [0.0] * len(elements)
    for left, right, order in bonds:
        numeric_order = 1.5 if order == 4 else float(order)
        adjacency[left].append((right, order))
        adjacency[right].append((left, order))
        bond_order_sum[left] += numeric_order
        bond_order_sum[right] += numeric_order
    implicit_hydrogens = 0
    for index, element in enumerate(elements):
        valence = TYPICAL_VALENCE.get(element)
        if valence is not None:
            implicit_hydrogens += max(0, int(round(valence - bond_order_sum[index])))
    molecular_weight = sum(ATOMIC_WEIGHTS.get(element, 0.0) for element in elements)
    molecular_weight += implicit_hydrogens * ATOMIC_WEIGHTS["H"]
    amide_bonds = 0
    for left, right, order in bonds:
        if order != 1:
            continue
        carbon, nitrogen = None, None
        if elements[left] == "C" and elements[right] == "N":
            carbon, nitrogen = left, right
        elif elements[right] == "C" and elements[left] == "N":
            carbon, nitrogen = right, left
        if carbon is not None and any(
            elements[neighbor] == "O" and neighbor_order == 2
            for neighbor, neighbor_order in adjacency[carbon]
            if neighbor != nitrogen
        ):
            amide_bonds += 1
    return molecular_weight, amide_bonds


def approximate_rotatable_bonds(elements, bonds):
    adjacency = defaultdict(set)
    bond_orders = {}
    for left, right, order in bonds:
        adjacency[left].add(right)
        adjacency[right].add(left)
        bond_orders[frozenset((left, right))] = order

    def connected_without(blocked):
        nodes = [i for i, element in enumerate(elements) if element != "H"]
        if not nodes:
            return True
        seen, stack = {nodes[0]}, [nodes[0]]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if frozenset((node, neighbor)) == blocked or elements[neighbor] == "H":
                    continue
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(nodes)

    count = 0
    for left, right, order in bonds:
        if order != 1 or elements[left] == "H" or elements[right] == "H":
            continue
        if len(adjacency[left]) <= 1 or len(adjacency[right]) <= 1:
            continue
        blocked = frozenset((left, right))
        if connected_without(blocked):
            continue
        carbon, nitrogen = None, None
        if elements[left] == "C" and elements[right] == "N":
            carbon, nitrogen = left, right
        elif elements[right] == "C" and elements[left] == "N":
            carbon, nitrogen = right, left
        if carbon is not None and any(
            elements[neighbor] == "O"
            and bond_orders.get(frozenset((carbon, neighbor))) == 2
            for neighbor in adjacency[carbon]
            if neighbor != nitrogen
        ):
            continue
        count += 1
    return count


def dihedral(points):
    p0, p1, p2, p3 = points
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if norm < 1e-8:
        return None
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    if np.linalg.norm(v) < 1e-8 or np.linalg.norm(w) < 1e-8:
        return None
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def residue_chis(residue):
    result = []
    atoms = residue["atoms"]
    for names in CHI_ATOMS.get(residue["resname"], ()):
        if not all(name in atoms for name in names):
            break
        value = dihedral([atoms[name] for name in names])
        if value is None:
            break
        result.append(value)
    return result


def pocket_residues(protein, ligand_coords, cutoff):
    result = set()
    cutoff_squared = cutoff * cutoff
    for key, residue in protein["residues"].items():
        atom_coords = np.asarray(list(residue["atoms"].values()))
        if atom_coords.size == 0:
            continue
        delta = atom_coords[:, None, :] - ligand_coords[None, :, :]
        if np.min(np.sum(delta * delta, axis=-1)) <= cutoff_squared:
            result.add(key)
    return result


def compact_protein(protein, pocket):
    compact = {"residues": {}, "order": protein["order"]}
    for key, residue in protein["residues"].items():
        atoms = residue["atoms"] if key in pocket else {"CA": residue["atoms"]["CA"]}
        compact["residues"][key] = {"resname": residue["resname"], "atoms": atoms}
    return compact


def kabsch(reference, mobile):
    ref_center = reference.mean(axis=0)
    mob_center = mobile.mean(axis=0)
    covariance = (mobile - mob_center).T @ (reference - ref_center)
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right_t
    return rotation, ref_center, mob_center


def transform(coords, alignment):
    rotation, ref_center, mob_center = alignment
    return (coords - mob_center) @ rotation + ref_center


def circular_difference(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


def pair_metrics(first, second):
    common = sorted(set(first["protein"]["residues"]) & set(second["protein"]["residues"]))
    if len(common) < 20:
        raise ValueError(f"only {len(common)} common CA residues")
    first_ca = np.asarray([first["protein"]["residues"][key]["atoms"]["CA"] for key in common])
    second_ca = np.asarray([second["protein"]["residues"][key]["atoms"]["CA"] for key in common])
    alignment = kabsch(first_ca, second_ca)
    second_ca_aligned = transform(second_ca, alignment)
    all_ca_rmsd = float(np.sqrt(np.mean(np.sum((first_ca - second_ca_aligned) ** 2, axis=1))))

    pocket_union = (first["pocket"] | second["pocket"]) & set(common)
    pocket_common = first["pocket"] & second["pocket"] & set(common)
    pocket_indices = [index for index, key in enumerate(common) if key in pocket_union]
    if len(pocket_indices) < 3:
        raise ValueError(f"only {len(pocket_indices)} pocket CA residues")
    pocket_delta = first_ca[pocket_indices] - second_ca_aligned[pocket_indices]
    pocket_rmsd = float(np.sqrt(np.mean(np.sum(pocket_delta ** 2, axis=1))))
    overlap = len(pocket_common) / max(len(pocket_union), 1)

    compared_rotamers, changed_rotamers = 0, 0
    chi_differences = []
    for key in pocket_common:
        first_chis = residue_chis(first["protein"]["residues"][key])
        second_chis = residue_chis(second["protein"]["residues"][key])
        if not first_chis or not second_chis:
            continue
        differences = [
            circular_difference(left, right)
            for left, right in zip(first_chis, second_chis)
        ]
        if differences:
            compared_rotamers += 1
            chi_differences.extend(differences)
            if any(value >= 60.0 for value in differences):
                changed_rotamers += 1

    second_ligand_aligned = transform(second["ligand"]["coords"], alignment)
    ligand_centroid_distance = float(
        np.linalg.norm(first["ligand"]["coords"].mean(axis=0) - second_ligand_aligned.mean(axis=0))
    )
    resolution_values = [
        value for value in (first["resolution"], second["resolution"]) if np.isfinite(value)
    ]
    mean_resolution = float(np.mean(resolution_values)) if resolution_values else math.nan
    gap_penalty = abs(first["filled_num"]) + abs(second["filled_num"])
    clean_score = pocket_rmsd * (0.5 + overlap) / (1.0 + 0.02 * gap_penalty)
    return {
        "uid": first["uid"],
        "entry_a": first["entry"],
        "entry_b": second["entry"],
        "ligand_a": first["comp_id"],
        "ligand_b": second["comp_id"],
        "common_ca": len(common),
        "pocket_union_residues": len(pocket_union),
        "pocket_intersection_residues": len(pocket_common),
        "pocket_site_jaccard": overlap,
        "holo_all_ca_rmsd": all_ca_rmsd,
        "holo_pocket_ca_rmsd": pocket_rmsd,
        "ligand_centroid_distance": ligand_centroid_distance,
        "rotamer_residues_compared": compared_rotamers,
        "rotamer_changed_residues": changed_rotamers,
        "rotamer_changed_fraction": changed_rotamers / compared_rotamers if compared_rotamers else math.nan,
        "mean_abs_chi_difference": float(np.mean(chi_differences)) if chi_differences else math.nan,
        "mw_a": first["mw"],
        "mw_b": second["mw"],
        "heavy_atoms_a": first["n_atoms"],
        "heavy_atoms_b": second["n_atoms"],
        "rotatable_bonds_a": first["ligand"]["rotatable_bonds"],
        "rotatable_bonds_b": second["ligand"]["rotatable_bonds"],
        "peptide_like_a": first["peptide_like"],
        "peptide_like_b": second["peptide_like"],
        "large_flexible_a": first["large_flexible"],
        "large_flexible_b": second["large_flexible"],
        "mean_resolution": mean_resolution,
        "clean_score": clean_score,
    }


def row_value(row, names, default):
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def load_records(csv_path, archive_path, cutoff, include_groups):
    table = pd.read_csv(csv_path)
    if include_groups and "group" in table:
        table = table[table["group"].astype(str).isin(include_groups)].copy()
    records, errors = [], []
    with zipfile.ZipFile(archive_path) as archive:
        member_index = build_member_index(archive)
        for progress, (index, row) in enumerate(table.iterrows(), start=1):
            uid = str(row["uid"])
            entry = str(row_value(row, ("entryName", "pdb"), index))
            try:
                protein_member = member_name(member_index, uid, row["pdbFile"])
                ligand_member = member_name(member_index, uid, row["ligandFile"])
                protein_text = archive.read(protein_member).decode("utf-8", errors="replace")
                ligand_text = archive.read(ligand_member).decode("utf-8", errors="replace")
                protein = parse_protein(protein_text, Path(protein_member).suffix, entry)
                ligand = parse_sdf(ligand_text)
                pocket = pocket_residues(protein, ligand["coords"], cutoff)
                protein = compact_protein(protein, pocket)
                molecular_weight = float(row_value(
                    row, ("wt",), ligand["molecular_weight"]
                ))
                n_atoms = int(row_value(
                    row, ("n_atoms",), sum(element != "H" for element in ligand["elements"])
                ))
                record = {
                    "uid": uid,
                    "entry": entry,
                    "comp_id": str(row_value(row, ("comp_id", "ligand"), "")),
                    "mw": molecular_weight,
                    "n_atoms": n_atoms,
                    "resolution": float(row_value(row, ("resolution",), math.nan)),
                    "filled_num": int(row_value(row, ("filled_num",), 0)),
                    "protein": protein,
                    "ligand": ligand,
                    "pocket": pocket,
                    "protein_member": protein_member,
                    "ligand_member": ligand_member,
                    "peptide_like": bool(ligand["amide_bonds"] >= 2 and n_atoms >= 10),
                    "large_flexible": bool(n_atoms >= 30 or ligand["rotatable_bonds"] >= 8),
                }
                records.append(record)
            except Exception as exc:
                errors.append({
                    "row": int(index),
                    "uid": uid,
                    "entry": entry,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if progress % 100 == 0:
                print(f"Parsed {progress}/{len(table)} selected metadata rows", flush=True)
    return table, records, errors


def select_candidates(pair_table, count):
    eligible = pair_table[
        (pair_table["pocket_union_residues"] >= 5)
        & (pair_table["pocket_site_jaccard"] >= 0.35)
        & (pair_table["holo_pocket_ca_rmsd"] >= 1.0)
    ].sort_values(["clean_score", "holo_pocket_ca_rmsd"], ascending=False)
    selected = []
    used_uids = set()
    for _, row in eligible.iterrows():
        if row["uid"] in used_uids:
            continue
        selected.append(row.to_dict())
        used_uids.add(row["uid"])
        if len(selected) >= count:
            break
    if len(selected) < count:
        fallback = pair_table.sort_values(
            ["clean_score", "holo_pocket_ca_rmsd"], ascending=False
        )
        for _, row in fallback.iterrows():
            if row["uid"] in used_uids:
                continue
            selected.append(row.to_dict())
            used_uids.add(row["uid"])
            if len(selected) >= count:
                break
    return pd.DataFrame(selected)


def plot_overlay(first, second, destination, title):
    common = sorted(set(first["protein"]["residues"]) & set(second["protein"]["residues"]))
    first_ca = np.asarray([first["protein"]["residues"][key]["atoms"]["CA"] for key in common])
    second_ca = np.asarray([second["protein"]["residues"][key]["atoms"]["CA"] for key in common])
    alignment = kabsch(first_ca, second_ca)
    second_ca = transform(second_ca, alignment)
    second_ligand = transform(second["ligand"]["coords"], alignment)
    center = np.vstack((first_ca, second_ca)).mean(axis=0)
    _, _, axes = np.linalg.svd(np.vstack((first_ca, second_ca)) - center, full_matrices=False)

    def project(coords):
        return (coords - center) @ axes[:2].T

    first_2d, second_2d = project(first_ca), project(second_ca)
    first_ligand_2d = project(first["ligand"]["coords"])
    second_ligand_2d = project(second_ligand)
    pocket_union = first["pocket"] | second["pocket"]
    pocket_indices = [index for index, key in enumerate(common) if key in pocket_union]

    figure, axis = plt.subplots(figsize=(8, 7))
    axis.plot(first_2d[:, 0], first_2d[:, 1], color="#d95f5f", alpha=0.32, linewidth=0.8)
    axis.plot(second_2d[:, 0], second_2d[:, 1], color="#3977d5", alpha=0.32, linewidth=0.8)
    axis.scatter(first_2d[pocket_indices, 0], first_2d[pocket_indices, 1], s=18, color="#c73535", label=f"{first['entry']} pocket")
    axis.scatter(second_2d[pocket_indices, 0], second_2d[pocket_indices, 1], s=18, color="#1f5bb5", label=f"{second['entry']} pocket")
    axis.scatter(first_ligand_2d[:, 0], first_ligand_2d[:, 1], s=25, color="#f3a712", marker="o", label=f"{first['comp_id']} ligand")
    axis.scatter(second_ligand_2d[:, 0], second_ligand_2d[:, 1], s=25, color="#28a87d", marker="^", label=f"{second['comp_id']} ligand")
    axis.set_title(title)
    axis.set_xlabel("PCA axis 1 (Å)")
    axis.set_ylabel("PCA axis 2 (Å)")
    axis.set_aspect("equal", adjustable="datalim")
    axis.legend(fontsize=8, loc="best")
    axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_summary(pair_table, destination):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(pair_table["holo_pocket_ca_rmsd"], bins=35, color="#3977d5", alpha=0.85)
    axes[0].axvline(1.0, color="#d95f5f", linestyle="--")
    axes[0].axvline(2.0, color="#8a2c70", linestyle="--")
    axes[0].set_xlabel("Pairwise holo pocket Cα RMSD (Å)")
    axes[0].set_ylabel("Pair count")
    scatter = axes[1].scatter(
        pair_table["pocket_site_jaccard"],
        pair_table["holo_pocket_ca_rmsd"],
        c=pair_table["rotamer_changed_fraction"].fillna(0),
        cmap="viridis",
        s=14,
        alpha=0.65,
    )
    axes[1].set_xlabel("Pocket residue-set Jaccard")
    axes[1].set_ylabel("Pairwise holo pocket Cα RMSD (Å)")
    figure.colorbar(scatter, ax=axes[1], label="Changed rotamer fraction")
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def write_report(path, summary, candidates, dataset_label, include_groups):
    group_text = ", ".join(include_groups) if include_groups else "all"
    lines = [
        f"# DynamicBind {dataset_label} multi-state audit",
        "",
        "## Scope",
        "",
        f"- Dataset label: {dataset_label}",
        f"- Included metadata groups: {group_text}",
        "- The audit measures structural signal in the supplied records. It does not",
        "  by itself prove that a trained checkpoint uses ligand identity causally;",
        "  that is tested with a common receptor anchor in Day 2.",
        "",
        "## Dataset summary",
        "",
        f"- Selected metadata rows: {summary['metadata_rows']}",
        f"- Successfully parsed complexes: {summary['parsed_complexes']}",
        f"- Parse failures: {summary['parse_failures']}",
        f"- Protein (UniProt) clusters: {summary['protein_clusters']}",
        f"- Clusters with at least two complexes: {summary['multi_ligand_clusters']}",
        f"- Valid within-protein pairs: {summary['valid_pairs']}",
        f"- Pocket Cα RMSD > 1 Å: {summary['pairs_pocket_rmsd_gt_1']}",
        f"- Pocket Cα RMSD > 2 Å: {summary['pairs_pocket_rmsd_gt_2']}",
        f"- Overlapping-site pairs (Jaccard ≥ 0.5) with RMSD > 1 Å: {summary['overlap_signal_pairs_gt_1']}",
        f"- Overlapping-site pairs (Jaccard ≥ 0.5) with RMSD > 2 Å: {summary['overlap_signal_pairs_gt_2']}",
        f"- Peptide-like complexes (transparent heuristic): {summary['peptide_like_complexes']}",
        f"- Large/flexible complexes (≥30 heavy atoms or ≥8 rotatable bonds): {summary['large_flexible_complexes']}",
        "",
        "Pocket residues are protein residues with any heavy atom within 5 Å of the",
        "native ligand. Pairwise structures are globally Kabsch-aligned on common",
        "Cα atoms before pocket RMSD is measured. A residue is counted as a rotamer",
        "change when any comparable χ angle differs by at least 60°.",
        "",
        "## Candidate systems",
        "",
        "| UID | holo A | ligand A | holo B | ligand B | pocket RMSD | site Jaccard | rotamer changed |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for _, row in candidates.iterrows():
        rotamer = row["rotamer_changed_fraction"]
        rotamer_text = "NA" if not np.isfinite(rotamer) else f"{rotamer:.2f}"
        lines.append(
            f"| {row['uid']} | {row['entry_a']} | {row['ligand_a']} | "
            f"{row['entry_b']} | {row['ligand_b']} | {row['holo_pocket_ca_rmsd']:.2f} | "
            f"{row['pocket_site_jaccard']:.2f} | {rotamer_text} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "",
        "The causal model test remains Day 2: hold the receptor anchor fixed, change",
        "only ligand identity, and compare predictions against both holo endpoints.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--structure-zip", required=True)
    parser.add_argument("--dataset-label", default="dataset")
    parser.add_argument(
        "--include-groups",
        default="",
        help="Comma-separated metadata group values, for example train,valid",
    )
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--pocket-cutoff", type=float, default=5.0)
    parser.add_argument("--candidate-count", type=int, default=8)
    args = parser.parse_args()

    require_clean_commit()
    csv_path, archive_path = Path(args.metadata_csv), Path(args.structure_zip)
    include_groups = tuple(
        value.strip() for value in args.include_groups.split(",") if value.strip()
    )
    started = datetime.now().astimezone()
    stamp = started.strftime("%Y%m%d-%H%M%S%z")
    result_dir = Path(args.output_root) / stamp
    result_dir.mkdir(parents=True, exist_ok=False)
    (result_dir / "overlays").mkdir()

    commit = git("rev-parse", "HEAD")
    commit_message = git("log", "-1", "--format=%B")
    metadata = {
        "started_at": started.isoformat(),
        "finished_at": None,
        "status": "running",
        "exit_code": None,
        "commit": commit,
        "branch": git("branch", "--show-current"),
        "commit_message": commit_message,
        "command": sys.argv,
        "python": sys.executable,
        "dataset_label": args.dataset_label,
        "include_groups": include_groups,
        "input_sha256": {
            str(csv_path): sha256(csv_path),
            str(archive_path): sha256(archive_path),
        },
    }
    (result_dir / "commit.txt").write_text(
        f"{commit}\n{commit_message.rstrip()}\n", encoding="utf-8"
    )
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    try:
        source_table, records, errors = load_records(
            csv_path, archive_path, args.pocket_cutoff, include_groups
        )
        if errors:
            pd.DataFrame(errors).to_csv(result_dir / "parse_errors.csv", index=False)
        records_by_uid = defaultdict(list)
        records_by_entry = {}
        for record in records:
            records_by_uid[record["uid"]].append(record)
            records_by_entry[record["entry"]] = record

        pairs, pair_errors = [], []
        for uid, group in records_by_uid.items():
            for first, second in itertools.combinations(group, 2):
                if first["comp_id"] == second["comp_id"]:
                    continue
                try:
                    pairs.append(pair_metrics(first, second))
                except Exception as exc:
                    pair_errors.append({
                        "uid": uid,
                        "entry_a": first["entry"],
                        "entry_b": second["entry"],
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        if not pairs:
            raise RuntimeError("no valid within-protein pairs")
        pair_table = pd.DataFrame(pairs).sort_values(
            ["uid", "holo_pocket_ca_rmsd"], ascending=[True, False]
        )
        pair_table.to_csv(result_dir / "pair_statistics.csv", index=False)
        if pair_errors:
            pd.DataFrame(pair_errors).to_csv(result_dir / "pair_errors.csv", index=False)

        clusters = []
        for uid, group in records_by_uid.items():
            subset = pair_table[pair_table["uid"] == uid]
            clusters.append({
                "uid": uid,
                "complexes": len(group),
                "distinct_ligands": len({record["comp_id"] for record in group}),
                "valid_pairs": len(subset),
                "pairs_pocket_rmsd_gt_1": int((subset["holo_pocket_ca_rmsd"] > 1).sum()),
                "pairs_pocket_rmsd_gt_2": int((subset["holo_pocket_ca_rmsd"] > 2).sum()),
                "median_pocket_rmsd": float(subset["holo_pocket_ca_rmsd"].median()) if len(subset) else math.nan,
                "max_pocket_rmsd": float(subset["holo_pocket_ca_rmsd"].max()) if len(subset) else math.nan,
                "peptide_like_complexes": sum(record["peptide_like"] for record in group),
                "large_flexible_complexes": sum(record["large_flexible"] for record in group),
            })
        cluster_table = pd.DataFrame(clusters).sort_values(
            ["complexes", "max_pocket_rmsd"], ascending=False
        )
        cluster_table.to_csv(result_dir / "cluster_statistics.csv", index=False)

        candidates = select_candidates(pair_table, args.candidate_count)
        candidates.to_csv(result_dir / "candidate_systems.csv", index=False)
        for rank, row in candidates.reset_index(drop=True).iterrows():
            first = records_by_entry[row["entry_a"]]
            second = records_by_entry[row["entry_b"]]
            name = f"{rank + 1:02d}_{row['uid']}_{row['entry_a']}_vs_{row['entry_b']}.png"
            title = (
                f"{row['uid']}: {row['ligand_a']} vs {row['ligand_b']}\n"
                f"pocket Cα RMSD={row['holo_pocket_ca_rmsd']:.2f} Å, "
                f"site Jaccard={row['pocket_site_jaccard']:.2f}"
            )
            plot_overlay(first, second, result_dir / "overlays" / name, title)
        plot_summary(pair_table, result_dir / "pair_summary.png")

        overlap = pair_table["pocket_site_jaccard"] >= 0.5
        strong_one = int((overlap & (pair_table["holo_pocket_ca_rmsd"] > 1)).sum())
        strong_two = int((overlap & (pair_table["holo_pocket_ca_rmsd"] > 2)).sum())
        if strong_two >= 10:
            interpretation = (
                "MDT contains a substantial set of same-protein, overlapping-site holo pairs "
                "with backbone differences above 2 Å. This is consistent with a real "
                "ligand-associated conformational signal rather than only alternate binding "
                "sites, but crystal/construct effects remain possible confounders."
            )
        elif strong_one >= 10:
            interpretation = (
                "MDT contains repeated same-protein, overlapping-site holo pairs above 1 Å, "
                "but large >2 Å transitions are limited. The signal is suitable for a cautious "
                "same-anchor inference test, not yet for a broad state-selection claim."
            )
        else:
            interpretation = (
                "After requiring overlapping binding sites, few pairs exceed 1 Å. The supplied "
                "data do not provide strong evidence for a broad multi-state benchmark."
            )
        summary = {
            "metadata_rows": int(len(source_table)),
            "parsed_complexes": len(records),
            "parse_failures": len(errors),
            "protein_clusters": len(records_by_uid),
            "multi_ligand_clusters": sum(
                len({record["comp_id"] for record in group}) >= 2
                for group in records_by_uid.values()
            ),
            "valid_pairs": len(pair_table),
            "pair_failures": len(pair_errors),
            "pairs_pocket_rmsd_gt_1": int((pair_table["holo_pocket_ca_rmsd"] > 1).sum()),
            "pairs_pocket_rmsd_gt_2": int((pair_table["holo_pocket_ca_rmsd"] > 2).sum()),
            "overlap_signal_pairs_gt_1": strong_one,
            "overlap_signal_pairs_gt_2": strong_two,
            "peptide_like_complexes": sum(record["peptide_like"] for record in records),
            "large_flexible_complexes": sum(record["large_flexible"] for record in records),
            "interpretation": interpretation,
        }
        (result_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        write_report(
            result_dir / "report.md",
            summary,
            candidates,
            args.dataset_label,
            include_groups,
        )
        metadata.update({
            "finished_at": datetime.now().astimezone().isoformat(),
            "status": "passed",
            "exit_code": 0,
            "summary": summary,
        })
        exit_code = 0
    except Exception as exc:
        metadata.update({
            "finished_at": datetime.now().astimezone().isoformat(),
            "status": "failed",
            "exit_code": 1,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        exit_code = 1
    (result_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Audit {metadata['status']}: {result_dir}")
    if exit_code:
        print(metadata["traceback"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
