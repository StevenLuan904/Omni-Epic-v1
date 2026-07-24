# PepFlow and DynamicBind reproduction 002

## Scope and governing rule

- Execution date: 2026-07-25 (Asia/Shanghai)
- Local repository: `D:\DWorkspace\yangyang\Omni-Epic\20250722-pepflow\Omni-Epic-v1`
- Remote repository: `/data0/luanhaoyang/Omni-Epic-v1`
- Governing rule:
  `D:\DWorkspace\yangyang\Omni-Epic\20250722-pepflow\test-requirement.md`
- Authorized physical GPUs: 0 and 1

Every recorded train or inference run rejected a dirty worktree, recorded the
full commit and message, and used a timestamped ignored results directory.
Local and remote commits were checked before remote execution.

## PepFlow minimum training

The original one-step training reproduction had passed at `b100d12`, but the
later repository move introduced two independent regressions:

1. At `829533f`, `pepflow/pepflow/utils/vc.py` required the working directory
   itself to be a Git repository. The clean remote run failed with
   `InvalidGitRepositoryError`.
2. After fixing repository discovery in `f91dc94`, the clean remote run reached
   dataset construction and failed because the moved smoke configuration still
   pointed to `./datasets/structures`.

The minimal fixes were committed separately:

- `f91dc94` — search parent directories for the Git repository;
- `86bd3f7` — point the moved smoke configuration to `../datasets`;
- `a4392e8` — record data hashes, elapsed time, and GPU snapshots in the smoke
  artifact.

The clean remote run at `a4392e8` passed:

- result: `pepflow/test/results/train-smoke/20260725-053431+0800`;
- Python: `/home/huangyueshan/.conda/envs/flex/bin/python`;
- visible GPUs: `0,1`; actual device: `cuda:0`;
- dataset entries: 9,849;
- model parameters: 6,880,353;
- completed forward/backward/optimizer iterations: 1;
- loss: 170.2016;
- gradient norm: 225.4276;
- forward/backward time: 1.049 / 0.431 seconds;
- total harness time: 9.346 seconds;
- exit code: 0.

The canonical final validation is the newest timestamped smoke result whose
`commit.txt` matches the repository HEAD containing this report.

## Data integrity

The source archives and stable training cache are stored below the remote
repository's `datasets/` directory:

```text
452240f8d60227c0959f7f3a8cf43a2f8a63e53806f4f47bdf8ccb1cb1f5ef08  PepMerge_lmdb.zip
eb0c9f6f81b85c399a32fe38e7f79274584805d7cafaac774a8d091792d0410c  PepMerge_release.zip
693037f65f02b8b48a513c337c12aa46905d85840b281a35058d54fd6c25a2f7  pep_pocket_train_structure_cache.lmdb
```

DynamicBind v1 inputs:

```text
5cf954d534d0fbdea15104f60e1f74db0fe9cd5982fe36d3598ddd03a2216b25  d3_with_clash_info.csv
518bfc0d745d620eab09f1bbecf61ba55e6cbb629601734df8749d01d8c3da77  pdbbind_v11_pocket_aligned_fill_missing.zip
80c47add23637f689ccdc97c382cd3c83d115430ca492d980d5216f6ef14dddd  workdir-v1.zip
3a8ed1e65b01aab71016520640a6b80dbd843afa9734ca4ecc9919a32fc30202  ema_inference_epoch314_model.pt
```

The official torus cache was regenerated with the official formula and
spot-checked:

```text
1b7656e9455daf4051566c4570f73d60f23163edb348363896631540849eda09  .p.npy
48b79d8d0ccca5adac87190c063494b62e7b2759ebdf9b17bc5a8bd982674fed  .score.npy
```

The official formula produces NaNs only in its tiny-sigma periodic tail
(777,312 values, 3.1%); this matches the unmodified formula and did not affect
the completed inference runs.

## DynamicBind Day 1: dataset audit

Canonical result:
`dynamicbind/test/results/day1-multistate-audit/20260725-033810+0800`.

- metadata rows: 13,560;
- parsed complexes: 13,318; parse failures: 242;
- protein clusters: 2,405;
- clusters with multiple ligands: 1,161;
- valid same-protein pairs: 355,876; pair failures: 4,593;
- pocket Cα RMSD >1 Å: 111,107 pairs;
- pocket Cα RMSD >2 Å: 32,890 pairs;
- overlapping-site signal (Jaccard >=0.5), >1 Å: 80,021 pairs;
- overlapping-site signal (Jaccard >=0.5), >2 Å: 22,448 pairs;
- peptide-like complexes: 4,139;
- large/flexible complexes: 6,143.

The cleanest candidate set included P07900, Q9H2K2, P00734, B0VD92,
Q16539, Q04609, P68400, and Q83883. The audit therefore supports structural
heterogeneity in the training distribution, but it does not by itself prove
that the checkpoint uses ligand identity to choose the endpoint.

## DynamicBind Day 2: same-anchor state selection

Three top-ranked systems were run from one AlphaFold anchor with ligand A,
ligand B, and a shuffled-ligand control. Each condition used seeds 11, 23, 37,
and 51 and 20 inference steps. All 36 expected predictions completed.

| Protein | Holo endpoints | Endpoint pocket RMSD | Mean directional margin | Correct endpoint fraction |
| --- | --- | ---: | ---: | ---: |
| P07900 | 2yjw / 3eko | 3.187 Å | +0.0825 Å | 0.50 |
| Q9H2K2 | 4ui7 / 4ufy | 2.340 Å | +0.0083 Å | 0.50 |
| P00734 | 4hfp / 1dwc | 4.450 Å | +0.0040 Å | 0.50 |

The per-system evaluations and two-dimensional endpoint scatter plots are:

- `day2-state-evaluation/20260725-051524+0800`;
- `day2-state-evaluation/20260725-052014+0800`;
- `day2-state-evaluation/20260725-052433+0800`.

Predictions were not byte-identical, but within each target all seeds converged
to essentially one receptor endpoint independent of ligand A/B. The shuffled
control commonly shared the same preference. All three systems are therefore
`negative-or-inconclusive`; there are no defensible “successful” examples to
promote. P07900 has the largest positive mean margin, while P00734 is the
clearest large-endpoint failure.

Per the implementation-plan gate, the frozen-trunk state head was deliberately
not trained. Training it after a 50% endpoint hit rate would only test
small-sample memorization and would not establish that the reproduced
checkpoint contains a usable state-selection signal.

## DynamicBind Day 3: native short peptides

Three dataset-native, connected, peptide-like ligands were selected after
checking resolution, clash score, atom count, amide topology, and flexibility:

| Entry | Length | Heavy atoms | Rotatable bonds | Resolution |
| --- | ---: | ---: | ---: | ---: |
| 1jet | 3-mer | 24 | 13 | 1.20 Å |
| 6m8w | 4-mer | 28 | 10 | 1.10 Å |
| 1oai | 9-mer | 66 | 30 | 1.00 Å |

Input export:
`day3-peptide-inputs/20260725-053747+0800`.

Inference used physical GPU 1, seed 11, and 20 real steps:
`day3-peptide-inference/20260725-053853+0800`.

- graph construction: 3/3;
- inference: 3/3;
- failed/skipped complexes: 0/0;
- inference elapsed time: 59.549 seconds.

Evaluation:
`day3-peptide-evaluation/20260725-054217+0800`.

| Entry | Ligand pose RMSD | Ligand conformer RMSD | Receptor pocket RMSD to holo |
| --- | ---: | ---: | ---: |
| 1jet | 0.887 Å | 0.806 Å | 0.105 Å |
| 6m8w | 1.856 Å | 1.587 Å | 0.263 Å |
| 1oai | 12.385 Å | 4.362 Å | 0.965 Å |

The 3-mer and 4-mer pass a practical pose smoke threshold; the flexible 9-mer
does not. This is evidence that the DynamicBind small-molecule representation
can technically ingest some short peptides, but its reliability degrades
strongly with peptide size and flexibility.

## GPU audit and decision

The server already had PID 4077042 using all eight GPUs. The task explicitly
authorized GPUs 0–1, and both retained sufficient memory headroom. PepFlow used
physical GPU 0 and DynamicBind used physical GPU 1. Before/after snapshots show
that each reproduction process exited and GPU memory returned to the prior
baseline; the unrelated existing process was not modified.

Decision:

- PepFlow minimum training: **passed and usable**.
- DynamicBind native short-peptide compatibility: **partial / Yellow**.
- Ligand-specific receptor-state recovery with this v1 checkpoint and
  evaluation: **not reproduced / Red**.

Do not train or advertise a state-selective design head yet. The next justified
work is to verify checkpoint/version and preprocessing alignment on official
paper cases, then repeat the same-anchor evaluation. PepFlow-to-DynamicBind
integration is worth exploring only as peptide-conditioned holo generation,
with a PepFlow-native peptide representation replacing the small-molecule
encoder for longer peptides.
