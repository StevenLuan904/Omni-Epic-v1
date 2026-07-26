# Frozen DynamicBind state-probe reproduction

## Outcome

The `implementation-plan-v0726-3.md` frozen-feature experiment is complete.
The last ligand-to-receptor interaction layer contains a robust linearly
readable signal for the two c-Met ligand/state examples, but the same protocol
does not generalize across held-out ligand placements for Q9H2K2.

| System | Primary test accuracy | Three-seed mean accuracy / AUC | Minimum swap exchange | Gate |
| --- | ---: | ---: | ---: | --- |
| c-Met | 1.000 | 1.000 / 1.000 | 1.000 | pass |
| Q9H2K2 | 0.500 | 0.556 / 0.778 | 0.500 | fail |

Because both systems had to pass before unfreezing interaction and receptor
motion layers, the planned LoRA experiment was not launched. This is the
case-dependent branch of the plan: the available checkpoint does not provide
a sufficiently stable Q9H2K2 state signal to justify a small decoder repair.

This result is not evidence that ligand-receptor *compatibility* has been
identified. Each state has only one ligand, so ligand identity and state are
confounded. The evaluator therefore records a ligand-only baseline and marks
`state_compatibility_identifiable` false for both systems.

## Code and commits

Every formal computation ran from a clean `main` worktree at
`ea7308a4b959c56a3dbdb2006b60d9156ca506b9`.

| Commit | Time (+08:00) | Change |
| --- | --- | --- |
| `252a6f0` | 2026-07-26 17:26:37 | Extract frozen interaction features |
| `e20b861` | 2026-07-26 17:30:01 | Make placement and extraction deterministic |
| `f664d5c` | 2026-07-26 17:52:52 | Add true row-swap and methane controls |
| `5881b66` | 2026-07-26 17:53:45 | Add the fixed split linear readout |
| `192d165` | 2026-07-26 17:55:03 | Align permutation and interval statistics |
| `ea7308a` | 2026-07-26 18:31:53 | Track required vendored dataset modules |

Entry points:

- `dynamicbind/test/extract_frozen_interactions.py`
- `dynamicbind/test/evaluate_frozen_state_probe.py`

The extractor discards native SDF coordinates, rebuilds conformers with
seeded ETKDG, places both conditions from the same seeded transforms, uses a
common receptor anchor, and pools mean/standard-deviation/maximum over the
invariant scalar channels from `lig_to_rec_conv_layers[-1]`. The DynamicBind
parameters are frozen and their pre/post SHA-256 values must match.

## Server, environment, and input audit

The primary and additional servers had pre-existing jobs. The formal runs
therefore used idle GPUs 5 and 6 on the authorized synth server without
interrupting any process:

- route: `eh002` to `synth@192.168.99.2`
- host: `admin.cluster.local`
- project: `/sdd_data/synth/Omni-Epic-v1/repo`
- environment: `/sdd_data/synth/Omni-Epic-v1/venvs/dynamicbind-frozen-probe`
- GPU: NVIDIA GeForce RTX 3090, 24,576 MiB
- Python 3.8.10, PyTorch 2.0.1+cu117, CUDA 11.7

Critical transferred artifacts were SHA-256 checked before use:

| Artifact | SHA-256 |
| --- | --- |
| Paper checkpoint | `3a8ed1e65b01aab71016520640a6b80dbd843afa9734ca4ecc9919a32fc30202` |
| c-Met ESM embedding | `7b03977566f2781ff622353dfa46f4e6b4629c7d475c1104c17d87822c0356ed` |
| Q9H2K2 ESM embedding | `3c6565119a2803e86e13fb4feec2e506e30f175fe26770d2560628f6fe8fd49b` |
| c-Met common anchor | `38b098eb8d165dfdc90dcbaf1b6fc3cf812b863d03ad6dac2bddc976a7c376b6` |
| Q9H2K2 common anchor | `e525a041ee60f4d7b270104751d211b3337ed1c55af1ec9fb9c685105e9744dd` |

The six SO(3)/torus cache files were also compared with their source-server
hashes after transfer. The model parameter hash was
`105ec2116bd4f08f0c1381d9e6dab64dc0a6670267ffb9cb8b2cafc1087cb0ca`
before and after both extraction runs.

## Frozen feature extraction

Each system used 24 paired placement seeds, diffusion time `t=0.6`, and 216
pooled scalar features per condition sample. Native ligand coordinates and
holo pockets were not used as model inputs.

| System | Visible GPU | Result directory | Feature shape | Paired L2 mean | Feature SHA-256 |
| --- | ---: | --- | ---: | ---: | --- |
| c-Met | 5 | `frozen-interaction-features/20260726-184607+0800-cmet` | 48 x 216 | 48.2637 | `e0b61ea7a3ff693c592d5b27f19353f0f5316ac11d8047f36621c49cf8599193` |
| Q9H2K2 | 6 | `frozen-interaction-features/20260726-185437+0800-q9h2k2` | 48 x 216 | 9.3484 | `c65f4b7ded0091a1230dc8328c60e689fa20bb4a58ae0ed7ab7bbc36f6fd8008` |

The independent row-swapped re-executions agreed with exchanged endpoints to
maximum absolute errors of `2.86e-5` (c-Met) and `1.14e-5` (Q9H2K2). When the
same seeded methane graph replaced both ligands, paired feature distances fell
to `6.98e-5` and `1.95e-4`, respectively.

## Fixed-pair readout

The readout was specified before formal evaluation: a single linear layer,
12 paired placement seeds for training, 6 for validation, and 6 untouched
seeds for final testing. Seeds `20260726`, `20260727`, and `20260728` test
head-initialization sensitivity. The fixed gate requires mean test accuracy
and AUC at least 0.80, minimum accuracy and swap exchange at least 0.75, and
maximum methane-control absolute logit at most 0.405.

The CPU-only evaluation ran from 2026-07-26 18:56:23 to 18:56:55 +08:00 and
completed with exit code 0. Its result directory is:

`dynamicbind/test/results/frozen-state-probe/20260726-185623+0800`

| Metric | c-Met | Q9H2K2 |
| --- | ---: | ---: |
| Primary test accuracy | 1.000 | 0.500 |
| Three-seed mean test accuracy | 1.000 | 0.556 |
| Three-seed minimum test accuracy | 1.000 | 0.500 |
| Three-seed mean test AUC | 1.000 | 0.778 |
| Minimum input-row swap exchange accuracy | 1.000 | 0.500 |
| Maximum methane-control absolute logit | 0.0993 | 0.1162 |
| Primary 6-pair Wilson 95% interval | [0.610, 1.000] | [0.188, 0.812] |
| 99-run matched-label permutation p-value | 0.19 | 0.56 |
| Ligand-only test accuracy | 1.000 | 1.000 |

The perfect c-Met point estimate has only six independent test pairs and is
not significant under the small 99-run permutation diagnostic. Q9H2K2 fits
all training and validation pairs but fails on held-out placement seeds,
which is the decisive failure for the predeclared gate.

Evaluation artifact hashes:

| Artifact | SHA-256 |
| --- | --- |
| `metrics.json` | `bdef1392b4f4d3267a820e7238c2d552efb62d545b2fbc36552aca0267506091` |
| `run_metrics.csv` | `a90cbc553b3c4578a8782d77303b70900610d187d54044d02d1e3de7e4ae360d` |
| `report.html` | `4b7667a0cd16b5b8265fbbc144d30696d2531da1f8cdf05cdf260f0ef428ef63` |

## GPU and next-step decision

Post-run audit showed GPUs 5, 6, and 7 each at 24,257 MiB free and 0%
utilization; the extraction processes had exited. Existing jobs on GPUs 0--4
were left untouched. The remote Git worktree remained clean at `ea7308a`.

The v0726-3 decision is therefore complete: do not run the conditional LoRA
stage on these four examples. A scientifically identifiable next experiment
needs crossed ligand/state observations or an explicit multistate training
set, followed by held-out proteins and ligands; extending the earlier
translation-head fine-tune or fitting more layers to these pairs is not
supported by this result.
