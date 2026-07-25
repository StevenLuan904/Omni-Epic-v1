# DynamicBind Figure 3 c-Met reproduction

## Outcome

The exact c-Met inputs from DynamicBind Figure 3 were reproduced with one
AlphaFold receptor anchor and two ligand conditions:

- P08581 / 6UBW / 84S / DFG-in
- P08581 / 7V3S / 5I9 / DFG-out

The evaluation implementation recovers the paper's AlphaFold pocket RMSD
baselines to numerical precision. A one-sample inference with the paper
checkpoint selects the correct endpoint for both ligands. Its ligand and
predicted-pocket RMSDs do not reproduce the paper's selected-pose values.

The planned low-cost paired correction was also tested. Five steps updating
only the receptor translation score head preserve correct state selection but
do not improve the two-condition mean pocket RMSD. The gate for expanding the
fine-tune to more layers or longer training therefore did not pass.

Primary references:

- [DynamicBind paper](https://www.nature.com/articles/s41467-024-45461-2)
- [Official DynamicBind repository](https://github.com/luwei0917/DynamicBind)

## Code and commits

All successful computations used a committed, clean `main` worktree.

| Commit | Time (+08:00) | Change |
| --- | --- | --- |
| `31468db34d4bf888991caeef5726c11646aa0bfe` | 2026-07-26 01:40:46 | Export exact Figure 3 c-Met inputs |
| `a7d435d56fbb8296a512055a49858051ff2e549e` | 2026-07-26 01:43:47 | Evaluate ligand and all-atom pocket RMSDs |
| `76014c28b0fa1e7702521aaa274e12e6c8198f5c` | 2026-07-26 01:56:46 | Add paired state fine-tuning |
| `cdf72d21c743c158d3ee97e34acc808fdfe59d84` | 2026-07-26 02:01:49 | Fix standalone imports |
| `edf585d7eb9830fee6da45a7b8ab111d21add57b` | 2026-07-26 02:05:47 | Validate the materialized graph cache |
| `bdc4fe7086cb942c3e3621bf1884b2592dea8efa` | 2026-07-26 02:10:27 | Use the low-memory receptor score-head scope |

The relevant entry points are:

- `dynamicbind/test/prepare_cmet_figure3_inputs.py`
- `dynamicbind/test/evaluate_cmet_figure3.py`
- `dynamicbind/test/finetune_cmet_state.py`
- `dynamicbind/test/run_same_anchor_inference.py`

## Data and checkpoint audit

The local files and both remote copies used these SHA-256 values:

| Artifact | SHA-256 |
| --- | --- |
| `major_drug_targets.csv` | `9891e1d441dd227b5c6ddbc3bd3de27b75247cc73509593f8397b24209fd86cd` |
| `MDT.zip` | `db6ffd2a3d7d5b7a46b402882a06babe88f25449102dc1c0a1741b8964e0b391` |
| `workdir-v2.zip` | `21ef07dfaa5e9f3f94e0b51e9a8670214937625f6d4221bb763bde039a960e7e` |
| Paper checkpoint, `ema_inference_epoch314_model.pt` | `3a8ed1e65b01aab71016520640a6b80dbd843afa9734ca4ecc9919a32fc30202` |
| c-Met ESM embedding | `7b03977566f2781ff622353dfa46f4e6b4629c7d475c1104c17d87822c0356ed` |

The two AF files overlap on 288 C-alpha atoms and those coordinates are
identical (direct RMSD 0.0 A). The 6UBW-aligned AF coordinates were used as
the common inference anchor; the second AF file was retained for audit.

## Paper-metric validation

The all-atom pocket metric maps residue and atom names, selects native holo
protein atoms within 5 A of the native ligand, and uses the same raw aligned
coordinate frame for the AF baseline.

| Case | Paper AF pocket RMSD | Computed AF pocket RMSD | Absolute error |
| --- | ---: | ---: | ---: |
| 6UBW / 84S / DFG-in | 9.440 | 9.440434 | 0.000434 |
| 7V3S / 5I9 / DFG-out | 6.020 | 6.027192 | 0.007192 |

This check passed the configured 0.02 A tolerance before prediction metrics
were accepted.

## Paper-checkpoint inference

Run:

- server: primary `eh019` route, `admin.cluster.local`
- project: `/data0/luanhaoyang/Omni-Epic-v1`
- Python: `datasets/dynamicbind-v1/env-torch20/bin/python`
- visible GPU: physical GPU 1
- seed: 11
- diffusion steps: 20
- samples per complex: 1
- started: `2026-07-26T01:47:29+08:00`
- finished: `2026-07-26T01:51:28+08:00`
- exit code: 0

Result directory:

`dynamicbind/test/results/day2-same-anchor-inference/20260726-014729+0800`

| Condition | Ligand RMSD | Pocket RMSD | Opposite-state RMSD | Correct-state margin | Selected correctly |
| --- | ---: | ---: | ---: | ---: | --- |
| DFG-in | 2.2549 | 2.4390 | 3.9257 | 1.4867 | yes |
| DFG-out | 2.0075 | 1.8456 | 3.0007 | 1.1552 | yes |

The state-selection accuracy is 2/2. For comparison, Figure 3 reports ligand
RMSDs 0.49/0.51 A and predicted pocket RMSDs 1.97/1.19 A. A single generated
pose is not asserted to reproduce the paper's selected pose.

Evaluation artifacts:

- baseline metrics SHA-256:
  `f078615b50a980d69e07f81e1aa01d7ef47022b694d8a00bd495bd23c7e78ac3`
- baseline summary SHA-256:
  `c67e269cde3d34d3644e856ddc7e2e23a0e4b8462096d869f027f1f658d7c139`

## Paired low-cost fine-tune

Correct pairs were common-anchor + 84S to 6UBW and common-anchor + 5I9 to
7V3S. Each ligand was also scored against the opposite endpoint as an
explicit cross negative. The loss combines correct-endpoint regression,
hinge state ranking, distal-residue preservation, and parameter-delta
regularization.

An initial cross-interaction-plus-head attempt was audited and failed with a
CUDA OOM while another user's PID 4077042 occupied all eight GPUs. No output
checkpoint was accepted. The successful low-cost scope froze 63,676,349 of
63,678,870 parameters and trained only these 2,521 parameters:

- `res_tr_final_layer.0.weight`
- `res_tr_final_layer.0.bias`
- `res_tr_final_layer.3.weight`
- `res_tr_final_layer.3.bias`

The frozen DynamicBind backbone still computes ligand-receptor cross
interactions and supplies ligand-conditioned receptor features to this head.

Successful run:

- visible GPU: physical GPU 2
- started: `2026-07-26T02:18:41+08:00`
- finished: `2026-07-26T02:19:02+08:00`
- steps: 5
- learning rate: `1e-5`
- exit code: 0
- result:
  `dynamicbind/test/results/figure3-cmet-finetune/20260726-021841+0800`

| Quantity | Step 1 | Step 5 |
| --- | ---: | ---: |
| Total loss | 0.346551 | 0.346185 |
| Correct-pair loss | 0.346551 | 0.346185 |
| DFG-in training margin | 0.538297 | 0.540046 |
| DFG-out training margin | 0.706603 | 0.706322 |
| Gradient norm | 0.992043 | 0.965891 |

Checkpoint hashes:

| Artifact | SHA-256 |
| --- | --- |
| Full merged checkpoint | `59918c8a85f0908fc5c2a7b199e4dc8a494989ec7434d4648f0503d3ddd29dc3` |
| Trainable delta | `0c358c740b7d5fde6b59c51b8917d95806e3f627670e286d5c4252c3c623059a` |
| Training metrics | `7c53dc60426f8afd4ab06d38c5965efb2af51b21958d9c0a0d025536394ae43a` |
| Fine-tune metadata | `19373553489c7474e424ddd10501abfb5212db8866c0781c1a270c7fe9e066a6` |

## Before/after coordinate evaluation

The merged checkpoint was sampled once with the same seed and inputs.

| Condition | Base pocket RMSD | Fine-tuned pocket RMSD | Change | Base ligand RMSD | Fine-tuned ligand RMSD |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFG-in | 2.4390 | 2.4697 | +0.0307 | 2.2549 | 2.2570 |
| DFG-out | 1.8456 | 1.8425 | -0.0031 | 2.0075 | 2.0033 |
| Mean | 2.1423 | 2.1561 | +0.0138 | 2.1312 | 2.1301 |

Both conditions remain correctly state-selected, but mean pocket RMSD
worsens. This is a negative result for the cheap-correction gate, so no
longer run or broader unfreezing was launched.

Fine-tuned evaluation artifacts:

- metrics SHA-256:
  `8e00718a264c6c6238e47c021d5da123718d7ce485c08fc92f6041a10c3191d3`
- summary SHA-256:
  `7f7f65f973a3bc36a3884cd647fb9735591686a6dbf89bdf59a459716ccc1496`

## GPU and process audit

All primary-server runs recorded device and process snapshots in JSON.
PID 4077042 belonged to a pre-existing job and was never interrupted. The
successful formal training temporarily added PID 1767233 on physical GPU 2;
the final post-inference audit contains only PID 4077042, confirming that the
reproduction processes exited.

The additional `eh050` route was checked using the required DPAPI/ASKPASS
helper. It had four pre-existing GPU jobs, so after the user directed a return
to `eh019`, no computation was launched there. A checksum-verified project
copy remains under its declared `/data0/luanhaoyang` root.

## Reproduction decision

- `implementation-plan-v0726.md`: exact inputs and AF baseline are validated;
  ligand-specific state selection succeeds for 2/2 cases. The paper's
  selected-pose ligand/pocket numbers are not reproduced by the one-sample
  run and are not claimed.
- `implementation-plan-v0726-2.md`: the paired correct/cross-negative
  fine-tune and before/after test are complete. The low-cost correction gate
  fails because mean pocket RMSD worsens, so scaling this configuration is
  not justified.
