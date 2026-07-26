# Joint peptide-receptor small-batch training

## Outcome

The retained 19.3M-parameter PepFlow-DynamicBind architecture now completes a
real 80-state small-batch training run and passes split-matched random-complex,
peptide-cluster, and receptor-family-proxy holdouts. Each formal model was
trained only on the training cases for its own split; the audited train/test
intersection is empty in all three runs.

The common retained configuration is 100 optimizer steps, two cases per step,
learning rate `3e-4`, ranking weight `2`, and minimal-relaxation weight `0.01`.
Both PepFlow and DynamicBind are checkpoint initialized. The optimized modules
remain the same 19,301,120 parameters used by the minimum-overfit experiment.

## Real small batch

The preparation script selected 40 distinct same-receptor, different-peptide
pairs from PepMerge, giving 80 real peptide-holo states. Peptides are 5--30
residues and receptors are 40--500 residues. Each state uses the other state's
holo receptor as a real cross-holo anchor, so the task explicitly contains:

```text
positive: peptide A + state A endpoint
negative: peptide A + state B endpoint
```

The loss combines peptide frame/torsion flow, receptor translation/rotation/
side-chain flow, correct-versus-wrong endpoint ranking, minimal receptor
relaxation, and shuffled/zero peptide controls.

- Prepared data:
  `/sdd_data/synth/Omni-Epic-v1/inputs/joint-small-batch/20260727-022909+0800`.
- Dataset manifest SHA-256:
  `7c7ef4e0c9dc956bf4b33b49e9205900155e72615df59b5c63df6c7ad542187e`.
- Transfer tar SHA-256:
  `9e356dc9ccab650b73b2c81d378b836cdee6445abfd5fdf8374de801b2acb5a5`.
- Source PepMerge archive SHA-256:
  `eb0c9f6f81b85c399a32fe38e7f79274584805d7cafaac774a8d091792d0410c`.
- Candidate-pair audit SHA-256:
  `5c27e88954cefa41f3608488bbb5537ccf59d06ccc820b8b974c08775f1f81f4`.

The source and destination hashes matched after transfer. AlloGen's public
sample contains only eight unique complexes and no trainable atomic-coordinate
corpus, so it was not used to fabricate apo or AlphaFold anchors.

## Strict held-out results

All runs used commit `aaabfd2e09fba4f050c1507f54d5a51b9535d096`
(`Bound small-batch state memory`) on synth GPUs 5--7. Every run exited 0,
recorded 100 real optimizer steps, 96 before/after evaluation states, an HTML
report, a trainable delta, full stdout/stderr, and GPU audits. There were no
preprocessing failures.

| Matching split | Train/test cases | Result suffix | Loss start -> end | Peptide pose RMSD before -> after (A) | Pocket improvement after (A) | Correct-state margin before -> after | Accuracy after | Shuffle degradation after | Movement after (A) |
| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| random complex | 28 / 8 | `20260727-024235+0800` | 2.50858 -> 0.79611 | 1.54174 -> 1.07002 | 0.00674 | 0.01669 -> 0.04308 | 0.9375 | -0.00000497 | 0.17631 |
| peptide cluster | 28 / 8 | `20260727-031436+0800` | 2.15894 -> 0.62336 | 2.26118 -> 1.81976 | 0.00055 | 0.02439 -> 0.04328 | 0.9375 | 0.00008070 | 0.20414 |
| receptor family proxy | 27 / 8 | `20260727-031438+0800` | 1.71828 -> 0.77175 | 2.81872 -> 1.81429 | 0.00474 | 0.02087 -> 0.04665 | 1.0000 | 0.00003916 | 0.15951 |

The result suffixes are under
`/sdd_data/synth/Omni-Epic-v1/repo/pepflow/test/results/joint-small-batch-train/`.
Each directory contains 101 CSV lines including the header for training and 97
CSV lines including the header for evaluation. The formal composite gate is
passed: unseen-peptide pose improves, unseen-family correct-state accuracy is
greater than 0.5, and unseen-family shuffle degradation is positive.

A selectivity-heavy family control (`lr=7e-4`, ranking weight 5) also passed at
`20260727-031440+0800`: accuracy was 1.0 and shuffle degradation was
`0.00005258`. It was not selected because it had higher final loss (1.12446),
larger receptor movement (0.28919 A), and less pocket improvement (0.00294 A)
than the common configuration.

The earlier `20260727-024233+0800` through `20260727-024237+0800` parallel
search trained on the random split and was useful for hyperparameter selection.
Its non-random metrics are not reported as strict holdouts because those split
test lists can overlap the random training list.

## Checkpoint provenance

- PepFlow `model1.pt` SHA-256:
  `ee3f0458cc47b63f2c5c8bc27f0e9897fab395592a2beb5e627a42f74754fb0a`.
- PepFlow configuration SHA-256:
  `35fb093d8e4537de0029244cbd92a079b9b0eb61347714fa38202c2579764dd7`.
- DynamicBind checkpoint SHA-256:
  `3a8ed1e65b01aab71016520640a6b80dbd843afa9734ca4ecc9919a32fc30202`.
- DynamicBind parameter YAML SHA-256:
  `195a050fff2de7dcc1deac9fc8a9cb63cb2b0c04b236a0c77146506734cd56f7`.

Every summary records `pepflow_initialization: checkpoint`; this experiment is
not inference-only and does not use a randomly initialized PepFlow branch.

## Memory iteration

The first three full 100-step attempts at commit `16024db` failed with CUDA OOM
because both state autograd graphs were retained before backward. Commit
`aaabfd2` performs an immediately scaled backward for each state. This is
mathematically the same two-case gradient accumulation while bounding live
memory to one state graph. A full 20-step validation covered every random-split
training case before the formal runs were launched.

## Portable environment

Docker CLI is installed on synth but its daemon is unavailable. The validated
environment was therefore exported as a self-contained Linux x86-64 archive
with an embedded Python 3.8.10 runtime:

- Archive:
  `/sdd_data/synth/Omni-Epic-v1/venvs/portable/dynamicbind-frozen-probe-aaabfd2-py38.tar.gz`.
- Archive SHA-256:
  `eb6ef5a5165ce2c9b16d8347dfaf14252b29b4474fccb9818f997c34254184a2`.
- Size: 2.0 GiB.
- Requirements lock:
  `/sdd_data/synth/Omni-Epic-v1/venvs/portable/dynamicbind-frozen-probe-aaabfd2.requirements-lock.txt`.
- Lock SHA-256:
  `22207cd218330593b806f6dc3918f1ab8970406dd60587f7329d682be2588d4f`.

The archive was extracted into a different path and invoked from `/tmp`.
`sys.prefix` followed the new extraction path, `sys.base_prefix` resolved to
the embedded runtime, PyTorch reported `2.0.1+cu117`, a CUDA tensor succeeded
on isolated GPU 5, and `pip check` reported no broken requirements. The bundled
runtime targets glibc 2.31; the primary server is glibc 2.31 and the additional
server is glibc 2.35, so both satisfy that runtime floor. Use the extracted
`bin/python` entry point rather than a target server's system Python.

## Reproduction commits

- `f73be16`: prepare the real 80-state cross-holo small batch and splits.
- `0ebe833`: train joint peptide/receptor flow on small batches.
- `bcd5b8d`: skip and backfill invalid graph candidates.
- `6d255ca`, `16024db`: match DynamicBind receptor embedding lengths with the
  same BioPython parser used during graph construction.
- `aaabfd2`: bound state-graph memory and enable complete 100-step runs.

## Remaining scope

The public inputs support only cross-holo anchors, so anchor-type holdout across
apo, AlphaFold, and cross-holo remains unavailable. `receptor_family_proxy` is
a sequence-derived proxy rather than a curated family annotation. Sequence
recovery is deliberately `null`: peptide sequence masking and de novo sequence
generation remain disabled until the pose/selectivity result is accepted and
repeated on a larger independently annotated dataset.
