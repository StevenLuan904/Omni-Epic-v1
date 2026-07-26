# Joint PepFlow–DynamicBind minimum overfit

## Outcome

The retained architecture now runs a real joint optimization step and passes the
two-state minimum-overfit gate:

```text
PepFlow peptide flow
  + peptide-condition adapter
  + DynamicBind cross-interaction trunk and receptor flow heads
```

The formal run trained peptide frame/torsion prediction from corrupted peptide
states together with peptide-conditioned receptor translation, rotation, and
side-chain torsion endpoints. It used swapped endpoint negatives plus normal,
shuffled-peptide, and zero-condition controls.

## Audited input pair

- State A: receptor `1eb1_A`, peptide `DYEPIPEEAF`.
- State B: receptor `3vxf_J`, peptide `DFEEIPEEYL`.
- Matched receptor C-alpha atoms: 257.
- Aligned global receptor RMSD: 1.3494 A.
- 10 A pocket RMSD: 2.6714 A.
- Source archive SHA-256:
  `eb0c9f6f81b85c399a32fe38e7f79274584805d7cafaac774a8d091792d0410c`.
- Prepared case manifest SHA-256:
  `6b42a79131a7514764dbd9ac8c609cb3a0fa6aebf6f234d09155f502ec6568e1`.
- Prepared case tar SHA-256:
  `460c557d8c97b2639c2c71f05f175f325b8dd9afde47588fe8f7148fcf791109`.

The source audit ran on the primary server at
`/data1/huangyueshan/Omni-Epic-v1/repo/pepflow/test/results/multipeptide-state-audit/20260726-224731+0800`.
The final aligned case was prepared at
`/data1/huangyueshan/Omni-Epic-v1/repo/pepflow/test/results/joint-peptide-states/20260726-230536+0800`
and checksum-verified after transfer to synth.

## Formal pretrained runs

The final audit used the official PepFlow `model1.pt`, loaded strictly into the
PepFlow branch before optimization. The checkpoint was independently hashed on
the local and remote machines:

- PepFlow `model1.pt` SHA-256:
  `ee3f0458cc47b63f2c5c8bc27f0e9897fab395592a2beb5e627a42f74754fb0a`.
- PepFlow `learn_angle.yaml` SHA-256:
  `35fb093d8e4537de0029244cbd92a079b9b0eb61347714fa38202c2579764dd7`.
- DynamicBind checkpoint SHA-256:
  `3a8ed1e65b01aab71016520640a6b80dbd843afa9734ca4ecc9919a32fc30202`.
- DynamicBind parameter YAML SHA-256:
  `195a050fff2de7dcc1deac9fc8a9cb63cb2b0c04b236a0c77146506734cd56f7`.
- Result commit: `f30764b6947a827688f2ee40586c225f8b292a1b`
  (`Record PepFlow checkpoint provenance`).
- Server: synth (`admin.cluster.local`), repository
  `/sdd_data/synth/Omni-Epic-v1/repo`.
- Python: isolated venv
  `/sdd_data/synth/Omni-Epic-v1/venvs/dynamicbind-frozen-probe` (Python 3.8).
- GPUs: physical GPUs 5, 6, and 7, NVIDIA GeForce RTX 3090. All three were
  empty before launch and returned to 2 MiB used after completion.

All three jobs ran concurrently with different seeds. Each produced 200 metric
rows from real backward/optimizer steps, a nonzero final gradient norm, and a
roughly 74 MB trainable delta containing 19,301,120 optimized parameters.

| Seed | GPU | Result directory suffix | Elapsed (s) | Loss start -> end | Reduction | Gate |
| --- | ---: | --- | ---: | --- | ---: | --- |
| 20260726 | 5 | `20260727-013433+0800` | 293.12 | 1.18177 -> 0.11475 | 90.29% | pass |
| 20260727 | 6 | `20260727-013435+0800` | 291.29 | 1.12332 -> 0.07132 | 93.65% | pass |
| 20260728 | 7 | `20260727-013437+0800` | 289.03 | 0.94309 -> 0.10108 | 89.28% | pass |

Each suffix is under
`/sdd_data/synth/Omni-Epic-v1/repo/pepflow/test/results/joint-overfit/`.

Command:

```bash
/sdd_data/synth/Omni-Epic-v1/venvs/dynamicbind-frozen-probe/bin/python \
  pepflow/test/train_joint_peptide_receptor_overfit.py \
  --case-dir /sdd_data/synth/Omni-Epic-v1/inputs/v0726-4/joint-case \
  --model-dir /sdd_data/synth/Omni-Epic-v1/model/workdir/big_score_model_sanyueqi_with_time \
  --graph-cache /sdd_data/synth/Omni-Epic-v1/cache/v0726-4-joint \
  --pepflow-config pepflow/configs/learn_angle.yaml \
  --pepflow-checkpoint /sdd_data/synth/Omni-Epic-v1/inputs/pepflow-official/model1.pt \
  --visible-gpus 5 --steps 200 --sample-steps 10 \
  --learning-rate 0.0003 --peptide-weight 0.1 --ranking-weight 5.0 \
  --seed 20260726
```

## Metrics

- All three seeds selected the correct receptor endpoint for both states.
  Correct endpoint errors across the three seeds were 0.05710--0.06350.
- Eight-draw peptide denoising improved for both states in every seed:
  state 0 changed from 1.14765--1.25735 A to 0.32538--0.46826 A, and state 1
  changed from 0.74873--0.87021 A to 0.27058--0.63556 A.
- Prior-to-native peptide samples across all seeds were 0.51853--0.81230 A
  for state 0 and 0.55929--0.69503 A for state 1.
- Shuffle controls degraded correct receptor endpoint errors to
  0.07841--0.08235. Zero-condition correct errors were 41,262--95,421.
- `joint_overfit_gate` was `true` for 3/3 seeds; process status was `passed`
  with exit code 0 for 3/3 seeds.
- The final CSV row for each run had nonzero PepFlow, adapter, and receptor
  gradient norms. The overall final gradient norms were 1.63305, 0.45640, and
  14.52225 respectively.

Artifact SHA-256 values:

- Seed 20260726 delta: `d9da181580ef6cb75b901621bfe1ceae9966ca14df54f7cf2e1d62baa0bbec4f`;
  metrics: `bf7a9f92343ca74ca3ef1fbe34f93756debfe13bc2cc3da568ad9cd11c4e9bda`;
  summary: `ae2fbc464c0951681a57d781f8d496eaa9cf1496e63cd17d546edfb364ce6680`.
- Seed 20260727 delta: `a36fc811ebcd579aa43bc648e832673cacff2183b038c4c11a49c54cc0057177`;
  metrics: `bbf20d552e8180f238f8648ee53aab6c0658d14f8514bc914e2f9a58366a5338`;
  summary: `b22b8428fad279539dba8e57d41eb04423adf2bc531f34ecb4f7376b783c5005`.
- Seed 20260728 delta: `a6a6f3ef819355895e8188e91bda2945b50d2b8f13d7b575edbe12bb3981fa64`;
  metrics: `0812162aa007472cb0dd82663b3fdad70edea314769df9e5da3993dc3474dfec`;
  summary: `a6db4b243a9b4b59b9ab4ee4a81c6ccfa22e84cd4dd341154caba259d893dcca`.

Every `worker_summary.json` records `pepflow_initialization: checkpoint` and
the official PepFlow SHA. Every `trainable_delta.pt` separately stores the same
SHA as `base_pepflow_sha256`, so the optimized delta cannot be mistaken for a
randomly initialized PepFlow base.

## Reproduction commits

- `9eac802`: audit same-receptor peptide state pairs.
- `b446e6b`: prepare aligned joint peptide states.
- `42fe539`: add target receptor embedding controls.
- `b73ec9f`: connect PepFlow to DynamicBind receptor flow.
- `3eb5ec1`: match DynamicBind ESM embedding format.
- `6fa8252`: train joint peptide and receptor flows.
- `e8f2ade`, `7316b36`: cache and memory-bound DynamicBind torus startup.
- `70d507e`, `503fb04`, `5f94b5a`, `aea2bc2`: Python 3.8 and missing
  tracked package/source fixes.
- `d2612ed`, `eb94998`: add fixed-seed, eight-draw peptide denoising gate.
- `f30764b`: bind summaries and trainable deltas to the PepFlow checkpoint SHA.

The earlier random-initialization run at commit `eb94998` and result directory
`20260727-005228+0800` is retained only as a historical control. It is not the
formal pretrained result.

## Scope and limitation

Both the PepFlow and DynamicBind branches are now strictly initialized from
pretrained checkpoints. These runs prove that the retained joint modules can
optimize corrupted peptide poses and select peptide-specific receptor endpoints
without discarding pretrained PepFlow initialization. This remains a two-state
minimum-overfit test, not evidence of large-dataset generalization or
production-quality unconditional peptide generation. That claim still requires
full-dataset training and held-out evaluation.
