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

## Formal run

- Server: synth (`admin.cluster.local`).
- Repository: `/sdd_data/synth/Omni-Epic-v1/repo`.
- Result commit: `eb9499808f955796654031853d1d781594dcdc19`.
- Commit message: `Average peptide denoising controls`.
- Result directory:
  `/sdd_data/synth/Omni-Epic-v1/repo/pepflow/test/results/joint-overfit/20260727-005228+0800`.
- Start: `2026-07-27T00:52:28.280707+08:00`.
- Finish: `2026-07-27T00:55:00.821253+08:00`.
- Elapsed: 152.54 seconds.
- GPU: physical GPU 5, NVIDIA GeForce RTX 3090; GPU 5 was empty before
  launch and returned to 2 MiB used / 24257 MiB free after exit.
- Python: isolated venv
  `/sdd_data/synth/Omni-Epic-v1/venvs/dynamicbind-frozen-probe` (Python 3.8).
- Added environment dependencies: `dm-tree==0.1.6`, `einops==0.8.1`.
- DynamicBind checkpoint SHA-256:
  `3a8ed1e65b01aab71016520640a6b80dbd843afa9734ca4ecc9919a32fc30202`.
- DynamicBind parameter YAML SHA-256:
  `195a050fff2de7dcc1deac9fc8a9cb63cb2b0c04b236a0c77146506734cd56f7`.
- PepFlow config SHA-256:
  `c3c688486c24355bafbe21a65a7c9b75b76323ab6d221265468745497df0925c`.

Command:

```bash
/sdd_data/synth/Omni-Epic-v1/venvs/dynamicbind-frozen-probe/bin/python \
  pepflow/test/train_joint_peptide_receptor_overfit.py \
  --case-dir /sdd_data/synth/Omni-Epic-v1/inputs/v0726-4/joint-case \
  --model-dir /sdd_data/synth/Omni-Epic-v1/model/workdir/big_score_model_sanyueqi_with_time \
  --graph-cache /sdd_data/synth/Omni-Epic-v1/cache/v0726-4-joint \
  --visible-gpus 5 --steps 100 --sample-steps 10 \
  --learning-rate 0.001 --peptide-weight 0.1 --ranking-weight 5.0
```

## Metrics

- Total loss: 5.6822 -> 4.2596 (25.04% reduction).
- Receptor state 0: correct error 0.05739, wrong error 0.15669,
  correct margin 0.09930.
- Receptor state 1: correct error 0.05768, wrong error 0.18574,
  correct margin 0.12806.
- Eight-draw peptide denoising, state 0: C-alpha RMSD
  3.3206 -> 2.8426 A; weighted flow loss 35.23 -> 29.30.
- Eight-draw peptide denoising, state 1: C-alpha RMSD
  4.9837 -> 4.5754 A; weighted flow loss 45.15 -> 39.54.
- Shuffle control increased correct endpoint errors to 0.07620 and 0.08642
  and reduced margins to 0.05674 and 0.06448.
- Zero-condition errors rose to 96277.97 and 44964.77; state 0 selected the
  wrong endpoint.
- `joint_overfit_gate`: `true`.
- Process status: `passed`; stderr was empty.

Artifact SHA-256 values:

- `trainable_delta.pt`: `f1fe288220257d30815882acf5cf5b648ef10663ce21a8e290655768d4ef1252`.
- `training_metrics.csv`: `d8619dedb40c7cb7508dfbb74d89c09c606234ec21894824d7f2eefb07d8621f`.
- `worker_summary.json`: `b149c612b5019ef434f5d2fd9cc12a438e2c79d0a4db921cbe69807410910cff`.
- `report.html`: `fd5dc0b77eacff58d98038b8963a4a31280baf734b48040f4a6e12006a9d3ef0`.

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

## Scope and limitation

No PepFlow pretrained checkpoint was present in the supplied assets, so the
PepFlow branch was initialized randomly; the DynamicBind receptor branch was
strictly initialized from the supplied pretrained checkpoint. This run proves
that the retained joint modules can optimize corrupted peptide poses and select
peptide-specific receptor endpoints. It does not establish production-quality
unconditional peptide generation: prior-to-native sampling remained at 6.34 A
and 8.87 A C-alpha RMSD. A pretrained PepFlow checkpoint and large-dataset
training are still required for that claim.
