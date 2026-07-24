# PepFlow smoke test

The smoke test performs one real forward/backward/optimizer step with
`configs/learn_angle.smoke.yaml`. It refuses to run unless every non-ignored
file is committed, and prints both staged and unstaged diffs when the check
fails.

Prepare the cached dataset with this layout:

```text
datasets/
├── structures/
└── lmdb/
    ├── pep_pocket_train_structure_cache.lmdb
    └── pep_pocket_test_structure_cache.lmdb
```

Then run:

```bash
python test/run_train_smoke.py --visible-gpus 0,1 --device cuda:0
```

Each run writes `commit.txt`, `metadata.json`, `train.log`, and an HTML metric
visualization under `test/results/train-smoke/<timestamp>/`. The result folders
are ignored by Git so they do not make later committed-worktree checks fail.
