# PepFlow reproduction 001

## Scope

- Date: 2026-07-23 (Asia/Shanghai)
- Original baseline commit: `16e0d267c2dbd96cdacbe5ac07c4dada0d61169b`
- Remote project: `/data0/luanhaoyang/Omni-Epic-v1`
- Python: `/home/huangyueshan/.conda/envs/flex/bin/python`
- Authorized visible GPUs: `0,1`
- Minimal train device: `cuda:0`

## Step zero: reproduce the original baseline

The unmodified baseline was synchronized to the remote server and launched with
the original training entry point. It failed during import, before dataset or
model construction:

```text
FileNotFoundError:
/datapool/data2/home/ruihan/data/jiahan/ResProj/PepDiff/pepflowww/Data/names.txt
```

The cause was a module-level read of an absolute path from the original
author's machine in `models_con/pep_dataloader.py`.

## Minimal fixes

The reproduction was advanced through independently committed changes:

1. Make the preprocessing exclusion list optional when the original absolute
   path does not exist.
2. Add a one-iteration, batch-size-one smoke configuration using relative data
   paths.
3. Skip `wandb.log()` in debug mode when no W&B run has been initialized.
4. Ignore remote datasets and generated test artifacts.
5. Add a smoke harness that enforces a committed worktree and records evidence.

## Data integrity

The source archives were stored under the remote project's `datasets/`
directory. The LMDB archive was extracted and its stable cache files were
placed under `datasets/lmdb/`.

```text
452240f8d60227c0959f7f3a8cf43a2f8a63e53806f4f47bdf8ccb1cb1f5ef08  PepMerge_lmdb.zip
eb0c9f6f81b85c399a32fe38e7f79274584805d7cafaac774a8d091792d0410c  PepMerge_release.zip
693037f65f02b8b48a513c337c12aa46905d85840b281a35058d54fd6c25a2f7  pep_pocket_train_structure_cache.lmdb
```

## Verified result before this report

Commit `b100d12e17d0ed2cbf733d40d94d12c6579ce52f` passed the remote smoke
test from a clean worktree:

- dataset entries: `9849`
- model parameters: `6880353`
- completed iterations: `1`
- loss: `170.2016`
- gradient norm: `225.4276`
- exit code: `0`
- result directory:
  `/data0/luanhaoyang/Omni-Epic-v1/test/results/train-smoke/20260723-003009+0800`

The canonical evidence for any later commit is the newest timestamped result
whose `commit.txt` matches that exact commit. After committing this report, the
smoke test must be run again so the report commit also has same-commit evidence.
