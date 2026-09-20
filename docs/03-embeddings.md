# 03 · Embedding caches

Input: `$CLIN_JEPA_DATA/trajectories` and the encoder checkpoints of stage 02. Output: `$CLIN_JEPA_DATA/embeddings/<model>/<split>/embeddings_cII_SSS.pt`, one file per trajectory shard. One GPU per job.

```bash
python -m clin_jepa.encode --model clin_jepa --split train [--task_id I --num_tasks N]
```

`--task_id I --num_tasks N` gives job `I` every `N`-th shard; shards already written are skipped.

| model | encoder (`configs/encode.yaml`) | splits |
|---|---|---|
| `clin_jepa` | `clin_jepa/checkpoints/final/online` | train, val, test |
| `vjepa2ac` | `vjepa2ac/checkpoints/best/online` | train, val, test |
| `sft_baseline` | `encoder_sft/final` | train, val, test |
| `no_warmup` | `no_warmup/checkpoints/after_sync/online` | test |
| `no_alignment` | `no_alignment/checkpoints/after_co_training/online` | test |

A file holds the last-token hidden state (4,096-d, fp16) of every hourly state text, every distinct action text and one demographics text per stay, plus the windows as indices into these tables. On one H200 a validation or test shard takes about 20 minutes and a full training shard about 45 (23 / 10 / 10 shards for train / val / test), about 22 GPU-hours per encoder; a full cache is about 96 GB.
