# 05 · Rollout

Input: embedding caches (03) and predictors (02, 04). Output: `$CLIN_JEPA_OUTPUT/rollout/<model>/<split>/`. One GPU per job; the test split takes about five minutes.

```bash
python -m clin_jepa.rollout --model clin_jepa --split test --save_predictions
```

Every window of at least 24 hours is rolled forward from its first 24 hourly states under the recorded actions, for at most 48 steps and never past the end of the stay. On the test split that is 39,413 windows, 39,127 with at least one step inside the stay and 33,144 that reach step 48.

| file | content |
|---|---|
| `index.parquet` | one row per window: `stay_id`, `start_step`, `length`, cache shard and position in it |
| `pooled.npy` | `(windows, horizons, statistics, 4096)` fp16: for horizons 8, 24 and 48, the mean, standard deviation and last of the predicted states; steps past the stay end enter as zero vectors |
| `errors.npz` | `model_l1`, `copy_forward_l1`, `linear_extrap_l1`, each `(windows, 48)`, NaN past the stay end |
| `predicted.npy` | with `--save_predictions`: the predicted states, `(windows, 48, 4096)` fp16, about 15 GB for the test split |

| model (`configs/rollout.yaml`) | cache | splits | `--save_predictions` | read by |
|---|---|---|---|---|
| `clin_jepa`, `vjepa2ac`, `sft_baseline` | own | train, val, test | test | 06, 08 |
| `separate_seed_{42,1337,2026}` | `clin_jepa` | test | yes | 06 |
| `no_warmup`, `no_alignment` | own | test | yes | 06 |
| `clin_jepa_after_{warmup,co_training,alignment}` | `sft_baseline`, `clin_jepa`, `clin_jepa` | test | no | 06 |
