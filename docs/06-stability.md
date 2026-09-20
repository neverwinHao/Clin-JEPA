# 06 · Stability of the encoder and predictor (§5.2)

Input: the test-split embedding caches (03) and the test-split rollouts (05, with `--save_predictions` where the guide of stage 05 says so). Output: `$CLIN_JEPA_OUTPUT/stability/`. Models, labels and checkpoints are listed in `configs/stability.yaml`.

```bash
# encoder side, once per encoder: clin_jepa vjepa2ac sft_baseline no_warmup no_alignment
python -m clin_jepa.stability.encoder_side --encoder clin_jepa

# predictor side, once per rollout with saved predictions:
# clin_jepa vjepa2ac sft_baseline separate_seed_{42,1337,2026} no_warmup no_alignment
python -m clin_jepa.stability.predictor_side --rollout clin_jepa

python -m clin_jepa.stability.drift
python -m clin_jepa.stability.report
```

| step | computes | writes |
|---|---|---|
| `encoder_side` | mean ‖z‖₁, z_std, static and dynamic RankMe, top-10 variance share, mean-state errors at h = 48 | `encoder_side/<encoder>.json` |
| `predictor_side` | RankMe of predicted and true states and their top-64 subspace overlap, h = 1 … 48 | `predictor_side/<rollout>.json` |
| `drift` | ℓ1(h), MASE(h), rollout drift = predictor degradation × state displacement, stay-clustered bootstrap intervals, paired Wilcoxon test | `drift.json` |
| `report` | Fig. 2, Table 2, App. Tables 1–3, App. Fig. 1, the numbers quoted in the text | `figures/`, `tables/` |

The first two steps use a GPU when one is present (each takes 5–15 minutes on an H200 and needs about 60 GB of host memory); `drift` and `report` run on CPU in a few minutes.
