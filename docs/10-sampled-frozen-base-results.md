# Sampled MIMIC-IV experiment: frozen Qwen3-8B

This is a small, exploratory pipeline check, not a reproduction of the paper's
SFT or jointly trained model. Qwen3-8B was loaded as a frozen base, and only the
92.5M-parameter Clin-JEPA predictor was trained.

## Setup

Patients were sampled from the existing patient-level train / validation / test
splits with seed 42. Every selected patient's stays and available windows were
retained. The first 24 hours of each test window form the rollout context; the
predictor is evaluated against Qwen's frozen encoding of subsequent hours.

| Split | Patients selected | ICU stays | Windows |
|---|---:|---:|---:|
| Train | 1,000 | 1,262 | 4,190 |
| Validation | 200 | 276 | 830 |
| Test | 200 | 260 | 759 |

Using three idle A30s for encoding and one A30 for training, encoding through
test rollout took about 51 minutes on this server. Predictor training used
20 epochs, batch 128, two gradient accumulation steps (effective batch 256),
and took 287 seconds. Best validation next-state L1 was 1653.03. The saved
checkpoint was reloaded and reproduced this validation loss.

## Test rollout versus copy-forward

L1 sums absolute errors across the 4,096 dimensions. Copy-forward uses the
last observed state for every future step. MASE here is the mean model L1
divided by the mean copy-forward L1 on the same eligible windows; below 1 is
better than copy-forward. The confidence interval uses 2,000 bootstrap draws
resampled by patient, with seed 42; windows remain weighted as in the original
rollout evaluation.

| Hours ahead | Eligible windows | Patients | Model L1 | Copy-forward L1 | MASE | Patient-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 702 | 162 | 1462.34 | 2006.34 | 0.729 | 0.681–0.784 |
| 8 | 662 | 135 | 1544.31 | 1847.19 | 0.836 | 0.769–0.901 |
| 24 | 620 | 102 | 1651.53 | 1897.84 | 0.870 | 0.821–0.927 |
| 48 | 573 | 64 | 1714.25 | 2048.10 | 0.837 | 0.796–0.877 |

The 48-hour windows come from just 64 patients; overlapping windows from long
stays contribute repeatedly. Bootstrap uncertainty is conditional on this
sample, encoder, and training seed and is not a clinical performance estimate.
Recorded actions, including future actions, condition the rollout, so these
results are retrospective and do not establish prospective predictive ability.
No clinical AUROC/AUPRC was measured in this run.

## Reproduction and artifacts

The code that generated the samples, caches, predictor and rollout was commit
`9f643c3`. The server's experiment directory is
`/media/cuichang/liuhao/clin-jepa/sample_3h/`; `provenance.json` records the
sample and software versions, `output/predictor_qwen3_base/best.pt` holds the
predictor, and `output/rollout_summary.json` and `output/rollout_bootstrap.json`
hold aggregate results. MIMIC-derived caches and model weights are not in Git.

See [the frozen-base guide](09-frozen-base.md) for the sampling, training and
rollout commands. To calculate the patient-clustered interval:

```bash
"$CLIN_JEPA_PYTHON" scripts/bootstrap_rollout_patients.py \
  --rollout "$EXP/output/rollout/qwen3_base/test" \
  --splits "$EXP/processed/splits/splits.csv" \
  --output "$EXP/output/rollout_bootstrap.json"
```
