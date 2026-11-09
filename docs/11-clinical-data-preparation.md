# Sampled predictor diagnostics and mortality data preparation

This stage prepares numbers and clinical features for later analysis; it does
not train a mortality classifier or claim clinical performance. Patient-level
files stay on the server and must not be pushed to Git.

## Predictor diagnostics

The test rollout was repeated with `--save_predictions` and compared against
the original rollout errors; all three saved error arrays matched. Baselines
use the same eligible windows as the predictor:

- copy-forward: the last observed state;
- context mean: the average of the first 24 observed states;
- training mean: a single mean fitted on training states only.

`scripts/prepare_predictor_diagnostics.py` prepares all 48 horizons of
aggregate L1 data, per-window baseline errors, and RankMe, top-10 variance
share and top-64 overlap for horizons 1, 8, 24 and 48.
`scripts/prepare_rollout_curve_data.py` adds 2,000 patient-clustered
bootstrap draws and exports plot-ready CSV/JSON; no figure is generated.

| Horizon | Patients | Model L1 | Context-mean L1 | Improvement | 95% CI |
|---|---:|---:|---:|---:|---:|
| 1 h | 162 | 1462.34 | 1489.33 | 1.81% | -0.91% to 4.10% |
| 8 h | 135 | 1544.31 | 1568.75 | 1.56% | -1.85% to 5.03% |
| 24 h | 102 | 1651.53 | 1669.06 | 1.05% | -1.95% to 4.03% |
| 48 h | 64 | 1714.25 | 1741.18 | 1.55% | -1.88% to 4.89% |

Intervals are bootstrap percentiles with patient clusters and windows weighted
as in the rollout metric. These intervals include zero; the small advantage
over the context mean is not conclusive in this sample. At 48 hours RankMe is
19.8 for predictions versus 119.0 for true states; predicted/true top-10
variance shares are 99.91% / 90.78%. This is evidence of strongly
concentrated predictions, not proof of clinical failure. There are only
64 patients and overlapping windows; raw ranks should not be compared directly
with those calculated on the paper's much larger test set. As in the original
rollout, recorded future actions were used in this retrospective diagnostic.

The data live at
`/media/cuichang/liuhao/clin-jepa/sample_3h/diagnostics/prepared/`:
`rollout_curves_bootstrap.csv`, `rollout_curves_bootstrap.json`,
`rank_diagnostics.json` and `baseline_errors.npz`.

## Clinical downstream inputs

The authors' label and history scripts were run on the sampled cohort. History
features summarize state and action embeddings over the first 24 hours plus
demographics; the hospital-mortality label matches the sampled cohort on all
1,798 stays. Only the admission window (start_step 0) with at least 24
hourly steps is retained for mortality probing.

| Split | Patients | ICU stays | Hospital-death stays |
|---|---:|---:|---:|
| Train | 867 | 1,035 | 105 |
| Validation | 167 | 226 | 27 |
| Test | 165 | 210 | 23 |

Splits are patient-disjoint. The fixed predictor was also rolled forward for
24 hours using observed actions through hour 23 and a fixed synthetic
"no active interventions" action for hours 24-46. No recorded future
observation, future treatment, or outcome label was used to construct these
features. This is an artificial, potentially out-of-distribution assumption,
not a treatment recommendation or validated causal counterfactual. The future
features are deterministic functions of the observed history and this fixed
assumption, not new patient evidence. A test verifies changing recorded future
states/actions does not change the prepared features.

The data are at
`/media/cuichang/liuhao/clin-jepa/sample_3h/clinical_data/`:

- `downstream/labels/`: original task label tables;
- `downstream/features/qwen3_base/`: original history feature blocks;
- `no_intervention_rollout/`: hypothetical rollout features and policy metadata;
- `mortality_dataset/{train,val,test}.npz`: aligned history, rollout,
  hospital-mortality labels and patient/stay identifiers;
- `mortality_dataset/metadata.json`: aggregate counts and assumptions.

The mortality probe has since been run; see [the probe protocol and results](12-mortality-probe.md).
It was tuned on train/validation and evaluated on the patient-disjoint test
split with patient-clustered resampling.

## Reproduction

Set `EXP` to the sampled experiment directory; source the repository `.env`
and point `CLIN_JEPA_DATA` at `$EXP/processed` and `CLIN_JEPA_OUTPUT` at
`$EXP/clinical_data`. The latter uses a symlink used only for reads from
`$EXP/processed/discretized` to the already prepared full discretized data;
the label script restricts its reads to the sampled cohort.

```bash
"$CLIN_JEPA_PYTHON" -m clin_jepa.downstream.labels
for split in train val test; do
  "$CLIN_JEPA_PYTHON" -m clin_jepa.downstream.history --encoder qwen3_base --split "$split"
done
"$CLIN_JEPA_PYTHON" -m scripts.prepare_clinical_rollout \
  --data "$EXP/processed" --history "$EXP/clinical_data/downstream/features/qwen3_base" \
  --checkpoint "$EXP/output/predictor_qwen3_base/best.pt" \
  --output "$EXP/clinical_data/no_intervention_rollout"
"$CLIN_JEPA_PYTHON" scripts/prepare_mortality_dataset.py \
  --history "$EXP/clinical_data/downstream/features/qwen3_base" \
  --future "$EXP/clinical_data/no_intervention_rollout" \
  --labels "$EXP/clinical_data/downstream/labels/stay.parquet" \
  --splits "$EXP/processed/splits/splits.csv" --output "$EXP/clinical_data/mortality_dataset"
```

For predictor diagnostic scripts, pass the saved test rollout directory
containing `index.parquet`, `predicted.npy` and `errors.npz`, the sampled
processed directory, and a fresh output directory. Commands should be run with
the configured Clin-JEPA Python environment.
