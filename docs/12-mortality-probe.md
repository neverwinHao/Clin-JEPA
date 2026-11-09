# Sampled hospital-mortality probe

This exploratory probe uses the patient-disjoint data prepared in
[the clinical-data guide](11-clinical-data-preparation.md). It is not the
paper's full-cohort clinical evaluation or a clinical risk score.

## Fixed evaluation protocol

One admission window with at least 24 hourly steps is used per ICU stay. The
label is hospital mortality. The first 24 hours of state/action embeddings and
static features form the history input. The predictor's 24-hour continuation
is generated using observed actions through hour 23 followed by synthetic
"no active interventions" actions. No real future action or observation is
used. This synthetic policy can be unrealistic; it is not medical advice or a
validated causal counterfactual. The generated continuation is a deterministic
transformation of the history and fixed policy, not additional patient data.

Preprocessing and PCA are fitted on training data only. Three L2-regularized,
class-balanced logistic classifiers compare:

1. History: 32 PCA dimensions.
2. History: 64 PCA dimensions, a same-capacity control.
3. History: 32 PCA dimensions plus 32 PCA dimensions from the predictor's
   hypothetical future features.

The PCA is randomized, seed 42, after training-only standardization. The
regularization is selected separately for each classifier by validation AUROC
from C = 0.01, 0.1, 1, 10. The untouched patient-disjoint test split reports
AUROC and average precision (AUPRC). Intervals and paired differences use
2,000 patient-clustered bootstrap draws. The probe tests the data pipeline and
an inductive-bias comparison; results from 23 positive test stays are uncertain.

## Run

With the repository environment active and `EXP` set to the sampled
experiment directory, run:

```bash
"$CLIN_JEPA_PYTHON" -m scripts.run_mortality_probe \
  --data "$EXP/clinical_data/mortality_dataset" \
  --output "$EXP/clinical_data/mortality_probe"
```

The output holds aggregate metrics, fitted preprocessing/classifiers and
patient-level test predictions. Keep the latter on the authorized server.

## Result on the sampled cohort

The fixed protocol above was executed with 1,035 train stays, 226 validation
stays, and 210 test stays. The test stays belong to 165 patients and include
23 hospital deaths. No observed future treatments or observations were used.

| Method | Validation AUROC | Test AUROC (95% patient-bootstrap CI) | Test AUPRC |
|---|---:|---:|---:|
| History, 32 dimensions | 0.656 | 0.756 [0.649, 0.860] | 0.305 |
| History, 64 dimensions | 0.673 | 0.799 [0.706, 0.884] | 0.313 |
| History 32 + hypothetical future 32 | 0.627 | 0.741 [0.623, 0.853] | 0.303 |

Each method selected C=0.01 using validation AUROC. The positive proportion
in the test stays was 23/210 = 0.110. Average precision is reported as AUPRC.
The added-future minus history-32 AUROC difference was -0.015, with paired
patient-bootstrap 95% CI [-0.062, 0.037]; the AUPRC difference was -0.002,
CI [-0.074, 0.072]. Relative to history-64, the AUROC difference was -0.058,
CI [-0.153, 0.030]; that comparison also changes the allocation of dimensions
between history and future, so it is a capacity control rather than an isolated
estimate of adding future features.

The model, PCA projections and all test scores were reloaded and independently
recomputed, matching to within 1e-6. These findings do not establish clinical
efficacy, nor rule out an effect in a larger study: there are few positive
patients and the no-intervention action policy is synthetic. The lower sampled
point estimate should not be presented as a statistically established decrease.

Detailed results and model artifacts are on the server at
`/media/cuichang/liuhao/clin-jepa/sample_3h/clinical_data/mortality_probe/`:
`summary.json`, `paired_comparisons.json`, `probe.joblib` and
`test_predictions.parquet`. Only aggregate results belong in Git.
