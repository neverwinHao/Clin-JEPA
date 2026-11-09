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
