# Sampled 34-task evaluation protocol

See [the aggregate results](13-all-downstream-results.md). This is an
exploratory sampled-cohort evaluation using the original repository's 34 task
labels, not a reproduction of the paper's full training and readout protocol.
Patient-level labels, predictions, features and fitted models stay on the server.

## Design

The sample and labels are those prepared in [the data guide](11-clinical-data-preparation.md).
Historical features use the first 24 hours of each available window; for
stay-level tasks only admission windows are used. Hourly labels are aligned to
`start_step + 23`, as in the released readout. Patient splits are disjoint.

Future features roll the fixed predictor forward for 8, 24 or 48 hours,
conditioning on the observed hour-23 action and then synthetic "no active
interventions" actions at the correct absolute ICU hours. No recorded future
state, treatment or outcome enters the model inputs. This is an artificial,
potentially out-of-distribution policy, not a treatment recommendation or a
validated causal counterfactual. The paper instead uses recorded future
actions for its retrospective rollout. Synthetic future features are a
deterministic function of the observed history and fixed policy.

The 64-dimensional history PCA trained on admission windows in the earlier
mortality experiment is reused. For each horizon a separate 32-dimensional
future PCA is fitted only on all sampled training windows. L2-regularized,
class-balanced logistic regression compares history 32, history 64, and
history 32 plus future 32. Each task/method chooses C from
{0.01, 0.1, 1, 10} by validation AUROC. Test AUROC and average precision
(AUPRC) are calculated per task; 500 patient-clustered bootstrap draws give
intervals and paired differences. Windows are weighted as in the original
readout. Single-class tasks are reported as unavailable; rare classes are
flagged. Very rare events and 34 simultaneous exploratory comparisons preclude
reliable claims from individual test results.

All 34 labels in this sample had both classes in all three splits. Some tasks
have extremely few positive patients: for example, tracheostomy has one in the
test split. Its interval uses only bootstrap draws containing both classes.
This evaluation has different encoders, predictor training, action policy,
readout and cohort size from the paper; comparing the absolute scores directly
would conflate all of these factors.

## Server artifacts

Under `/media/cuichang/liuhao/clin-jepa/sample_3h/clinical_data/all_tasks/`:

- `task_label_audit.json` is in the parent `clinical_data/` directory.
- `policy_actions.pt`, `parts/`, and `future/` contain synthetic action and
  projected future features.
- `evaluation/results.json` and `evaluation/results.csv` contain aggregate
  task-level metrics, counts, intervals and validation settings.
- `evaluation/predictions/` and `evaluation/models/` contain patient-level
  outputs and fitted probes. Keep these on the authorized server.

The Chinese Markdown report can be regenerated with:

```bash
"$CLIN_JEPA_PYTHON" scripts/write_downstream_report.py \
  --input "$EXP/clinical_data/all_tasks/evaluation/results.json" \
  --output docs/13-all-downstream-results.md
```

Run `prepare_hypothetical_actions.py` once; then run
`prepare_all_task_rollouts.py` with task IDs 0, 1 and 2, each on a distinct
idle GPU. `merge_all_task_rollouts.py` checks that all rows are present,
finite and aligned with history. `evaluate_all_downstream.py` takes the
sampled history, merged future, labels, splits and the earlier mortality
`probe.joblib` as its history PCA projector. These scripts refuse to overwrite
completed artifacts. The earlier 24-hour admission-only future features and
the new h=24 features were checked to have minimum cosine above 0.99999 on
sampled overlapping rows. AUROC and AUPRC for all 102 model/task pairs were
independently recalculated; saved models were reloaded for representative
8-, 24- and 48-hour tasks. See the script `--help` for complete arguments.
