# 01 · Data

From MIMIC-IV to trajectory shards. All steps read `configs/data/paths.yaml`; run them from the repository root.

| step | reads | command | writes |
|---|---|---|---|
| 1 cohort | `icustay_detail`, `vitalsign`, `sepsis3`, `configs/data/cohort.yaml` | `python -m clin_jepa.data.step01_cohort` | `cohort/cohort.csv` |
| 2 observations | cohort, concept tables under `observations:` and the ICD tables under `diagnoses:` in `configs/data/features.yaml` | `python -m clin_jepa.data.step02_observations --chunk_idx I --total_chunks N` | `observations/<stay_id>.parquet` |
| 3 interventions | cohort, tables under `actions:` in `features.yaml`, `d_items` | `python -m clin_jepa.data.step03_actions --chunk_idx I --total_chunks N` | `actions/<stay_id>.parquet` |
| 4 hourly bins | cohort, steps 2–3, `temporal:` in `features.yaml` | `python -m clin_jepa.data.step04_discretization --chunk_idx I --total_chunks N` | `discretized/<stay_id>.pkl` |
| 5 split | cohort | `python -m clin_jepa.data.step05_split` | `splits/splits.csv` |
| 6 trajectories | steps 4–5, `configs/data/trajectories.yaml` | `python -m clin_jepa.data.step06_trajectories [--chunk_idx I]` | `trajectories/{train,val,test}/trajectories_cII_SSS.pt` |

Order: 1, then 2 and 3 (in parallel if wanted), then 4 once both are complete, 5 any time after 1, and 6 once 4 and 5 are complete. Steps 2–4 are independent across chunks (`I = 0 … N-1`) and any `N` gives the same files. Step 6 always writes the ten chunks fixed by `n_chunks` in its config, one per call with `--chunk_idx`, or all ten without it. With ten-way arrays (8 cores, 64 GB each) the whole stage takes well under an hour.

Result: 84,497 ICU stays, 280,882 trajectory windows of up to 72 hours (stride 12), split 70 / 15 / 15 by patient. Each shard stores every (stay, hour) once, with its state text, action text and SOFA score, and the windows as indices into those hours. What is extracted but kept out of the state and action texts is listed in `configs/data/trajectories.yaml` and `clin_jepa/data/helpers/action_filter.py` (paper, Appendix A).

The feature tables of Appendix A are generated from `configs/data/features.yaml`:

```bash
python -m clin_jepa.data.feature_table    # writes $CLIN_JEPA_OUTPUT/tables/appendix_features_table.tex
```
