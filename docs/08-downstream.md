# 08 · Downstream tasks (§5.4)

Input: hourly bins, cohort and trajectory shards (01), the embedding caches of `clin_jepa`, `vjepa2ac` and `sft_baseline` (03), their rollouts on all three splits (05), and from `$MIMIC_RAW` the `admissions`, `patients`, `icustays` and `ventilation` tables. Output: `$CLIN_JEPA_OUTPUT/downstream/`. Tasks, read-out settings, baseline grids and the bootstrap are in `clin_jepa/downstream/tasks.py` and `configs/downstream.yaml`; CLMBR-T-base is configured in `configs/clmbr.yaml`.

```bash
python -m clin_jepa.downstream.labels

python -m clin_jepa.downstream.history --encoder clin_jepa --split train        # every encoder and split
python -m clin_jepa.downstream.readout --benchmark E1 --encoder clin_jepa --variant full   # E1 E2 E3 x encoder x hist|full

python -m clin_jepa.downstream.tabular census
python -m clin_jepa.downstream.tabular extract                                  # --task_id i --num_tasks n to split the work
python -m clin_jepa.downstream.baselines --model lightgbm --task_set acute_hourly   # every model and task set

python -m clin_jepa.downstream.clmbr.timelines --rows stay --split train       # stay | hourly  x  train | val | test
python -m clin_jepa.downstream.clmbr.extract --rows stay --split train         # femr environment; --task_id i --num_tasks n
python -m clin_jepa.downstream.clmbr.readout --rows stay                       # stay | hourly

python -m clin_jepa.downstream.bootstrap
python -m clin_jepa.downstream.report
```

| step | computes | writes |
|---|---|---|
| `labels` | the task labels per stay-hour or per stay, NaN where a task is undefined | `labels/hourly.parquet`, `labels/stay.parquet` |
| `history` | per window: mean, standard deviation and last of the 24 context state and action embeddings, and the demographics embedding | `features/<encoder>/<split>/` |
| `readout` | one MLP per task on the context block (`hist`) or on the context block and the pooled rollout at the task's horizon (`full`) | `predictions/<task>/<encoder>_<variant>.parquet` |
| `tabular` | the record the encoders read as 400 hourly columns and 7 static columns; `census` fixes the item and ventilator-mode columns from the training split | `tabular/` |
| `baselines` | Ridge and LightGBM on 24-hour summary statistics, LSTM and TCN on the 24-hour sequence | `predictions/<task>/<model>.parquet` |
| `clmbr.timelines` | per subject, the laboratory, charted, input, output and procedure events that map into the model's vocabulary, up to the cutoff (24 h into the first stay, or the end of the subject's last hourly row), as MEDS patients; events read, mapped and in vocabulary per table | `clmbr/timelines/<rows>/<split>/` |
| `clmbr.extract` | the 768-d representation of the last token at or before each row's cutoff, from the last 496 tokens of the timeline | `clmbr/features/<rows>/<split>/` |
| `clmbr.readout` | the probes of `configs/clmbr.yaml` per task, the best on validation AUROC | `predictions/<task>/clmbr.parquet` |
| `bootstrap` | 500 stay-clustered resamples of AUROC and AUPRC per prediction file | `bootstrap/<task>/<method>.npz` |
| `report` | Fig. 4, Table 4, the six per-task tables, the numbers quoted in the text | `figures/`, `tables/` |

Models: `ridge`, `lightgbm`, `lstm`, `tcn`. Task sets: `acute_hourly`, `treatment_hourly`, `prognosis`, `treatment_stay`, `icu_mortality`. A prediction file holds `stay_id`, `target_hour`, `label` and one probability column per configuration; existing files are skipped, so interrupted runs can be resumed. `--task` restricts Ridge or LightGBM to one task of the set.

| step | resources (full cohort) |
|---|---|
| `labels` | under 10 min on 16 cores |
| `history` | CPU, a few minutes per encoder and split |
| `readout` | one GPU, 1–8 min per call, up to 80 GB of host memory |
| `tabular` | `census` under 10 min on 16 cores; `extract` under 10 min per task as an 8-task array |
| `baselines` | Ridge and LightGBM 5–20 min per task on 16 cores, 40 GB; LSTM and TCN one GPU, 5–45 min per task set, up to 90 GB of host memory |
| `clmbr` | `timelines` 20 min per call on 8 cores, up to 60 GB; `extract` 5 min per 2,000 subjects on 8 cores; `readout` one GPU, 10–20 min per call |
| `bootstrap`, `report` | 15 min on 16 cores |

`n_workers` in the config (16) sets the process pools and LightGBM's threads. Baseline results differ slightly between machines, Ridge and LightGBM between CPU types and LSTM and TCN between GPU models, and TCN also between runs, in the third or fourth decimal of AUROC.

## CLMBR-T-base

| input | source | variable |
|---|---|---|
| CLMBR-T-base (`config.json`, `dictionary.msgpack`, `model.safetensors`) | [StanfordShahLab/clmbr-t-base](https://huggingface.co/StanfordShahLab/clmbr-t-base) on Hugging Face, access on request | `CLMBR_MODEL` |
| MIMIC-IV item to OMOP concept mappings | `mimic-iv/mapping/` of [mimic-code](https://github.com/MIT-LCP/mimic-code) | `MIMIC_CODE` (the checkout) |
| `labevents`, `chartevents`, `inputevents`, `outputevents`, `procedureevents`, `patients` | MIMIC-IV v3.1 | `MIMIC_RAW` |

`clmbr.extract` runs in the model's own environment; every other step uses the project environment.

```bash
conda create -n femr python=3.10 && conda activate femr
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install femr==0.2.3 transformers==4.35.2 datasets==2.15.0 xformers==0.0.23.post1 pandas pyarrow pyyaml
pip install -e . --no-deps
```

`bootstrap` and `report` cover the prediction files that exist.
