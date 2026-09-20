# 07 · Latent geometry (§5.3)

Input: the test trajectory shards (01) and the test-split embedding caches of `clin_jepa`, `vjepa2ac` and `sft_baseline` (03). Output: `$CLIN_JEPA_OUTPUT/geometry/`. CPU only; cohort rules, sample sizes and seeds are in `configs/geometry.yaml`.

```bash
python -m clin_jepa.geometry.cohorts
python -m clin_jepa.geometry.metrics --encoder clin_jepa      # also vjepa2ac, sft_baseline
python -m clin_jepa.geometry.umap_view --encoder clin_jepa    # also vjepa2ac, sft_baseline
python -m clin_jepa.geometry.report
```

| step | computes | writes |
|---|---|---|
| `cohorts` | admission windows with 72 hourly SOFA totals; deteriorating / stable labels under the four definitions; the 50 + 50 subsample; 20 random relabelings | `cohorts.json` |
| `metrics` | per encoder, in its own 4,096-d space: typical inter-patient distance, net displacement and Cohen's d, centroid divergence, leave-one-out nearest-centroid and direction accuracy, kNN purity, bootstrap intervals, robustness configurations, null control | `metrics/<encoder>.json` |
| `umap_view` | UMAP fitted on 100,000 random test states of the encoder; projection of the subsample | `umap/<encoder>.npz` |
| `report` | Fig. 3 a–d, Table 3, App. Table 4, App. Figs. 2–3, the numbers quoted in the text | `figures/`, `tables/` |

`cohorts` finds 3,205 eligible admission windows, 1,074 deteriorating and 892 stable under the main definition. `metrics` and `umap_view` take 10–25 minutes per encoder and need about 96 GB of memory; `metrics` uses 16 cores, the seeded UMAP fit one. UMAP coordinates depend on the library versions and the CPU.
