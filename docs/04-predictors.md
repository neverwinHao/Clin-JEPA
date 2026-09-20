# 04 · Predictors on frozen encoders

Input: the embedding caches of stage 03. One GPU and about 80 GB of host memory each; the training split of a cache is held in memory.

```bash
python -m clin_jepa.training.train_predictor --config configs/train/predictor_vjepa2ac.yaml
python -m clin_jepa.training.train_predictor --config configs/train/predictor_sft_baseline.yaml
python -m clin_jepa.training.train_predictor --config configs/train/predictor_separate.yaml
```

| model | cache | output |
|---|---|---|
| V-JEPA 2-AC style | `embeddings/vjepa2ac` | `predictor_vjepa2ac/best.pt` |
| SFT baseline w/o JEPA | `embeddings/sft_baseline` | `predictor_sft_baseline/best.pt` |
| Clin-JEPA w/ separately trained predictor | `embeddings/clin_jepa` | `predictor_separate/seed_{42,1337,2026}/best.pt` |

Twenty epochs (15,400 steps) each, under an hour per training.
