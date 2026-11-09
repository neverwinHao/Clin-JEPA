"""Encode a fixed hypothetical no-intervention policy, indexed by ICU hour."""
import argparse
from pathlib import Path

import pandas as pd
import torch

from clin_jepa.frozen_base import encode_texts_batched, load_frozen_base
from clin_jepa.utils import load_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--history', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', default='configs/encode_frozen_base.yaml')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    maximum = max(int(pd.read_parquet(args.history / s / 'index.parquet').start_step.max()) for s in ('train', 'val', 'test')) + 70
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, tokenizer = load_frozen_base(config, device)
    texts = [f't={h}h | No active interventions.' for h in range(maximum + 1)]
    z = encode_texts_batched(model, tokenizer, texts, device, max_seq_len=config['data']['max_seq_len'],
                             output_dtype=torch.float16, batch_size=config['data']['batch_size'],
                             max_batch_tokens=config['data']['max_batch_tokens'])
    if z.shape != (maximum + 1, 4096) or not torch.isfinite(z).all():
        raise ValueError('Invalid synthetic action embeddings')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'embeddings': z.cpu(), 'max_hour': maximum, 'base_model': config['model']['base_model'],
                'policy': 'Synthetic no active interventions. Not clinical advice or a validated counterfactual.'}, args.output)
    print(f'Prepared no-intervention action embeddings for hours 0..{maximum}', flush=True)


if __name__ == '__main__':
    main()
