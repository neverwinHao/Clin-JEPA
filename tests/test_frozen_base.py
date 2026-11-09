"""Checks for frozen-base pooling and compatibility with predictor caches."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from clin_jepa.encode import encode_shard
from clin_jepa.frozen_base import encode_texts_batched
from clin_jepa.training.predictor_dataset import PredictorDataset


class Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, **kwargs):
        ids = [[int(x) for x in text.split()] for text in texts]
        if kwargs.get("padding") is False:
            return {"input_ids": ids}
        width = max(len(row) for row in ids)
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in ids]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in ids]),
        }


class Model:
    config = SimpleNamespace(hidden_size=2)

    def __init__(self):
        self.batch_tokens = []

    def __call__(self, input_ids, attention_mask, use_cache):
        self.batch_tokens.append(attention_mask.numel())
        assert not torch.is_grad_enabled()
        assert use_cache is False
        return SimpleNamespace(last_hidden_state=torch.stack([input_ids, input_ids + 1], dim=-1).float())


class FrozenBaseTest(unittest.TestCase):
    def test_last_non_padding_token_is_stable_across_batches(self):
        texts = ["2 3", "7", "4 5 6"]
        results = [
            encode_texts_batched(Model(), Tokenizer(), texts, torch.device("cpu"), batch_size=size)
            for size in (1, 3)
        ]
        expected = torch.tensor([[3, 4], [7, 8], [6, 7]], dtype=torch.float16)
        self.assertTrue(torch.equal(results[0], expected))
        self.assertTrue(torch.equal(results[1], expected))
        model = Model()
        bounded = encode_texts_batched(model, Tokenizer(), texts, torch.device("cpu"), batch_size=3, max_batch_tokens=3)
        self.assertTrue(torch.equal(bounded, expected))
        self.assertLessEqual(max(model.batch_tokens), 3)

    def test_trajectory_cache_is_readable_by_predictor_dataset(self):
        shard = {
            "per_hour": {"state_texts": ["2 3", "7"], "action_texts": ["8", "8"]},
            "per_stay": {"ages": [60], "genders": ["M"], "races": ["OTHER"], "stay_ids": [1]},
            "windows": [{"hour_indices": np.array([0, 1]), "stay_index": 0, "start_step": 0, "length": 2}],
        }
        def encode(model, tokenizer, texts, device, **kwargs):
            if texts and texts[0].startswith("Age:"):
                texts = ["9"] * len(texts)
            return encode_texts_batched(model, tokenizer, texts, device, **kwargs)
        cached = encode_shard(shard, Model(), Tokenizer(), torch.device("cpu"), 16, encode)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "train"
            directory.mkdir()
            torch.save(cached, directory / "embeddings.pt")
            example = PredictorDataset(tmp, "train")[0]
        self.assertEqual(tuple(example["z_states"].shape), (2, 2))
        self.assertTrue(torch.equal(example["z_actions"][0], example["z_actions"][1]))
        self.assertEqual(example["length"], 2)


if __name__ == "__main__":
    unittest.main()
