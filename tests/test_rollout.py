import numpy as np
import torch

from clin_jepa.model.predictor import build_block_causal_mask
from clin_jepa.rollout import rollout_batch


class AddAction:
    def forward_rollout_step(self, z_hist, static, a_hist, a_next, z_current):
        return z_current + a_next


def test_rollout_pools_and_masks_past_the_stay_end():
    torch.manual_seed(0)
    context, steps, horizons = 3, 4, [2, 4]
    states, actions = torch.randn(3, context + steps, 5), torch.randn(3, context + steps, 5)
    n_inside = torch.tensor([4, 2, 0])
    out = rollout_batch(AddAction(), states, actions, torch.zeros(3, 5), n_inside, context, steps, horizons)

    predicted = states[:, context - 1 : context] + torch.cumsum(actions[:, context - 1 : context - 1 + steps], dim=1)
    inside = torch.arange(steps)[None, :] < n_inside[:, None]
    padded = torch.where(inside[:, :, None], predicted, torch.zeros_like(predicted))
    for i, h in enumerate(horizons):
        np.testing.assert_allclose(out["pooled"][:, i, 0], padded[:, :h].mean(dim=1), atol=1e-5)
        np.testing.assert_allclose(out["pooled"][:, i, 1], padded[:, :h].std(dim=1), atol=1e-5)
        last = torch.where((n_inside >= h)[:, None], predicted[:, h - 1], torch.zeros(3, 5))
        np.testing.assert_allclose(out["pooled"][:, i, 2], last, atol=1e-5)

    truth = states[:, context:]
    expected = (predicted - truth).abs().sum(-1)
    np.testing.assert_allclose(out["model_l1"][inside], expected[inside], rtol=1e-5)
    assert torch.isnan(out["model_l1"][~inside]).all() and torch.isnan(out["copy_forward_l1"][~inside]).all()
    copy_forward = (states[:, context - 1 : context] - truth).abs().sum(-1)
    np.testing.assert_allclose(out["copy_forward_l1"][inside], copy_forward[inside], rtol=1e-5)


def test_block_causal_mask_hides_later_hours_only():
    mask = build_block_causal_mask(3, torch.device("cpu"))
    hour = [0, 1, 1, 2, 2, 3, 3]
    for i in range(7):
        for j in range(7):
            assert (mask[i, j] == 0) == (hour[j] <= hour[i])
