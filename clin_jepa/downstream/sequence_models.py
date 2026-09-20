"""LSTM and TCN baselines on the hourly sequence and the demographics."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

MIN_ROWS = 30


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, dropout=dropout if layers > 1 else 0.0, batch_first=True)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lstm(x)[1][0][-1]


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.trim = (kernel - 1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_in, n_out, kernel, padding=self.trim, dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(n_out, n_out, kernel, padding=self.trim, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(F.relu(self.conv1(x)[:, :, : -self.trim].contiguous()))
        out = self.dropout(F.relu(self.conv2(out)[:, :, : -self.trim].contiguous()))
        return F.relu(out + (x if self.skip is None else self.skip(x)))


class TCNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[TemporalBlock(input_dim if i == 0 else hidden, hidden, kernel, 2**i, dropout)
                                      for i in range(layers)])
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x.transpose(1, 2))[:, :, -1]


class SequenceModel(nn.Module):
    def __init__(self, encoder: nn.Module, static_dim: int, n_heads: int, head_hidden: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Linear(encoder.out_dim + static_dim, head_hidden), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(head_hidden, n_heads))

    def forward(self, sequence: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(sequence), statics], dim=-1))


def build(model: str, input_dim: int, static_dim: int, n_heads: int, config: dict, spec: dict) -> SequenceModel:
    hidden, layers, dropout = config["hidden_size"], config["num_layers"], config["dropout"]
    if model == "lstm":
        encoder = LSTMEncoder(input_dim, hidden, layers, dropout)
    else:
        encoder = TCNEncoder(input_dim, hidden, layers + spec["tcn_base_layers"], spec["tcn_kernel"], dropout)
    return SequenceModel(encoder, static_dim, n_heads, spec["head_hidden"], dropout)


@torch.no_grad()
def logits(
    net: SequenceModel, sequence: np.ndarray, statics: np.ndarray, batch_size: int, device: torch.device,
) -> np.ndarray:
    net.eval()
    sequence, statics = torch.from_numpy(sequence), torch.from_numpy(statics)
    out = [net(sequence[i : i + batch_size].to(device), statics[i : i + batch_size].to(device)).float().cpu().numpy()
           for i in range(0, len(sequence), batch_size)]
    return np.concatenate(out)


def mean_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mean AUROC over the heads with at least ``MIN_ROWS`` labelled rows of both classes; NaN if there is none."""
    values = []
    for j in range(labels.shape[1]):
        known = np.isfinite(labels[:, j]) & np.isfinite(scores[:, j])
        if known.sum() >= MIN_ROWS and len(np.unique(labels[known, j])) == 2:
            values.append(roc_auc_score(labels[known, j], scores[known, j]))
    return float(np.mean(values)) if values else float("nan")


def fit(
    model: str, config: dict, spec: dict, train: dict, val: dict, seed: int, device: torch.device,
) -> SequenceModel:
    """Train one configuration and restore the epoch with the best mean validation AUROC.

    ``train`` and ``val`` hold ``sequence`` (n, hours, channels), ``statics`` (n, d) and ``labels`` (n, heads) with
    NaN where a head has no label.
    """
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sequence, statics = torch.from_numpy(train["sequence"]), torch.from_numpy(train["statics"])
    known = np.isfinite(train["labels"])
    target = torch.from_numpy(np.nan_to_num(train["labels"], nan=0.0).astype(np.float32))
    mask = torch.from_numpy(known.astype(np.float32))
    positives, negatives = (known & (train["labels"] == 1)).sum(axis=0), (known & (train["labels"] == 0)).sum(axis=0)
    pos_weight = torch.from_numpy(np.maximum(1.0, negatives / np.maximum(1, positives)).astype(np.float32)).to(device)

    net = build(model, sequence.shape[2], statics.shape[1], target.shape[1], config, spec).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=spec["max_epochs"], eta_min=config["lr"] * spec["final_lr_ratio"])
    generator = torch.Generator().manual_seed(seed)
    n, batch_size = len(sequence), spec["batch_size"]
    best, best_state, stale = -1.0, None, 0
    for _ in range(spec["max_epochs"]):
        net.train()
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            rows = order[start : start + batch_size]
            out = net(sequence[rows].to(device, non_blocking=True), statics[rows].to(device, non_blocking=True))
            weight = mask[rows].to(device)
            loss = F.binary_cross_entropy_with_logits(out, target[rows].to(device), pos_weight=pos_weight,
                                                      reduction="none")
            loss = (loss * weight).sum() / weight.sum().clamp(min=1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), spec["grad_clip"])
            optimizer.step()
        scheduler.step()
        score = mean_auroc(val["labels"], logits(net, val["sequence"], val["statics"], batch_size, device))
        if np.isfinite(score) and score > best + spec["min_improvement"]:
            best, stale = score, 0
            best_state = {k: v.detach().clone().cpu() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= spec["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net
