"""Covariance spectrum of an embedding matrix and the rank statistics read from it."""

from dataclasses import dataclass

import torch

CHUNK = 65_536


@dataclass
class Spectrum:
    eigenvalues: torch.Tensor
    basis: torch.Tensor | None = None


def covariance_spectrum(z: torch.Tensor, device: torch.device | str, n_basis: int = 0) -> Spectrum:
    """Eigenvalues (descending) of the sample covariance of the rows of ``z``, accumulated in float64."""
    n, d = z.shape
    mean = torch.zeros(d, dtype=torch.float64, device=device)
    for start in range(0, n, CHUNK):
        mean += z[start : start + CHUNK].to(device=device, dtype=torch.float64).sum(dim=0)
    mean /= n

    cov = torch.zeros(d, d, dtype=torch.float64, device=device)
    for start in range(0, n, CHUNK):
        block = z[start : start + CHUNK].to(device=device, dtype=torch.float64) - mean
        cov += block.T @ block
    cov /= n - 1
    cov = 0.5 * (cov + cov.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    order = torch.argsort(eigenvalues, descending=True)
    basis = eigenvectors[:, order[:n_basis]].contiguous().cpu() if n_basis else None
    return Spectrum(eigenvalues[order].clamp_min(0.0).cpu(), basis)


def _exp_entropy(weights: torch.Tensor) -> float:
    p = weights / weights.sum()
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()))


def rankme(spectrum: Spectrum) -> float:
    """RankMe (Garrido et al., 2023): exponential of the entropy of the normalized singular values."""
    singular = spectrum.eigenvalues.sqrt()
    return _exp_entropy(singular / singular.sum() + 1e-7)


def effective_rank(spectrum: Spectrum) -> float:
    """Exponential of the entropy of the variance shares."""
    return _exp_entropy(spectrum.eigenvalues)


def top_share(spectrum: Spectrum, k: int = 10) -> float:
    return float(spectrum.eigenvalues[:k].sum() / spectrum.eigenvalues.sum())


def subspace_overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared cosine of the principal angles between two orthonormal bases ``(D, k)``."""
    cross = a.T @ b
    return float((cross * cross).sum() / a.shape[1])
