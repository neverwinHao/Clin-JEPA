"""Config loading, path expansion, logging and seeding."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        value = os.path.expandvars(obj)
        if "${" in value:
            raise ValueError(f"unset environment variable in {obj!r}")
        return value
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def repo_path(path: str | Path) -> Path:
    """A relative path that does not exist from the working directory is taken relative to the repository."""
    path = Path(path)
    return path if path.is_absolute() or path.exists() else REPO_ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config and expand ``${VAR}`` references against the environment."""
    with repo_path(path).open() as f:
        return _expand(yaml.safe_load(f) or {})


def expand_path(path: str | Path) -> Path:
    """Expand ``~`` and ``${VAR}`` in a path and make it absolute."""
    return Path(_expand(os.path.expanduser(str(path)))).resolve()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
