"""Demonstration dataset with episode-level train/validation splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from robotics.common.config import PROJECT_ROOT
from robotics.learning.features import ACTION_DIM, OBS_DIM


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> Normalizer:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        # Binary / constant features (one-hot, latch) keep a unit scale.
        std = np.where(std < eps, 1.0, std)
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> Normalizer:
        return cls(np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32))


def load_demos(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"Demo file {p} not found. Run: uv run python scripts/collect_demos.py"
        )
    data = dict(np.load(p))
    obs, act = data["observations"], data["actions"]
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(
            f"observations must be (N, {OBS_DIM}), got {obs.shape} (feature version mismatch?)"
        )
    if act.shape != (obs.shape[0], ACTION_DIM):
        raise ValueError(f"actions must be (N, {ACTION_DIM}), got {act.shape}")
    return data


def split_by_episode(
    episodes: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks for train / validation.  Whole episodes go to one side only, so no
    near-duplicate consecutive frames leak from training into validation."""
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction))) if len(unique) > 1 else 0
    val_eps = set(unique[:n_val].tolist())
    val = np.array([e in val_eps for e in episodes])
    return ~val, val


class DemoDataset(Dataset):
    def __init__(self, obs: np.ndarray, act: np.ndarray, normalizer: Normalizer):
        self.x = torch.from_numpy(normalizer(obs))
        self.y = torch.from_numpy(act.astype(np.float32))

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]
