"""MLP behaviour-cloning policy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from robotics.common.config import PROJECT_ROOT
from robotics.learning.dataset import Normalizer
from robotics.learning.features import ACTION_DIM, FEATURE_VERSION, OBS_DIM


class MLPPolicy(nn.Module):
    """obs → (translation action ∈ [-1, 1]^3, gripper logit).

    Translation is regressed (tanh output, MSE loss).  The gripper command is binary in the
    demonstrations (open / close) and is learned as a classification problem (BCE loss), which
    avoids the "half-closed" averages a regression head produces at phase boundaries.
    """

    def __init__(
        self, obs_dim: int = OBS_DIM, hidden: list[int] | None = None, dropout: float = 0.0
    ):
        super().__init__()
        hidden = hidden or [256, 256, 256]
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, ACTION_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.head(self.body(x))
        return torch.tanh(out[:, :3]), out[:, 3]


class LoadedPolicy:
    """Inference wrapper: raw features in, environment action pieces out."""

    def __init__(self, model: MLPPolicy, normalizer: Normalizer, meta: dict):
        self.model = model.eval()
        self.normalizer = normalizer
        self.meta = meta

    @torch.no_grad()
    def act(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        x = torch.from_numpy(self.normalizer(features[None, :]))
        xyz, logit = self.model(x)
        return xyz[0].numpy().astype(np.float64), float(torch.sigmoid(logit[0]))


def save_checkpoint(path: str | Path, model: MLPPolicy, normalizer: Normalizer, meta: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "normalizer": normalizer.to_dict(),
            "hidden": meta.get("hidden_sizes"),
            "dropout": meta.get("dropout", 0.0),
            "feature_version": FEATURE_VERSION,
            "meta": meta,
        },
        p,
    )
    return p


def load_policy(path: str | Path) -> LoadedPolicy:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint {p} not found. Train one with: uv run python scripts/train_bc.py"
        )
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if ckpt.get("feature_version") != FEATURE_VERSION:
        raise ValueError(
            f"checkpoint feature version {ckpt.get('feature_version')} != {FEATURE_VERSION}"
        )
    model = MLPPolicy(OBS_DIM, ckpt["hidden"], ckpt.get("dropout", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return LoadedPolicy(model, Normalizer.from_dict(ckpt["normalizer"]), ckpt.get("meta", {}))
