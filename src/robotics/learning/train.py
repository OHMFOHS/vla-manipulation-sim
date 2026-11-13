"""Behaviour-cloning training loop with validation and checkpointing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from robotics.common.config import PROJECT_ROOT, TrainingConfig
from robotics.learning.dataset import DemoDataset, Normalizer, load_demos, split_by_episode
from robotics.learning.policy import MLPPolicy, save_checkpoint

logger = logging.getLogger(__name__)


def _evaluate(model, loader, bce, gripper_w) -> dict:
    model.eval()
    tot = {"loss": 0.0, "xyz_mse": 0.0, "grip_acc": 0.0}
    n = 0
    with torch.no_grad():
        for x, y in loader:
            xyz, logit = model(x)
            target_g = (y[:, 3] > 0).float()
            mse = ((xyz - y[:, :3]) ** 2).mean(dim=1)
            g = bce(logit, target_g)
            tot["loss"] += float((mse + gripper_w * g).sum())
            tot["xyz_mse"] += float(mse.sum())
            tot["grip_acc"] += float(((logit > 0).float() == target_g).float().sum())
            n += len(x)
    return {k: v / max(1, n) for k, v in tot.items()}


def train_bc(
    cfg: TrainingConfig, out_dir: str | Path | None = None, max_episodes: int | None = None
) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    data = load_demos(cfg.demo_path)
    obs, act, episodes = data["observations"], data["actions"], data["episodes"]
    if max_episodes is not None:
        keep_eps = np.unique(episodes)[:max_episodes]
        m = np.isin(episodes, keep_eps)
        obs, act, episodes = obs[m], act[m], episodes[m]
    train_mask, val_mask = split_by_episode(episodes, cfg.val_fraction, cfg.seed)
    normalizer = Normalizer.fit(obs[train_mask])  # fit on training split only
    train_ds = DemoDataset(obs[train_mask], act[train_mask], normalizer)
    val_ds = DemoDataset(obs[val_mask], act[val_mask], normalizer)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=4096)

    model = MLPPolicy(obs.shape[1], cfg.hidden_sizes, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    out = Path(out_dir or cfg.checkpoint_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = float("inf")
    meta = {
        **cfg.model_dump(),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "train_episodes": int(len(np.unique(episodes[train_mask]))),
        "val_episodes": int(len(np.unique(episodes[val_mask]))),
    }
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run, n = 0.0, 0
        for x, y in train_loader:
            xyz, logit = model(x)
            loss = ((xyz - y[:, :3]) ** 2).mean(dim=1) + cfg.gripper_loss_weight * bce(
                logit, (y[:, 3] > 0).float()
            )
            loss = loss.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * len(x)
            n += len(x)
        sched.step()
        val = _evaluate(model, val_loader, bce, cfg.gripper_loss_weight)
        rec = {"epoch": epoch, "train_loss": run / n, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        if val["loss"] < best:
            best = val["loss"]
            save_checkpoint(
                out / "bc_policy.pt",
                model,
                normalizer,
                {**meta, "best_epoch": epoch, "best_val": val},
            )
        if epoch % 10 == 0 or epoch == 1:
            logger.warning(
                "epoch %3d train %.4f | val loss %.4f xyz_mse %.4f grip_acc %.4f",
                epoch, rec["train_loss"], val["loss"], val["xyz_mse"], val["grip_acc"],
            )  # fmt: skip
    save_checkpoint(out / "bc_policy_last.pt", model, normalizer, {**meta, "epoch": cfg.epochs})
    summary = {
        **meta,
        "best_val_loss": best,
        "train_time_s": time.perf_counter() - t0,
        "history": history,
    }
    with open(out / "training_log.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary
