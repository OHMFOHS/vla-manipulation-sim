"""Module F building blocks: features, dataset split, normaliser, policy."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robotics.learning.dataset import Normalizer, split_by_episode
from robotics.learning.features import ACTION_DIM, OBS_DIM, HeldLatch, build_features
from robotics.learning.policy import LoadedPolicy, MLPPolicy, load_policy, save_checkpoint


def test_feature_vector_layout():
    f = build_features(
        np.array([0.0, 0.0, 1.0]),
        np.array([0.1, 0.0, -0.2]),
        0.08,
        np.array([0.1, -0.2, 0.86]),
        0.9,
        np.array([0.0, 0.16, 0.82]),
        -1.0,
        0.0,
        "can",
    )
    assert f.shape == (OBS_DIM,) and f.dtype == np.float32
    assert f[3:6] == pytest.approx([0.1, 0.0, -0.2])
    assert f[6:9] == pytest.approx([0.1, -0.2, -0.14])
    assert f[9] == pytest.approx(-0.1)
    assert f[-4:].tolist() == [0, 0, 0, 1]


def test_velocity_estimator():
    from robotics.learning.features import VelocityEstimator

    v = VelocityEstimator(dt=0.05)
    assert np.allclose(v.update(np.zeros(3)), 0.0)
    assert np.allclose(v.update(np.array([0.01, 0.0, -0.005])), [0.2, 0.0, -0.1])


def test_held_latch_requires_consecutive_hold():
    latch = HeldLatch(0.012, 0.068, steps=3)
    assert latch.update(0.042, 1.0) == 0.0
    assert latch.update(0.08, 1.0) == 0.0  # opened: counter resets
    for _ in range(3):
        v = latch.update(0.042, 1.0)
    assert v == 1.0
    assert latch.update(0.08, -1.0) == 1.0  # stays latched after release


def test_episode_split_has_no_leakage():
    episodes = np.repeat(np.arange(20), 50)
    tr, va = split_by_episode(episodes, 0.2, seed=0)
    assert set(episodes[tr]).isdisjoint(set(episodes[va]))
    assert len(set(episodes[va])) == 4
    assert tr.sum() + va.sum() == len(episodes)


def test_normalizer_handles_constant_features():
    x = np.stack([np.arange(10.0), np.ones(10)], axis=1)
    n = Normalizer.fit(x)
    z = n(x)
    assert abs(z[:, 0].mean()) < 1e-6 and z[:, 1].std() == 0
    assert np.isfinite(z).all()
    assert np.allclose(Normalizer.from_dict(n.to_dict()).mean, n.mean)


def test_policy_output_ranges():
    model = MLPPolicy(OBS_DIM, [32, 32])
    xyz, logit = model(torch.randn(8, OBS_DIM) * 10)
    assert xyz.shape == (8, 3) and logit.shape == (8,)
    assert torch.all(xyz.abs() <= 1.0)


def test_checkpoint_roundtrip(tmp_path):
    model = MLPPolicy(OBS_DIM, [16])
    norm = Normalizer(np.zeros(OBS_DIM, np.float32), np.ones(OBS_DIM, np.float32))
    path = save_checkpoint(tmp_path / "p.pt", model, norm, {"hidden_sizes": [16]})
    pol = load_policy(path)
    assert isinstance(pol, LoadedPolicy)
    xyz, p = pol.act(np.zeros(OBS_DIM, np.float32))
    assert xyz.shape == (3,) and 0.0 <= p <= 1.0
    assert ACTION_DIM == 4


def test_missing_checkpoint_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_bc.py"):
        load_policy(tmp_path / "nope.pt")
