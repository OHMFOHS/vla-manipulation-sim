"""Smooth Cartesian trajectories (minimum-jerk interpolation with a speed limit)."""

from __future__ import annotations

import numpy as np


def min_jerk_profile(tau: np.ndarray) -> np.ndarray:
    """Normalised minimum-jerk position profile s(τ) = 10τ³ − 15τ⁴ + 6τ⁵, τ ∈ [0, 1].

    It has zero velocity and acceleration at both ends, which avoids the jerky starts/stops
    of a raw P-controller chasing a far-away goal.
    """
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


def min_jerk_segment(
    start: np.ndarray, goal: np.ndarray, max_speed: float, dt: float, min_steps: int = 3
) -> np.ndarray:
    """Return an (N, 3) array of set-points from ``start`` (exclusive) to ``goal`` (inclusive).

    Peak speed of a minimum-jerk profile is 1.875 × the average speed, so the duration is
    chosen as T = 1.875 · d / v_max to respect ``max_speed``.
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if max_speed <= 0 or dt <= 0:
        raise ValueError("max_speed and dt must be positive")
    dist = float(np.linalg.norm(goal - start))
    duration = 1.875 * dist / max_speed
    n = max(min_steps, int(np.ceil(duration / dt)))
    tau = np.arange(1, n + 1) / n
    s = min_jerk_profile(tau)[:, None]
    return start[None, :] + s * (goal - start)[None, :]


def max_step_speed(points: np.ndarray, dt: float) -> float:
    """Largest per-step speed along a set-point sequence (used in tests)."""
    if len(points) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(np.diff(points, axis=0), axis=1)) / dt)
