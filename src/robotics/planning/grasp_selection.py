"""Collision-aware choice of the grasp yaw.

A square object can be grasped at ``yaw`` or ``yaw + 90°``.  When an object rests against a
bin wall (common after a drop) one of the two choices drives a finger into the wall.  We
evaluate the clearance of both finger tips to the known static walls and to the other
detected objects and keep the yaw with the larger clearance.
"""

from __future__ import annotations

import numpy as np

from robotics.simulation.scene import CONTAINERS

FINGER_REACH = (
    0.05  # m from the grasp centre to a finger pad when the gripper is open (+ pad width)
)


def _wrap_half_pi(yaw: float) -> float:
    """Wrap to [-π/2, π/2): equivalent for a parallel gripper, gentle on the wrist joint."""
    return float((yaw + np.pi / 2) % np.pi - np.pi / 2)


def _wall_clearance(p: np.ndarray) -> float:
    """Distance from an xy point to the nearest wall of the container it lies in.

    Points outside every container (on the table edge, above a wall) get negative clearance.
    """
    best = -np.inf
    for c in CONTAINERS.values():
        dx = c.half_extents[0] - abs(p[0] - c.center[0])
        dy = c.half_extents[1] - abs(p[1] - c.center[1])
        best = max(best, min(dx, dy))
    return float(best)


def finger_points(center_xy: np.ndarray, yaw: float, reach: float = FINGER_REACH) -> np.ndarray:
    """At yaw θ the Panda fingers open along world (−sin θ, cos θ)."""
    d = np.array([-np.sin(yaw), np.cos(yaw)])
    return np.stack([center_xy + reach * d, center_xy - reach * d])


def grasp_clearance(
    center_xy: np.ndarray, yaw: float, obstacles_xy: list[np.ndarray], obstacle_radius: float = 0.03
) -> float:
    clear = np.inf
    for f in finger_points(np.asarray(center_xy, float), yaw):
        clear = min(clear, _wall_clearance(f))
        for o in obstacles_xy:
            clear = min(clear, float(np.linalg.norm(f - np.asarray(o)[:2])) - obstacle_radius)
    return float(clear)


def choose_grasp_yaw(
    center_xy: np.ndarray, base_yaw: float, elongated: bool, obstacles_xy: list[np.ndarray]
) -> tuple[float, float]:
    """Return ``(yaw, clearance)``.  Elongated footprints keep their single valid yaw."""
    candidates = [base_yaw] if elongated else [base_yaw, base_yaw + np.pi / 2]
    scored = [
        (grasp_clearance(center_xy, _wrap_half_pi(y), obstacles_xy), _wrap_half_pi(y))
        for y in candidates
    ]
    # prefer the smaller wrist rotation when clearances are equivalent (within 5 mm)
    scored.sort(key=lambda t: (-round(t[0] / 0.005), abs(t[1])))
    clearance, yaw = scored[0]
    return yaw, clearance
