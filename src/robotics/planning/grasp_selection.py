"""Collision-aware choice of the grasp yaw.

A square object can be grasped at ``yaw`` or ``yaw + 90°``.  When an object rests against a
bin wall (common after a drop) one of the two choices drives a finger into the wall.  We
evaluate the clearance of both finger tips to the known static walls and to the other
detected objects and keep the yaw with the larger clearance.
"""

from __future__ import annotations

import numpy as np

from robotics.simulation.scene import BIN_WALL_TOP_Z, CONTAINERS

FINGER_REACH = (
    0.05  # m from the grasp centre to a finger pad when the gripper is open (+ pad width)
)


MIN_CLEARANCE = 0.005  # m
HAND_HALF_LENGTH = 0.105  # m, Panda hand housing half-length along the finger axis
HAND_BOTTOM_ABOVE_TIP = 0.045  # m, hand housing bottom above the fingertip centre (grip site)


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
    center_xy: np.ndarray,
    yaw: float,
    obstacles_xy: list[np.ndarray],
    obstacle_radius: float = 0.03,
    grasp_z: float | None = None,
) -> float:
    """Smallest clearance of the open fingers (and, for low grasps, the hand housing).

    The Panda hand housing is ~21 cm long along the finger axis and sits ~4.5 cm above the
    fingertips.  When that is below the bin wall tops, its ends must clear the walls too.
    """
    center_xy = np.asarray(center_xy, float)
    clear = np.inf
    for f in finger_points(center_xy, yaw):
        clear = min(clear, _wall_clearance(f))
        for o in obstacles_xy:
            clear = min(clear, float(np.linalg.norm(f - np.asarray(o)[:2])) - obstacle_radius)
    if grasp_z is not None and grasp_z + HAND_BOTTOM_ABOVE_TIP < BIN_WALL_TOP_Z + 0.015:
        for h in finger_points(center_xy, yaw, HAND_HALF_LENGTH):
            clear = min(clear, _wall_clearance(h))
    return float(clear)


def choose_grasp_yaw(
    center_xy: np.ndarray,
    base_yaw: float,
    elongated: bool,
    obstacles_xy: list[np.ndarray],
    grasp_z: float | None = None,
) -> tuple[float, float]:
    """Return ``(yaw, clearance)``.  Elongated footprints keep their single valid yaw."""
    candidates = [base_yaw] if elongated else [base_yaw, base_yaw + np.pi / 2]
    scored = [
        (
            grasp_clearance(center_xy, _wrap_half_pi(y), obstacles_xy, grasp_z=grasp_z),
            _wrap_half_pi(y),
        )
        for y in candidates
    ]
    # Among collision-free candidates take the smallest wrist rotation: large yaws push the
    # Panda's last joint toward its ±2.9 rad limit during far transports.  If no candidate is
    # free, take the one with the most clearance.
    feasible = [t for t in scored if t[0] > MIN_CLEARANCE]
    if feasible:
        scored = sorted(feasible, key=lambda t: abs(t[1]))
    else:
        scored.sort(key=lambda t: -t[0])
    clearance, yaw = scored[0]
    return yaw, clearance
