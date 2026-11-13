"""Choosing a reachable release point inside a container."""

from __future__ import annotations

import numpy as np

from robotics.simulation.scene import WORKSPACE, ContainerSpec, Workspace


def select_place_point(
    container: ContainerSpec,
    workspace: Workspace = WORKSPACE,
    margin: float = 0.045,
    occupied: list[np.ndarray] | None = None,
    min_separation: float = 0.06,
    resolution: float = 0.01,
) -> np.ndarray | None:
    """Return the release point (world, floor height) or ``None`` if the container is unreachable.

    Candidates form a grid over the container footprint shrunk by ``margin`` (object half-width
    plus clearance from the walls).  We keep candidates within the robot's comfortable reach and
    away from objects already in the container, then pick the one closest to the container
    centre.  Far quadrants of bin2 lie at the edge of the Panda's reach, so in practice the
    release point is pulled towards the robot.
    """
    cx, cy = container.center
    hx, hy = container.half_extents[0] - margin, container.half_extents[1] - margin
    if hx <= 0 or hy <= 0:
        return None
    xs = np.arange(cx - hx, cx + hx + 1e-9, resolution)
    ys = np.arange(cy - hy, cy + hy + 1e-9, resolution)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    reach = np.linalg.norm(pts - np.asarray(workspace.robot_base_xy)[None, :], axis=1)
    ok = reach <= workspace.comfortable_reach
    if occupied:
        for o in occupied:
            ok &= np.linalg.norm(pts - np.asarray(o[:2])[None, :], axis=1) >= min_separation
    if not np.any(ok):
        return None
    cand = pts[ok]
    best = cand[np.argmin(np.linalg.norm(cand - np.array([cx, cy])[None, :], axis=1))]
    return np.array([best[0], best[1], container.floor_z])
