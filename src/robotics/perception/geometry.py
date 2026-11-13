"""Pinhole camera geometry for RGB-D back-projection (pure NumPy, no simulator access)."""

from __future__ import annotations

import numpy as np


def backproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a metric depth image into camera-frame 3D points.

    For pixel (u, v) with depth z (distance along the optical axis):
        x = (u - cx) · z / fx,   y = (v - cy) · z / fy
    Returns an (H, W, 3) array in the OpenCV camera frame (x right, y down, z forward).
    """
    h, w = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    # Pixel centres are at integer + 0.5 in MuJoCo's rasteriser convention.
    u = np.arange(w, dtype=np.float64) + 0.5
    v = np.arange(h, dtype=np.float64) + 0.5
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float64)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def camera_to_world(points_cam: np.ndarray, cam_to_world: np.ndarray) -> np.ndarray:
    """Apply a 4×4 rigid transform to an (..., 3) array of points."""
    R = cam_to_world[:3, :3]
    t = cam_to_world[:3, 3]
    return points_cam @ R.T + t


def depth_to_world(
    depth: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray
) -> np.ndarray:
    return camera_to_world(backproject_depth(depth, intrinsics), cam_to_world)


def project_to_pixels(
    points_world: np.ndarray, intrinsics: np.ndarray, cam_to_world: np.ndarray
) -> np.ndarray:
    """World points (N, 3) → pixel coordinates (N, 2) as (u, v) floats."""
    world_to_cam = np.linalg.inv(cam_to_world)
    pc = points_world @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    uv = pc @ intrinsics.T
    return uv[:, :2] / uv[:, 2:3] - 0.5


def valid_depth_mask(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    """Pixels with finite depth inside the sensor's working range."""
    return np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)


def grasp_yaw_from_footprint(
    points_xy: np.ndarray, square_aspect: float = 1.25
) -> tuple[float, float]:
    """Gripper yaw that closes the fingers across the short side of an object's footprint.

    ``points_xy`` are world-frame (N, 2) points of the object's top view.  At yaw θ the Panda
    fingers close along the world direction (−sin θ, cos θ), so for a short axis at angle φ we
    need θ = φ − π/2 (mod π).  Near-square footprints have 90° symmetry, so we pick the
    smallest rotation in [−π/4, π/4); elongated ones (e.g. a toppled can) use [−π/2, π/2).

    Returns ``(yaw, aspect_ratio)``.
    """
    import cv2

    pts = np.asarray(points_xy, dtype=np.float64)
    if len(pts) < 3:
        return 0.0, 1.0
    # minAreaRect needs float32; work in millimetres for numerical resolution.
    box = cv2.boxPoints(cv2.minAreaRect((pts * 1000.0).astype(np.float32))) / 1000.0
    e1, e2 = box[1] - box[0], box[2] - box[1]
    l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
    short = e1 if l1 <= l2 else e2
    aspect = max(l1, l2) / max(1e-6, min(l1, l2))
    phi = float(np.arctan2(short[1], short[0]))
    theta = phi - np.pi / 2
    if aspect < square_aspect:
        theta = (theta + np.pi / 4) % (np.pi / 2) - np.pi / 4
    else:
        theta = (theta + np.pi / 2) % np.pi - np.pi / 2
    return float(theta), float(aspect)
