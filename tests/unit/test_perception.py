"""Module B: RGB-D perception on synthetic frames (no simulator)."""

from __future__ import annotations

import numpy as np
import pytest

from robotics.perception.geometry import (
    backproject_depth,
    camera_to_world,
    grasp_yaw_from_footprint,
    project_to_pixels,
    valid_depth_mask,
)
from robotics.perception.vision import VisionPerception
from robotics.simulation.env import RGBDFrame, opencv_cam_to_world, pinhole_intrinsics

CAM_POS = np.array([0.1, 0.015, 2.0])
FLOOR = 0.82
COLORS = {
    "milk": (26, 89, 242),
    "bread": (242, 204, 26),
    "cereal": (230, 31, 31),
    "can": (26, 204, 64),
}


def make_frame(objects, size=200, floor_rgb=(150, 120, 90)):
    """Render a synthetic top-down RGB-D frame.

    ``objects``: list of (rgb, center_xy, half_size, top_z).  Later objects occlude earlier ones.
    """
    K = pinhole_intrinsics(52.0, size, size)
    T = opencv_cam_to_world(CAM_POS, np.eye(3))
    rgb = np.zeros((size, size, 3), np.uint8)
    rgb[:] = floor_rgb
    depth = np.full((size, size), CAM_POS[2] - FLOOR, np.float32)
    v, u = np.mgrid[0:size, 0:size]
    dx = (u + 0.5 - K[0, 2]) / K[0, 0]
    dy = (v + 0.5 - K[1, 2]) / K[1, 1]
    for color, (cx, cy), half, top in objects:
        zc = CAM_POS[2] - top
        wx = CAM_POS[0] + dx * zc
        wy = CAM_POS[1] - dy * zc
        m = (np.abs(wx - cx) < half) & (np.abs(wy - cy) < half)
        rgb[m] = color
        depth[m] = zc
    return RGBDFrame(
        rgb=rgb, depth=depth, intrinsics=K, cam_to_world=T, camera_name="synthetic", timestamp=0.0
    )


def test_backprojection_roundtrip():
    K = pinhole_intrinsics(52.0, 120, 160)
    T = opencv_cam_to_world(CAM_POS, np.eye(3))
    depth = np.full((120, 160), 1.18, np.float32)
    pts = camera_to_world(backproject_depth(depth, K), T)
    assert np.allclose(pts[..., 2], FLOOR, atol=1e-6)
    uv = project_to_pixels(pts.reshape(-1, 3), K, T).reshape(120, 160, 2)
    v, u = np.mgrid[0:120, 0:160]
    assert np.allclose(uv[..., 0], u, atol=1e-6) and np.allclose(uv[..., 1], v, atol=1e-6)


def test_multi_object_detection_and_3d_position():
    truth = {
        "milk": (0.05, -0.2),
        "bread": (0.15, -0.3),
        "cereal": (0.0, -0.35),
        "can": (0.2, -0.1),
    }
    frame = make_frame([(COLORS[n], xy, 0.021, FLOOR + 0.08) for n, xy in truth.items()])
    res = VisionPerception().detect(frame)
    assert res.missing == []
    for name, xy in truth.items():
        det = res.get(name)
        assert det.category == name
        assert np.allclose(det.position[:2], xy, atol=0.004), (name, det.position)
        assert det.top_z == pytest.approx(FLOOR + 0.08, abs=0.003)
        assert det.position[2] == pytest.approx(FLOOR + 0.04, abs=0.003)
        assert det.confidence > 0.8
        assert det.bbox is not None and det.bbox[2] > 0
        assert det.source == "vision"


def test_missing_object_reported():
    frame = make_frame([(COLORS["milk"], (0.05, -0.2), 0.021, FLOOR + 0.08)])
    res = VisionPerception().detect(frame)
    assert set(res.missing) == {"bread", "cereal", "can"}


def test_flat_marker_is_not_an_object():
    # same colour as milk but only 3 mm above the floor (a container marker)
    frame = make_frame([(COLORS["milk"], (0.05, 0.2), 0.03, FLOOR + 0.003)])
    assert VisionPerception().detect(frame).get("milk") is None


def test_partial_occlusion_is_flagged():
    # Grey box 20 cm above the floor hiding ~70 % of the milk's top face (perspective included).
    occluder = ((200, 200, 200), (0.05, -0.194), 0.025, FLOOR + 0.2)
    frame = make_frame([(COLORS["milk"], (0.05, -0.2), 0.021, FLOOR + 0.08), occluder], size=300)
    det = VisionPerception().detect(frame).get("milk")
    assert det is not None
    assert "partially_occluded" in det.flags
    assert det.confidence < 0.5


def test_invalid_depth_is_handled():
    frame = make_frame([(COLORS["can"], (0.2, -0.1), 0.021, FLOOR + 0.08)])
    frame.depth[:, : frame.depth.shape[1] // 2] = np.nan  # sensor dropout on the left half
    frame.depth[0, -1] = np.inf
    res = VisionPerception().detect(frame)
    assert res.get("can") is not None  # the can is on the right half
    assert np.isfinite(res.get("can").position).all()
    assert 0.4 < res.debug["valid_depth_fraction"] < 0.6


def test_object_with_all_depth_invalid_is_missing():
    frame = make_frame([(COLORS["bread"], (0.15, -0.3), 0.021, FLOOR + 0.06)])
    frame.depth[:] = 0.0
    assert "bread" in VisionPerception().detect(frame).missing


def test_mismatched_shapes_raise():
    frame = make_frame([])
    frame.depth = frame.depth[:-1]
    with pytest.raises(ValueError):
        VisionPerception().detect(frame)


def test_valid_depth_mask():
    d = np.array([np.nan, 0.05, 1.0, 5.0, np.inf])
    assert valid_depth_mask(d, 0.1, 3.0).tolist() == [False, False, True, False, False]


def test_grasp_yaw_footprint():
    rect = np.array([[-0.04, -0.02], [0.04, -0.02], [0.04, 0.02], [-0.04, 0.02]])
    yaw, aspect = grasp_yaw_from_footprint(rect)
    assert aspect == pytest.approx(2.0, rel=1e-3)
    assert yaw == pytest.approx(
        0.0, abs=1e-5
    )  # fingers (closing along y at yaw 0) span the short side
    yaw, _ = grasp_yaw_from_footprint(rect[:, ::-1])
    assert abs(abs(yaw) - np.pi / 2) < 1e-5
    th = 0.3
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    square = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * 0.02
    yaw, aspect = grasp_yaw_from_footprint(square @ R.T)
    assert aspect == pytest.approx(1.0, abs=1e-3)
    assert yaw == pytest.approx(th, abs=1e-4)


def test_vision_detector_has_no_simulator_handle():
    """Vision mode must not be able to read object state: the detector only holds config."""
    det = VisionPerception()
    assert not any(hasattr(det, a) for a in ("env", "sim", "_env"))
