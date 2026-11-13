"""Oracle perception: privileged simulator state (debugging tool and performance upper bound)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from robotics.perception.geometry import grasp_yaw_from_footprint
from robotics.perception.types import Detection, PerceptionResult
from robotics.simulation.scene import OBJECT_CATALOG


def object_extent_z(spec, rot: Rotation) -> float:
    """Half extent along world z of a box / cylinder with orientation ``rot``.

    Needed so that a toppled object gets the correct top surface height.
    """
    R = rot.as_matrix()
    if spec.shape == "box":
        return float(np.sum(np.abs(R[2, :]) * np.asarray(spec.size)))
    r, hl = spec.size
    axis_z = abs(R[2, 2])  # cylinder axis is local z
    return float(hl * axis_z + r * np.sqrt(max(0.0, 1 - axis_z**2)))


def footprint_points(spec, pos: np.ndarray, rot: Rotation) -> np.ndarray:
    """World xy points outlining a box / cylinder (8 corners or two 16-point rings)."""
    if spec.shape == "box":
        sx, sy, sz = spec.size
        local = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
    else:
        r, hl = spec.size
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ring = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        local = np.concatenate([np.c_[ring, np.full(16, -hl)], np.c_[ring, np.full(16, hl)]])
    return (rot.apply(local) + pos)[:, :2]


class OraclePerception:
    """Reads object poses directly from MuJoCo.  Never used in Vision mode."""

    mode = "oracle"

    def __init__(self, env):
        self.env = env

    def perceive(self) -> PerceptionResult:
        sim = self.env.env.sim
        detections: dict[str, Detection] = {}
        for spec in OBJECT_CATALOG:
            bid = self.env.env.obj_body_id[spec.sim_name]
            pos = np.array(sim.data.body_xpos[bid], dtype=np.float64)
            w, x, y, z = sim.data.body_xquat[bid]
            rot = Rotation.from_quat([x, y, z, w])
            R = rot.as_matrix()
            yaw, _ = grasp_yaw_from_footprint(footprint_points(spec, pos, rot))
            flags = [] if abs(R[2, 2]) > 0.9 else ["toppled"]
            detections[spec.name] = Detection(
                category=spec.name,
                position=pos,
                top_z=float(pos[2] + object_extent_z(spec, rot)),
                confidence=1.0,
                yaw=yaw,
                source="oracle",
                flags=flags,
            )
        return PerceptionResult(
            mode=self.mode, detections=detections, missing=[], timestamp=self.env.sim_time
        )
