"""RGB-D object detection and 3D localisation with OpenCV.

Vision mode receives *only* an :class:`~robotics.simulation.env.RGBDFrame` (colour image, metric
depth and camera calibration).  It never touches the simulator, so it cannot read ground-truth
object poses.

Pipeline (per frame):

1. Validate depth and back-project every pixel to a world-frame point cloud.
2. Keep pixels that rise above the support surface (rejects bin floor, floor markers, texture).
3. Per category: HSV colour threshold → morphology → connected components.
4. Per component: 3D points → top surface (high depth percentile) → centre, yaw, bbox.
5. Score confidence from visible area vs. the expected top-face area (occlusion) and from
   the fraction of valid depth pixels.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from robotics.common.config import ColorClass, PerceptionConfig
from robotics.perception.geometry import depth_to_world, grasp_yaw_from_footprint, valid_depth_mask
from robotics.perception.types import Detection, PerceptionResult
from robotics.simulation.scene import BIN_FLOOR_Z, OBJECT_CATALOG

logger = logging.getLogger(__name__)

# Default HSV thresholds (OpenCV ranges: H∈[0,180), S,V∈[0,255]).  High V rejects the dark bin
# wood, high S rejects the light wood and the white robot.
DEFAULT_CLASSES: list[ColorClass] = [
    ColorClass(name="milk", hsv_low=(98, 110, 140), hsv_high=(125, 255, 255)),
    ColorClass(name="bread", hsv_low=(22, 120, 150), hsv_high=(38, 255, 255)),
    ColorClass(
        name="cereal",
        hsv_low=(0, 110, 150),
        hsv_high=(7, 255, 255),
        hsv_low2=(172, 110, 150),
        hsv_high2=(180, 255, 255),
    ),
    ColorClass(name="can", hsv_low=(55, 110, 120), hsv_high=(80, 255, 255)),
]

# Top-face area of each category (m²) - used only to estimate how much of an object is visible.
_TOP_AREA = {
    s.name: (4 * s.size[0] * s.size[1]) if s.shape == "box" else (np.pi * s.size[0] ** 2)
    for s in OBJECT_CATALOG
}


class VisionPerception:
    """Colour + depth object detector.  Stateless apart from its configuration."""

    mode = "vision"

    def __init__(self, config: PerceptionConfig | None = None, support_z: float = BIN_FLOOR_Z):
        self.cfg = config or PerceptionConfig()
        self.classes = self.cfg.classes or DEFAULT_CLASSES
        self.support_z = support_z
        k = max(1, int(self.cfg.morph_kernel))
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ------------------------------------------------------------------ public API
    def detect(self, frame) -> PerceptionResult:
        rgb, depth = frame.rgb, frame.depth
        if rgb.ndim != 3 or rgb.shape[:2] != depth.shape[:2]:
            raise ValueError(f"RGB {rgb.shape} and depth {depth.shape} sizes do not match")
        depth = depth.reshape(depth.shape[0], depth.shape[1])
        valid = valid_depth_mask(depth, self.cfg.min_valid_depth, self.cfg.max_valid_depth)
        safe_depth = np.where(valid, depth, 0.0)
        world = depth_to_world(safe_depth, frame.intrinsics, frame.cam_to_world)
        height = world[..., 2] - self.support_z
        raised = valid & (height > self.cfg.min_height_above_floor) & (height < 0.45)

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        detections: dict[str, Detection] = {}
        missing: list[str] = []
        masks: dict[str, np.ndarray] = {}
        for cls in self.classes:
            color_mask = self._color_mask(hsv, cls)
            # Pixels with the right colour but invalid depth are counted to flag sensor dropout.
            invalid_colored = int(np.count_nonzero(color_mask & ~valid))
            mask = color_mask & raised
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, self._kernel).astype(
                bool
            )
            masks[cls.name] = mask
            det = self._best_component(cls.name, mask, world, frame, invalid_colored)
            if det is None:
                missing.append(cls.name)
            else:
                detections[cls.name] = det
        return PerceptionResult(
            mode=self.mode,
            detections=detections,
            missing=missing,
            timestamp=float(getattr(frame, "timestamp", 0.0)),
            debug={"masks": masks, "valid_depth_fraction": float(valid.mean())},
        )

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _color_mask(hsv: np.ndarray, cls: ColorClass) -> np.ndarray:
        m = cv2.inRange(hsv, np.array(cls.hsv_low, np.uint8), np.array(cls.hsv_high, np.uint8)) > 0
        if cls.hsv_low2 is not None and cls.hsv_high2 is not None:
            m |= (
                cv2.inRange(
                    hsv, np.array(cls.hsv_low2, np.uint8), np.array(cls.hsv_high2, np.uint8)
                )
                > 0
            )
        return m

    def _best_component(self, name, mask, world, frame, invalid_colored) -> Detection | None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if n <= 1:
            return None
        # Largest component wins; others are reported through a flag (e.g. split by occlusion).
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(areas)[::-1]
        best = 1 + int(order[0])
        area = int(stats[best, cv2.CC_STAT_AREA])
        if area < self.cfg.min_pixels:
            return None
        flags: list[str] = []
        if len(order) > 1 and areas[order[1]] >= self.cfg.min_pixels:
            flags.append("multiple_candidates")

        comp = labels == best
        pts = world[comp]
        z = pts[:, 2]
        top_z = float(np.percentile(z, self.cfg.top_surface_percentile))
        # Top face = points within 1 cm of the top.  Its centroid is the object's xy centre for
        # an upright object seen from above; side faces (perspective) are excluded this way.
        top_sel = z > top_z - 0.01
        top_pts = pts[top_sel]
        xy = np.median(top_pts[:, :2], axis=0)
        # Objects rest on the support surface, so the geometric centre is halfway up.
        center_z = 0.5 * (top_z + self.support_z) if top_z - self.support_z < 0.2 else top_z - 0.05
        position = np.array([xy[0], xy[1], center_z])

        # Grasp yaw from the footprint of the top face (world xy), see grasp_yaw_from_footprint.
        yaw, aspect = grasp_yaw_from_footprint(top_pts[:, :2]) if len(top_pts) >= 5 else (0.0, 1.0)
        if aspect >= 1.25:
            flags.append("elongated_footprint")

        # Expected top-face pixel count at this depth: area / (pixel footprint)².
        depth_here = float(np.median(np.linalg.norm(top_pts - frame.cam_to_world[:3, 3], axis=1)))
        pixel_size = depth_here / frame.intrinsics[0, 0]
        expected = max(1.0, _TOP_AREA.get(name, 0.0016) / pixel_size**2)
        visible = float(len(top_pts) / expected)
        if visible < self.cfg.occlusion_visible_ratio:
            flags.append("partially_occluded")
        depth_ok = area / max(1, area + invalid_colored)
        if depth_ok < 0.9:
            flags.append("invalid_depth_pixels")
        if top_z - self.support_z > 0.2:
            flags.append("not_on_support_surface")
        confidence = float(np.clip(min(visible, 1.0) * depth_ok, 0.0, 1.0))

        x, y, w, h = (int(v) for v in stats[best, :4])
        return Detection(
            category=name,
            position=position,
            top_z=top_z,
            confidence=confidence,
            yaw=float(yaw),
            bbox=(x, y, w, h),
            pixel_count=area,
            visible_ratio=visible,
            source="vision",
            flags=flags,
        )


class VisionPerceptionBackend:
    """Glue between the simulator camera and :class:`VisionPerception`.

    The backend only calls ``env.capture_rgbd()`` - a sensor read - and hands the frame to the
    detector; it never reads object state.
    """

    mode = "vision"

    def __init__(self, env, config: PerceptionConfig | None = None, camera: str | None = None):
        self._capture = env.capture_rgbd  # capture function only: no access to object state
        self.detector = VisionPerception(config)
        self.camera = camera
        self.last_frame = None

    def perceive(self) -> PerceptionResult:
        frame = self._capture(self.camera)
        self.last_frame = frame
        result = self.detector.detect(frame)
        logger.debug("vision: detected=%s missing=%s", sorted(result.detections), result.missing)
        return result
