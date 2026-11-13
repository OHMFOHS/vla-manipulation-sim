"""Overlay perception results on camera images."""

from __future__ import annotations

import cv2
import numpy as np

from robotics.perception.types import PerceptionResult

_COLORS = {
    "milk": (40, 90, 255),
    "bread": (255, 210, 40),
    "cereal": (235, 40, 40),
    "can": (40, 210, 70),
}


def draw_detections(rgb: np.ndarray, result: PerceptionResult, scale: int = 2) -> np.ndarray:
    """Return an upscaled RGB image with boxes, labels, 3D positions and confidences."""
    img = cv2.resize(
        rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_NEAREST
    )
    for name, det in result.detections.items():
        color = _COLORS.get(name, (255, 255, 255))
        if det.bbox is None:
            continue
        x, y, w, h = (v * scale for v in det.bbox)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label = f"{name} {det.confidence:.2f}"
        pos = "({:.2f},{:.2f},{:.2f})".format(*det.position)
        for i, text in enumerate((label, pos)):
            org = (x, max(12, y - 6 - 14 * (1 - i)))
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    if result.missing:
        cv2.putText(
            img,
            "missing: " + ", ".join(result.missing),
            (8, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return img


def depth_colormap(depth: np.ndarray, near: float = 0.9, far: float = 1.3) -> np.ndarray:
    d = np.clip((depth - near) / (far - near), 0, 1)
    d = (255 * (1 - d)).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(d, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
