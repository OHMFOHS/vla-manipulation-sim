"""Data types shared by the oracle and vision perception backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Detection:
    """One detected object instance.

    ``position`` is the estimated object centre in the world frame and ``top_z`` the height of
    its top surface - the two quantities the grasp planner needs.
    """

    category: str
    position: np.ndarray
    top_z: float
    confidence: float
    yaw: float = 0.0
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h (pixels)
    pixel_count: int = 0
    visible_ratio: float = 1.0
    source: str = "vision"
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "position": [round(float(v), 4) for v in self.position],
            "top_z": round(float(self.top_z), 4),
            "yaw_deg": round(float(np.degrees(self.yaw)), 1),
            "confidence": round(float(self.confidence), 3),
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "pixel_count": int(self.pixel_count),
            "visible_ratio": round(float(self.visible_ratio), 3),
            "source": self.source,
            "flags": list(self.flags),
        }


@dataclass
class PerceptionResult:
    mode: str
    detections: dict[str, Detection]
    missing: list[str]
    timestamp: float = 0.0
    debug: dict = field(default_factory=dict)

    def get(self, category: str) -> Detection | None:
        return self.detections.get(category)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "timestamp": self.timestamp,
            "detections": {k: d.to_dict() for k, d in self.detections.items()},
            "missing": list(self.missing),
        }


class PerceptionBackend(Protocol):
    """Anything that can produce a :class:`PerceptionResult` of the current scene."""

    mode: str

    def perceive(self) -> PerceptionResult: ...
