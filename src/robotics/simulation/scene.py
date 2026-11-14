"""Static description of the sorting scene: objects, containers and workspace.

The layout mirrors robosuite's ``PickPlace`` task: objects start in ``bin1`` (the source bin)
and must be sorted into the four quadrants of ``bin2``.  Each quadrant is the designated
container for one object category.  Geometry values are taken from robosuite's
``bins_arena.xml`` (bin floors are 2 cm thick boxes centred at z=0.80, walls are 10 cm tall).

This module contains *static, known-a-priori* information only (what a warehouse operator
would put in a config file).  It never contains live object poses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# robosuite PickPlace defaults
BIN1_POS = np.array([0.1, -0.25, 0.8])
BIN2_POS = np.array([0.1, 0.28, 0.8])
BIN_SIZE = np.array([0.39, 0.49])  # PickPlace uses table_full_size[:2] as the bin footprint
BIN_FLOOR_Z = 0.82  # top surface of the bin floor (0.80 + 0.02 half thickness)
BIN_WALL_TOP_Z = 0.90  # top of bin walls and bin2 dividers


@dataclass(frozen=True)
class ObjectSpec:
    """A graspable object category.

    Attributes:
        name: canonical lower-case name used everywhere in the code base.
        sim_name: body name inside robosuite (PickPlace capitalised names).
        shape: ``box`` or ``cylinder``.
        size: robosuite primitive size (box: half extents xyz; cylinder: radius, half-length).
        rgba: render colour.  Colours are deliberately saturated so that the
            vision pipeline can classify objects with simple HSV thresholds.
        aliases: words that refer to this object in natural language.
    """

    name: str
    sim_name: str
    shape: str
    size: tuple[float, ...]
    rgba: tuple[float, float, float, float]
    aliases: tuple[str, ...] = ()

    @property
    def half_height(self) -> float:
        return float(self.size[2] if self.shape == "box" else self.size[1])


# Order matters: index i is the robosuite object id and the id of its target quadrant in bin2.
OBJECT_CATALOG: tuple[ObjectSpec, ...] = (
    ObjectSpec(
        "milk",
        "Milk",
        "box",
        (0.021, 0.021, 0.045),
        (0.10, 0.35, 0.95, 1.0),  # blue carton
        ("milk", "milk carton", "carton", "dairy"),
    ),
    ObjectSpec(
        "bread",
        "Bread",
        "box",
        (0.022, 0.022, 0.03),
        (0.95, 0.80, 0.10, 1.0),  # yellow loaf
        ("bread", "loaf", "bun", "toast"),
    ),
    ObjectSpec(
        "cereal",
        "Cereal",
        "box",
        (0.021, 0.021, 0.05),
        (0.90, 0.12, 0.12, 1.0),  # red box
        ("cereal", "cereal box", "cornflakes", "box of cereal"),
    ),
    ObjectSpec(
        "can",
        "Can",
        "cylinder",
        (0.021, 0.04),
        (0.10, 0.80, 0.25, 1.0),  # green soda can
        ("can", "soda", "soda can", "coke", "tin", "drink"),
    ),
)

OBJECT_NAMES: tuple[str, ...] = tuple(o.name for o in OBJECT_CATALOG)


def get_object_spec(name: str) -> ObjectSpec:
    for spec in OBJECT_CATALOG:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown object '{name}'. Known objects: {list(OBJECT_NAMES)}")


@dataclass(frozen=True)
class ContainerSpec:
    """An axis-aligned container region (one quadrant of bin2 or the whole source bin)."""

    name: str
    center: tuple[float, float]
    half_extents: tuple[float, float]
    floor_z: float
    aliases: tuple[str, ...] = ()
    designated_for: str | None = None

    def contains(self, xyz: np.ndarray, margin: float = 0.0, max_height: float = 0.12) -> bool:
        """True if ``xyz`` lies inside the container footprint and close to its floor."""
        dx = abs(float(xyz[0]) - self.center[0])
        dy = abs(float(xyz[1]) - self.center[1])
        in_xy = dx < self.half_extents[0] - margin and dy < self.half_extents[1] - margin
        in_z = self.floor_z - 0.02 < float(xyz[2]) < self.floor_z + max_height
        return bool(in_xy and in_z)

    @property
    def center_xyz(self) -> np.ndarray:
        return np.array([self.center[0], self.center[1], self.floor_z])


def _quadrant(index: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Replicates robosuite PickPlace.not_in_bin() quadrant geometry for object id ``index``."""
    x_low, y_low = BIN2_POS[0], BIN2_POS[1]
    if index in (0, 2):
        x_low -= BIN_SIZE[0] / 2
    if index < 2:
        y_low -= BIN_SIZE[1] / 2
    half = (BIN_SIZE[0] / 4, BIN_SIZE[1] / 4)
    return (x_low + half[0], y_low + half[1]), half


# The robot base sits at x=-0.5 looking along +x, so +x is "far" and +y is "left".
_QUADRANT_WORDS = {0: "near right", 1: "far right", 2: "near left", 3: "far left"}


def _build_containers() -> dict[str, ContainerSpec]:
    containers: dict[str, ContainerSpec] = {}
    for i, obj in enumerate(OBJECT_CATALOG):
        center, half = _quadrant(i)
        words = _QUADRANT_WORDS[i]
        name = f"{obj.name}_bin"
        aliases = (
            f"{obj.name} bin",
            f"{obj.name} container",
            f"{obj.name} compartment",
            f"{words} bin",
            f"{words} compartment",
            f"{words.replace(' ', '-')} bin",
        )
        containers[name] = ContainerSpec(
            name=name,
            center=(float(center[0]), float(center[1])),
            half_extents=(float(half[0]), float(half[1])),
            floor_z=BIN_FLOOR_Z,
            aliases=aliases,
            designated_for=obj.name,
        )
    containers["source_bin"] = ContainerSpec(
        name="source_bin",
        center=(float(BIN1_POS[0]), float(BIN1_POS[1])),
        half_extents=(0.19, 0.24),
        floor_z=BIN_FLOOR_Z,
        aliases=("source bin", "starting bin", "first bin", "pick bin"),
        designated_for=None,
    )
    return containers


CONTAINERS: dict[str, ContainerSpec] = _build_containers()


def designated_container(object_name: str) -> ContainerSpec:
    return CONTAINERS[f"{object_name}_bin"]


@dataclass(frozen=True)
class Workspace:
    """Conservative Cartesian box the end-effector is allowed to reach (world frame)."""

    low: tuple[float, float, float] = (-0.15, -0.52, 0.83)
    high: tuple[float, float, float] = (0.32, 0.55, 1.25)
    safe_height: float = 1.03  # transport height: held objects clear the 0.90 m bin walls by >4 cm
    home: tuple[float, float, float] = (-0.10, -0.10, 1.10)
    # Arm pose used while taking pictures: retracted behind the bins so it does not occlude them.
    # Higher poses were tried (e.g. (-0.08, -0.10, 1.25)): they start far-corner grasps from a
    # greater height, which drives the elbow to full extension during the approach.
    perception_pose: tuple[float, float, float] = (-0.22, -0.10, 1.18)
    robot_base_xy: tuple[float, float] = (-0.50, -0.10)
    # Empirical horizontal reach of the Panda (from the base axis) at bin height with a
    # top-down gripper; targets beyond this are reported as unreachable.
    max_reach: float = 0.83
    comfortable_reach: float = 0.79
    extra: dict[str, float] = field(default_factory=dict)

    def contains(self, xyz: np.ndarray, margin: float = 0.0) -> bool:
        lo = np.asarray(self.low) + margin
        hi = np.asarray(self.high) - margin
        return bool(np.all(xyz >= lo) and np.all(xyz <= hi))

    def clip(self, xyz: np.ndarray) -> np.ndarray:
        return np.clip(xyz, self.low, self.high)

    def horizontal_reach(self, xyz: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(xyz[:2]) - np.asarray(self.robot_base_xy)))

    def is_reachable(self, xyz: np.ndarray) -> bool:
        return self.contains(np.asarray(xyz)) and self.horizontal_reach(xyz) <= self.max_reach


WORKSPACE = Workspace()
