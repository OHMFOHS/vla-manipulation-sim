"""robosuite / MuJoCo environment for the language-guided sorting task.

``ColorSortPickPlace`` subclasses robosuite's ``PickPlace`` (Panda + two bins, four object
categories, four target quadrants) and only swaps the textured meshes for colour-coded
primitives and adds a calibrated overhead RGB-D camera.  Task semantics, bins, physics and the
built-in success check are inherited unchanged.

``SimEnv`` is a thin wrapper that exposes exactly what the rest of the stack needs:

* ``step`` / ``reset`` with explicit seeds,
* proprioception (end-effector pose, gripper opening),
* ``capture_rgbd`` - sensor data for the vision pipeline (images + camera calibration only),
* ``oracle_*`` methods - privileged simulator state.  These are used by the Oracle perception
  mode and by the evaluator for ground-truth scoring, and are never passed to Vision mode.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass

import numpy as np

from robotics.common.config import EnvConfig
from robotics.simulation.scene import OBJECT_CATALOG, OBJECT_NAMES, designated_container

logger = logging.getLogger(__name__)

# Import robosuite lazily-ish but at module level: it prints macro warnings on import.
import robosuite  # noqa: E402
from robosuite.environments.manipulation.pick_place import PickPlace  # noqa: E402
from robosuite.models.objects import BoxObject, CylinderObject  # noqa: E402
from robosuite.utils.mjcf_utils import new_element  # noqa: E402

logging.getLogger("robosuite_logs").setLevel(logging.WARNING)


def _make_primitive(spec, name: str, visual: bool, friction, density):
    kwargs: dict = {"name": name, "size": list(spec.size), "rgba": list(spec.rgba)}
    if visual:
        # Flat translucent marker on the floor of the designated quadrant (no collision).
        kwargs.update(obj_type="visual", joints=None)
        size = list(spec.size)
        size = [0.03, 0.03, 0.0115] if spec.shape == "box" else [0.03, 0.0115]
        kwargs["size"] = size
        kwargs["rgba"] = [*spec.rgba[:3], 0.5]
    else:
        kwargs.update(friction=list(friction), density=density)
    cls = BoxObject if spec.shape == "box" else CylinderObject
    return cls(**kwargs)


class ColorSortPickPlace(PickPlace):
    """robosuite PickPlace with colour-coded primitive objects and an overhead RGB-D camera."""

    def __init__(
        self,
        *args,
        overhead_camera_pos=(0.1, 0.015, 2.0),
        overhead_camera_fovy: float = 52.0,
        object_friction=(1.0, 0.005, 0.0001),
        object_density: float = 500.0,
        **kwargs,
    ):
        self._overhead_pos = tuple(overhead_camera_pos)
        self._overhead_fovy = float(overhead_camera_fovy)
        self._object_friction = tuple(object_friction)
        self._object_density = float(object_density)
        super().__init__(*args, **kwargs)

    def _construct_visual_objects(self):
        self.visual_objects = [
            _make_primitive(spec, "Visual" + spec.sim_name, True, None, None)
            for spec in OBJECT_CATALOG
        ]

    def _construct_objects(self):
        self.objects = [
            _make_primitive(spec, spec.sim_name, False, self._object_friction, self._object_density)
            for spec in OBJECT_CATALOG
        ]

    def _load_model(self):
        super()._load_model()
        # Straight-down camera above both bins.  MuJoCo cameras look along their local -z axis,
        # so the identity orientation points the camera at the table with image-up = world +y.
        self.model.worldbody.append(
            new_element(
                tag="camera",
                name="overhead",
                mode="fixed",
                pos=" ".join(str(v) for v in self._overhead_pos),
                quat="1 0 0 0",
                fovy=str(self._overhead_fovy),
            )
        )


def linearize_depth(depth01: np.ndarray, extent: float, znear: float, zfar: float) -> np.ndarray:
    """Convert MuJoCo's normalised [0, 1] z-buffer into metric depth (m).

    MuJoCo stores near/far planes as fractions of the model extent; the z-buffer is
    non-linear: d = near / (1 - z * (1 - near / far)).
    """
    near = znear * extent
    far = zfar * extent
    return (near / (1.0 - depth01 * (1.0 - near / far))).astype(np.float32)


def pinhole_intrinsics(fovy_deg: float, height: int, width: int) -> np.ndarray:
    """MuJoCo cameras have square pixels and a vertical field of view ``fovy``."""
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def opencv_cam_to_world(cam_pos: np.ndarray, cam_rot: np.ndarray) -> np.ndarray:
    """MuJoCo camera frame (x right, y up, looking along -z) -> OpenCV frame (x right, y down, z fwd)."""
    T = np.eye(4)
    T[:3, :3] = cam_rot @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = cam_pos
    return T


@dataclass
class RGBDFrame:
    """One calibrated RGB-D observation.  This is everything Vision mode is allowed to see."""

    rgb: np.ndarray  # (H, W, 3) uint8, row 0 = top of the image
    depth: np.ndarray  # (H, W) float32 metric depth along the optical axis (m)
    intrinsics: np.ndarray  # (3, 3) pinhole K
    cam_to_world: np.ndarray  # (4, 4) OpenCV-convention camera pose (z forward, y down)
    camera_name: str
    timestamp: float


def running_under_mjpython() -> bool:
    return "mjpython" in os.path.basename(sys.executable) or bool(os.environ.get("MJPYTHON_BIN"))


class SimEnv:
    """Seeded wrapper around :class:`ColorSortPickPlace`."""

    def __init__(
        self, config: EnvConfig | None = None, gui: bool = False, gui_backend: str | None = None
    ):
        """
        Args:
            gui: open a live viewer.
            gui_backend: ``mjviewer`` (interactive MuJoCo viewer; on macOS this requires launching
                the script with ``mjpython``) or ``opencv`` (a plain OpenCV window that works with
                the normal interpreter).  Chosen automatically when ``None``.
        """
        self.config = config or EnvConfig()
        self.gui = gui
        if gui and gui_backend is None:
            gui_backend = (
                "mjviewer" if (sys.platform != "darwin" or running_under_mjpython()) else "opencv"
            )
            if gui_backend == "opencv":
                logger.warning(
                    "macOS without mjpython: showing the simulation in an OpenCV window. "
                    "For the interactive MuJoCo viewer run: uv run mjpython scripts/run_demo.py --gui"
                )
        self.gui_backend = gui_backend if gui else None
        use_mjviewer = self.gui_backend == "mjviewer"
        cfg = self.config
        self.env = ColorSortPickPlace(
            robots="Panda",
            has_renderer=use_mjviewer,
            has_offscreen_renderer=True,
            render_camera=cfg.render_camera,
            use_camera_obs=False,  # images are rendered on demand by capture_rgbd()
            use_object_obs=True,
            control_freq=cfg.control_freq,
            horizon=100_000,
            ignore_done=True,
            hard_reset=cfg.hard_reset,
            initialization_noise={"magnitude": cfg.initialization_noise, "type": "gaussian"},
            renderer="mjviewer",
            overhead_camera_pos=cfg.overhead_camera_pos,
            overhead_camera_fovy=cfg.overhead_camera_fovy,
            object_friction=cfg.object_friction,
            object_density=cfg.object_density,
            seed=0,
        )
        self.robot = self.env.robots[0]
        self.arm = self.robot.arms[0]
        self.control_dt = 1.0 / cfg.control_freq
        self.step_count = 0
        self.seed: int | None = None
        self._obs: dict = {}
        self._wall_start = time.perf_counter()

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int) -> dict:
        """Reset with a fixed seed: object placement and robot init noise are reproducible."""
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.env.rng = rng
        if getattr(self.env, "placement_initializer", None) is not None:
            for sampler in self.env.placement_initializer.samplers.values():
                sampler.rng = rng
        self._obs = self.env.reset()
        self.step_count = 0
        self._wall_start = time.perf_counter()
        # Let objects settle onto the bin floor before anything is perceived.
        for _ in range(10):
            self.step(self.hold_action())
        self.step_count = 0
        return self._obs

    def step(self, action: np.ndarray) -> dict:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_dim,):
            raise ValueError(f"Expected action of shape ({self.action_dim},), got {action.shape}")
        self._obs, _, _, _ = self.env.step(np.clip(action, -1.0, 1.0))
        self.step_count += 1
        if self.gui_backend == "mjviewer":
            self.env.render()
        elif self.gui_backend == "opencv" and self.step_count % 2 == 0:
            self._show_opencv()
        return self._obs

    def _show_opencv(self) -> None:
        import cv2

        frame = self.render_rgb(self.config.render_camera, 480, 640)
        cv2.imshow(
            "vla-manipulation-sim (q to close window)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        )
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            self.gui_backend = None

    def close(self) -> None:
        if self.gui_backend == "opencv":
            import cv2

            cv2.destroyAllWindows()
        try:
            self.env.close()
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.debug("env.close() failed: %s", exc)

    # ------------------------------------------------------------------ robot state
    @property
    def action_dim(self) -> int:
        return int(self.env.action_dim)

    def hold_action(self, gripper: float = -1.0) -> np.ndarray:
        a = np.zeros(self.action_dim)
        a[-1] = gripper
        return a

    @property
    def eef_pos(self) -> np.ndarray:
        return np.array(self._obs["robot0_eef_pos"], dtype=np.float64)

    @property
    def eef_quat(self) -> np.ndarray:
        """(x, y, z, w) quaternion of the end-effector site."""
        return np.array(self._obs["robot0_eef_quat"], dtype=np.float64)

    @property
    def gripper_qpos(self) -> np.ndarray:
        return np.array(self._obs["robot0_gripper_qpos"], dtype=np.float64)

    @property
    def gripper_width(self) -> float:
        """Distance between the two Panda fingers (m).  ~0.08 open, ~0.0 closed on nothing."""
        q = self.gripper_qpos
        return float(q[0] - q[1])

    @property
    def sim_time(self) -> float:
        return self.step_count * self.control_dt

    @property
    def wall_time(self) -> float:
        return time.perf_counter() - self._wall_start

    # ------------------------------------------------------------------ sensors
    def capture_rgbd(
        self, camera: str | None = None, height: int | None = None, width: int | None = None
    ) -> RGBDFrame:
        """Render a calibrated RGB-D frame.  Camera calibration is sensor metadata, not object state."""
        camera = camera or self.config.perception_camera
        h = height or self.config.camera_height
        w = width or self.config.camera_width
        sim = self.env.sim
        self._make_render_context_current()
        rgb, depth = sim.render(width=w, height=h, camera_name=camera, depth=True)
        rgb = np.ascontiguousarray(rgb[::-1])  # OpenGL origin is bottom-left
        depth = np.ascontiguousarray(depth[::-1])
        depth_m = linearize_depth(
            depth, sim.model.stat.extent, sim.model.vis.map.znear, sim.model.vis.map.zfar
        )
        cam_id = sim.model.camera_name2id(camera)
        K = pinhole_intrinsics(float(sim.model.cam_fovy[cam_id]), h, w)
        cam_to_world = opencv_cam_to_world(
            np.array(sim.data.cam_xpos[cam_id]), np.array(sim.data.cam_xmat[cam_id]).reshape(3, 3)
        )
        return RGBDFrame(
            rgb=rgb.astype(np.uint8),
            depth=depth_m,
            intrinsics=K,
            cam_to_world=cam_to_world,
            camera_name=camera,
            timestamp=self.sim_time,
        )

    def _make_render_context_current(self) -> None:
        """robosuite only makes its offscreen GL context current when it is created.  With
        several environments in one process (benchmarks, tests) another env's context may be
        current - or already freed - so we re-bind ours before every render."""
        ctx = getattr(self.env.sim, "_render_context_offscreen", None)
        if ctx is not None:
            ctx.gl_ctx.make_current()

    def render_rgb(
        self, camera: str | None = None, height: int = 360, width: int = 480
    ) -> np.ndarray:
        self._make_render_context_current()
        rgb = self.env.sim.render(
            width=width, height=height, camera_name=camera or self.config.render_camera
        )
        return np.ascontiguousarray(rgb[::-1]).astype(np.uint8)

    # ------------------------------------------------------------------ privileged state
    def oracle_object_position(self, name: str) -> np.ndarray:
        """Ground-truth body position.  ORACLE / EVALUATION USE ONLY."""
        spec = next(s for s in OBJECT_CATALOG if s.name == name)
        bid = self.env.obj_body_id[spec.sim_name]
        return np.array(self.env.sim.data.body_xpos[bid], dtype=np.float64)

    def oracle_object_positions(self) -> dict[str, np.ndarray]:
        return {n: self.oracle_object_position(n) for n in OBJECT_NAMES}

    def oracle_is_in_container(self, name: str, container_name: str | None = None) -> bool:
        """Ground-truth placement check.

        For an object's designated quadrant we use robosuite's own ``not_in_bin`` test so that
        our success metric matches the official PickPlace definition.
        """
        pos = self.oracle_object_position(name)
        idx = OBJECT_NAMES.index(name)
        designated = designated_container(name).name
        if container_name is None or container_name == designated:
            return not self.env.not_in_bin(pos, idx)
        from robotics.simulation.scene import CONTAINERS

        return CONTAINERS[container_name].contains(pos)

    def oracle_task_success(self) -> bool:
        """robosuite's official PickPlace success (all objects in their bins, gripper away)."""
        return bool(self.env._check_success())

    def robot_environment_contacts(self) -> list[tuple[str, str]]:
        """Contacts between robot arm links / gripper and the static scene (bins, table, floor).

        Gripper-object contacts are expected during grasping and are excluded.  Used to compute
        the collision rate metric.
        """
        sim = self.env.sim
        model = sim.model
        robot_geoms = set(self.robot.robot_model.contact_geoms) | set(
            self.robot.gripper[self.arm].contact_geoms
        )
        object_geoms = set()
        for obj in self.env.objects:
            object_geoms |= set(obj.contact_geoms)
        hits: list[tuple[str, str]] = []
        for i in range(sim.data.ncon):
            c = sim.data.contact[i]
            g1 = model.geom_id2name(c.geom1)
            g2 = model.geom_id2name(c.geom2)
            n1, n2 = g1 or "", g2 or ""
            r1, r2 = n1 in robot_geoms, n2 in robot_geoms
            if r1 == r2:
                continue  # robot-robot or env-env
            other = n2 if r1 else n1
            if other in object_geoms:
                continue
            # Panda links resting against their own mount are not collisions.
            if "mount" in other or "base" in other:
                continue
            hits.append((n1, n2))
        return hits


__all__ = ["SimEnv", "RGBDFrame", "ColorSortPickPlace", "robosuite", "running_under_mjpython"]
