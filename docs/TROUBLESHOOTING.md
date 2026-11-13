# Troubleshooting

### `robosuite WARNING No private macro file found!`
Harmless.  robosuite prints it on import; the project does not need a private macro file.

### GUI on macOS
`uv run python scripts/run_demo.py --gui` opens an OpenCV window with the live front camera
(press `q` to close it).  This works in both Vision and Oracle mode and is the recommended GUI
on macOS.

The interactive MuJoCo viewer (`mujoco.viewer.launch_passive`) requires `mjpython` on macOS.
Two caveats were verified on macOS 27 / Apple Silicon:

1. uv's standalone Python keeps `libpython3.11.dylib` in its own `lib/` directory, so
   `uv run mjpython ...` fails with `Library not loaded: @rpath/libpython3.11.dylib`.
   `scripts/mjpython.sh` sets `DYLD_LIBRARY_PATH` and launches mjpython for you.
2. Under mjpython, robosuite's offscreen renderer crashes (`NSWindow should only be
   instantiated on the main thread`) because it opens a hidden GLFW window off the main
   thread.  `SimEnv` therefore disables offscreen rendering under the native viewer, so only
   Oracle perception is available there:

```bash
./scripts/mjpython.sh scripts/run_demo.py --gui --perception oracle
```

On Linux and Windows `--gui` selects the native viewer with the normal interpreter (this path
was not tested by the author; only headless Linux is exercised in CI).  Headless runs
(`evaluate.py`, tests) need no display at all.

### Black images / NaN depth when several environments live in one process
robosuite makes its offscreen OpenGL context current only once, at creation.  Creating and
closing another environment can leave a dead context current.  `SimEnv` re-binds its own
context before every render (`_make_render_context_current`); keep using `SimEnv.capture_rgbd`
/ `render_rgb` instead of calling `env.sim.render` directly.

### Linux / CI rendering
Use software rendering:

```bash
sudo apt-get install -y libosmesa6 libosmesa6-dev libgl1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
```

`MUJOCO_GL=egl` also works on machines with an EGL driver.

### `Checkpoint ... not found`
`C_vision_bc` and `run_demo.py --controller bc` need a trained policy:

```bash
uv run python scripts/collect_demos.py --episodes 100
uv run python scripts/train_bc.py
```

### The LLM planner falls back to the rule-based planner
`--planner auto` silently uses the rule-based parser when the `anthropic` package or
credentials are missing.  Install the extra (`uv sync --extra llm`) and set
`ANTHROPIC_API_KEY` (see `.env.example`), or use `--planner llm` to get the error instead.

### A far-left (can) placement looks close to the divider
The far quadrants of the target bin are at the edge of the Panda's reach from its base at
`(-0.5, -0.1)`.  `select_place_point` pulls the release point toward the robot, inside the
quadrant.  If you move the bins, update `Workspace.robot_base_xy` / `comfortable_reach`
in `simulation/scene.py`.

### Dependency versions
The lock file pins robosuite 1.5.2 with MuJoCo 3.3.7 (robosuite 1.5.x requires
`mujoco>=3.3.0`; MuJoCo 4.x was not validated and is excluded by `mujoco<3.4`).  NumPy is
held below 2.3 for numba compatibility.  Always install with `uv sync` so `uv.lock` is used.
