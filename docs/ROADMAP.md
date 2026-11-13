# Roadmap

## Delivered

| Phase | Scope | Status |
|---|---|---|
| 1 · Environment | uv project, locked robosuite 1.5.2 / MuJoCo 3.3.7, Panda PickPlace, headless + macOS GUI | done |
| 2 · Manipulation | OSC_POSE controller, min-jerk trajectories, scripted pick & place, physical grasp checks | done |
| 3 · Embodied AI | language → TaskPlan, RGB-D perception (vision / oracle), task planner, closed-loop verification, recovery | done |
| 4 · Learning | scripted-expert demos with DART noise, MLP behaviour cloning, train/val by episode, policy comparison | done |
| 5 · Portfolio | 50-episode benchmark, plots, demo GIF, README, GitHub Actions | done |

## Next steps (not implemented)

1. **ACT (Action Chunking with Transformers).**  Replace the per-step MLP with a chunked
   transformer policy over a short observation history; expected to reduce the
   phase-boundary errors seen in BC (see EXPERIMENTS.md).
2. **Image-conditioned policy.**  Feed the overhead RGB-D crop (or a frozen visual encoder)
   instead of perceived target coordinates, to learn perception and control end to end.
3. **Learned detector.**  Train a small segmentation network on rendered images so the
   original textured robosuite meshes can be used in Vision mode.
4. **Domain randomisation.**  Lighting, textures, camera pose jitter and object sizes, to
   measure robustness of the HSV pipeline and of BC.
5. **Motion planning.**  Swap the structured "lift–transport–lower" strategy for a
   sampling-based planner (e.g. OMPL / curobo) when clutter grows.
6. **Additional robots / tasks.**  robosuite's other arms (Sawyer, UR5e) and tasks
   (NutAssembly, Stack) behind the same `Skill` interface.
7. **LLM evaluation set.**  A labelled instruction set to compare the rule-based and LLM
   planners on accuracy and refusal behaviour.
