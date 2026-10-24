# tests/

`test_smoke.py` runs every scripted plan (`agents/scripted.py`) against its task
on the canonical layout (seed 0) through the real `eval.runner.run_episode`, and
asserts ground-truth success. It needs the G1 assets and a working GL backend:

```bash
scripts/fetch_assets.sh
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa pytest -q    # or MUJOCO_GL=egl on a GPU box
```

A failure here after a change to `sim/`, `control/` or `skills/` means the
skill layer broke. Fix the skill, don't edit the plan to route around it.

For robustness beyond seed 0, run the jittered scripted eval
(`python -m humanoid_agent.eval.runner --agent scripted --seeds 5`); the README
records the expected baseline. There are no unit tests for individual skills
or IK yet; if you add some, build a `HumanoidEnv` once per module (model build
is the slow part).
