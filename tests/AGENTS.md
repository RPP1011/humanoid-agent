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
records the expected baseline. `test_skills.py` holds skill-level guards
(refusals, routing, report names). `test_bean_bag.py` checks the soft support
is not a renamed table: a tower shifted 1 cm per level stands on a table but
falls on the bag (lump seed 1); a centred one stands on it.
