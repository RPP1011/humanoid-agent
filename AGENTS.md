# humanoid-agent — agent guide

An LLM plans over a fixed library of blocking skills; the skills drive a MuJoCo
simulation of a Unitree G1 (with hands). The LLM never sees joints. This file is
the entry point; each package directory has its own `AGENTS.md` with the details
for that layer. Read the one for the directory you are editing before you edit.

```
agents/   planner: ScriptedAgent | AnthropicAgent | OpenAICompatAgent
   │  call(tool_name, args) -> SkillResult
tools/    registry.py: the ONE tool list (feeds MCP server + function calling)
   │
skills/   SkillLibrary: blocking skills, each returns SkillResult(success, reason, data)
   │
control/  LocomotionController (KinematicGlide placeholder) · arm DLS IK
   │
sim/      HumanoidEnv (MjData wrapper, magic-grasp attachments) · scene builder (MjSpec)

eval/     tasks (ground-truth success checks) + runner (results.jsonl, summary.md)
server/   MCP stdio server: same tools + list_tasks / reset_task / check_task
```

Dependency direction is strictly downward: `sim` ← `control` ← `skills` ← `tools`
← `agents`/`eval`/`server`. (`control/locomotion.py` takes `env` untyped to avoid
an import cycle with `sim/env.py`; `control/arm_ik.py` does import `sim.env`.)

## Setup and commands

```bash
scripts/fetch_assets.sh                      # G1 model -> assets/unitree_g1 (gitignored, ~sparse clone of mujoco_menagerie)
pip install -e ".[dev]"                      # add ,anthropic / ,openai / ,policy as needed

# Rendering backend MUST work before anything imports sim/env.py.
#   GPU: MUJOCO_GL=egl (env.py defaults to this if unset)
#   CPU: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa   (apt: libosmesa6)
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa pytest -q            # 27 tests, ~70 s
python -m humanoid_agent.eval.runner --agent scripted --seeds 5  # harness check
python -m humanoid_agent.server.mcp_server                        # MCP over stdio
```

If pytest dies at collection with `AttributeError: 'NoneType' object has no
attribute 'eglQueryString'` (or `glGetError`), the GL backend is missing, not the
code: switch to osmesa and install `libosmesa6`.

## Ground rules

1. **Scripted plans are the regression gate.** `tests/test_smoke.py` runs every
   plan in `agents/scripted.py` on the canonical layout (seed 0). Any change to
   `sim/`, `control/` or `skills/` must keep it green. If a scripted plan fails,
   the bug is in skills/sim; never "fix" it by editing the prompt or the plan.
2. **One tool surface.** Add or change planner-visible tools only in
   `tools/registry.py`. The MCP server and both LLM agents read from it.
3. **Every skill returns a `SkillResult`, never raises.** `reason` is
   `"<code>: <detail>"`; the code (prefix before `:`) is aggregated by the eval
   runner, so treat codes as a stable API. See `skills/AGENTS.md`.
4. **Success is judged on sim state, not on what the agent says.** Only query
   and refusal tasks read `finish(answer)`, and they also require nothing moved.
5. **Placeholders are labelled, keep them labelled.** Locomotion
   (`KinematicGlide`) and grasping (`env.attach`) are faked; the bean bag is an
   approximation (sprung cell bed, not a deformable); the walking person is a
   capsule mannequin on a scripted path with no collision geometry. `is_physical=False`
   flows into every eval row and summary. Any new fake must do the same.
6. **The LLM never emits joint targets or velocities.** Keep composite skills
   (`pick`, `place_on`) available; low-level ones (`reach`, `grasp`) are extras.

## Conventions

- Python ≥3.10, `from __future__ import annotations` at the top of every module.
- Units: metres, radians, seconds of **sim** time. Poses are `(x, y, yaw)` in the
  world frame; "body frame" means (forward, left) relative to the robot base.
- Sides are the strings `"left"` / `"right"`. Object/table/location names are
  plain strings defined in `sim/scene.py` or a task's `scene()`.
- Numbers returned to the planner are rounded (3 dp for poses) to keep tool
  results short; do the same in new skills.
- Comments explain *why* (a failure mode, a frame convention), not *what*.
- No formatter/linter is configured; match the surrounding style (dense, short
  helpers, private `_methods` for non-skill internals).

## Where things go

| Change | File(s) | Also update |
|---|---|---|
| New skill | `skills/library.py` (decorate with `@skill`) | `tools/registry.py`; failure codes in `eval/runner.py:PLANNER_CODES` if planner-caused |
| New task | `eval/tasks.py` | a plan in `agents/scripted.py` (the smoke test runs every task) |
| New scene layout | a `scene=` callable on the `Task` | locations for every table the planner must stand at |
| New kind of support (bag, shelf) | `sim/scene.py` + its geoms in `env._support_name` | `place_on` branch in `skills/library.py`; see `sim/AGENTS.md` (Supports) |
| Real walking policy | `control/locomotion.py:PolicyLocomotion` | expect `pick`/`place_on` regressions; see `control/AGENTS.md` |
| Real grasping | `sim/env.py:attach` + `skills/library.py:grasp` | `pick` descent offsets |
| New LLM backend | `agents/llm.py` | `eval/runner.py:make_agent` |

## Outputs

`runs/<stamp>_<agent>/` (gitignored): `results.jsonl` (one row per episode, full
skill log), `summary.md`, and `<task>_s<seed>_final.png` overview renders.
