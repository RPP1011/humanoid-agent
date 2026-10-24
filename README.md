# humanoid-agent

An LLM plans; a skill library acts; MuJoCo simulates a Unitree G1 (with hands).
The LLM never touches joints. It calls ~14 blocking skills over MCP or function
calling, and an eval harness scores it against ground-truth sim state.

```
LLM (any: Claude, vLLM, llama.cpp, ...)
   │  tool calls (MCP or function calling) — one registry: tools/registry.py
   ▼
SkillLibrary  go_to_location · pick · place_on · look_at · get_frame · reach · grasp · …
   │  every skill returns SkillResult(success, reason code, data)
   ▼
Controllers   LocomotionController (placeholder: KinematicGlide) · arm IK (DLS)
   ▼
MuJoCo        G1 + tables + objects, ego & overview cameras
```

## Quick start

```bash
scripts/fetch_assets.sh          # G1 model from MuJoCo Menagerie
pip install -e ".[anthropic,openai,dev]"

# Harness check: hand-written plans. Should be 100% on seed 0.
python -m humanoid_agent.eval.runner --agent scripted --seeds 5

# Your LLM server (any OpenAI-compatible endpoint with tool calling)
python -m humanoid_agent.eval.runner --agent openai --model <name> --base-url http://localhost:8000/v1 --seeds 3

# Claude
ANTHROPIC_API_KEY=... python -m humanoid_agent.eval.runner --agent anthropic --model claude-sonnet-5-5

# Interactive: MCP server (stdio) for Claude Desktop / Claude Code / any MCP client
python -m humanoid_agent.server.mcp_server
```

Headless rendering: `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` (CPU, needs libosmesa6).
macOS: leave `MUJOCO_GL` unset.

MCP client config:
```json
{"mcpServers": {"humanoid": {"command": "python",
  "args": ["-m", "humanoid_agent.server.mcp_server"], "cwd": "/path/to/humanoid-agent"}}}
```
The server also exposes `list_tasks`, `reset_task(task_id)` and `check_task(answer?)` for manual eval.

## Status — what is real and what is faked

| Component | State |
|---|---|
| G1 model, physics, table/object contact | Real (MuJoCo Menagerie G1 with hands) |
| Arm reaching | Real: Cartesian waypoints → DLS IK → position actuators; collisions matter |
| Placement / retract | Real motion; skill picks a clear spot and side-steps to it |
| **Locomotion** | **FAKE.** `KinematicGlide` overwrites the base pose; legs are frozen. |
| **Grasping** | **FAKE.** "Magic grasp": object is kinematically attached; collisions disabled while held. |
| Perception | Ground truth (`list_objects`). Camera frames are real renders, not used by skills. |

Every eval row and summary carries `locomotion_is_physical`, so placeholder results
cannot be mistaken for physical ones.

## Baseline (scripted plans, kinematic locomotion)

| jitter | result | residual failures |
|---|---|---|
| 3 cm, 5 seeds | 35/35 | — |
| 6 cm, 8 seeds | 54/56 | `approach_tracking` (arm blocked by table/self) |

Run this before any LLM eval. If scripted fails, fix skills, not prompts.

## Eval

`humanoid_agent/eval/tasks.py` — 7 tasks: navigation, single and multi pick-and-place,
grounding ("the other table"), a query task (answer without disturbing anything), and an
impossible task (purple cube) that rewards honest refusal.

Output: `runs/<stamp>_<agent>/results.jsonl` (full skill log per episode),
`summary.md` (success per task; failure codes split into planner errors vs skill/sim errors),
and an overview PNG of each episode's final state.

## Next steps, in order

1. **Run your LLM against the scripted baseline.** Gap between them is the planner's.
2. **Real locomotion.** Implement `PolicyLocomotion._obs/_act` for a G1 velocity-tracking
   policy (mjlab or Unitree RL release, ONNX). Expect pick/place to regress — the arm
   currently assumes a perfectly still base. This is the first real test of the stack.
3. **Real grasping.** Replace `env.attach` with finger closure + friction. Hardest step;
   defer until 1–2 are stable.
4. **Perception.** Replace `list_objects` ground truth with a detector on `get_frame`, behind
   the same signature. Then eval tasks become partially-observable.
5. **Harder tasks.** Clutter, drawers, more objects than hands, distractors.

## Dead ends to avoid

- LLM emitting joint targets or velocities. Latency gap is ~1000× vs balance control.
- Exposing low-level primitives only (`reach`/`grasp`). Keep composites (`pick`, `place_on`) —
  agent performance drops as abstraction is removed.
- Tuning prompts while scripted plans fail.

## Layout

```
humanoid_agent/
  sim/scene.py         MjSpec scene builder (tables, objects, cameras, named locations)
  sim/env.py           HumanoidEnv: stepping, state, attachments, rendering
  control/locomotion.py  LocomotionController ABC, KinematicGlide placeholder, PolicyLocomotion slot
  control/arm_ik.py    damped least-squares palm IK
  skills/library.py    the planner's action space
  tools/registry.py    single tool registry → MCP + function calling
  agents/              scripted plans, AnthropicAgent, OpenAICompatAgent
  eval/                tasks + runner
  server/mcp_server.py MCP stdio server (mcp>=2.3)
tests/test_smoke.py    scripted plans must pass
```

Coding agents: start at `AGENTS.md` (each package has its own `AGENTS.md`; `CLAUDE.md` files import them).
