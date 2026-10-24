# server/ — MCP stdio server

`python -m humanoid_agent.server.mcp_server [--task TASK_ID]` (or the
`humanoid-mcp` console script). Uses the **low-level** `mcp.server.lowlevel.Server`
API from `mcp>=2.3` (handlers passed as `on_list_tools` / `on_call_tool`
constructor args). That API changed across 2.x; if you bump `mcp`, check the
constructor and handler signatures first.

## Behaviour
- One `Session` per process: an env, a `SkillLibrary`, and an `Episode`.
  `--task` loads that task's scene; otherwise the default scene ("free play").
- Tools = every `ToolSpec` in `tools/registry.py` + three control tools:
  `list_tasks`, `reset_task(task_id?)` (rebuilds the env, returns the prompt),
  `check_task(answer?)` (runs the task's ground-truth checks and lists each
  condition as met or not).
  `finish` is **not** exposed.
- Sim work runs in a worker thread (`anyio.to_thread.run_sync`) so the event
  loop stays responsive. Results carry an `ImageContent` block when the skill
  returned `png_bytes`, plus the JSON text; `is_error` = not success.

## Caveats
- No lock around the session: if a client issues tool calls concurrently, two
  threads would step the same `MjData`. Clients that call tools one at a time
  are fine. Add an `anyio.Lock` if that changes.
- There is no tool-call budget here; budgets only exist in the eval runner.
- Rendering needs a working `MUJOCO_GL` in the server's environment (put it in
  the MCP client config `env`).
