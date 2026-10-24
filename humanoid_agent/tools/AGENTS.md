# tools/ — the single tool registry

`registry.py` is the only place planner-visible tools are defined. The MCP
server (`server/mcp_server.py`) and both LLM agents (`agents/llm.py`) build
their tool lists from `TOOLS` (+ `FINISH_TOOL` for agents), so an MCP client and
the eval harness see identical tools.

## Pieces

- `ToolSpec(name, description, params, required, call)` — `params` is the
  JSON-schema `properties` dict; `input_schema()` wraps it. `call` is
  `lambda L, **kwargs: L.<skill>(...)` and is where optional-arg defaults live.
- `TOOLS` — 14 skill tools, in the order models see them.
- `FINISH_TOOL` — not a skill. Ends the episode with an answer. The eval runner
  intercepts `finish` before it reaches `call_tool`; the MCP server does not
  expose it (humans use `check_task` instead).
- `call_tool(lib, name, args)` — dispatch; unknown names → `unknown_tool`,
  wrong/missing kwargs (`TypeError`) → `bad_args`.
- `result_to_text` — JSON of `SkillResult.to_dict()` with `png_bytes` stripped.
- `result_to_anthropic_content` — image block (if any) + JSON text block.

## Adding a tool

1. Implement the skill in `skills/library.py` with `@skill`.
2. Add a `ToolSpec` here. Descriptions are prompt text: state preconditions
   ("stand at the table first") and units. Use `SIDE` for hand arguments and
   `_num(desc)` for numbers.
3. Nothing else needs wiring; MCP and agents pick it up.

Removing or renaming a tool changes every planner's action space and breaks
scripted plans that use it; grep `agents/scripted.py` first.
