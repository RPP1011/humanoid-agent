# agents/ — planners

An agent implements `run(task_id, prompt, call)` (see `base.py:Agent`) and drives
the episode only through `call(tool_name, args) -> SkillResult`. `call` raises
`EpisodeDone` after `finish` or when the tool budget is exhausted; agents should
let it propagate.

## base.py
`ToolCaller`, `EpisodeDone`, the `Agent` protocol, and `SYSTEM_PROMPT` (shared
by both LLM agents). The prompt is short on purpose: check the world first,
stand at a table before pick/place, adapt on failure codes, call `finish`.

## scripted.py — `ScriptedAgent`
`PLANS[task_id]` is a fixed list of `(tool, args)`. Stops at the first failure
and finishes with the failure reason. It is a **harness check, not a baseline
for intelligence**: every task in `eval/tasks.py` must have a plan here,
because `tests/test_smoke.py` runs all of them.

## llm.py
- `AnthropicAgent` — Messages API with tool use. Model from arg, else
  `HUMANOID_AGENT_MODEL`, else `claude-haiku-5-5`. Images from `get_frame` go
  inside the `tool_result` content. A turn with no tool calls is treated as
  `finish(text)`.
- `OpenAICompatAgent` — any OpenAI-compatible server with tool calling (vLLM,
  llama.cpp, SGLang, Ollama…). Images are sent as a follow-up **user** message
  because most compatible servers reject images in tool messages;
  `--no-images` disables them. Malformed tool-call JSON gets a `bad_json` tool
  reply instead of a crash.

Both keep the full conversation, including every image, for the whole episode;
there is no context trimming.

## Known sharp edges
- In `OpenAICompatAgent`, a `bad_json` reply does not go through `call`, so it
  doesn't count against the task's tool budget. A model that keeps emitting
  bad JSON loops until the server or API errors.
- Agent exceptions are caught by the runner and recorded as `error` (episode
  fails), not raised.

## Adding a backend
Implement the protocol in `llm.py` (lazy-import the SDK inside `__init__` so
it stays an optional dependency), build tools from `ALL_TOOLS`, add a branch in
`eval/runner.py:make_agent` and a `--agent` choice.
