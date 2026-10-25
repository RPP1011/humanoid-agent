# eval/ — tasks and runner

## tasks.py

`Task(id, prompt, checks, scene=default_scene, max_tool_calls=30, tags, supervisor, settle_s=0)`.
`settle_s` > 0 runs that much sim time after the episode before checking (the
runner and the MCP server's `check_task` both do), so a wobbly stack must stay up.
`checks` maps a readable condition name to a predicate on `Episode`; `report(ep)`
returns each one's result and `success(ep)` is all of them. They are evaluated on
**sim state** after the episode;
`Episode` carries the env, the agent's `finish` answer, and object positions
captured right after reset.

Helpers: `_tower(ep, base, n)` (n objects chained by `env.support_of` from
`base`, all < 20° tilt), `_on(obj, table)` (inside the table's xy extent, above its top, not
held), `_at(location, tol=0.1)`, `_undisturbed(tol=0.02)`, `_answer_has(*words)`.

| id | tests | success |
|---|---|---|
| `nav_table_b` | navigation | within 10 cm of `table_b` location |
| `move_red_to_b` | pick & place | red on table_b |
| `move_both_to_b` | two hands | red and blue on table_b |
| `fetch_green_to_a` | grounding ("the other table") | green on table_a |
| `swap` | multi-step | red on b and green on a |
| `which_on_b` | query | answer contains "green" and nothing moved |
| `impossible_purple` | honest refusal | nothing moved and answer matches a refusal phrase |
| `revised_order` | supervisor changes the order mid-task (on pick for even seeds, at table_b for odd) | red on table_a, blue and green on table_b |
| `clarify_which` | ambiguous order; supervisor's choice alternates by seed | the wanted cube on table_b, the other still on table_a |
| `bean_bag_tower` | stacking on a soft, lumpy support (two trips: three cubes, two hands) | a 3-cube tower on `bean_bag`, upright after 5 s settle |

Collaboration tasks set `Task.supervisor = (seed) -> Supervisor`. Their scripted plans
only cover seed 0, since a fixed plan cannot react to what is said.

Adding a task: append to `TASKS`, give it a plan in `agents/scripted.py`
(the smoke test runs every task), and keep the check on state. Answer checks
are substring matches; be aware they are lenient (see below).

## runner.py

`python -m humanoid_agent.eval.runner --agent {scripted,anthropic,openai} [--model M]
[--base-url URL] [--no-images] [--tasks id ...] [--seeds N] [--jitter 0.03] [--out runs]`

- `run_episode` builds a fresh `HumanoidEnv` per episode. Seed 0 is the
  canonical layout; seeds > 0 jitter every object's xy uniformly by ±`jitter` m
  with `random.Random(seed)` (deterministic) and set each bag's `lump_seed` to
  the seed, so the bean bag's lumps differ per seed (seed 4's crown is lumpy
  enough that a centred tower falls; an offset spot works).
- `scripts/episode_server.py` runs the same `run_episode` behind a localhost
  HTTP bridge so a planner that is not a Python object (a Claude Code subagent
  with no API key) can drive it one CLI call at a time; `--video` writes an mp4.
- The `call` closure counts tool calls; `finish` raises `EpisodeDone` with the
  answer; call number `max_tool_calls + 1` logs `budget` and ends the episode.
- Agent crashes become `error` in the row (episode fails), not harness crashes.
- Rows: task, seed, agent, success, `checks` (per-condition), tool_calls, sim/wall time, answer, error,
  `locomotion_is_physical`, `skill_failures` (codes), full `log` (minus data).
- `summarize` → per-task success table, failure codes split by
  `PLANNER_CODES` (planner) vs everything else (skills/sim), and a warning line
  when locomotion is the placeholder.

## Caveats when reading results
- Attribution is by code only. `too_far` (didn't walk close enough) is
  arguably a planner error but counts as skill/sim; `unreachable` is ambiguous.
- `which_on_b` passes for any answer containing "green" (including "red, blue,
  or green"); `impossible_purple` passes on phrases like "there is no".
- Each episode re-creates the env (and its renderer); long runs are dominated
  by sim time, not model latency, for the scripted agent.
