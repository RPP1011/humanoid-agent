# skills/ — the planner's action space

`SkillLibrary(env)` holds every action a planner can take. `types.py` defines
`SkillResult`, the only thing a planner ever sees.

## SkillResult

```python
SkillResult(success: bool, reason: str = "ok", data: dict = {}, sim_time_s: float = 0.0)
```

`reason` is `"<code>: <human detail>"`, e.g. `"unreachable: ik residual 0.140m; move closer"`.
`.code` is the prefix before `:`. The eval runner groups failures by code and
splits them into planner vs skill errors using `eval/runner.py:PLANNER_CODES`,
so **codes are an API**: don't rename them casually, and add new
planner-caused codes to that set.

Current codes: `ok`, `fallen`, `bad_args`, `unknown_location`, `unknown_target`,
`unknown_object`, `unknown_surface`, `unreachable`, `tracking`, `too_far`,
`hand_full`, `not_holding`, `retract_blocked`, `no_free_space`, `no_support`, `placement`,
`timeout`; composites prefix sub-step failures: `approach_*`, `descend_*`,
`align_*` (e.g. `approach_tracking`).

## The `@skill` decorator

Every public skill method is wrapped by `@skill`, which:

1. refuses to run if the robot has fallen;
2. converts `KeyError`/`ValueError` into `bad_args` (other exceptions propagate
   and end the episode as an agent error, so don't let them escape);
3. downgrades success to `fallen` if the robot fell during the skill;
4. stamps `sim_time_s` and appends a log entry to `self.log` (the eval log).

`skills()` discovers skills by the `is_skill` attribute it sets.

Composites call other skills' bodies through `SkillLibrary.grasp.__wrapped__(self, ...)`
so the sub-call is not logged as a separate planner action. Internal helpers
(`_reach`, `_retract`, `_walk_to`, `_align_for`, `_step_to`) are undecorated.

## Skill groups

| Group | Skills | Notes |
|---|---|---|
| Query | `get_state`, `list_objects`, `list_locations`, `get_frame` | `list_objects` is ground-truth perception (with `on`, `tilt_deg`, and a `surfaces` list of tables and bags); `get_frame` returns PNG bytes under `data["png_bytes"]` |
| Locomotion | `go_to_location`, `walk_to`, `turn_to`, `look_at` | `_walk_to`: turn-then-drive P-controller, 2 cm / 0.05 rad tolerance; a straight line that would enter a table (10 cm keep-out) is routed via table corners (`_route`, visibility graph) or refused `blocked_path`; yaw > 2π is `bad_args` |
| Primitives | `reach`, `grasp`, `release`, `retract` | `grasp` = magic attach if object within 8 cm of palm |
| Composites | `pick`, `place_on` | what planners should normally use |
| People | `ask_supervisor`, `report` | `no_supervisor` if the task has none; asking costs 2 s of sim time |

When `env.supervisor` is set (see `sim/supervisor.py`), the `@skill` wrapper ticks it after
every skill and attaches anything it said to that result as `data["supervisor_says"]`.

### How `pick` works
choose side (object's side of the body, else the free hand) → `_align_for`
side-steps so the object sits at (`REACH_FWD`=0.30, ±`REACH_LAT`=0.15) in the
body frame (fails `too_far` if that's >0.5 m away: walk first) → reach 10 cm
above → descend to 4.5 cm above → magic grasp → lift 12 cm → carry pose (palm at `CARRY_Z` = 1.05 m, high
enough that the carried cube clears a two-cube stack on the bean bag).
Lift/carry failures are reported in `data` but don't fail the skill.

### How `place_on` works
pick the hand: `obj` names the held object; else `side`; with both hands full
and neither given, fail `ambiguous` (never guess). Then by `surface`:

- **table**: fail `too_far` if the table is >0.5 m from the base → `_free_spot`
  grid-searches it (3 cm grid, 6 cm edge margin) for the point closest to the
  preferred spot (30 cm ahead in front of the placing hand, plus `dx`/`dy`) that
  is ≥15 cm from every other object and within 45 cm of it (`no_free_space`).
- **bag or object** (stacking): aim at its centre plus `dx`/`dy`; `no_support`
  if that point is not over it.

Height comes from `env.surface_below` over the object's footprint. Lift every
carried object to `max(CARRY_Z, top + 0.17)` (side-stepping otherwise sweeps
them through a stack) → align (`_align_for` refuses stances within
`STANCE_MARGIN` = 9 cm of a table: the knees hit it) → approach 10 cm above →
descend → **servo on the held object's position** (up to 4 corrections to
<5 mm; the palm alone stops up to 3 cm off and the wrist swings the object) →
release 1 cm above → `_retract` → settle 0.5 s → verify with `env.support_of`
(`placement` failure otherwise). Success data: landed `pos`, `tilt_deg`,
`release_miss_m`. Physics decides whether a stack stays up; the task checks
again after its `settle_s`.

### Why `_reach` and `_retract` look the way they do
`_reach` lifts to target height near the body before moving horizontally, and
streams IK waypoints every 2 cm at control rate, because straight joint-space
moves sweep through table tops. `_retract` lifts, pulls back behind the table
edge, lowers beside the hip, then snaps to rest, because going straight to the
rest pose knocks objects off the table.

## Rules for new skills

- Blocking, bounded by a `timeout_s` in **sim** time, always return a
  `SkillResult`; validate names up front and return `unknown_*` codes.
- Make failure reasons actionable for a planner ("walk closer first", "free a
  hand"). That text is the only feedback an LLM gets.
- Keep returned numbers rounded and payloads small; images only via `png_bytes`.
- Register it in `tools/registry.py`, and if it is meant for planners, add a
  scripted plan/task that exercises it.
