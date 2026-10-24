# sim/ — MuJoCo scene and environment

## scene.py

`SceneSpec` = `tables` + `objects` + `locations`. `build_model(spec)` loads
`assets/unitree_g1/scene_with_hands.xml` via `MjSpec`, adds:

- `ego` camera on `torso_link` (approx. head), `overview` fixed world camera;
- each `Table` as a static body (box top + 4 legs, no joint);
- each `Obj` as a free body: joint `<name>_free`, geom `<name>_geom`, 50 g box,
  high friction. **These naming patterns are load-bearing**: `env.py` looks them
  up by name.

`locations` are named standing poses `(x, y, yaw)`: where the robot should
stand to work at something, not the thing itself (e.g. `table_a` location is
0.52 m in front of the table centre, facing it). Tasks that add tables must add
a matching location or `go_to_location` can't get the robot there.

## env.py — `HumanoidEnv`

Thin wrapper around `MjModel`/`MjData` that skills drive.

- **Stepping.** `step(n)` advances `n` control steps of `control_dt` (0.02 s →
  `n_sub` physics substeps). Arm/hand position targets (`arm_target`,
  `hand_target`) are written to `ctrl` once per control step; the locomotion
  controller's `pre_physics`/`post_physics` and `_apply_attachments` run every
  substep. Skills never call `mj_step` directly.
- **Reset.** Loads the `stand` keyframe (robot only), restores object poses from
  the spec, detaches everything, resets targets and the ego camera pitch, then
  steps 25 times so objects settle. Initial object positions for "undisturbed"
  checks are captured *after* this settle (in the runner/server).
- **Index maps** built once in `__init__`: `act_index`, `arm_act`/`arm_dof`/
  `arm_qadr` per side (7 arm joints, see `ARM_JOINTS`), `hand_act`,
  `leg_waist_act`, `_obj_qadr`/`_obj_dadr`/`_obj_geom`. Use these rather than
  re-deriving ids.
- **Palm point** = `wrist_yaw_link` origin + `PALM_OFFSET` (10 cm along local x).
  IK, grasp distance and placement all use this one point.
- **Magic grasp** (`attach`/`detach`/`_apply_attachments`): an attached object's
  pose is rewritten every substep to stay rigidly fixed to the hand, its velocity
  zeroed, and its geom's `contype`/`conaffinity` set to 0 (it would otherwise
  inject huge penetration forces). `detach` restores the saved collision bits.
  Note this mutates the **model**, so it is per-env state.
- `is_fallen()` = base height < 0.5 m. Every skill checks it.
- Rendering: `MUJOCO_GL` defaults to `egl` *at import time* of this module. The
  `Renderer` is created lazily on first `render()`. `set_ego_pitch` rewrites the
  camera's local quaternion (clamped −0.3…1.2 rad, positive = down).

## Gotchas

- `base_pose()` yaw comes from the free-joint quaternion; under
  `KinematicGlide` the base never pitches/rolls, under a real policy it will.
- Tables are static and the kinematic base ignores contact, so the robot can
  walk *through* tables today (verified: `walk_to(2.0, 0.0)` from start passes
  through `table_a` and reports success). Don't build tests that rely on
  collisions blocking navigation until locomotion is physical.

## supervisor.py

`Supervisor` is a **placeholder human** (`is_physical=False`): no body, just
one-shot `Rule(when(env), say)` triggers plus a keyword answer table for
`ask_supervisor`. It is deterministic on purpose so collaboration tasks stay
gradeable on sim state. It is heard only between skills; a rule that becomes
true mid-skill is delivered when that skill returns. `transcript` (who said
what, at what sim time) goes into every eval row.
