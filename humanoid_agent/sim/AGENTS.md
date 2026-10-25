# sim/ — MuJoCo scene and environment

## scene.py

`SceneSpec` = `tables` + `objects` + `bags` + `locations`. `build_model(spec)` loads
`assets/unitree_g1/scene_with_hands.xml` via `MjSpec`, adds:

- `ego` camera on `torso_link` (approx. head), `overview` fixed world camera;
- each `Table` as a static body (box top + 4 legs, no joint);
- each `Obj` as a free body: joint `<name>_free`, geom `<name>_geom`, 50 g box,
  high friction. **These naming patterns are load-bearing**: `env.py` looks them
  up by name.

- each `BeanBag` as a bed of cells `<name>_cell<k>` (2 cm plates, 3 cm pitch,
  gravcomp) on vertical slide joints with stiffness, damping and frictionloss,
  shaped as a flat-topped dome plus seeded random lumps (`lump_seed`; the
  runner sets it to the episode seed), resting on a table top (`base_z`), and a
  `<name>_cam` close-up camera.
- `option.cone = elliptic`, `impratio = 10`: with the defaults, resting contacts
  creep, and a cube tower on a 4° slope slid apart in seconds even on a rigid support.
- World geoms (floor, tables, cells, objects) use collision bits `WORLD_BITS = 3`;
  the robot keeps 1. See "release ghosting" below.

### Why the bean bag is a cell bed, not a MuJoCo flex
A 3D `flexcomp` ellipsoid was tried first (MuJoCo 3.17). 5×5×3 vertices cost
7–15× the sim time of the base scene, needed `integrator="discrete"` for flex
elasticity (the G1 uses implicitfast) or edge equality constraints, and at that
resolution the vertex spacing (9 cm) is coarser than a 5 cm cube: the cube sits
on one vertex peak and a perfectly centred two-cube stack topples. Conforming
to a cube needs ~2 cm spacing, i.e. thousands of vertices, which is too slow on
CPU. The cell bed costs ~2.4× and gives local compliance at 3 cm; what it can't
do is flow sideways, bulge, or wrap around an object.

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
  position is rewritten every substep to stay fixed to the hand, its orientation
  kept level (base yaw + the offset at grasp time; position-only IK lets the wrist
  roll, which released cubes ~30° tilted), its velocity zeroed, and its geom's
  `contype`/`conaffinity` set to 0 (it would otherwise inject huge penetration
  forces). **Release ghosting:** `detach` sets the bits to 2, so the object
  lands on world geoms but ignores the hand it overlaps (colliding kicked cubes
  off stacks); `step` restores the saved bits once it is 15 cm from both palms.
  Note this mutates the **model**, so it is per-env state.
- **Supports.** `surface_below(x, y, z, exclude, half)` ray-casts down over
  group-0 geoms only (robot collision meshes are group 3) and returns
  `(height, owner)`; with `half` > 0 the height is the max over a 3×3 footprint
  grid, so a flat-bottomed object is never lowered into a lump. Owners come from
  `_support_name` (geom id → table / bag / object name, or `floor`): register new
  support geoms there. `support_of(obj)` = owner under the centre if within
  ~1.8·half, else `air`; `held` while attached. `tilt_deg(obj)` = angle from upright.
- `is_fallen()` = base height < 0.5 m. Every skill checks it.
- Rendering: `MUJOCO_GL` defaults to `egl` *at import time* of this module. The
  `Renderer` is created lazily on first `render()`. `set_ego_pitch` rewrites the
  camera's local quaternion (clamped −0.3…1.2 rad, positive = down).

## Gotchas

- `base_pose()` yaw comes from the free-joint quaternion; under
  `KinematicGlide` the base never pitches/rolls, under a real policy it will.
- Tables are static and the kinematic base ignores contact: if the base is driven
  into a table, the body contorts over it and knocks objects to the floor while
  reporting success. `SkillLibrary._walk_to` therefore routes around table
  footprints; anything else that commands the base (`_step_to`, a new skill)
  must not drive into a table either.

## supervisor.py

`Supervisor` is a **placeholder human** (`is_physical=False`): no body, just
one-shot `Rule(when(env), say)` triggers plus a keyword answer table for
`ask_supervisor`. It is deterministic on purpose so collaboration tasks stay
gradeable on sim state. It is heard only between skills; a rule that becomes
true mid-skill is delivered when that skill returns. `transcript` (who said
what, at what sim time) goes into every eval row.
