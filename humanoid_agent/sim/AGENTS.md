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

### Fridge (`Fridge`, `kitchen_scene()`)

Static carcass (back/sides/roof, solid block below the lowest shelf) + shelf geoms
`<fridge>_<shelf>` + child body `<fridge>_door` with hinge joint `<fridge>_door`,
handle geom/site `<fridge>_handle` and a close-up camera `<fridge>_cam`. `yaw` is
the heading of the front face (multiples of π/2: footprints are axis-aligned);
the hinge is on the robot's right. `shelf_tables()` returns each shelf as a
Table-like record (`fridge_top_shelf`, ...) that `place_on` treats like a table;
`footprint` / `door_footprint(angle)` are obstacle records for walk routing.
Shelf heights (0.70, 1.05 m) are the band a *standing* G1 can reach; real fridges
put shelves lower and higher than that. Shelves stop 5 cm short of the front so a
hand at carry height in the doorway clears them.

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
- **Fridge doors are kinematic.** `door_angle[fridge]` is written into the hinge
  qpos every substep (`_apply_doors`), so contact never moves a door and a closed
  door really walls off the shelves. `hidden(obj)` = inside a fridge whose door is
  shut; `list_objects` omits those. Shelf geoms are registered in `_support_name`
  under the shelf name, the rest of the carcass as the fridge name.
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

## person.py

`Person` (spec, on `SceneSpec.people`) + `Walker` (runtime, `env.people`). A
**placeholder pedestrian** (`is_physical=False`): a mocap body of capsules named
`<name>` (group 2, no contact bits, so it never collides and `surface_below` ray
casts ignore it) that glides back and forth along `path` at `speed`, pausing
`pause_s` at each end. It yields (stops) while the robot base is within 0.9 m
ahead and 0.5 m of its line, and after `patience_s` of waiting turns back, so a
parked robot can't deadlock it. `seed` picks the start point on its cycle (the
runner sets it to the episode seed; `reset` fast-forwards to it while yielding to
the robot's start pose). Deterministic, so `episode_server.py replay` reproduces it.

`env.step` advances walkers once per control step and records contact *onsets*
in `env.person_contacts` when base-to-person distance < `ROBOT_RADIUS` (0.2) +
`radius` (0.25), plus `env.min_person_dist`. That is a 2D cylinder test on the
base, not a physics contact: an arm swung into the person is not detected.

## supervisor.py

`Supervisor` is a **placeholder human** (`is_physical=False`): no body, just
one-shot `Rule(when(env), say)` triggers plus a keyword answer table for
`ask_supervisor`. It is deterministic on purpose so collaboration tasks stay
gradeable on sim state. It is heard only between skills; a rule that becomes
true mid-skill is delivered when that skill returns. `transcript` (who said
what, at what sim time) goes into every eval row.
