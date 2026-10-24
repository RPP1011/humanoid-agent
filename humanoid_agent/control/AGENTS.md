# control/ — locomotion and arm IK

## locomotion.py

Contract: a `LocomotionController` takes a body-frame velocity command
`(vx, vy, wz)` via `set_command` / `stop`, and moves the base when the env calls
`pre_physics(env)` (write leg/waist `ctrl`) and `post_physics(env)` every
physics substep. Skills only use `set_command`/`stop`, so swapping controllers
changes nothing above this layer.

- `KinematicGlide` — **placeholder**. Legs/waist held at the `stand` keyframe;
  after each `mj_step` the base pose is overwritten with the integrated command
  (clipped to 0.6 m/s, 1.2 rad/s), height frozen, base velocity zeroed. The
  robot cannot fall, ignores contacts, and the arm works from a perfectly still
  base. `is_physical` returns `False`; keep it that way.
- `PolicyLocomotion` — slot for an ONNX velocity-tracking policy (needs the
  `policy` extra). You must implement `_obs` (match the training observation
  layout, joint order, scaling) and `_act` (policy action → leg/waist `ctrl`).
  Inference runs every `decimation` substeps and the action is held between.
  The G1 scene's physics timestep is 0.002 s, so the default `decimation=10`
  gives 50 Hz; set it to match the policy's training rate.

When you wire in a real policy, expect regressions in `skills/` rather than
here: `_align_for` / `_step_to` tolerances (2 cm, 0.05 rad) assume perfect
tracking, and `_reach` assumes the base doesn't move during the arm motion.
Run the scripted eval with `--seeds 5` before and after.

## arm_ik.py

`solve_arm_ik(env, side, target, seed=...)` → `(joint_targets, residual_m)`.
Damped least squares on the palm **position** only (no orientation), 7 arm
DOFs, step-clipped to 0.2 rad and joint-limit clipped each iteration. It runs on
a scratch `MjData` copied from the live one, so it never disturbs the sim.

Callers (see `skills/library.py:_reach`) warm-start each waypoint from the
previous solution so joint targets along a Cartesian path stay continuous, and
treat residual > 4 cm as `unreachable`. Because orientation is unconstrained,
the wrist/hand orientation at the target is whatever falls out; with magic grasp
that doesn't matter, with real grasping it will.
