"""Locomotion controllers.

The contract every controller satisfies: accept a body-frame velocity command
(vx, vy, wz) and advance the robot's base. Skills only ever talk to this
interface, so swapping the placeholder for a learned policy changes nothing above.

`KinematicGlide` is a PLACEHOLDER. It drives the floating base kinematically
(legs held in the stand pose, base pose overwritten each step). It exists so the
agent/skill/eval loop can be built and tested before a real walking policy is
wired in. Do not draw locomotion conclusions from it.

`PolicyLocomotion` is the slot for a real velocity-tracking policy (e.g. an
mjlab / Unitree RL velocity policy exported to ONNX).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class LocomotionController(ABC):
    def __init__(self) -> None:
        self.cmd = np.zeros(3)  # vx, vy (body frame, m/s), wz (rad/s)

    def set_command(self, vx: float, vy: float, wz: float) -> None:
        self.cmd[:] = (vx, vy, wz)

    def stop(self) -> None:
        self.cmd[:] = 0.0

    def reset(self, env) -> None:  # noqa: B027
        pass

    @abstractmethod
    def pre_physics(self, env) -> None:
        """Write leg/waist actuator targets before mj_step."""

    def post_physics(self, env) -> None:  # noqa: B027
        """Optional correction after mj_step (used by the kinematic placeholder)."""

    @property
    def is_physical(self) -> bool:
        """False if base motion is faked. Eval reports carry this flag."""
        return True


class KinematicGlide(LocomotionController):
    """Placeholder: base follows the integrated command exactly; legs stay at stand pose."""

    max_lin = 0.6
    max_ang = 1.2

    def reset(self, env) -> None:
        self.stop()
        x, y, yaw = env.base_pose()
        self._pose = np.array([x, y, yaw])
        self._z = env.data.qpos[2]

    def pre_physics(self, env) -> None:
        env.data.ctrl[env.leg_waist_act] = env.stand_ctrl[env.leg_waist_act]

    def post_physics(self, env) -> None:
        dt = env.model.opt.timestep
        vx, vy = np.clip(self.cmd[:2], -self.max_lin, self.max_lin)
        wz = np.clip(self.cmd[2], -self.max_ang, self.max_ang)
        c, s = np.cos(self._pose[2]), np.sin(self._pose[2])
        self._pose[0] += (c * vx - s * vy) * dt
        self._pose[1] += (s * vx + c * vy) * dt
        self._pose[2] = (self._pose[2] + wz * dt + np.pi) % (2 * np.pi) - np.pi
        q = env.data.qpos
        q[0], q[1], q[2] = self._pose[0], self._pose[1], self._z
        h = self._pose[2] / 2
        q[3:7] = (np.cos(h), 0.0, 0.0, np.sin(h))
        env.data.qvel[0:6] = 0.0

    @property
    def is_physical(self) -> bool:
        return False


class PolicyLocomotion(LocomotionController):
    """Slot for a learned velocity-tracking policy.

    Implement `_obs` and `_act` to match your policy's observation layout, action
    scaling and joint ordering. Typical policies run at 50 Hz; call the policy
    every `decimation` physics steps and hold the action in between.
    """

    def __init__(self, onnx_path: str, decimation: int = 10) -> None:
        super().__init__()
        import onnxruntime as ort  # optional dependency

        self.sess = ort.InferenceSession(onnx_path)
        self.decimation = decimation
        self._k = 0
        self._last_action = None

    def _obs(self, env) -> np.ndarray:
        raise NotImplementedError("Match this to the policy's training observation layout.")

    def _act(self, env, action: np.ndarray) -> None:
        raise NotImplementedError("Map policy action -> env.data.ctrl for legs/waist.")

    def pre_physics(self, env) -> None:
        if self._k % self.decimation == 0:
            obs = self._obs(env)[None].astype(np.float32)
            self._last_action = self.sess.run(None, {self.sess.get_inputs()[0].name: obs})[0][0]
        self._act(env, self._last_action)
        self._k += 1
