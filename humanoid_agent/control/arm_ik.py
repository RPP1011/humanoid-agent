"""Damped-least-squares position IK for one 7-DOF arm (palm point only).

Solved on a scratch MjData so the live sim is untouched; the result is a joint
target the position actuators then track.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..sim.env import PALM_OFFSET, HumanoidEnv


def solve_arm_ik(
    env: HumanoidEnv,
    side: str,
    target: np.ndarray,
    iters: int = 200,
    damping: float = 0.05,
    tol: float = 0.01,
    seed: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Returns (joint_targets, residual_error_m). `seed` warm-starts the arm joints."""
    m = env.model
    d = mujoco.MjData(m)
    d.qpos[:] = env.data.qpos
    qadr, dadr = env.arm_qadr[side], env.arm_dof[side]
    if seed is not None:
        d.qpos[qadr] = seed
    jids = [m.dof_jntid[i] for i in dadr]
    lo, hi = m.jnt_range[jids, 0], m.jnt_range[jids, 1]
    body = env.hand_body[side]
    jacp = np.zeros((3, m.nv))
    err = np.inf
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.xpos[body] + d.xmat[body].reshape(3, 3) @ PALM_OFFSET
        e = target - p
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        mujoco.mj_jac(m, d, jacp, None, p, body)
        J = jacp[:, dadr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), e)
        d.qpos[qadr] = np.clip(d.qpos[qadr] + np.clip(dq, -0.2, 0.2), lo, hi)
    return d.qpos[qadr].copy(), err
