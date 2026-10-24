"""HumanoidEnv: thin wrapper around MuJoCo model/data with the hooks skills need."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from ..control.locomotion import KinematicGlide, LocomotionController
from .scene import SceneSpec, build_model, default_scene

ARM_JOINTS = {
    side: [f"{side}_{j}_joint" for j in (
        "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw",
    )]
    for side in ("left", "right")
}
HAND_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}
# Approximate palm centre offset in the wrist_yaw_link frame.
PALM_OFFSET = np.array([0.10, 0.0, 0.0])


def _yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class HumanoidEnv:
    def __init__(
        self,
        scene: SceneSpec | None = None,
        locomotion: LocomotionController | None = None,
        control_dt: float = 0.02,
        render_size: tuple[int, int] = (320, 240),
    ) -> None:
        self.scene = scene or default_scene()
        self.model = build_model(self.scene)
        self.data = mujoco.MjData(self.model)
        self.loco = locomotion or KinematicGlide()
        self.n_sub = max(1, round(control_dt / self.model.opt.timestep))
        self.control_dt = self.n_sub * self.model.opt.timestep
        self._render_size = render_size
        self._renderer = None

        m = self.model
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        self._key = key
        self.stand_ctrl = m.key_ctrl[key].copy()

        act_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
        self.act_index = {n: i for i, n in enumerate(act_names)}
        self.leg_waist_act = np.array(
            [i for i, n in enumerate(act_names) if ("hip" in n or "knee" in n or "ankle" in n or "waist" in n)]
        )
        self.arm_act = {s: np.array([self.act_index[j] for j in js]) for s, js in ARM_JOINTS.items()}
        self.arm_dof = {
            s: np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in js])
            for s, js in ARM_JOINTS.items()
        }
        self.arm_qadr = {
            s: np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in js])
            for s, js in ARM_JOINTS.items()
        }
        self.hand_act = {
            s: np.array([i for n, i in self.act_index.items() if n.startswith(f"{s}_hand_")])
            for s in ("left", "right")
        }
        self.hand_body = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for s, b in HAND_BODY.items()}
        self.object_names = [o.name for o in self.scene.objects]
        self._obj_qadr = {
            o: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{o}_free")]
            for o in self.object_names
        }
        self._obj_dadr = {
            o: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{o}_free")]
            for o in self.object_names
        }
        self._obj_geom = {
            o: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{o}_geom") for o in self.object_names
        }
        self._obj_contact = {
            o: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for o, g in self._obj_geom.items()
        }
        # object -> (side, pos_in_hand, quat_in_hand). "Magic grasp" attachments.
        self.attached: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        self.arm_target = {s: self.stand_ctrl[self.arm_act[s]].copy() for s in ("left", "right")}
        self.hand_target = {s: self.stand_ctrl[self.hand_act[s]].copy() for s in ("left", "right")}
        # Optional scripted human (sim/supervisor.py), set per task. Skills tick it.
        self.supervisor = None
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        # Keyframe only covers the robot; restore object poses from the spec.
        for o in self.scene.objects:
            a = self._obj_qadr[o.name]
            self.data.qpos[a:a + 3] = o.pos
            self.data.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(self.model, self.data)
        for o in list(self.attached):
            self.detach(o)
        for s in ("left", "right"):
            self.arm_target[s] = self.stand_ctrl[self.arm_act[s]].copy()
            self.hand_target[s] = self.stand_ctrl[self.hand_act[s]].copy()
        self.loco.reset(self)
        self.set_ego_pitch(0.6)
        # Let objects settle onto tables.
        for _ in range(25):
            self.step()

    def step(self, n: int = 1) -> None:
        """Advance n control steps (control_dt each)."""
        for _ in range(n):
            for s in ("left", "right"):
                self.data.ctrl[self.arm_act[s]] = self.arm_target[s]
                self.data.ctrl[self.hand_act[s]] = self.hand_target[s]
            for _ in range(self.n_sub):
                self.loco.pre_physics(self)
                mujoco.mj_step(self.model, self.data)
                self.loco.post_physics(self)
                self._apply_attachments()
            mujoco.mj_forward(self.model, self.data)

    # ---------------------------------------------------------------- state
    @property
    def time(self) -> float:
        return float(self.data.time)

    def base_pose(self) -> tuple[float, float, float]:
        q = self.data.qpos
        return float(q[0]), float(q[1]), _yaw_from_quat(q[3:7])

    def base_height(self) -> float:
        return float(self.data.qpos[2])

    def is_fallen(self) -> bool:
        return self.base_height() < 0.5

    def object_pos(self, name: str) -> np.ndarray:
        a = self._obj_qadr[name]
        return self.data.qpos[a:a + 3].copy()

    def palm_pos(self, side: str) -> np.ndarray:
        b = self.hand_body[side]
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ PALM_OFFSET

    # ---------------------------------------------------------------- attachments
    def attach(self, obj: str, side: str) -> None:
        b = self.hand_body[side]
        R = self.data.xmat[b].reshape(3, 3)
        a = self._obj_qadr[obj]
        rel_p = R.T @ (self.data.qpos[a:a + 3] - self.data.xpos[b])
        hq_inv = np.zeros(4)
        mujoco.mju_negQuat(hq_inv, self.data.xquat[b])
        rel_q = np.zeros(4)
        mujoco.mju_mulQuat(rel_q, hq_inv, self.data.qpos[a + 3:a + 7])
        self.attached[obj] = (side, rel_p, rel_q)
        # Held objects are moved kinematically; letting them collide would inject
        # unbounded penetration forces into the scene. Disable until released.
        g = self._obj_geom[obj]
        self.model.geom_contype[g] = 0
        self.model.geom_conaffinity[g] = 0

    def detach(self, obj: str) -> None:
        if self.attached.pop(obj, None) is not None:
            g = self._obj_geom[obj]
            self.model.geom_contype[g], self.model.geom_conaffinity[g] = self._obj_contact[obj]

    def _apply_attachments(self) -> None:
        if not self.attached:
            return
        mujoco.mj_kinematics(self.model, self.data)
        for obj, (side, rel_p, rel_q) in self.attached.items():
            b = self.hand_body[side]
            a, d = self._obj_qadr[obj], self._obj_dadr[obj]
            self.data.qpos[a:a + 3] = self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ rel_p
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, self.data.xquat[b], rel_q)
            self.data.qpos[a + 3:a + 7] = q
            self.data.qvel[d:d + 6] = 0.0

    # ---------------------------------------------------------------- rendering
    def render(self, camera: str = "ego") -> np.ndarray:
        if self._renderer is None:
            w, h = self._render_size
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def set_ego_pitch(self, pitch: float) -> None:
        """Tilt the ego camera (positive = look down). Radians, clamped."""
        pitch = float(np.clip(pitch, -0.3, 1.2))
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")
        # MuJoCo cameras look along their local -z, with +y up and +x right.
        fwd = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
        up = np.array([np.sin(pitch), 0.0, np.cos(pitch)])
        right = np.cross(fwd, up)
        mat = np.column_stack([right, up, -fwd]).flatten()
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, mat)
        self.model.cam_quat[cid] = q
