"""HumanoidEnv: thin wrapper around MuJoCo model/data with the hooks skills need."""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from ..control.locomotion import KinematicGlide, LocomotionController
from .person import Walker
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
        # Support lookup: world geoms (group 0) -> the table/bag/object/floor they belong to.
        self._support_name = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"): "floor"}
        for t in self.scene.tables:
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t.name)
            self._support_name.update({g: t.name for g in range(m.ngeom) if m.geom_bodyid[g] == bid})
        for b in self.scene.bags:
            self._support_name.update({g: b.name for g in range(m.ngeom)
                                       if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith(f"{b.name}_cell")})
        for f in self.scene.fridges:
            for g in range(m.ngeom):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                if n.startswith(f"{f.name}_"):
                    shelf = next((s for s in f.shelf_names if n == f"{f.name}_{s}"), None)
                    self._support_name[g] = f"{f.name}_{shelf}" if shelf else f.name
        self._support_name.update({g: o for o, g in self._obj_geom.items()})
        # Fridge doors are kinematic: the hinge angle is rewritten every substep, so
        # the arm can't shove a door open and a closed door really blocks the inside.
        self._door_qadr = {f.name: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f.name}_door")]
                           for f in self.scene.fridges}
        self._door_dadr = {f.name: m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{f.name}_door")]
                           for f in self.scene.fridges}
        self.door_angle = {f.name: 0.0 for f in self.scene.fridges}
        self._obj_body = {o: int(m.geom_bodyid[g]) for o, g in self._obj_geom.items()}
        self._bag_qadr = np.array([m.jnt_qposadr[j] for j in range(m.njnt)
                                   if any((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith(f"{b.name}_cell")
                                          for b in self.scene.bags)], dtype=int)
        # object -> (side, pos_in_hand, yaw relative to the base). "Magic grasp" attachments.
        self.attached: dict[str, tuple[str, np.ndarray, float]] = {}
        self._ghosts: set[str] = set()  # released, not yet clear of the hand
        self.arm_target = {s: self.stand_ctrl[self.arm_act[s]].copy() for s in ("left", "right")}
        self.hand_target = {s: self.stand_ctrl[self.hand_act[s]].copy() for s in ("left", "right")}
        # Scripted pedestrians (sim/person.py): mocap bodies moved every control step.
        self.people = [Walker(p) for p in self.scene.people]
        self._person_mocap = [int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, p.name)])
                              for p in self.scene.people]
        self.person_contacts: list[dict] = []  # contact onsets: {t, person, pos, robot}
        self.min_person_dist = float("inf")
        self._touching: set[str] = set()
        # Optional scripted human (sim/supervisor.py), set per task. Skills tick it.
        self.supervisor = None
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._key)
        self.data.qpos[self._bag_qadr] = 0.0  # keyframe only covers the robot; un-dent the bags
        for f in self.door_angle:
            self.door_angle[f] = 0.0
            self.data.qpos[self._door_qadr[f]] = 0.0
        # Keyframe only covers the robot; restore object poses from the spec.
        for o in self.scene.objects:
            a = self._obj_qadr[o.name]
            self.data.qpos[a:a + 3] = o.pos
            self.data.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        mujoco.mj_forward(self.model, self.data)
        for o in list(self.attached):
            self.detach(o)
        for o in list(self._ghosts):
            self._unghost(o)
        for s in ("left", "right"):
            self.arm_target[s] = self.stand_ctrl[self.arm_act[s]].copy()
            self.hand_target[s] = self.stand_ctrl[self.hand_act[s]].copy()
        self.loco.reset(self)
        for w in self.people:
            w.reset(self.base_pose()[:2])
        self.person_contacts, self.min_person_dist, self._touching = [], float("inf"), set()
        self._apply_people()
        mujoco.mj_forward(self.model, self.data)
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
                self._apply_doors()
                self._apply_attachments()
            if self.people:
                self._tick_people()
            mujoco.mj_forward(self.model, self.data)
            for o in [o for o in self._ghosts
                      if min(np.linalg.norm(self.object_pos(o) - self.palm_pos(s)) for s in ("left", "right")) > 0.15]:
                self._unghost(o)

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

    def tilt_deg(self, name: str) -> float:
        """Angle between the object's local z axis and world up."""
        a = self._obj_qadr[name]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, self.data.qpos[a + 3:a + 7])
        return float(np.degrees(np.arccos(np.clip(R[8], -1.0, 1.0))))

    def surface_below(self, x: float, y: float, z: float = 2.0, exclude: str | None = None,
                      half: float = 0.0) -> tuple[float, str]:
        """Highest world surface under (x, y) below height z: (height, owner name).

        With `half` > 0 the height is the max over a 3x3 grid spanning a square
        footprint of that half-size (the owner is still what is under the centre):
        lowering a flat-bottomed object to the centre height alone would sink its
        corners into a lumpy support, and it would be ejected on release.
        Only group-0 geoms (floor, tables, bags, objects) are hit; the robot's
        collision meshes live in group 3, so an arm over the spot does not count.
        """
        gid = np.zeros(1, np.int32)
        mask = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        body = self._obj_body[exclude] if exclude else -1
        top, owner = 0.0, "floor"
        offs = (-half, 0.0, half) if half > 0 else (0.0,)
        for ox in offs:
            for oy in offs:
                d = mujoco.mj_ray(self.model, self.data, np.array([x + ox, y + oy, z]), np.array([0.0, 0.0, -1.0]),
                                  mask, 1, body, gid)
                if d >= 0:
                    top = max(top, z - d)
                    if ox == 0.0 and oy == 0.0:
                        owner = self._support_name.get(int(gid[0]), "unknown")
        return top, owner

    def support_of_point(self, x: float, y: float, z: float) -> str:
        return self.surface_below(x, y, z + 0.01)[1]

    def support_of(self, name: str) -> str:
        """What an object rests on: a table, bag or object name, 'floor', 'air' or 'held'."""
        if name in self.attached:
            return "held"
        p = self.object_pos(name)
        half = next(o.half_size for o in self.scene.objects if o.name == name)
        z, owner = self.surface_below(p[0], p[1], p[2], exclude=name)
        # A cube balanced on an edge or corner sits up to half*sqrt(3) above what holds it.
        return owner if p[2] - z < 1.8 * half + 0.01 else "air"

    def fridge(self, name: str):
        return next((f for f in self.scene.fridges if f.name == name), None)

    def fridge_of(self, support: str):
        """The fridge a support name (shelf or carcass) belongs to, or None."""
        return next((f for f in self.scene.fridges
                     if support == f.name or support in (f"{f.name}_{s}" for s in f.shelf_names)), None)

    def door_is_open(self, fridge: str) -> bool:
        return self.door_angle[fridge] > self.fridge(fridge).open_angle - 0.3

    def hidden(self, name: str) -> bool:
        """Inside a closed fridge: a real camera could not see it."""
        f = self.fridge_of(self.support_of(name))
        return f is not None and self.door_angle[f.name] < 0.3

    # Robot base keep-out radius for contact with a person (shoulder half-width ~0.2 m).
    ROBOT_RADIUS = 0.2

    def _tick_people(self) -> None:
        robot = np.array(self.base_pose()[:2])
        for w in self.people:
            w.advance(self.control_dt, robot)
            d = float(np.linalg.norm(w.pos - robot))
            self.min_person_dist = min(self.min_person_dist, d)
            if d < self.ROBOT_RADIUS + w.spec.radius:
                if w.spec.name not in self._touching:
                    self.person_contacts.append({"t": round(self.time, 2), "person": w.spec.name,
                                                 "pos": [round(float(v), 3) for v in w.pos],
                                                 "robot": [round(float(v), 3) for v in robot]})
                self._touching.add(w.spec.name)
            else:
                self._touching.discard(w.spec.name)
        self._apply_people()

    def _apply_people(self) -> None:
        for w, mid in zip(self.people, self._person_mocap):
            self.data.mocap_pos[mid] = (*w.pos, 0.0)
            h = float(np.arctan2(w.heading[1], w.heading[0])) / 2
            self.data.mocap_quat[mid] = (np.cos(h), 0.0, 0.0, np.sin(h))

    def _apply_doors(self) -> None:
        for f, a in self.door_angle.items():
            self.data.qpos[self._door_qadr[f]] = a
            self.data.qvel[self._door_dadr[f]] = 0.0

    def palm_pos(self, side: str) -> np.ndarray:
        b = self.hand_body[side]
        return self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ PALM_OFFSET

    # ---------------------------------------------------------------- attachments
    def attach(self, obj: str, side: str) -> None:
        b = self.hand_body[side]
        R = self.data.xmat[b].reshape(3, 3)
        a = self._obj_qadr[obj]
        rel_p = R.T @ (self.data.qpos[a:a + 3] - self.data.xpos[b])
        rel_yaw = _yaw_from_quat(self.data.qpos[a + 3:a + 7]) - self.base_pose()[2]
        self.attached[obj] = (side, rel_p, rel_yaw)
        self._ghosts.discard(obj)
        # Held objects are moved kinematically; letting them collide would inject
        # unbounded penetration forces into the scene. Disable until released.
        g = self._obj_geom[obj]
        self.model.geom_contype[g] = 0
        self.model.geom_conaffinity[g] = 0

    def detach(self, obj: str) -> None:
        if self.attached.pop(obj, None) is not None:
            # The magic grasp leaves the object overlapping the fingers/wrist; colliding
            # with them on release kicks it off whatever it was set down on (fatal for
            # a stack). Collide with the world only until it is clear of both hands.
            g = self._obj_geom[obj]
            self.model.geom_contype[g] = self.model.geom_conaffinity[g] = 2
            self._ghosts.add(obj)

    def _unghost(self, obj: str) -> None:
        self._ghosts.discard(obj)
        g = self._obj_geom[obj]
        self.model.geom_contype[g], self.model.geom_conaffinity[g] = self._obj_contact[obj]

    def _apply_attachments(self) -> None:
        if not self.attached:
            return
        mujoco.mj_kinematics(self.model, self.data)
        base_yaw = self.base_pose()[2]
        for obj, (side, rel_p, rel_yaw) in self.attached.items():
            b = self.hand_body[side]
            a, d = self._obj_qadr[obj], self._obj_dadr[obj]
            self.data.qpos[a:a + 3] = self.data.xpos[b] + self.data.xmat[b].reshape(3, 3) @ rel_p
            # Position follows the hand; orientation is kept level and turns with the body.
            # Position-only arm IK lets the wrist roll freely, so a rigid hand-frame
            # attachment would release cubes up to ~30 deg tilted. Part of the placeholder.
            h = (base_yaw + rel_yaw) / 2
            self.data.qpos[a + 3:a + 7] = (np.cos(h), 0.0, 0.0, np.sin(h))
            self.data.qvel[d:d + 6] = 0.0

    # ---------------------------------------------------------------- rendering
    def render(self, camera: str = "ego") -> np.ndarray:
        if self._renderer is None:
            w, h = self._render_size
            self._renderer = mujoco.Renderer(self.model, h, w)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        """Free the GL context. A server running many episodes ran out of memory without this."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

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
