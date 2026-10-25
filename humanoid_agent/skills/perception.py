"""Camera-only perception: the robot's belief about objects, built from what it has seen.

SIMULATED DETECTOR, not a vision model. An object counts as seen when its centre
(or top) is inside the ego camera's field of view, within MAX_RANGE, and the first
world geom on the ray from the camera to it is the object itself; the reported
position then gets noise that grows with distance. This isolates the planner's
problem (it only knows what it has looked at, and that can go stale) from the
separate problem of reading pixels.

Held objects are always known (proprioception). People are seen by field of view
and range only: the walker has no collision geometry to occlude or be occluded.
"""
from __future__ import annotations

import mujoco
import numpy as np

from ..sim.env import HumanoidEnv


class CameraMemory:
    MAX_RANGE = 4.0     # m; beyond this the detector reports nothing
    NOISE_BASE = 0.005  # m position noise (1 sigma) at the camera...
    NOISE_PER_M = 0.01  # ...plus this much per metre of distance

    def __init__(self, env: HumanoidEnv, seed: int = 0) -> None:
        self.env = env
        self.rng = np.random.default_rng(10_007 + seed)
        self.seen: dict[str, dict] = {}  # name -> {"pos", "on", "tilt_deg", "t"}
        self._cam = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "ego")

    def knows(self, name: str) -> bool:
        return name in self.seen

    def _view(self):
        d, m = self.env.data, self.env.model
        w, h = self.env._render_size
        ty = np.tan(np.radians(m.cam_fovy[self._cam]) / 2)
        return d.cam_xpos[self._cam].copy(), d.cam_xmat[self._cam].reshape(3, 3), ty * w / h, ty

    def _in_fov(self, p, view) -> bool:
        cp, R, tx, ty = view
        loc = R.T @ (np.asarray(p, float) - cp)
        depth = -loc[2]  # MuJoCo cameras look along their local -z
        return (0.05 < depth and np.linalg.norm(p - cp) < self.MAX_RANGE
                and abs(loc[0]) < tx * depth and abs(loc[1]) < ty * depth)

    def _unoccluded(self, name: str, p, cp) -> bool:
        v = p - cp
        dist = float(np.linalg.norm(v))
        gid = np.zeros(1, np.int32)
        # Group 0 only: world geoms (the robot's own meshes are group 3).
        r = mujoco.mj_ray(self.env.model, self.env.data, cp, v / dist, np.array([1, 0, 0, 0, 0, 0], np.uint8),
                          1, -1, gid)
        return r < 0 or (gid[0] >= 0 and int(self.env.model.geom_bodyid[gid[0]]) == self.env._obj_body[name])

    def visible(self, name: str, view=None) -> bool:
        view = view or self._view()
        p = self.env.object_pos(name)
        half = next(o.half_size for o in self.env.scene.objects if o.name == name)
        return any(self._in_fov(q, view) and self._unoccluded(name, q, view[0])
                   for q in (p, p + [0.0, 0.0, 0.8 * half]))

    def observe(self) -> list[str]:
        """Update beliefs from the current camera view; returns names seen for the first time."""
        env = self.env
        mujoco.mj_kinematics(env.model, env.data)
        mujoco.mj_camlight(env.model, env.data)
        view, new = self._view(), []
        for name in env.object_names:
            held = name in env.attached
            if not held and not self.visible(name, view):
                continue
            p = env.object_pos(name)
            if not held:
                sigma = self.NOISE_BASE + self.NOISE_PER_M * float(np.linalg.norm(p - view[0]))
                p = p + self.rng.normal(0.0, sigma, 3)
            if name not in self.seen:
                new.append(name)
            self.seen[name] = {"pos": p, "on": env.support_of(name), "tilt_deg": env.tilt_deg(name), "t": env.time}
        return new

    def in_view(self) -> list[str]:
        view = self._view()
        return [n for n in self.env.object_names if self.visible(n, view)]

    def people_in_view(self) -> list:
        view = self._view()
        return [w for w in self.env.people if self._in_fov(np.array([*w.pos, 1.0]), view)]
