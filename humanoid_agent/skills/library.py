"""Skill library: the planner's entire action space.

Design rules:
- Skills are blocking, bounded in sim time, and always return a SkillResult.
- Failure reasons are specific enough for a planner to recover (or for you to
  tell planner errors from skill errors in eval logs).
- Perception is ground truth for now (`list_objects`). Swap in a detector later
  behind the same signature.
"""
from __future__ import annotations

import io
from functools import wraps

import numpy as np
from PIL import Image

from ..control.arm_ik import solve_arm_ik
from ..sim.env import HumanoidEnv
from .types import SkillResult

SIDES = ("left", "right")


def _wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def skill(fn):
    """Common guards + timing for every skill."""

    @wraps(fn)
    def inner(self: "SkillLibrary", *args, **kwargs) -> SkillResult:
        if self.env.is_fallen():
            return SkillResult(False, "fallen: robot is on the ground")
        t0 = self.env.time
        try:
            res = fn(self, *args, **kwargs)
        except (KeyError, ValueError) as e:
            res = SkillResult(False, f"bad_args: {e}")
        if self.env.is_fallen() and res.success:
            res = SkillResult(False, "fallen: robot fell during skill", res.data)
        res.sim_time_s = round(self.env.time - t0, 3)
        self.log.append({"skill": fn.__name__, "args": args, "kwargs": kwargs, **res.to_dict()})
        return res

    inner.is_skill = True
    return inner


class SkillLibrary:
    def __init__(self, env: HumanoidEnv, frame_dir: str | None = None) -> None:
        self.env = env
        self.log: list[dict] = []
        self.frame_dir = frame_dir
        self._frame_count = 0

    # ================================================================ queries
    @skill
    def get_state(self) -> SkillResult:
        x, y, yaw = self.env.base_pose()
        held = {s: [o for o, (side, *_) in self.env.attached.items() if side == s] for s in SIDES}
        return SkillResult(True, data={
            "base": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)},
            "holding": held,
            "sim_time_s": round(self.env.time, 2),
            "locomotion_is_physical": self.env.loco.is_physical,
        })

    @skill
    def list_objects(self) -> SkillResult:
        x, y, _ = self.env.base_pose()
        out = []
        for name in self.env.object_names:
            p = self.env.object_pos(name)
            out.append({
                "name": name,
                "pos": [round(float(v), 3) for v in p],
                "on": "held" if name in self.env.attached else self._support_of(p),
                "dist_m": round(float(np.hypot(p[0] - x, p[1] - y)), 2),
                "held_by": self.env.attached.get(name, (None,))[0],
            })
        return SkillResult(True, data={"objects": out})

    @skill
    def list_locations(self) -> SkillResult:
        return SkillResult(True, data={"locations": {
            k: {"x": v[0], "y": v[1], "yaw": v[2]} for k, v in self.env.scene.locations.items()
        }})

    @skill
    def get_frame(self, camera: str = "ego") -> SkillResult:
        img = self.env.render(camera)
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        data = {"png_bytes": buf.getvalue(), "camera": camera, "shape": list(img.shape)}
        if self.frame_dir:
            path = f"{self.frame_dir}/frame_{self._frame_count:04d}_{camera}.png"
            Image.fromarray(img).save(path)
            data["path"] = path
            self._frame_count += 1
        return SkillResult(True, data=data)

    # ================================================================ locomotion
    @skill
    def walk_to(self, x: float, y: float, yaw: float | None = None, timeout_s: float = 20.0) -> SkillResult:
        if yaw is not None and abs(float(yaw)) > 2 * np.pi + 1e-6:
            return SkillResult(False, f"bad_args: yaw is in radians, got {yaw} (degrees?)")
        return self._walk_to(float(x), float(y), None if yaw is None else float(yaw), timeout_s)

    @skill
    def go_to_location(self, name: str, timeout_s: float = 20.0) -> SkillResult:
        if name not in self.env.scene.locations:
            return SkillResult(False, f"unknown_location: {name}; known={list(self.env.scene.locations)}")
        x, y, yaw = self.env.scene.locations[name]
        return self._walk_to(x, y, yaw, timeout_s)

    @skill
    def turn_to(self, yaw: float, timeout_s: float = 8.0) -> SkillResult:
        if abs(float(yaw)) > 2 * np.pi + 1e-6:
            return SkillResult(False, f"bad_args: yaw is in radians, got {yaw} (degrees?)")
        x, y, _ = self.env.base_pose()
        return self._walk_to(x, y, float(yaw), timeout_s)

    def _walk_to(self, gx: float, gy: float, gyaw: float | None, timeout_s: float,
                 pos_tol: float = 0.02, yaw_tol: float = 0.05) -> SkillResult:
        # The kinematic base passes through furniture (and the body then wrecks the
        # table), so refuse straight lines through tables; detour via one corner if possible.
        x, y, _ = self.env.base_pose()
        blocker = self._path_blocked((x, y), (gx, gy))
        if blocker is not None:
            route = self._route((x, y), (gx, gy))
            if route is None:
                return SkillResult(False, f"blocked_path: no clear route to ({gx:.2f}, {gy:.2f}); "
                                          f"straight line crosses {blocker.name}")
            for wx, wy in route:
                r = self._walk_to(float(wx), float(wy), None, timeout_s)
                if not r.success:
                    return r
            r = self._walk_to(gx, gy, gyaw, timeout_s)
            if r.success:
                r.data["detour_via"] = [[round(float(v), 2) for v in w] for w in route]
            return r
        env, loco = self.env, self.env.loco
        t_end = env.time + timeout_s
        while env.time < t_end:
            x, y, yaw = env.base_pose()
            dx, dy = gx - x, gy - y
            dist = float(np.hypot(dx, dy))
            if dist > pos_tol:
                heading = np.arctan2(dy, dx)
                herr = _wrap_angle(heading - yaw)
                # Turn first if badly misaligned, then walk forward with heading correction.
                vx = 0.0 if abs(herr) > 0.6 else min(0.5, 1.2 * dist)
                loco.set_command(vx, 0.0, float(np.clip(2.0 * herr, -1.0, 1.0)))
            elif gyaw is not None and abs(_wrap_angle(gyaw - yaw)) > yaw_tol:
                loco.set_command(0.0, 0.0, float(np.clip(2.0 * _wrap_angle(gyaw - yaw), -1.0, 1.0)))
            else:
                loco.stop()
                env.step(5)
                x, y, yaw = env.base_pose()
                return SkillResult(True, data={"base": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}})
            env.step()
            if env.is_fallen():
                loco.stop()
                return SkillResult(False, "fallen: lost balance while walking")
        loco.stop()
        x, y, yaw = env.base_pose()
        return SkillResult(False, f"timeout: {np.hypot(gx - x, gy - y):.2f}m from goal",
                           {"base": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}})

    @skill
    def look_at(self, target: str) -> SkillResult:
        """Turn the body to face an object or location and tilt the ego camera toward it."""
        env = self.env
        if target in env.object_names:
            p = env.object_pos(target)
        elif target in env.scene.locations:
            lx, ly, _ = env.scene.locations[target]
            p = np.array([lx, ly, 0.75])
        else:
            return SkillResult(False, f"unknown_target: {target}")
        x, y, _ = env.base_pose()
        yaw = float(np.arctan2(p[1] - y, p[0] - x))
        res = self._walk_to(x, y, yaw, 8.0)
        if not res.success:
            return res
        cam_h = 1.25  # approx ego camera height
        pitch = float(np.arctan2(cam_h - p[2], max(np.hypot(p[0] - x, p[1] - y), 0.05)))
        env.set_ego_pitch(pitch)
        env.step()
        return SkillResult(True, data={"yaw": round(yaw, 3), "camera_pitch": round(pitch, 3)})

    # ================================================================ manipulation primitives
    @skill
    def reach(self, side: str, x: float, y: float, z: float, timeout_s: float = 3.0) -> SkillResult:
        return self._reach(side, np.array([x, y, z], dtype=float), timeout_s)

    def _reach(self, side: str, target: np.ndarray, timeout_s: float = 3.0, tol: float = 0.03) -> SkillResult:
        if side not in SIDES:
            return SkillResult(False, f"bad_args: side must be one of {SIDES}")
        env = self.env
        # Cartesian path: lift to target height near the body first (avoids sweeping
        # through table tops), then a straight line to the target. IK is warm-started
        # along the path so consecutive joint targets stay close.
        start = env.palm_pos(side)
        corners = [start]
        if start[2] < target[2] - 0.02:
            corners.append(np.array([start[0], start[1], target[2] + 0.03]))
        corners.append(target)
        path = []
        for a, b in zip(corners[:-1], corners[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.02))
            path += [a + (b - a) * (i / n) for i in range(1, n + 1)]
        seed, qs = env.data.qpos[env.arm_qadr[side]].copy(), []
        for wp in path:
            seed, resid = solve_arm_ik(env, side, wp, seed=seed, iters=60)
            if resid > 0.04:
                return SkillResult(False, f"unreachable: ik residual {resid:.3f}m; move closer",
                                   {"ik_residual_m": round(resid, 3)})
            qs.append(seed)
        for q in qs:  # stream waypoints at control rate
            env.arm_target[side] = q
            env.step()
        t_end = env.time + timeout_s
        while env.time < t_end:
            if np.linalg.norm(env.palm_pos(side) - target) < tol:
                return SkillResult(True, data={"palm": [round(float(v), 3) for v in env.palm_pos(side)]})
            env.step()
        err = float(np.linalg.norm(env.palm_pos(side) - target))
        return SkillResult(False, f"tracking: palm {err:.3f}m from target (collision or torque limit?)",
                           {"palm_err_m": round(err, 3)})

    @skill
    def grasp(self, side: str, obj: str, max_dist: float = 0.08) -> SkillResult:
        """Close the hand on an object already at the palm. PLACEHOLDER: 'magic' attach."""
        env = self.env
        if obj not in env.object_names:
            return SkillResult(False, f"unknown_object: {obj}")
        if any(s == side for s, *_ in env.attached.values()):
            return SkillResult(False, f"hand_full: {side} hand is already holding something")
        d = float(np.linalg.norm(env.object_pos(obj) - env.palm_pos(side)))
        if d > max_dist:
            return SkillResult(False, f"too_far: {obj} is {d:.3f}m from {side} palm; reach it first",
                               {"dist_m": round(d, 3)})
        env.attach(obj, side)
        env.hand_target[side] = self._closed_hand(side)
        env.step(10)
        return SkillResult(True, data={"holding": obj})

    @skill
    def release(self, side: str) -> SkillResult:
        env = self.env
        held = [o for o, (s, *_) in env.attached.items() if s == side]
        if not held:
            return SkillResult(False, f"not_holding: {side} hand is empty")
        for o in held:
            env.detach(o)
        env.hand_target[side] = env.stand_ctrl[env.hand_act[side]].copy()
        env.step(25)
        return SkillResult(True, data={"released": held,
                                       "landed_on": [self._support_of(env.object_pos(o)) for o in held]})

    @skill
    def retract(self, side: str) -> SkillResult:
        return self._retract(side)

    def _retract(self, side: str) -> SkillResult:
        """Lift clear, pull the hand back beside the body, then return to the rest pose.

        Going straight to the rest pose in joint space sweeps the hand across the
        table and knocks things off.
        """
        env = self.env
        self._reach(side, env.palm_pos(side) + [0, 0, 0.10], timeout_s=1.0, tol=0.05)
        lat = 0.22 if side == "left" else -0.22
        # Pull back behind the table edge (fingers extend ~8cm past the palm point).
        self._reach(side, self._body_to_world(-0.03, lat, env.palm_pos(side)[2]), timeout_s=1.5, tol=0.03)
        # Lower beside the hip in Cartesian space before snapping to the rest joints.
        self._reach(side, self._body_to_world(-0.02, 1.1 * lat, 0.66), timeout_s=1.5, tol=0.04)
        env.arm_target[side] = env.stand_ctrl[env.arm_act[side]].copy()
        env.step(40)
        rest = self._body_to_world(0.0, 1.09 * lat, 0.62)
        if np.linalg.norm(env.palm_pos(side) - rest) > 0.08:
            return SkillResult(False, f"retract_blocked: {side} arm did not reach rest pose")
        return SkillResult(True)

    # ================================================================ composites
    @skill
    def pick(self, obj: str, side: str | None = None) -> SkillResult:
        """Approach from above, grasp, lift. Requires standing within reach."""
        env = self.env
        if obj not in env.object_names:
            return SkillResult(False, f"unknown_object: {obj}")
        p = env.object_pos(obj)
        if len({s for s, *_ in env.attached.values()}) == len(SIDES):
            return SkillResult(False, "hand_full: both hands are holding something; place one first")
        if side is None:
            side = self._side_for(p)
            if any(s == side for s, *_ in env.attached.values()):
                side = "left" if side == "right" else "right"
        if any(s == side for s, *_ in env.attached.values()):
            return SkillResult(False, f"hand_full: {side} hand is already holding something")
        a = self._align_for(p, side)
        if not a.success:
            return a
        r = self._reach(side, p + [0, 0, 0.10], tol=0.05)
        if not r.success:
            return SkillResult(False, f"approach_{r.reason}", r.data)
        r = self._reach(side, p + [0, 0, 0.045])  # palm just above the object; grasp is magic
        if not r.success:
            return SkillResult(False, f"descend_{r.reason}", r.data)
        g = SkillLibrary.grasp.__wrapped__(self, side, obj)
        if not g.success:
            return g
        lift = self._reach(side, env.palm_pos(side) + [0, 0, 0.12])
        carry = self._reach(side, self._body_to_world(0.22, 0.18 if side == "left" else -0.18, 0.95))
        return SkillResult(True, data={"holding": obj, "side": side,
                                       "lifted": lift.success, "in_carry_pose": carry.success})

    @skill
    def place_on(self, surface: str, side: str | None = None,
                 dx: float = 0.0, dy: float = 0.0, obj: str | None = None) -> SkillResult:
        """Place a held object on a table, in a free spot ~30cm in front of the robot.

        Select the object with `obj` (or the hand with `side`); with both hands full
        and neither given, refuse rather than guess. (dx, dy) shift the preferred
        spot in the robot's body frame (forward, left).
        """
        env = self.env
        held = {s: o for o, (s, *_) in env.attached.items()}
        if not held:
            return SkillResult(False, "not_holding: nothing in either hand")
        if obj is not None:
            if obj not in env.attached:
                return SkillResult(False, f"not_holding: {obj} is not in a hand; holding {held}")
            if side is not None and held.get(side) != obj:
                return SkillResult(False, f"bad_args: {obj} is in the {env.attached[obj][0]} hand, not {side}")
            side = env.attached[obj][0]
        elif side is None:
            if len(held) > 1:
                return SkillResult(False, f"ambiguous: holding {held}; pass obj to say which to place")
            side = next(iter(held))
        if side not in held:
            return SkillResult(False, f"not_holding: {side} hand is empty")
        table = next((t for t in env.scene.tables if t.name == surface), None)
        if table is None:
            return SkillResult(False, f"unknown_surface: {surface}")
        x, y, _ = env.base_pose()
        gap = np.hypot(max(abs(x - table.pos[0]) - table.size[0], 0), max(abs(y - table.pos[1]) - table.size[1], 0))
        if gap > 0.5:
            return SkillResult(False, f"too_far: {surface} is {gap:.2f}m away; go_to_location {surface} first")
        obj = held[side]
        spot = self._free_spot(table, obj, side, dx, dy)
        if spot is None:
            return SkillResult(False, f"no_free_space: no clear reachable spot on {surface}; "
                                      f"reposition or clear the table")
        a = self._align_for(np.array([spot[0], spot[1], table.height]), side)
        if not a.success:
            return a
        # Palm target such that the held object's centre ends up just above the surface.
        half = next(o.half_size for o in env.scene.objects if o.name == obj)
        palm_minus_obj = env.palm_pos(side) - env.object_pos(obj)
        target = np.array([spot[0], spot[1], table.height + half + 0.01]) + palm_minus_obj
        r = self._reach(side, target + [0, 0, 0.10], tol=0.05)
        if not r.success:
            return SkillResult(False, f"approach_{r.reason}", r.data)
        r = self._reach(side, target)
        if not r.success:
            return SkillResult(False, f"descend_{r.reason}", r.data)
        rel = SkillLibrary.release.__wrapped__(self, side)
        ret = self._retract(side)
        landed = self._support_of(env.object_pos(obj))
        if landed != surface:
            return SkillResult(False, f"placement: {obj} ended up on {landed}", rel.data)
        return SkillResult(True, data={"placed": obj, "on": surface, "arm_retracted": ret.success})

    # ================================================================ helpers
    BODY_RADIUS = 0.10  # base keep-out around table footprints (m)

    def _in_table(self, t, p, margin: float) -> bool:
        return abs(p[0] - t.pos[0]) <= t.size[0] + margin and abs(p[1] - t.pos[1]) <= t.size[1] + margin

    def _path_blocked(self, a, b):
        """First table the straight base path a->b enters, or None.

        Starting inside a table's keep-out band (e.g. after stepping up to place)
        only forbids entering the table itself, so the robot can always back away.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        n = max(2, int(np.linalg.norm(b - a) / 0.02))
        pts = [a + (b - a) * (i / n) for i in range(1, n + 1)]
        for t in self.env.scene.tables:
            margin = 0.0 if self._in_table(t, a, self.BODY_RADIUS) else self.BODY_RADIUS
            if any(self._in_table(t, p, margin) for p in pts):
                return t
        return None

    def _route(self, a, b, clearance: float = 0.25):
        """Shortest waypoint list a->b through table corners (visibility graph), or None."""
        nodes = [np.asarray(a, float), np.asarray(b, float)] + [
            np.array([t.pos[0] + sx * (t.size[0] + clearance), t.pos[1] + sy * (t.size[1] + clearance)])
            for t in self.env.scene.tables for sx in (-1, 1) for sy in (-1, 1)]
        dist, prev, todo = {0: 0.0}, {}, set(range(len(nodes)))
        while todo:
            i = min((j for j in todo if j in dist), key=dist.get, default=None)
            if i is None or i == 1:
                break
            todo.discard(i)
            for j in todo:
                if self._path_blocked(nodes[i], nodes[j]) is None:
                    d = dist[i] + float(np.linalg.norm(nodes[j] - nodes[i]))
                    if d < dist.get(j, np.inf):
                        dist[j], prev[j] = d, i
        if 1 not in dist:
            return None
        route, i = [], prev[1]
        while i != 0:
            route.append(nodes[i])
            i = prev[i]
        return route[::-1]

    def _free_spot(self, table, obj: str, side: str, dx: float, dy: float,
                   margin: float = 0.06, clearance: float = 0.15, max_shift: float = 0.45):
        """Clear point on the table within a short side-step, closest to the preferred spot.

        Preferred spot: 30cm ahead, in front of the placing hand, shifted by (dx, dy)
        in the body frame. Candidates are scored by distance to that spot.
        """
        lat_sign = 1.0 if side == "left" else -1.0
        pref = self._body_to_world(self.REACH_FWD + dx, lat_sign * self.REACH_LAT + dy, 0.0)[:2]
        others = [self.env.object_pos(o)[:2] for o in self.env.object_names if o != obj]
        best, best_d = None, np.inf
        for gx in np.arange(-table.size[0] + margin, table.size[0] - margin + 1e-6, 0.03):
            for gy in np.arange(-table.size[1] + margin, table.size[1] - margin + 1e-6, 0.03):
                q = np.array([table.pos[0] + gx, table.pos[1] + gy])
                d = float(np.linalg.norm(q - pref))
                if d > max_shift or d >= best_d:
                    continue
                if any(np.linalg.norm(q - o) < clearance for o in others):
                    continue
                best, best_d = q, d
        return best

    REACH_FWD = 0.30   # comfortable forward reach at table height (m, body frame)
    REACH_LAT = 0.15   # lateral offset that puts a point in front of one hand

    def _align_for(self, p: np.ndarray, side: str, tol: float = 0.04) -> SkillResult:
        """Side-step / step so world point p sits at (REACH_FWD, ±REACH_LAT) in the body frame."""
        x, y, yaw = self.env.base_pose()
        c, s_ = np.cos(yaw), np.sin(yaw)
        lat = self.REACH_LAT if side == "left" else -self.REACH_LAT
        bx = p[0] - (c * self.REACH_FWD - s_ * lat)
        by = p[1] - (s_ * self.REACH_FWD + c * lat)
        if np.hypot(bx - x, by - y) < tol:
            return SkillResult(True)
        if np.hypot(bx - x, by - y) > 0.5:
            return SkillResult(False, "too_far: target is more than 0.5m from a reachable stance; walk closer first")
        r = self._step_to(bx, by, yaw)
        if not r.success:
            return SkillResult(False, f"align_{r.reason}", r.data)
        return r

    def _step_to(self, gx: float, gy: float, gyaw: float, timeout_s: float = 6.0, tol: float = 0.02) -> SkillResult:
        """Short holonomic step (side-step allowed) without turning toward the goal."""
        env, loco = self.env, self.env.loco
        t_end = env.time + timeout_s
        while env.time < t_end:
            x, y, yaw = env.base_pose()
            ex, ey = gx - x, gy - y
            if np.hypot(ex, ey) < tol and abs(_wrap_angle(gyaw - yaw)) < 0.05:
                loco.stop()
                env.step(3)
                return SkillResult(True)
            c, s_ = np.cos(yaw), np.sin(yaw)
            vx, vy = c * ex + s_ * ey, -s_ * ex + c * ey
            loco.set_command(float(np.clip(1.5 * vx, -0.3, 0.3)), float(np.clip(1.5 * vy, -0.3, 0.3)),
                             float(np.clip(2.0 * _wrap_angle(gyaw - yaw), -0.8, 0.8)))
            env.step()
        loco.stop()
        return SkillResult(False, "timeout: side-step did not converge")

    def _support_of(self, p: np.ndarray) -> str:
        for t in self.env.scene.tables:
            if (abs(p[0] - t.pos[0]) <= t.size[0] and abs(p[1] - t.pos[1]) <= t.size[1]
                    and p[2] > t.height - 0.02):
                return t.name
        return "floor" if p[2] < 0.15 else "air"

    def _body_to_world(self, fx: float, fy: float, z: float) -> np.ndarray:
        x, y, yaw = self.env.base_pose()
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([x + c * fx - s * fy, y + s * fx + c * fy, z])

    def _side_for(self, p: np.ndarray) -> str:
        x, y, yaw = self.env.base_pose()
        local_y = -np.sin(yaw) * (p[0] - x) + np.cos(yaw) * (p[1] - y)
        return "left" if local_y > 0 else "right"

    def _closed_hand(self, side: str) -> np.ndarray:
        m, env = self.env.model, self.env
        ids = env.hand_act[side]
        lo, hi = m.actuator_ctrlrange[ids, 0], m.actuator_ctrlrange[ids, 1]
        # Move each finger 60% toward whichever limit is farther from the open pose.
        open_ = env.stand_ctrl[ids]
        far = np.where(np.abs(hi - open_) > np.abs(lo - open_), hi, lo)
        return open_ + 0.6 * (far - open_)

    def skills(self) -> dict[str, callable]:
        return {n: getattr(self, n) for n in dir(self)
                if not n.startswith("_") and getattr(getattr(self, n), "is_skill", False)}
