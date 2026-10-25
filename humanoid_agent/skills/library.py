"""Skill library: the planner's entire action space.

Design rules:
- Skills are blocking, bounded in sim time, and always return a SkillResult.
- Failure reasons are specific enough for a planner to recover (or for you to
  tell planner errors from skill errors in eval logs).
- Perception is ground truth (`list_objects`) unless the `camera` difficulty
  switch is on: then it is what the head camera has seen (`perception.py`), and
  skills refuse object names the robot has not seen yet (`not_seen`).
"""
from __future__ import annotations

import io
import random
from functools import wraps

import mujoco
import numpy as np
from PIL import Image

from ..control.arm_ik import solve_arm_ik
from ..sim.env import HumanoidEnv
from .difficulty import Difficulty
from .perception import CameraMemory
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
        if self.memory is not None:  # decide on what is in view now, not on what was there last skill
            self.memory.observe()
        try:
            res = fn(self, *args, **kwargs)
        except (KeyError, ValueError) as e:
            res = SkillResult(False, f"bad_args: {e}")
        if self.env.is_fallen() and res.success:
            res = SkillResult(False, "fallen: robot fell during skill", res.data)
        res.sim_time_s = round(self.env.time - t0, 3)
        if self.memory is not None and (new := self.memory.observe()):
            res.data["newly_seen"] = new
        sup = self.env.supervisor
        if sup is not None:
            sup.tick(self.env)
            if said := sup.drain():
                res.data["supervisor_says"] = said
        self.log.append({"skill": fn.__name__, "args": args, "kwargs": kwargs, **res.to_dict()})
        return res

    inner.is_skill = True
    return inner


class SkillLibrary:
    def __init__(self, env: HumanoidEnv, frame_dir: str | None = None,
                 difficulty: Difficulty | None = None, seed: int = 0) -> None:
        self.env = env
        self.log: list[dict] = []
        self.frame_dir = frame_dir
        self._frame_count = 0
        self.difficulty = difficulty or Difficulty()
        self.memory = CameraMemory(env, seed) if self.difficulty.camera else None
        self.faults: list[dict] = []  # injected failures, for the eval row (never shown to the planner)
        self._fault_rng = random.Random(7919 * seed + 1)
        self._grasps = 0

    def _unseen(self, name: str) -> SkillResult | None:
        """not_seen for a name the camera has not seen. Checked before existence, so a
        made-up name and a real but unseen one get the same answer (no ground-truth leak)."""
        if self.memory is None or self.memory.knows(name):
            return None
        return SkillResult(False, f"not_seen: {name} has not been seen yet; look around for it "
                                  f"(look_around, go_to_location, open_door) or check list_objects")

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
            **({"doors": {f: ("open" if self.env.door_is_open(f) else "closed" if a < 0.05 else "ajar")
                          for f, a in self.env.door_angle.items()}} if self.env.door_angle else {}),
        })

    @skill
    def list_objects(self) -> SkillResult:
        x, y, _ = self.env.base_pose()
        if self.memory is not None:
            return self._list_seen()
        out, unseen = [], sorted({self.env.fridge_of(self.env.support_of(o)).name
                                  for o in self.env.object_names if self.env.hidden(o)})
        for name in self.env.object_names:
            if self.env.hidden(name):
                continue
            p = self.env.object_pos(name)
            out.append({
                "name": name,
                "pos": [round(float(v), 3) for v in p],
                "on": self.env.support_of(name),
                "tilt_deg": round(self.env.tilt_deg(name)),
                "dist_m": round(float(np.hypot(p[0] - x, p[1] - y)), 2),
                "held_by": self.env.attached.get(name, (None,))[0],
            })
        # Things to put objects on, so a planner can learn names like "bean_bag".
        data = {"objects": out, "surfaces": self._surfaces()}
        if self.env.people:  # ground truth, like objects: position and current velocity
            data["people"] = [w.to_dict((x, y)) for w in self.env.people]
        if unseen:
            data["not_visible"] = f"{', '.join(unseen)} closed: open_door to see inside"
        return SkillResult(True, data=data)

    def _surfaces(self) -> list[dict]:
        # Furniture is the map, known in every perception mode. Half extents let a
        # planner work out where to stand for the far side of a table.
        surfaces = [{"name": t.name, "pos": [round(t.pos[0], 3), round(t.pos[1], 3), t.height],
                     "half_size": [round(t.size[0], 3), round(t.size[1], 3)]} for t in self._supports()]
        for b in self.env.scene.bags:
            top, owner = self.env.surface_below(b.pos[0], b.pos[1], b.base_z + b.height + 0.02)
            surfaces.append({"name": b.name, "pos": [round(b.pos[0], 3), round(b.pos[1], 3), round(top, 3)],
                             "on": self.env.support_of_point(b.pos[0], b.pos[1], b.base_z), "soft": True})
        return surfaces

    def _list_seen(self) -> SkillResult:
        """Camera mode: beliefs, not ground truth. Positions are as last seen (noisy, maybe stale)."""
        env, mem = self.env, self.memory
        x, y, _ = env.base_pose()
        now = set(mem.in_view())
        out = [{"name": n, "pos": [round(float(v), 3) for v in b["pos"]], "on": b["on"],
                "tilt_deg": round(b["tilt_deg"]), "dist_m": round(float(np.hypot(b["pos"][0] - x, b["pos"][1] - y)), 2),
                "held_by": env.attached.get(n, (None,))[0],
                "in_view": n in now or n in env.attached, "last_seen_s_ago": round(env.time - b["t"], 1)}
               for n, b in mem.seen.items()]
        data = {"perception": "camera: only objects your head camera has seen so far, where they were when last "
                              "seen. Anything else is unknown, not absent: look around (look_around, "
                              "go_to_location, look_at) to see more.",
                "objects": out, "surfaces": self._surfaces()}
        if env.people:
            data["people"] = [w.to_dict((x, y)) for w in mem.people_in_view()]
        closed = [f.name for f in env.scene.fridges if env.door_angle[f.name] < 0.3]
        if closed:  # every closed fridge, empty or not: listing only non-empty ones would leak contents
            data["not_visible"] = f"{', '.join(closed)} closed: contents unknown until you open_door"
        return SkillResult(True, data=data)

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
        if pos_tol < np.hypot(gx - x, gy - y) < self.SHORT_STEP:
            # Turning to face a goal a few cm away swings the arms through whatever the robot
            # stands at (it knocked a can out of an open fridge): side-step without turning.
            r = self._step_to(gx, gy, env.base_pose()[2], timeout_s)
            if not r.success:
                return r
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
                if vx > 0 and (blocked := self._person_blocking((np.cos(yaw), np.sin(yaw)), vx)):
                    return blocked
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
        if target not in env.scene.locations and (r := self._unseen(target)):
            return r
        if self.memory is not None and target in self.memory.seen:
            p = self.memory.seen[target]["pos"]
        elif target in env.object_names:
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

    DEFAULT_PITCH = 0.52  # rad down: the ego camera's mounting tilt (scene.py), which sees table tops at 1-3 m

    @skill
    def look_around(self) -> SkillResult:
        """Turn in place through four headings, looking at each, then face the original heading."""
        env = self.env
        x, y, yaw0 = env.base_pose()
        env.set_ego_pitch(self.DEFAULT_PITCH)
        seen = set()
        for k in (1, 2, 3, 4):
            r = self._walk_to(x, y, float(_wrap_angle(yaw0 + k * np.pi / 2)), 8.0)
            if not r.success:
                return r
            if self.memory is not None:
                self.memory.observe()
                seen.update(self.memory.in_view())
        data = {"heading": round(env.base_pose()[2], 3)}
        if self.memory is not None:
            data["seen"] = sorted(seen)
        return SkillResult(True, data=data)

    # ================================================================ fixtures
    @skill
    def open_door(self, fridge: str, timeout_s: float = 20.0) -> SkillResult:
        """Touch the handle, step back out of the door's swing, swing it open, step back in.

        PLACEHOLDER: the hand only touches the handle; the door itself is swung
        kinematically (no pulling force, no contact while it moves).
        """
        env = self.env
        f = env.fridge(fridge)
        if f is None:
            return SkillResult(False, f"unknown_target: {fridge}; fridges={[g.name for g in env.scene.fridges]}")
        if env.door_is_open(fridge):
            return SkillResult(True, data={"door": "open", "already": True})
        free = [s for s in SIDES if s not in {s_ for s_, *_ in env.attached.values()}]
        if not free:
            return SkillResult(False, "hand_full: both hands are full; a hand is needed for the handle")
        stance = f.stance(self.FRIDGE_STANDOFF)
        x, y, _ = env.base_pose()
        if np.hypot(stance[0] - x, stance[1] - y) > 1.0:
            return SkillResult(False, f"too_far: {fridge} is out of reach; go_to_location {fridge} first")
        r = self._walk_to(*stance, timeout_s)
        if not r.success:
            return r
        # The handle is on the robot's left (hinge on the right): prefer the left hand.
        side = "left" if "left" in free else free[0]
        hid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, f"{fridge}_handle")
        handle = env.data.site_xpos[hid].copy()
        grip = handle - 0.05 * self._fwd()
        r = self._align_for(grip, side)  # side-steps over when only the far hand is free
        if r.success:
            r = self._reach(side, grip, tol=0.05)
        if not r.success:
            self._retract(side)
            return SkillResult(False, f"handle_{r.reason}", r.data)
        env.hand_target[side] = self._closed_hand(side)
        env.step(10)
        env.hand_target[side] = env.stand_ctrl[env.hand_act[side]].copy()
        self._retract(side)
        r = self._swing(f, f.open_angle, stance, timeout_s)
        if not r.success:
            return r
        return SkillResult(True, data={"door": "open", "hand": side})

    @skill
    def close_door(self, fridge: str, timeout_s: float = 20.0) -> SkillResult:
        """Step back out of the door's swing and swing it shut. PLACEHOLDER: kinematic swing."""
        env = self.env
        f = env.fridge(fridge)
        if f is None:
            return SkillResult(False, f"unknown_target: {fridge}; fridges={[g.name for g in env.scene.fridges]}")
        if env.door_angle[fridge] < 0.05:
            return SkillResult(True, data={"door": "closed", "already": True})
        x, y, _ = env.base_pose()
        back = f.stance(self._door_clearance(f))
        if np.hypot(back[0] - x, back[1] - y) > 1.5:
            return SkillResult(False, f"too_far: {fridge} is out of reach; go_to_location {fridge} first")
        for s_ in SIDES:  # an arm left inside would be hit by the door
            if not any(s == s_ for s, *_ in env.attached.values()):
                self._retract(s_)
        r = self._swing(f, 0.0, None, timeout_s)
        if not r.success:
            return r
        return SkillResult(True, data={"door": "closed"})

    FRIDGE_STANDOFF = 0.16  # base to door face when working at a fridge (m)

    def _door_clearance(self, f) -> float:
        """Standoff that keeps the base ~0.25m outside the door's swing radius."""
        w = 2 * f.size[1]
        return float(np.sqrt((w + 0.25) ** 2 - f.size[1] ** 2)) - f.door_thick / 2

    def _swing(self, f, angle: float, then, timeout_s: float, duration: float = 1.5) -> SkillResult:
        """Back out of the swing, move the door to `angle`, then walk to pose `then` (if given)."""
        env = self.env
        back = f.stance(self._door_clearance(f))
        r = self._walk_to(*back, timeout_s)
        if not r.success:
            return SkillResult(False, f"backup_{r.reason}", r.data)
        a0, n = env.door_angle[f.name], max(1, int(duration / env.control_dt))
        for i in range(1, n + 1):
            u = i / n
            env.door_angle[f.name] = a0 + (angle - a0) * (3 * u * u - 2 * u ** 3)
            env.step()
        env.step(10)
        if then is not None:
            r = self._walk_to(*then, timeout_s)
            if not r.success:
                return r
        return SkillResult(True)

    def _fwd(self) -> np.ndarray:
        _, _, yaw = self.env.base_pose()
        return np.array([np.cos(yaw), np.sin(yaw), 0.0])

    def _outside(self, side: str, p: np.ndarray, fwd: float = 0.10) -> SkillResult:
        """Move the palm to p's height and lateral offset, but only `fwd` ahead of the base.

        Entering or leaving a fridge on that waypoint keeps the hand from sweeping
        diagonally through the shelf edge above or below.
        """
        x, y, yaw = self.env.base_pose()
        lat = -np.sin(yaw) * (p[0] - x) + np.cos(yaw) * (p[1] - y)
        return self._reach(side, self._body_to_world(fwd, lat, float(p[2])), tol=0.04)

    def _half(self, obj: str) -> float:
        return next(o.half_size for o in self.env.scene.objects if o.name == obj)

    def _supports(self) -> list:
        return self.env.scene.tables + [t for f in self.env.scene.fridges for t in f.shelf_tables()]

    def _obstacles(self) -> list:
        """Footprints the base must route around: tables, fridges, and any open door."""
        out = list(self.env.scene.tables)
        for f in self.env.scene.fridges:
            out.append(f.footprint)
            if self.env.door_angle[f.name] > 0.05:
                out.append(f.door_footprint(self.env.door_angle[f.name]))
        return out

    SHORT_STEP = 0.35  # goals closer than this are side-stepped to, not walked to (m)

    PERSON_STOP = 0.9    # stop if a person is this close (base to body centre) in the walking cone
    PERSON_CONE = 1.05    # rad half-angle of that cone (~60 deg)
    PERSON_PASS = 0.6     # ...or if the predicted closest approach over the next second is below this
    PERSON_LOOK = 1.2     # (only for people this close: farther ones are reassessed as they approach)

    def _person_blocking(self, direction, speed: float) -> SkillResult | None:
        """Stop the base for a person ahead (or about to cut across) and say where they are.

        Ground-truth positions and velocities (no perception yet). Turning in place is
        never blocked; only translation is checked.
        """
        env = self.env
        if not env.people:
            return None
        x, y, yaw = env.base_pose()
        u = np.asarray(direction, float)
        u /= max(np.linalg.norm(u), 1e-9)
        for w in env.people:
            rel = w.pos - (x, y)
            d = float(np.linalg.norm(rel))
            ahead = d < self.PERSON_STOP and rel @ u > np.cos(self.PERSON_CONE) * d
            vrel = w.vel - speed * u
            t = float(np.clip(-(rel @ vrel) / max(vrel @ vrel, 1e-9), 0.0, 1.0))
            crossing = (d < self.PERSON_LOOK and rel @ u > 0
                        and float(np.linalg.norm(rel + vrel * t)) < self.PERSON_PASS)
            if ahead or crossing:
                env.loco.stop()
                env.step()
                p = w.to_dict((x, y))
                where = "ahead" if ahead else "about to cross your path"
                moving = "standing still" if not np.any(p["velocity"]) else f"moving {p['velocity']} m/s"
                return SkillResult(False, f"blocked_by_person: {w.spec.name} {where} at {p['pos']}, {p['dist_m']} m "
                                          f"away, {moving}; wait, or go around them",
                                   {"person": p, "base": {"x": round(x, 3), "y": round(y, 3), "yaw": round(yaw, 3)}})
        return None

    # ================================================================ waiting
    @skill
    def wait(self, seconds: float = 2.0) -> SkillResult:
        """Stand still for up to 10 s of sim time (people keep moving)."""
        seconds = float(seconds)
        if not 0 < seconds <= 10:
            return SkillResult(False, f"bad_args: seconds must be in (0, 10], got {seconds}")
        self.env.loco.stop()
        self.env.step(max(1, int(round(seconds / self.env.control_dt))))
        x, y, _ = self.env.base_pose()
        people = self.env.people if self.memory is None else self.memory.people_in_view()
        return SkillResult(True, data={"people": [w.to_dict((x, y)) for w in people]}
                           if self.env.people else {})

    # ================================================================ talking to people
    @skill
    def ask_supervisor(self, question: str) -> SkillResult:
        sup = self.env.supervisor
        if sup is None:
            return SkillResult(False, "no_supervisor: nobody is here to ask")
        answer = sup.ask(self.env, question)
        self.env.step(int(2.0 / self.env.control_dt))  # a spoken exchange is not free
        return SkillResult(True, data={"answer": answer})

    @skill
    def report(self, message: str) -> SkillResult:
        sup = self.env.supervisor
        if sup is None:
            return SkillResult(False, "no_supervisor: nobody is here to hear it")
        sup.hear(self.env, message)
        return SkillResult(True)

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
        if r := self._unseen(obj):
            return r
        if obj not in env.object_names:
            return SkillResult(False, f"unknown_object: {obj}")
        if any(s == side for s, *_ in env.attached.values()):
            return SkillResult(False, f"hand_full: {side} hand is already holding something")
        d = float(np.linalg.norm(env.object_pos(obj) - env.palm_pos(side)))
        if d > max_dist:
            return SkillResult(False, f"too_far: {obj} is {d:.3f}m from {side} palm; reach it first",
                               {"dist_m": round(d, 3)})
        if self._slips():
            # INJECTED FAULT (difficulty `slip`). The fingers stay open: closing them on an
            # object that is not attached squeezes it across the table.
            env.step(20)
            self.faults.append({"t": round(env.time, 2), "fault": "grasp_slip", "obj": obj, "side": side})
            return SkillResult(False, f"grasp_slip: {obj} slipped out of the {side} hand as it closed; "
                                      f"it is still on {env.support_of(obj)}")
        env.attach(obj, side)
        env.hand_target[side] = self._closed_hand(side)
        env.step(10)
        return SkillResult(True, data={"holding": obj})

    def _slips(self) -> bool:
        if not self.difficulty.slip:
            return False
        self._grasps += 1
        # The first grasp always slips so every episode with the switch on tests recovery.
        return self._grasps == 1 or self._fault_rng.random() < self.difficulty.slip

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
                                       "landed_on": [env.support_of(o) for o in held]})

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
        if r := self._unseen(obj):
            return r
        if obj not in env.object_names:
            return SkillResult(False, f"unknown_object: {obj}")
        p = env.object_pos(obj)
        fridge = env.fridge_of(env.support_of(obj))
        if fridge is not None and not env.door_is_open(fridge.name):
            return SkillResult(False, f"door_closed: {obj} is inside {fridge.name}; open_door {fridge.name} first")
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
        # Clearances scale with object height (identical to the old fixed ones for 5cm cubes).
        half = self._half(obj)
        above = p + [0, 0, half + 0.075]
        if fridge is not None:
            r = self._outside(side, above)
            if not r.success:
                return SkillResult(False, f"approach_{r.reason}", r.data)
        r = self._reach(side, above, tol=0.05)
        if not r.success:
            return SkillResult(False, f"approach_{r.reason}", r.data)
        r = self._reach(side, p + [0, 0, half + 0.02])  # palm just above the object; grasp is magic
        if not r.success:
            return SkillResult(False, f"descend_{r.reason}", r.data)
        g = SkillLibrary.grasp.__wrapped__(self, side, obj, max(0.08, half + 0.06))
        if not g.success:
            if fridge is not None:  # same way out as a successful pick: up clear of the object, then out
                self._reach(side, env.palm_pos(side) + [0, 0, 0.06])
                self._outside(side, env.palm_pos(side))
            self._retract(side)
            return g
        # Inside a fridge the shelf above caps the lift; pull straight out before raising to carry.
        lift = self._reach(side, env.palm_pos(side) + [0, 0, 0.12 if fridge is None else 0.06])
        if fridge is not None:
            self._outside(side, env.palm_pos(side))
            # Fingers hang below the palm and catch the shelf edge on the way to carry height.
            _, _, yaw = env.base_pose()
            self._step_to(*self._body_to_world(-0.12, 0.0, 0.0)[:2], yaw)
        carry = self._reach(side, self._carry_point(side, self.CARRY_Z))
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
        obj = held[side]
        table = next((t for t in self._supports() if t.name == surface), None)
        fridge = env.fridge_of(surface)
        if fridge is not None and not env.door_is_open(fridge.name):
            return SkillResult(False, f"door_closed: {surface} is inside {fridge.name}; open_door {fridge.name} first")
        bag = next((b for b in env.scene.bags if b.name == surface), None)
        x, y, _ = env.base_pose()
        if table is not None:
            gap = np.hypot(max(abs(x - table.pos[0]) - table.size[0], 0), max(abs(y - table.pos[1]) - table.size[1], 0))
            if gap > 0.5:
                return SkillResult(False, f"too_far: {surface} is {gap:.2f}m away; go_to_location {surface} first")
            # Fridge side walls: the wrist and elbow flare outboard of the object.
            spot = self._free_spot(table, obj, side, dx, dy, margin=0.06 if fridge is None else 0.09)
            if spot is None:
                return SkillResult(False, f"no_free_space: no clear reachable spot on {surface}; "
                                          f"reposition or clear the table")
        elif bag is None and (r := self._unseen(surface)):
            return r
        elif bag is not None or surface in env.object_names:
            if surface == obj:
                return SkillResult(False, f"bad_args: cannot place {obj} on itself")
            if surface in env.attached:
                return SkillResult(False, f"bad_args: {surface} is held; place it first")
            # Soft or small supports: aim at their centre (the flattest, most supported
            # point), shifted by (dx, dy) in the body frame.
            c = np.array(bag.pos) if bag is not None else env.object_pos(surface)[:2]
            _, _, yaw = env.base_pose()
            spot = c + np.array([np.cos(yaw) * dx - np.sin(yaw) * dy, np.sin(yaw) * dx + np.cos(yaw) * dy])
            if np.hypot(spot[0] - x, spot[1] - y) > self.REACH_FWD + 0.5:
                return SkillResult(False, f"too_far: {surface} is out of reach; walk to it first")
        else:
            return SkillResult(False, f"unknown_surface: {surface}; use a table, fridge shelf, bag or object name")
        half = self._half(obj)
        # Cast from just above a table-like surface: from 2m a fridge shelf is under the roof.
        top, hit = env.surface_below(spot[0], spot[1], z=2.0 if table is None else table.height + 0.05,
                                     exclude=obj, half=half)
        if table is None and hit != surface:
            return SkillResult(False, f"no_support: the spot above {surface} is over {hit}; use dx/dy closer to its centre",
                               {"spot": [round(float(v), 3) for v in spot]})
        # Side-stepping sweeps the carried objects (and fingers ~8cm below the palm)
        # sideways: lift them clear of a tall support or stack first.
        for s_, o in held.items():
            if fridge is None:
                self._reach(s_, self._carry_point(s_, max(self.CARRY_Z, top + 0.17)))
            else:  # at a fridge the carry pose is level with the top shelf's front edge: pull back
                self._outside(s_, env.palm_pos(s_))
        a = self._align_for(np.array([spot[0], spot[1], top]), side)
        if not a.success:
            return a
        # Palm target such that the held object's centre ends up just above the surface.
        palm_minus_obj = env.palm_pos(side) - env.object_pos(obj)
        target = np.array([spot[0], spot[1], top + half + 0.01]) + palm_minus_obj
        if fridge is not None:
            r = self._outside(side, target + [0, 0, 0.10])
            if not r.success:
                return SkillResult(False, f"approach_{r.reason}", r.data)
        r = self._reach(side, target + [0, 0, 0.10], tol=0.05)
        if not r.success:
            return SkillResult(False, f"approach_{r.reason}", r.data)
        r = self._reach(side, target)
        if not r.success:
            return SkillResult(False, f"descend_{r.reason}", r.data)
        goal = np.array([spot[0], spot[1], top + half + 0.01])
        # The palm stops up to 3cm off, and position-only IK lets the wrist (and so the
        # held object) swing on the way down: servo on the object itself. Stacking needs
        # ~1cm; near a table edge a few cm decides whether it lands on the table at all.
        for _ in range(4):
            err = goal - env.object_pos(obj)
            if np.linalg.norm(err) < 0.005:
                break
            self._reach(side, env.palm_pos(side) + err, timeout_s=0.6, tol=0.003)
        miss = float(np.linalg.norm((goal - env.object_pos(obj))[:2]))
        rel = SkillLibrary.release.__wrapped__(self, side)
        ret = self._retract(side)
        env.step(25)  # let soft supports and stacks settle before judging the result
        landed = env.support_of(obj)
        if landed != surface:
            return SkillResult(False, f"placement: {obj} ended up on {landed}", rel.data)
        return SkillResult(True, data={"placed": obj, "on": surface,
                                       "pos": [round(float(v), 3) for v in env.object_pos(obj)],
                                       "tilt_deg": round(env.tilt_deg(obj)), "release_miss_m": round(miss, 3),
                                       "arm_retracted": ret.success})

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
        for t in self._obstacles():
            margin = 0.0 if self._in_table(t, a, self.BODY_RADIUS) else self.BODY_RADIUS
            if any(self._in_table(t, p, margin) for p in pts):
                return t
        return None

    def _route(self, a, b, clearance: float = 0.25):
        """Shortest waypoint list a->b through table corners (visibility graph), or None."""
        nodes = [np.asarray(a, float), np.asarray(b, float)] + [
            np.array([t.pos[0] + sx * (t.size[0] + clearance), t.pos[1] + sy * (t.size[1] + clearance)])
            for t in self._obstacles() for sx in (-1, 1) for sy in (-1, 1)]
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
        # By release the wrist has rolled so the hand sits ~5cm inboard of the object.
        palm_off = self._body_to_world(0.0, -lat_sign * 0.05, 0.0)[:2] - self._body_to_world(0.0, 0.0, 0.0)[:2]
        pref = self._body_to_world(self.REACH_FWD + dx, lat_sign * self.REACH_LAT + dy, 0.0)[:2]
        others = [self.env.object_pos(o)[:2] for o in self.env.object_names
                  if o != obj and abs(self.env.object_pos(o)[2] - table.height) < 0.3]
        best, best_d = None, np.inf
        for gx in np.arange(-table.size[0] + margin, table.size[0] - margin + 1e-6, 0.03):
            for gy in np.arange(-table.size[1] + margin, table.size[1] - margin + 1e-6, 0.03):
                q = np.array([table.pos[0] + gx, table.pos[1] + gy])
                d = float(np.linalg.norm(q - pref))
                if d > max_shift or d >= best_d:
                    continue
                if any(np.linalg.norm(q - o) < clearance for o in others):
                    continue
                # Opening fingers sweep the inboard side: keep that clear too (they knocked a
                # neighbour over on a crowded fridge shelf).
                if any(np.linalg.norm(q + palm_off - o) < clearance - 0.01 for o in others):
                    continue
                best, best_d = q, d
        return best

    # Carried objects hang ~7cm below the palm and the fingers ~8cm; at 1.05m they
    # clear a two-cube stack on the bean bag when the robot turns beside it.
    CARRY_Z = 1.05

    def _carry_point(self, side: str, z: float) -> np.ndarray:
        return self._body_to_world(0.22, 0.18 if side == "left" else -0.18, z)

    REACH_FWD = 0.30   # comfortable forward reach at table height (m, body frame)
    REACH_LAT = 0.15   # lateral offset that puts a point in front of one hand
    STANCE_MARGIN = 0.09  # base this close to a table edge puts the knees into it

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
        if any(self._in_table(t, (bx, by), self.STANCE_MARGIN) for t in self._obstacles()):
            # The kinematic base would drive the legs into the table and wreck the arm.
            return SkillResult(False, "too_far: target is too deep into the table to reach from this side")
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
            if np.hypot(ex, ey) > tol and (blocked := self._person_blocking((ex, ey), min(0.3, np.hypot(ex, ey)))):
                return blocked
            loco.set_command(float(np.clip(1.5 * vx, -0.3, 0.3)), float(np.clip(1.5 * vy, -0.3, 0.3)),
                             float(np.clip(2.0 * _wrap_angle(gyaw - yaw), -0.8, 0.8)))
            env.step()
        loco.stop()
        return SkillResult(False, "timeout: side-step did not converge")

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
