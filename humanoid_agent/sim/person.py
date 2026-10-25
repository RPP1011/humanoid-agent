"""Scripted pedestrian: a moving obstacle for the robot to deal with.

PLACEHOLDER human (`is_physical=False`): a mocap body of capsules that glides
back and forth along a fixed polyline at constant speed (no gait, no balance),
pausing at each end. It has no collision geometry; "contact" is a 2D distance
test between the robot base and the person (env.person_contacts), so it never
pushes anything and nothing pushes it.

Behaviour, in order of priority:
- pause `pause_s` at each end of the path;
- yield: stop while the robot base is within YIELD_AHEAD metres ahead of it and
  within YIELD_LATERAL of its line of travel (people step around robots, they
  don't walk through them);
- after waiting `patience_s` for the robot, turn back and walk the other way,
  so a robot parked in the corridor can never deadlock the scene.

Deterministic given the robot trajectory: the start point on the cycle comes
from `seed`, so episodes replay exactly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

YIELD_AHEAD = 0.9
YIELD_LATERAL = 0.5


@dataclass
class Person:
    name: str
    path: tuple[tuple[float, float], ...]   # polyline, walked end to end and back
    speed: float = 0.8                      # m/s, a slow indoor walk
    pause_s: float = 1.5                    # standing at each end
    patience_s: float = 2.0                 # waiting for the robot before turning back
    seed: int = 0                           # picks where on the cycle the episode starts
    radius: float = 0.25                    # shoulders, for contact tests and rendering
    is_physical: bool = False

    @property
    def length(self) -> float:
        p = np.asarray(self.path, float)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    @property
    def period(self) -> float:
        return 2 * (self.length / self.speed + self.pause_s)


class Walker:
    """Runtime state of one Person. `advance` is called once per control step."""

    def __init__(self, spec: Person) -> None:
        self.spec = spec
        self._pts = np.asarray(spec.path, float)
        seg = np.linalg.norm(np.diff(self._pts, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.reset()

    def reset(self, robot_xy=None) -> None:
        self.s, self.dir, self.pause, self.waited = 0.0, 1, self.spec.pause_s, 0.0
        self.vel = np.zeros(2)
        self.yielding = False
        # Fast-forward to a seeded point on the cycle, yielding to the robot's start pose
        # so the episode never begins with the person standing inside the robot.
        t0 = random.Random(self.spec.seed).uniform(0.0, self.spec.period)
        for _ in range(int(t0 / 0.02)):
            self.advance(0.02, robot_xy)

    @property
    def pos(self) -> np.ndarray:
        i = int(np.clip(np.searchsorted(self._cum, self.s) - 1, 0, len(self._pts) - 2))
        u = (self.s - self._cum[i]) / max(self._cum[i + 1] - self._cum[i], 1e-9)
        return self._pts[i] + u * (self._pts[i + 1] - self._pts[i])

    @property
    def heading(self) -> np.ndarray:
        """Unit direction of travel (the way it faces, even while stopped)."""
        i = int(np.clip(np.searchsorted(self._cum, self.s, side="right" if self.dir > 0 else "left") - 1,
                        0, len(self._pts) - 2))
        d = self._pts[i + 1] - self._pts[i]
        return self.dir * d / max(np.linalg.norm(d), 1e-9)

    def robot_in_way(self, robot_xy) -> bool:
        if robot_xy is None:
            return False
        rel = np.asarray(robot_xy, float) - self.pos
        h = self.heading
        ahead, lat = float(rel @ h), float(abs(h[0] * rel[1] - h[1] * rel[0]))
        return 0.0 < ahead < YIELD_AHEAD and lat < YIELD_LATERAL

    def advance(self, dt: float, robot_xy) -> None:
        self.vel[:] = 0.0
        self.yielding = False
        if self.pause > 0:
            self.pause -= dt
            return
        if self.robot_in_way(robot_xy):
            self.yielding = True
            self.waited += dt
            if self.waited >= self.spec.patience_s:
                self.dir, self.waited = -self.dir, 0.0
            return
        self.waited = 0.0
        p0 = self.pos
        L = self._cum[-1]
        self.s += self.dir * self.spec.speed * dt
        if self.s >= L or self.s <= 0.0:
            self.s = float(np.clip(self.s, 0.0, L))
            self.dir, self.pause = -self.dir, self.spec.pause_s
        self.vel = (self.pos - p0) / dt

    def to_dict(self, robot_xy=None) -> dict:
        p = self.pos
        d = {"name": self.spec.name, "pos": [round(float(v), 3) for v in p],
             "velocity": [round(float(v), 2) for v in self.vel],
             "heading": round(float(np.arctan2(*self.heading[::-1])), 3)}
        if robot_xy is not None:
            d["dist_m"] = round(float(np.linalg.norm(p - np.asarray(robot_xy))), 2)
        return d
