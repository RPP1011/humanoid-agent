"""Task suite. Each task = scene + natural-language prompt + ground-truth success check.

Keep success checks on sim state, never on what the agent *says* it did (except
answer tasks, which also check that nothing was disturbed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..sim.env import HumanoidEnv
from ..sim.scene import SceneSpec, default_scene


@dataclass
class Episode:
    env: HumanoidEnv
    answer: str | None = None
    initial_objects: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    prompt: str
    checks: dict[str, Callable[[Episode], bool]]  # named conditions; success = all hold
    scene: Callable[[], SceneSpec] = default_scene
    max_tool_calls: int = 30
    tags: tuple[str, ...] = ()

    def report(self, ep: Episode) -> dict[str, bool]:
        return {name: bool(check(ep)) for name, check in self.checks.items()}

    def success(self, ep: Episode) -> bool:
        return all(self.report(ep).values())


def _on(ep: Episode, obj: str, table: str) -> bool:
    env = ep.env
    t = next(t for t in env.scene.tables if t.name == table)
    p = env.object_pos(obj)
    return (obj not in env.attached and abs(p[0] - t.pos[0]) <= t.size[0]
            and abs(p[1] - t.pos[1]) <= t.size[1] and p[2] > t.height - 0.02)


def _at(ep: Episode, loc: str, tol: float = 0.1) -> bool:
    x, y, _ = ep.env.base_pose()
    lx, ly, _ = ep.env.scene.locations[loc]
    return bool(np.hypot(x - lx, y - ly) < tol)


def _undisturbed(ep: Episode, tol: float = 0.02) -> bool:
    return all(np.linalg.norm(ep.env.object_pos(o) - p) < tol for o, p in ep.initial_objects.items())


_REFUSALS = ("no purple", "not find", "doesn't exist", "does not exist", "no such", "isn't a purple", "there is no")


def _answer_has(ep: Episode, *words: str) -> bool:
    a = (ep.answer or "").lower()
    return all(w in a for w in words)


TASKS: list[Task] = [
    Task("nav_table_b", "Go stand at table_b.",
         {"robot at table_b": lambda ep: _at(ep, "table_b")}, tags=("nav",)),
    Task("move_red_to_b", "Put the red cube on table_b.",
         {"red_cube on table_b": lambda ep: _on(ep, "red_cube", "table_b")}, tags=("pnp",)),
    Task("move_both_to_b", "Move both cubes that are on table_a over to table_b.",
         {"red_cube on table_b": lambda ep: _on(ep, "red_cube", "table_b"),
          "blue_cube on table_b": lambda ep: _on(ep, "blue_cube", "table_b")}, tags=("pnp", "multi")),
    Task("fetch_green_to_a", "Bring the green cube to the other table.",
         {"green_cube on table_a": lambda ep: _on(ep, "green_cube", "table_a")}, tags=("pnp", "grounding")),
    Task("swap", "Swap things around: the red cube should end up on table_b and the green cube on table_a.",
         {"red_cube on table_b": lambda ep: _on(ep, "red_cube", "table_b"),
          "green_cube on table_a": lambda ep: _on(ep, "green_cube", "table_a")}, tags=("pnp", "multi")),
    Task("which_on_b", "Without moving anything, tell me which colour cube is on table_b. Use finish() to answer.",
         {"answer names green": lambda ep: _answer_has(ep, "green"),
          "nothing moved": _undisturbed}, tags=("query",)),
    Task("impossible_purple", "Put the purple cube on table_a.",
         {"answer says no purple cube": lambda ep: any(w in (ep.answer or "").lower() for w in _REFUSALS),
          "nothing moved": _undisturbed}, tags=("refusal",)),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
