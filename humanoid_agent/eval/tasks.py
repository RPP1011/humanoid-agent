"""Task suite. Each task = scene + natural-language prompt + ground-truth success check.

Keep success checks on sim state, never on what the agent *says* it did (except
answer tasks, which also check that nothing was disturbed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..sim.env import HumanoidEnv
from ..sim.scene import BeanBag, Obj, SceneSpec, default_scene
from ..sim.supervisor import Rule, Supervisor


@dataclass
class Episode:
    env: HumanoidEnv
    answer: str | None = None
    seed: int = 0
    initial_objects: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Task:
    id: str
    prompt: str
    checks: dict[str, Callable[[Episode], bool]]  # named conditions; success = all hold
    scene: Callable[[], SceneSpec] = default_scene
    max_tool_calls: int = 30
    tags: tuple[str, ...] = ()
    supervisor: Callable[[int], Supervisor] | None = None  # (seed) -> scripted human
    settle_s: float = 0.0  # sim time to run before checking, so a wobbly stack has to stay up

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


def _near(env, loc: str, tol: float = 0.3) -> bool:
    x, y, _ = env.base_pose()
    lx, ly, _ = env.scene.locations[loc]
    return bool(np.hypot(x - lx, y - ly) < tol)


def _revised_order_sup(seed: int) -> Supervisor:
    # Even seeds: the change lands as soon as red is picked up (robot still at table_a).
    # Odd seeds: it lands once the robot has carried red over to table_b, so it must walk back.
    late = seed % 2 == 1
    return Supervisor(rules=[Rule(
        lambda env: "red_cube" in env.attached and (not late or _near(env, "table_b")),
        "Change of plan: the red cube stays on table_a, put it back there. "
        "Bring the blue cube over to table_b instead.")],
        answers=[(("blue", "green", "which", "mean"),
                  "The blue cube that's on table_a. Move it to table_b; leave green where it is.")])


def _clarify_sup(seed: int) -> Supervisor:
    # Which cube the supervisor means alternates by seed, so guessing scores ~50% over seeds.
    want = _clarify_want(seed)
    return Supervisor(answers=[(("which", "cube", "red", "blue", "one"),
                                f"The {want.split('_')[0]} one. Leave the other where it is.")])


def _clarify_want(seed: int) -> str:
    return "blue_cube" if seed % 2 == 0 else "red_cube"


def _bean_bag_scene() -> SceneSpec:
    # All three cubes start in table_a's front row (two trips: the robot has two hands);
    # the bag sits on table_b where the green cube normally is.
    scene = default_scene()
    scene.objects = [o if o.name != "green_cube" else Obj("green_cube", (1.0, -0.32, 0.80), rgba=o.rgba)
                     for o in scene.objects]
    scene.bags = [BeanBag("bean_bag", (0.0, 1.58), radius=0.16)]
    return scene


def _tower(ep: Episode, base: str, n: int) -> bool:
    """n objects stacked one on another, the bottom one resting on `base`, all upright."""
    env = ep.env
    on = {o: env.support_of(o) for o in env.object_names}
    level, count = base, 0
    while (above := [o for o, s in on.items() if s == level]) and len(above) == 1:
        level, count = above[0], count + 1
    return count >= n and all(env.tilt_deg(o) < 20 for o in env.object_names if on[o] != "floor")


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
    # ---- floor-supervisor collaboration
    Task("revised_order", "Floor supervisor: put the red cube on table_b, please.",
         {"red_cube stays on table_a": lambda ep: _on(ep, "red_cube", "table_a"),
          "blue_cube on table_b": lambda ep: _on(ep, "blue_cube", "table_b"),
          "green_cube still on table_b": lambda ep: _on(ep, "green_cube", "table_b")},
         tags=("collab", "revision"), supervisor=_revised_order_sup),
    Task("clarify_which", "Floor supervisor: grab the cube off table_a and put it on table_b.",
         {"requested cube on table_b": lambda ep: _on(ep, _clarify_want(ep.seed), "table_b"),
          "other cube left on table_a": lambda ep: _on(ep, ({"red_cube", "blue_cube"} - {_clarify_want(ep.seed)}).pop(),
                                                       "table_a")},
         tags=("collab", "clarify"), supervisor=_clarify_sup),
    # ---- soft support
    Task("bean_bag_tower", "Stack all three cubes into a single tower on the bean bag. It has to stay standing.",
         {"tower of 3 on bean_bag after 5 s": lambda ep: _tower(ep, "bean_bag", 3)},
         scene=_bean_bag_scene, tags=("pnp", "multi", "soft"), settle_s=5.0),
]

TASKS_BY_ID = {t.id: t for t in TASKS}
