"""The bean bag must behave like a soft support, not a renamed table."""
import dataclasses
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco

from humanoid_agent.eval.tasks import TASKS_BY_ID
from humanoid_agent.sim.env import HumanoidEnv

CUBES = ("red_cube", "blue_cube", "green_cube")


def _env(lump_seed: int) -> HumanoidEnv:
    scene = TASKS_BY_ID["bean_bag_tower"].scene()
    scene.bags = [dataclasses.replace(b, lump_seed=lump_seed) for b in scene.bags]
    return HumanoidEnv(scene)


def _tower_stands(env: HumanoidEnv, x: float, y: float, step_x: float) -> bool:
    """Set each cube down 1 cm above the one below, shifted step_x per level as a sloppy
    placement would, let it settle for the task's 5 s, and report whether it is intact."""
    m, d = env.model, env.data
    for i, o in enumerate(CUBES):
        top, _ = env.surface_below(x + i * step_x, y, exclude=o, half=0.025)
        a, da = env._obj_qadr[o], env._obj_dadr[o]
        d.qpos[a:a + 3] = (x + i * step_x, y, top + 0.035)
        d.qpos[a + 3:a + 7] = (1, 0, 0, 0)
        d.qvel[da:da + 6] = 0
        mujoco.mj_forward(m, d)
        env.step(50)
    env.step(250)
    return ([env.support_of(o) for o in CUBES[1:]] == list(CUBES[:2])
            and all(env.tilt_deg(o) < 20 for o in CUBES))


def test_centred_tower_stands_on_bag():
    env = _env(0)
    bag = env.scene.bags[0]
    assert _tower_stands(env, *bag.pos, 0.0)
    assert env.support_of("red_cube") == "bean_bag"


def test_sloppy_tower_survives_table_but_not_bag():
    # 1 cm per level is well inside a rigid tower's margin (2.5 cm half-width).
    env = _env(1)
    assert _tower_stands(env, 0.2, 2.0, 0.01) and env.support_of("red_cube") == "table_b"
    env = _env(1)
    bag = env.scene.bags[0]
    assert not _tower_stands(env, *bag.pos, 0.01)
