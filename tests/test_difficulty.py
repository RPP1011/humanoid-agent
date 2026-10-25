"""Difficulty switches: each one bites, is reported honestly, and leaves the task solvable."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

from humanoid_agent.eval.runner import _jittered
from humanoid_agent.eval.tasks import TASKS_BY_ID
from humanoid_agent.sim.env import HumanoidEnv
from humanoid_agent.sim.scene import kitchen_scene
from humanoid_agent.skills.difficulty import Difficulty
from humanoid_agent.skills.library import SkillLibrary


def _lib(diff: str, task: str | None = None, seed: int = 0, scene=None):
    d = Difficulty.parse(diff)
    if task is not None:
        scene = _jittered(TASKS_BY_ID[task], seed, 0.0, d)
    return SkillLibrary(HumanoidEnv(scene=scene), difficulty=d, seed=seed)


def test_parse_and_label():
    assert Difficulty.parse(None).label == "none"
    assert Difficulty.parse("camera,slip").label == "slip+camera"
    assert Difficulty.parse("slip=0.3").slip == 0.3
    assert Difficulty.parse("all").label == "slip+deep+camera"
    with pytest.raises(ValueError):
        Difficulty.parse("lava")


def test_first_grasp_slips_in_place_then_retry_works():
    lib = _lib("slip")
    env = lib.env
    assert lib.go_to_location("table_a").success
    before = env.object_pos("red_cube")
    res = lib.pick("red_cube")
    assert res.code == "grasp_slip" and "table_a" in res.reason
    # A slip must not shove the object: the retry has to be a fair second attempt.
    assert np.linalg.norm(env.object_pos("red_cube") - before) < 0.01
    assert lib.faults and lib.faults[0]["obj"] == "red_cube"
    for _ in range(5):  # later grasps slip with p=0.15
        if (res := lib.pick("red_cube")).success:
            break
    assert res.success and "red_cube" in env.attached


def test_deep_object_needs_the_far_side():
    lib = _lib("deep", task="move_red_to_b")
    env = lib.env
    assert env.object_pos("red_cube")[0] > 1.4  # pushed to table_a's far edge
    assert lib.go_to_location("table_a").success
    assert lib.pick("red_cube").code == "too_far"
    t = next(t for t in env.scene.tables if t.name == "table_a")
    assert lib.walk_to(t.pos[0] + t.size[0] + 0.25, float(env.object_pos("red_cube")[1]), np.pi).success
    assert lib.pick("red_cube").success
    assert lib.go_to_location("table_b").success
    assert lib.place_on("table_b").success


def test_camera_only_knows_what_it_has_seen():
    lib = _lib("camera")
    seen = {o["name"] for o in lib.list_objects().data["objects"]}
    assert seen == {"red_cube", "blue_cube"}  # table_a is ahead at the start; table_b is off to the left
    # An unseen real object and a made-up one get the same answer: no ground-truth leak.
    assert lib.pick("green_cube").code == "not_seen"
    assert lib.pick("purple_cube").code == "not_seen"
    assert lib.look_at("green_cube").code == "not_seen"
    res = lib.look_around()
    assert res.success and "green_cube" in res.data["seen"]
    assert "green_cube" in {o["name"] for o in lib.list_objects().data["objects"]}


def test_camera_fridge_contents_unknown_until_opened():
    lib = _lib("camera", scene=kitchen_scene())
    res = lib.list_objects()
    assert not res.data["objects"] and "fridge closed" in res.data["not_visible"]
    assert lib.go_to_location("fridge").success
    assert lib.pick("soda_can").code == "not_seen"
    res = lib.open_door("fridge")
    assert res.success and {"soda_can", "juice_box", "yogurt"} <= set(res.data["newly_seen"])
    assert lib.pick("soda_can").success
