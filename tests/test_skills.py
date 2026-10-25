"""Skill-level guards that keep a planner from silently wrecking an episode."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from humanoid_agent.eval.tasks import TASKS_BY_ID, Episode
from humanoid_agent.sim.env import HumanoidEnv
from humanoid_agent.skills.library import SkillLibrary


@pytest.fixture
def lib():
    return SkillLibrary(HumanoidEnv())


def _holding_red_and_blue(lib):
    assert lib.go_to_location("table_a").success
    assert lib.pick("red_cube").success
    assert lib.pick("blue_cube").success
    assert lib.go_to_location("table_b").success


def test_place_on_names_the_object(lib):
    _holding_red_and_blue(lib)
    res = lib.place_on("table_b", obj="blue_cube")
    assert res.success and res.data["placed"] == "blue_cube"


def test_place_on_refuses_to_guess_with_both_hands_full(lib):
    _holding_red_and_blue(lib)
    res = lib.place_on("table_b")
    assert not res.success and res.code == "ambiguous"


def test_place_on_far_table_is_too_far_not_no_space(lib):
    assert lib.go_to_location("table_a").success
    assert lib.pick("red_cube").success
    res = lib.place_on("table_b")
    assert not res.success and res.code == "too_far"


def test_walk_to_detours_around_table(lib):
    res = lib.walk_to(1.8, 0.0)  # straight line runs through table_a
    assert res.success and "detour_via" in res.data
    objs = {o["name"]: o for o in lib.list_objects().data["objects"]}
    assert objs["red_cube"]["on"] == "table_a" and objs["blue_cube"]["on"] == "table_a"


def test_walk_into_table_is_refused(lib):
    res = lib.walk_to(1.2, 0.0)  # table_a centre
    assert not res.success and res.code == "blocked_path"


def test_yaw_in_degrees_is_rejected(lib):
    res = lib.turn_to(90)
    assert not res.success and res.code == "bad_args"


def test_task_report_names_unmet_conditions(lib):
    ep = Episode(lib.env, initial_objects={o: lib.env.object_pos(o) for o in lib.env.object_names})
    report = TASKS_BY_ID["swap"].report(ep)
    assert report == {"red_cube on table_b": False, "green_cube on table_a": False}


@pytest.fixture
def kitchen():
    from humanoid_agent.sim.scene import kitchen_scene
    return SkillLibrary(HumanoidEnv(scene=kitchen_scene()))


def test_closed_fridge_hides_and_blocks_its_contents(kitchen):
    names = {o["name"] for o in kitchen.list_objects().data["objects"]}
    assert "soda_can" not in names and "milk" in names
    assert kitchen.go_to_location("fridge").success
    res = kitchen.pick("soda_can")
    assert not res.success and res.code == "door_closed"
    assert kitchen.open_door("fridge").success
    assert "soda_can" in {o["name"] for o in kitchen.list_objects().data["objects"]}
    assert kitchen.get_state().data["doors"] == {"fridge": "open"}
