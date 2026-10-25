"""Scripted pedestrian: walk_to stops for it, it yields to a parked robot, contact is detected."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from humanoid_agent.sim.env import HumanoidEnv
from humanoid_agent.sim.person import Person
from humanoid_agent.sim.scene import default_scene
from humanoid_agent.skills.library import SkillLibrary


def _lib(path, seed=0):
    scene = default_scene()
    scene.people = [Person("person", path, seed=seed)]
    return SkillLibrary(HumanoidEnv(scene=scene))


def test_walk_to_stops_for_person_standing_in_the_way():
    # A one-point "path" never moves: someone standing between the robot and table_a.
    lib = _lib(((0.45, 0.0), (0.45, 0.001)), seed=0)
    lib.env.people[0].pause = 1e9
    res = lib.go_to_location("table_a")
    assert not res.success and res.code == "blocked_by_person"
    assert res.data["person"]["dist_m"] > 0.3 and not lib.env.person_contacts


def test_person_yields_to_parked_robot_and_turns_back():
    # Lane straight through the robot's start pose: the person must stop short and turn around.
    lib = _lib(((-1.5, 0.0), (1.5, 0.0)), seed=0)
    for _ in range(6):
        assert lib.wait(5.0).success
    assert not lib.env.person_contacts and lib.env.min_person_dist > 0.6


def test_wait_lets_person_walk_past_and_reports_them():
    lib = _lib(((-1.0, 1.0), (1.0, 1.0)), seed=3)
    p0 = np.array(lib.list_objects().data["people"][0]["pos"])
    res = lib.wait(2.0)
    assert res.success and res.sim_time_s >= 1.99
    assert np.linalg.norm(np.array(res.data["people"][0]["pos"]) - p0) > 0.3


def test_contact_is_recorded():
    lib = _lib(((-1.5, 0.0), (1.5, 0.0)), seed=0)
    env = lib.env
    env.people[0].s, env.people[0].pause = 1.5, 1e9  # put the person on top of the robot
    env.step()
    assert len(env.person_contacts) == 1
