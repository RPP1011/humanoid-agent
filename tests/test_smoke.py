"""Fast regression check: every scripted plan passes on the canonical layout.

If this fails after a skill change, the skill broke — not the planner.
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from humanoid_agent.agents.scripted import ScriptedAgent
from humanoid_agent.eval.runner import run_episode
from humanoid_agent.eval.tasks import TASKS


@pytest.mark.parametrize("task", TASKS, ids=[t.id for t in TASKS])
def test_scripted_plan_passes(task, tmp_path):
    row = run_episode(ScriptedAgent(), task, seed=0, out_dir=tmp_path)
    assert row["success"], row["skill_failures"]
