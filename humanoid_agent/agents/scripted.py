"""Hand-written plans. Not a baseline for intelligence: a harness/skill check.

If a scripted plan fails, the bug is in the skills or sim, not the planner.
Run these before blaming any LLM.
"""
from __future__ import annotations

from .base import ToolCaller

PLANS: dict[str, list[tuple[str, dict]]] = {
    "nav_table_b": [("go_to_location", {"name": "table_b"})],
    "move_red_to_b": [
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "red_cube"}),
        ("go_to_location", {"name": "table_b"}), ("place_on", {"surface": "table_b"}),
    ],
    "move_both_to_b": [
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "red_cube"}), ("pick", {"obj": "blue_cube"}),
        ("go_to_location", {"name": "table_b"}),
        ("place_on", {"surface": "table_b", "side": "right"}), ("place_on", {"surface": "table_b", "side": "left"}),
    ],
    "fetch_green_to_a": [
        ("go_to_location", {"name": "table_b"}), ("pick", {"obj": "green_cube"}),
        ("go_to_location", {"name": "table_a"}), ("place_on", {"surface": "table_a"}),
    ],
    "swap": [
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "red_cube"}),
        ("go_to_location", {"name": "table_b"}), ("place_on", {"surface": "table_b"}),
        ("pick", {"obj": "green_cube"}),
        ("go_to_location", {"name": "table_a"}), ("place_on", {"surface": "table_a"}),
    ],
    # Seed-0 branches only: a fixed plan cannot react, so it encodes what seed 0's supervisor says.
    "revised_order": [
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "red_cube"}),
        ("place_on", {"surface": "table_a"}), ("pick", {"obj": "blue_cube"}),
        ("go_to_location", {"name": "table_b"}), ("place_on", {"surface": "table_b"}),
    ],
    "clarify_which": [
        ("ask_supervisor", {"question": "Which cube on table_a do you mean, red or blue?"}),
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "blue_cube"}),
        ("go_to_location", {"name": "table_b"}), ("place_on", {"surface": "table_b"}),
    ],
    "bean_bag_tower": [
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "red_cube"}), ("pick", {"obj": "blue_cube"}),
        ("go_to_location", {"name": "table_b"}),
        ("place_on", {"surface": "bean_bag", "obj": "red_cube"}), ("place_on", {"surface": "red_cube", "obj": "blue_cube"}),
        ("go_to_location", {"name": "table_a"}), ("pick", {"obj": "green_cube"}),
        ("go_to_location", {"name": "table_b"}), ("place_on", {"surface": "blue_cube"}),
    ],
    "fridge_soda": [
        ("go_to_location", {"name": "fridge"}), ("open_door", {"fridge": "fridge"}), ("pick", {"obj": "soda_can"}),
        ("close_door", {"fridge": "fridge"}),
        ("go_to_location", {"name": "kitchen_table"}), ("place_on", {"surface": "kitchen_table"}),
    ],
    "fridge_put_milk_away": [
        ("go_to_location", {"name": "counter"}), ("pick", {"obj": "milk"}),
        ("go_to_location", {"name": "fridge"}), ("open_door", {"fridge": "fridge"}),
        ("place_on", {"surface": "fridge_top_shelf"}), ("close_door", {"fridge": "fridge"}),
    ],
    "which_on_b": [("list_objects", {}), ("finish", {"answer": "green"})],
    "impossible_purple": [("list_objects", {}), ("finish", {"answer": "There is no purple cube in the room."})],
}


class ScriptedAgent:
    name = "scripted"

    def run(self, task_id: str, prompt: str, call: ToolCaller) -> None:
        for tool, args in PLANS.get(task_id, []):
            res = call(tool, args)
            if not res.success:
                call("finish", {"answer": f"scripted plan failed at {tool}: {res.reason}"})
                return
        call("finish", {"answer": "done"})
