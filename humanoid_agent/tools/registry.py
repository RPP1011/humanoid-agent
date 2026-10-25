"""Single source of truth for the planner-facing tool surface.

The same specs feed the MCP server and the function-calling agent, so an MCP
client and the eval harness see identical tools. Edit tools here only.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any, Callable

from ..skills.library import SkillLibrary
from ..skills.types import SkillResult

SIDE = {"type": "string", "enum": ["left", "right"]}


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict[str, Any]          # JSON-schema "properties"
    required: list[str]
    call: Callable[..., SkillResult]  # (lib, **kwargs) -> SkillResult

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": self.params, "required": self.required}


def _num(desc: str) -> dict:
    return {"type": "number", "description": desc}


TOOLS: list[ToolSpec] = [
    ToolSpec("get_state", "Robot base pose (x, y metres; yaw radians), what each hand holds, sim time, "
                          "and whether each fridge door is open.",
             {}, [], lambda L: L.get_state()),
    ToolSpec("list_objects", "Objects with world position, what they rest on (a table, the bean bag, another "
                             "object, floor), tilt from upright in degrees, distance from robot, and holder; plus the "
                             "surfaces things can be placed on (tables with half extents, bean bag), and any people "
                             "walking around (position, velocity in m/s, distance). If the result has a 'perception' "
                             "note, it lists only what the robot's camera has seen so far.",
             {}, [], lambda L: L.list_objects()),
    ToolSpec("list_locations", "Named places the robot can go to with go_to_location.",
             {}, [], lambda L: L.list_locations()),
    ToolSpec("get_frame", "Camera image. 'ego' is the robot's head view; 'overview' is a fixed room camera.",
             {"camera": {"type": "string", "enum": ["ego", "overview"]}}, [],
             lambda L, camera="ego": L.get_frame(camera)),
    ToolSpec("go_to_location", "Walk to a named location and face its stored heading. "
                               "Stops early (blocked_by_person) if a person is in the way.",
             {"name": {"type": "string"}}, ["name"], lambda L, name: L.go_to_location(name)),
    ToolSpec("walk_to", "Walk to world (x, y), optionally ending at heading yaw (radians). Routes around tables. "
                         "Stops early (blocked_by_person) if a person is in the way.",
             {"x": _num("metres"), "y": _num("metres"), "yaw": _num("radians, optional")}, ["x", "y"],
             lambda L, x, y, yaw=None: L.walk_to(x, y, yaw)),
    ToolSpec("wait", "Stand still for a few seconds (max 10) and let the world move, e.g. until a person has "
                     "walked past. Returns where people are afterwards.",
             {"seconds": _num("sim seconds, 0-10")}, ["seconds"], lambda L, seconds: L.wait(seconds)),
    ToolSpec("turn_to", "Rotate in place to heading yaw (radians).",
             {"yaw": _num("radians")}, ["yaw"], lambda L, yaw: L.turn_to(yaw)),
    ToolSpec("look_at", "Face an object or named location and tilt the head camera toward it.",
             {"target": {"type": "string"}}, ["target"], lambda L, target: L.look_at(target)),
    ToolSpec("look_around", "Turn in place through a full circle, looking in four directions, and end facing the "
                            "original heading. Returns what the camera saw.",
             {}, [], lambda L: L.look_around()),
    ToolSpec("pick", "Pick up an object within arm's reach (stand at the table first). Side auto-chosen if omitted.",
             {"obj": {"type": "string"}, "side": SIDE}, ["obj"],
             lambda L, obj, side=None: L.pick(obj, side)),
    ToolSpec("place_on", "Place a held object on a surface: a table or fridge shelf (auto-picks a spot clear of other objects near the "
                         "robot), a bean bag (aims at its centre), or another object to stack on top of it (aims at its "
                         "centre). Side-steps if needed; stand at the table first. The object is let go and physics "
                         "decides whether it stays: check the returned tilt_deg. "
                         "Name the object with obj (required if both hands are full). "
                         "dx/dy shift the spot forward/left in metres.",
             {"surface": {"type": "string", "description": "table, fridge shelf, bag or object name"},
              "obj": {"type": "string", "description": "held object to place"},
              "side": SIDE, "dx": _num("forward offset"), "dy": _num("left offset")},
             ["surface"], lambda L, surface, obj=None, side=None, dx=0.0, dy=0.0: L.place_on(surface, side, dx, dy, obj)),
    ToolSpec("open_door", "Open a fridge's door (you must be near it; needs one free hand). The robot steps back "
                          "while the door swings, then returns to the fridge. Its contents are hidden while it is closed.",
             {"fridge": {"type": "string"}}, ["fridge"], lambda L, fridge: L.open_door(fridge)),
    ToolSpec("close_door", "Close a fridge's door. The robot steps back out of the door's swing to do it.",
             {"fridge": {"type": "string"}}, ["fridge"], lambda L, fridge: L.close_door(fridge)),
    ToolSpec("ask_supervisor", "Ask the floor supervisor a question (takes ~2 s). Their unprompted messages "
                               "arrive in any tool result under supervisor_says.",
             {"question": {"type": "string"}}, ["question"], lambda L, question: L.ask_supervisor(question)),
    ToolSpec("report", "Tell the floor supervisor something (status, a problem, that you are done).",
             {"message": {"type": "string"}}, ["message"], lambda L, message: L.report(message)),
    ToolSpec("reach", "Low-level: move a palm to world point (x, y, z).",
             {"side": SIDE, "x": _num("m"), "y": _num("m"), "z": _num("m")}, ["side", "x", "y", "z"],
             lambda L, side, x, y, z: L.reach(side, x, y, z)),
    ToolSpec("grasp", "Low-level: close hand on an object already at the palm (within ~8cm).",
             {"side": SIDE, "obj": {"type": "string"}}, ["side", "obj"],
             lambda L, side, obj: L.grasp(side, obj)),
    ToolSpec("release", "Low-level: open a hand, dropping whatever it holds.",
             {"side": SIDE}, ["side"], lambda L, side: L.release(side)),
    ToolSpec("retract", "Return an arm to its resting pose.",
             {"side": SIDE}, ["side"], lambda L, side: L.retract(side)),
]

TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Not a skill: lets an agent end an episode and report an answer. Handled by the runner.
FINISH_TOOL = ToolSpec(
    "finish", "Call when the task is complete or impossible. Give a short answer/explanation.",
    {"answer": {"type": "string"}}, ["answer"], lambda L, answer: SkillResult(True, data={"answer": answer}),
)


def call_tool(lib: SkillLibrary, name: str, args: dict[str, Any]) -> SkillResult:
    spec = TOOLS_BY_NAME.get(name) or (FINISH_TOOL if name == "finish" else None)
    if spec is None:
        return SkillResult(False, f"unknown_tool: {name}")
    try:
        return spec.call(lib, **args)
    except TypeError as e:
        return SkillResult(False, f"bad_args: {e}")


def result_to_text(res: SkillResult) -> str:
    d = res.to_dict()
    d["data"] = {k: v for k, v in d["data"].items() if k != "png_bytes"}
    return json.dumps(d, default=str)


def result_to_anthropic_content(res: SkillResult) -> list[dict]:
    """Tool-result content blocks for the Anthropic Messages API (image + JSON)."""
    blocks: list[dict] = []
    png = res.data.get("png_bytes")
    if png:
        blocks.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}})
    blocks.append({"type": "text", "text": result_to_text(res)})
    return blocks
