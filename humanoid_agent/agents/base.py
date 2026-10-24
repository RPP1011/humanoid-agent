from __future__ import annotations

from typing import Any, Callable, Protocol

from ..skills.types import SkillResult

# call(tool_name, args) -> SkillResult. Raises EpisodeDone after finish() or budget exhaustion.
ToolCaller = Callable[[str, dict[str, Any]], SkillResult]


class EpisodeDone(Exception):
    pass


class Agent(Protocol):
    name: str

    def run(self, task_id: str, prompt: str, call: ToolCaller) -> None:
        """Drive the episode via `call`. Return when done (or let EpisodeDone propagate)."""


SYSTEM_PROMPT = """You control a Unitree G1 humanoid robot in a simulated room through tools.
Tools are blocking skills; each returns JSON with success, a reason code, and data.
Rules:
- Check the world first (list_objects / list_locations) instead of guessing names or positions.
- You must stand at a table (go_to_location) before pick or place_on on it.
- On failure, read the reason code and adapt (move closer, use the other hand, free a hand, etc.).
- When the task is done, or impossible, call finish(answer) with a short explanation.
"""
