"""MCP server exposing the skill tools, plus episode controls for manual eval.

    python -m humanoid_agent.server.mcp_server            # stdio
    python -m humanoid_agent.server.mcp_server --task move_red_to_b

Claude Desktop / Claude Code config:
    {"command": "python", "args": ["-m", "humanoid_agent.server.mcp_server"],
     "cwd": "/path/to/humanoid-agent", "env": {"MUJOCO_GL": "egl"}}
"""
from __future__ import annotations

import argparse
import base64

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from ..eval.tasks import TASKS, TASKS_BY_ID, Episode
from ..sim.env import HumanoidEnv
from ..skills.library import SkillLibrary
from ..tools.registry import TOOLS, call_tool, result_to_text


class Session:
    def __init__(self, task_id: str | None) -> None:
        self.reset(task_id)

    def reset(self, task_id: str | None) -> str:
        self.task = TASKS_BY_ID[task_id] if task_id else None
        self.env = HumanoidEnv(scene=self.task.scene() if self.task else None)
        self.env.supervisor = self.task.supervisor(0) if self.task and self.task.supervisor else None
        self.lib = SkillLibrary(self.env)
        self.episode = Episode(self.env, initial_objects={o: self.env.object_pos(o) for o in self.env.object_names})
        return self.task.prompt if self.task else "Free play: default scene loaded."


CONTROL_TOOLS = [
    types.Tool(name="list_tasks", description="List eval tasks (id and prompt).",
               input_schema={"type": "object", "properties": {}}),
    types.Tool(name="reset_task", description="Reset the sim to an eval task's scene; returns the task prompt. "
                                              "Omit task_id for free play.",
               input_schema={"type": "object", "properties": {"task_id": {"type": "string"}}}),
    types.Tool(name="check_task", description="Ground-truth success check for the current task. "
                                              "Optional answer for question tasks.",
               input_schema={"type": "object", "properties": {"answer": {"type": "string"}}}),
]


def build_server(task_id: str | None) -> Server:
    sess = Session(task_id)

    async def list_tools(ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=[
            types.Tool(name=t.name, description=t.description, input_schema=t.input_schema()) for t in TOOLS
        ] + CONTROL_TOOLS)

    def text(s: str, err: bool = False) -> types.CallToolResult:
        return types.CallToolResult(content=[types.TextContent(type="text", text=s)], is_error=err)

    async def call(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        name, args = params.name, dict(params.arguments or {})
        if name == "list_tasks":
            return text("\n".join(f"{t.id}: {t.prompt}" for t in TASKS))
        if name == "reset_task":
            tid = args.get("task_id")
            if tid and tid not in TASKS_BY_ID:
                return text(f"unknown task {tid}", True)
            return text(await anyio.to_thread.run_sync(sess.reset, tid))
        if name == "check_task":
            if sess.task is None:
                return text("no task loaded; call reset_task first", True)
            sess.episode.answer = args.get("answer")
            return text(f"{sess.task.id}: {'SUCCESS' if sess.task.success(sess.episode) else 'NOT YET'}")

        res = await anyio.to_thread.run_sync(lambda: call_tool(sess.lib, name, args))
        content: list = []
        if res.data.get("png_bytes"):
            content.append(types.ImageContent(type="image", data=base64.b64encode(res.data["png_bytes"]).decode(),
                                              mime_type="image/png"))
        content.append(types.TextContent(type="text", text=result_to_text(res)))
        return types.CallToolResult(content=content, is_error=not res.success)

    return Server("humanoid-agent", version="0.1.0",
                  instructions="Tools control a simulated Unitree G1 humanoid. Use list_objects and "
                               "list_locations first. Stand at a table before pick/place_on.",
                  on_list_tools=list_tools, on_call_tool=call)


async def _amain(task_id: str | None) -> None:
    server = build_server(task_id)
    async with stdio_server() as (r, w):
        await server.run(r, w, server.create_initialization_options())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    anyio.run(_amain, ap.parse_args().task)


if __name__ == "__main__":
    main()
