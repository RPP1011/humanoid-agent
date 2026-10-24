"""LLM planners over the shared tool registry.

- AnthropicAgent: Claude via the Messages API (needs ANTHROPIC_API_KEY).
- OpenAICompatAgent: any OpenAI-compatible chat server with tool calling —
  vLLM, llama.cpp server, SGLang, Ollama, LM Studio. This is the "slot in your
  own LLM server" path.

Both send ego frames as images when the agent calls get_frame.
"""
from __future__ import annotations

import base64
import json
import os

from ..tools.registry import FINISH_TOOL, TOOLS, result_to_anthropic_content, result_to_text
from .base import SYSTEM_PROMPT, ToolCaller

ALL_TOOLS = TOOLS + [FINISH_TOOL]


class AnthropicAgent:
    def __init__(self, model: str | None = None, max_tokens: int = 2048) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("HUMANOID_AGENT_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.name = f"anthropic:{self.model}"
        self.tools = [{"name": t.name, "description": t.description, "input_schema": t.input_schema()}
                      for t in ALL_TOOLS]

    def run(self, task_id: str, prompt: str, call: ToolCaller) -> None:
        messages: list[dict] = [{"role": "user", "content": prompt}]
        while True:
            resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                               system=SYSTEM_PROMPT, tools=self.tools, messages=messages)
            messages.append({"role": "assistant", "content": resp.content})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if not uses:
                text = "".join(b.text for b in resp.content if b.type == "text")
                call("finish", {"answer": text or "(no answer)"})
                return
            results = []
            for u in uses:
                res = call(u.name, dict(u.input))  # may raise EpisodeDone
                results.append({"type": "tool_result", "tool_use_id": u.id,
                                "content": result_to_anthropic_content(res), "is_error": not res.success})
            messages.append({"role": "user", "content": results})


class OpenAICompatAgent:
    """Point base_url at your server, e.g. http://localhost:8000/v1 for vLLM."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 send_images: bool = True) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "not-needed"))
        self.model = model
        self.send_images = send_images
        self.name = f"openai_compat:{model}"
        self.tools = [{"type": "function", "function": {
            "name": t.name, "description": t.description, "parameters": t.input_schema()}} for t in ALL_TOOLS]

    def run(self, task_id: str, prompt: str, call: ToolCaller) -> None:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        while True:
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=self.tools)
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                call("finish", {"answer": msg.content or "(no answer)"})
                return
            pending_images = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps({"success": False, "reason": f"bad_json: {e}"})})
                    continue
                res = call(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_to_text(res)})
                if self.send_images and res.data.get("png_bytes"):
                    pending_images.append(res.data["png_bytes"])
            # Most OpenAI-compatible servers only accept images in user turns.
            if pending_images:
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": "Camera frame(s) from your last tool call(s):"},
                    *[{"type": "image_url", "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(p).decode()}} for p in pending_images],
                ]})
