"""Scripted floor supervisor: a human who organizes the robot's work by talking to it.

PLACEHOLDER human: no body, no voice, no perception of its own. It is a set of
rules on sim state, so episodes stay reproducible and success checks never
depend on what an LLM-played human happened to say. `is_physical=False`.

Delivery model: the robot "hears" only between skills (skills are blocking).
Rules are evaluated after every skill; anything said is attached to that
skill's result under `supervisor_says`. A rule whose condition became true
mid-skill is therefore heard when the skill returns, not at that instant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Rule:
    when: Callable[[Any], bool]   # (env) -> bool; fires once, the first time it is true
    say: str
    fired: bool = False


@dataclass
class Supervisor:
    rules: list[Rule] = field(default_factory=list)
    # (keywords, answer): first entry with any keyword in the question wins.
    answers: list[tuple[tuple[str, ...], str]] = field(default_factory=list)
    default_answer: str = "Not sure. Use your judgement."
    name: str = "supervisor"
    is_physical: bool = False
    transcript: list[dict] = field(default_factory=list)
    _inbox: list[str] = field(default_factory=list)

    def tick(self, env) -> None:
        for r in self.rules:
            if not r.fired and r.when(env):
                r.fired = True
                self._inbox.append(r.say)
                self._log(env, self.name, r.say)

    def drain(self) -> list[str]:
        out, self._inbox = self._inbox, []
        return out

    def ask(self, env, question: str) -> str:
        self._log(env, "robot", question)
        q = question.lower()
        a = next((ans for kws, ans in self.answers if any(k in q for k in kws)), self.default_answer)
        self._log(env, self.name, a)
        return a

    def hear(self, env, message: str) -> None:
        self._log(env, "robot", message)

    def _log(self, env, who: str, text: str) -> None:
        self.transcript.append({"t": round(env.time, 2), "who": who, "text": text})
