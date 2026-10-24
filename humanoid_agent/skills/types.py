from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SkillResult:
    """Every skill returns one of these. The planner (LLM or scripted) only sees this.

    `reason` is a short machine-ish code plus detail, e.g. "unreachable: ik residual 0.14m".
    Keep codes stable: the eval harness aggregates failures by the prefix before ':'.
    """

    success: bool
    reason: str = "ok"
    data: dict[str, Any] = field(default_factory=dict)
    sim_time_s: float = 0.0

    @property
    def code(self) -> str:
        return self.reason.split(":", 1)[0].strip()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["code"] = self.code
        return d
