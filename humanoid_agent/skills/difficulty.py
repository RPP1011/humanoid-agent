"""Difficulty switches: per-run toggles that each make every task harder in one way.

One switch per ability, so a drop in pass rate says which ability broke (a task
that stacks every challenge only says that something did). All are off by default
and none changes a task's success checks.

    Difficulty.parse("slip,camera")   # or "slip=0.3", "deep", "all", "none"

- slip:   grasps fail and say so (`grasp_slip`). The first grasp of an episode
          always slips, so every episode measures recovery; later ones slip with
          probability `slip` (default 0.15). INJECTED FAULT, not grasp physics.
- deep:   one of the task's target objects starts at the far edge of its table,
          out of reach from the usual side but reachable from the opposite one.
          A real scene change (applied by the eval runner), not a fake failure.
- camera: perception is what the head camera has seen (`skills/perception.py`),
          not `list_objects` ground truth. SIMULATED DETECTOR: field of view,
          range and occlusion are computed from sim state, then noise is added.

Later switches (supervisor revisions, a false supervisor belief, a time budget)
go here as fields too, with their effect applied wherever they act.
"""
from __future__ import annotations

from dataclasses import dataclass

SWITCHES = ("slip", "deep", "camera")
DEFAULT_SLIP = 0.15


@dataclass(frozen=True)
class Difficulty:
    slip: float = 0.0
    deep: bool = False
    camera: bool = False

    @classmethod
    def parse(cls, spec: str | None) -> "Difficulty":
        kw: dict = {}
        for part in (spec or "").replace("+", ",").split(","):
            name, _, val = part.strip().partition("=")
            if name in ("", "none", "off"):
                continue
            if name == "all":
                kw.update(slip=DEFAULT_SLIP, deep=True, camera=True)
            elif name == "slip":
                kw["slip"] = float(val) if val else DEFAULT_SLIP
            elif name in ("deep", "camera"):
                kw[name] = True
            else:
                raise ValueError(f"unknown difficulty switch {name!r}; known: {', '.join(SWITCHES)}, all, none")
        return cls(**kw)

    @property
    def label(self) -> str:
        on = [("slip" if self.slip == DEFAULT_SLIP else f"slip={self.slip:g}")] if self.slip else []
        on += [n for n in ("deep", "camera") if getattr(self, n)]
        return "+".join(on) or "none"
