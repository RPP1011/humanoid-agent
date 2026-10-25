"""Eval runner.

    python -m humanoid_agent.eval.runner --agent scripted
    python -m humanoid_agent.eval.runner --agent anthropic --model claude-haiku-5-5 --seeds 3
    python -m humanoid_agent.eval.runner --agent openai --model qwen3-vl --base-url http://localhost:8000/v1

Writes runs/<stamp>_<agent>/results.jsonl (one row per episode, full skill log)
and summary.md (per-task success, and failure codes split by skill vs planner).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import random
import time
from pathlib import Path

from PIL import Image

from ..agents.base import EpisodeDone
from ..sim.env import HumanoidEnv
from ..skills.library import SkillLibrary
from ..skills.types import SkillResult
from ..tools.registry import call_tool
from .tasks import TASKS, TASKS_BY_ID, Episode, Task

PLANNER_CODES = {"unknown_tool", "bad_args", "unknown_object", "unknown_location", "unknown_target",
                 "unknown_surface", "not_holding", "hand_full", "budget", "no_supervisor", "no_support",
                 "door_closed"}
# Neither planner nor skill error: the world got in the way and the planner has to react.
WORLD_CODES = {"blocked_by_person"}


def _jittered(task: Task, seed: int, jitter: float):
    scene = task.scene()
    scene.people = [dataclasses.replace(p, seed=seed) for p in scene.people]
    if seed == 0 or jitter <= 0:
        return scene
    rng = random.Random(seed)
    scene.bags = [dataclasses.replace(b, lump_seed=seed) for b in scene.bags]
    scene.objects = [dataclasses.replace(o, pos=(o.pos[0] + rng.uniform(-jitter, jitter),
                                                  o.pos[1] + rng.uniform(-jitter, jitter), o.pos[2]))
                     for o in scene.objects]
    return scene


def run_episode(agent, task: Task, seed: int, out_dir: Path, jitter: float = 0.03) -> dict:
    env = HumanoidEnv(scene=_jittered(task, seed, jitter))
    env.supervisor = task.supervisor(seed) if task.supervisor else None
    lib = SkillLibrary(env)
    ep = Episode(env, seed=seed, initial_objects={o: env.object_pos(o) for o in env.object_names})
    n_calls = 0

    def call(name: str, args: dict) -> SkillResult:
        nonlocal n_calls
        n_calls += 1
        if name == "finish":
            ep.answer = str(args.get("answer", ""))
            lib.log.append({"skill": "finish", "kwargs": args, "success": True, "reason": "ok", "code": "ok"})
            raise EpisodeDone()
        if n_calls > task.max_tool_calls:
            lib.log.append({"skill": name, "kwargs": args, "success": False, "reason": "budget", "code": "budget"})
            raise EpisodeDone()
        return call_tool(lib, name, args)

    t0 = time.time()
    error = None
    try:
        agent.run(task.id, task.prompt, call)
    except EpisodeDone:
        pass
    except Exception as e:  # agent/API crash is a result, not a harness crash
        error = f"{type(e).__name__}: {e}"
    if task.settle_s:
        env.step(int(task.settle_s / env.control_dt))
    checks = task.report(ep)
    ok = all(checks.values()) and error is None

    tag = f"{task.id}_s{seed}"
    Image.fromarray(env.render("overview")).save(out_dir / f"{tag}_final.png")
    return {
        "task": task.id, "seed": seed, "agent": agent.name, "success": ok,
        "tool_calls": n_calls, "sim_time_s": round(env.time, 2), "wall_s": round(time.time() - t0, 2),
        "answer": ep.answer, "error": error, "checks": checks,
        "locomotion_is_physical": env.loco.is_physical,
        "transcript": env.supervisor.transcript if env.supervisor else None,
        **({"person_contacts": env.person_contacts, "min_person_dist_m": round(env.min_person_dist, 3),
            "people_is_physical": False} if env.people else {}),
        "skill_failures": [r["code"] for r in lib.log if not r.get("success", True)],
        "log": [{k: v for k, v in r.items() if k not in ("data",)} for r in lib.log],
    }


def summarize(rows: list[dict]) -> str:
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    lines = ["| task | success | mean tool calls | mean sim s |", "|---|---|---|---|"]
    for t, rs in by_task.items():
        sr = sum(r["success"] for r in rs) / len(rs)
        lines.append(f"| {t} | {sr:.0%} ({sum(r['success'] for r in rs)}/{len(rs)}) | "
                     f"{sum(r['tool_calls'] for r in rs) / len(rs):.1f} | "
                     f"{sum(r['sim_time_s'] for r in rs) / len(rs):.1f} |")
    total = sum(r["success"] for r in rows)
    lines.append(f"\n**Overall: {total}/{len(rows)}**")
    codes = collections.Counter(c for r in rows for c in r["skill_failures"])
    if codes:
        world = {c: n for c, n in codes.items() if c.endswith(tuple(WORLD_CODES))}
        planner = {c: n for c, n in codes.items() if c in PLANNER_CODES}
        skill = {c: n for c, n in codes.items() if c not in PLANNER_CODES and c not in world}
        lines.append(f"\nFailure codes attributed to planner: {planner or '{}'}")
        lines.append(f"Failure codes attributed to skills/sim: {skill or '{}'}")
        if world:
            lines.append(f"World events the planner had to handle: {world}")
    errs = [r for r in rows if r["error"]]
    if errs:
        lines.append(f"\nAgent errors: {len(errs)} (first: {errs[0]['error']})")
    if any("person_contacts" in r for r in rows):
        lines.append("\n_People are scripted mocap walkers; contact is a base-distance test, not physics._")
    if rows and not rows[0]["locomotion_is_physical"]:
        lines.append("\n_Locomotion is the kinematic placeholder: walking results are not physical._")
    return "\n".join(lines)


def make_agent(args):
    if args.agent == "scripted":
        from ..agents.scripted import ScriptedAgent
        return ScriptedAgent()
    if args.agent == "anthropic":
        from ..agents.llm import AnthropicAgent
        return AnthropicAgent(model=args.model)
    if args.agent == "openai":
        from ..agents.llm import OpenAICompatAgent
        if not args.model:
            raise SystemExit("--model is required for --agent openai")
        return OpenAICompatAgent(model=args.model, base_url=args.base_url, send_images=not args.no_images)
    raise SystemExit(f"unknown agent {args.agent}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="scripted", choices=["scripted", "anthropic", "openai"])
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--tasks", nargs="*", help="task ids (default: all)")
    ap.add_argument("--seeds", type=int, default=1, help="seed 0 = canonical layout; >0 jitters objects")
    ap.add_argument("--jitter", type=float, default=0.03, help="object xy jitter (m) for seeds > 0")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    agent = make_agent(args)
    tasks = [TASKS_BY_ID[t] for t in args.tasks] if args.tasks else TASKS
    out = Path(args.out) / f"{time.strftime('%Y%m%d-%H%M%S')}_{agent.name.replace(':', '_').replace('/', '_')}"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(out / "results.jsonl", "w") as f:
        for task in tasks:
            for seed in range(args.seeds):
                row = run_episode(agent, task, seed, out, args.jitter)
                rows.append(row)
                f.write(json.dumps(row, default=str) + "\n")
                f.flush()
                mark = "PASS" if row["success"] else "FAIL"
                print(f"[{mark}] {task.id} seed={seed} calls={row['tool_calls']} "
                      f"fails={row['skill_failures']} answer={row['answer']!r}")
    summary = summarize(rows)
    (out / "summary.md").write_text(summary + "\n")
    print("\n" + summary + f"\n\nResults: {out}")


if __name__ == "__main__":
    main()
