"""Drive eval episodes one tool call at a time from outside the process.

For planners that cannot be called from Python, e.g. a Claude Code subagent with
no API key: it calls tools through this CLI and the server runs the real
`run_episode` (same budget, finish, settle and checks as eval/runner.py).

    python scripts/episode_server.py serve --out runs/subagent [--video]   # leave running
    python scripts/episode_server.py start bean_bag_tower 3                # -> episode id + prompt
    python scripts/episode_server.py call <id> list_objects '{}'          # -> tool result JSON
    python scripts/episode_server.py call <id> finish '{"answer": "done"}'
    python scripts/episode_server.py summary                               # summary.md of finished rows

--video writes <task>_s<seed>.mp4 (overview of the robot plus a close-up of the
first bean bag, if any) with ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def serve(port: int, out: Path, video: bool, jitter: float) -> None:
    # Sim imports live here so the client commands start fast and need no GL backend.
    import numpy as np

    from humanoid_agent.agents.base import EpisodeDone
    from humanoid_agent.eval import runner
    from humanoid_agent.eval.tasks import TASKS_BY_ID
    from humanoid_agent.sim.env import HumanoidEnv
    from humanoid_agent.tools.registry import result_to_text

    recorders: dict[int, HumanoidEnv] = {}  # thread id -> env, so each episode's frames can be found

    class VideoEnv(HumanoidEnv):
        every = 5  # control steps per frame: 10 fps of sim time

        def __init__(self, *a, **kw) -> None:
            self.frames: list[np.ndarray] = []
            self._k = 0
            super().__init__(*a, render_size=(480, 360), **kw)
            recorders[threading.get_ident()] = self

        def step(self, n: int = 1) -> None:
            for _ in range(n):
                super().step(1)
                self._k += 1
                if self._k % self.every == 0:
                    cams = ["overview"] + [f"{b.name}_cam" for b in self.scene.bags[:1]]
                    self.frames.append(np.concatenate([self.render(c) for c in cams], axis=1))

    class RemoteAgent:
        name = "subagent"

        def __init__(self) -> None:
            self.inbox: queue.Queue = queue.Queue()
            self.outbox: queue.Queue = queue.Queue()

        def run(self, task_id: str, prompt: str, call) -> None:
            while True:
                tool, args = self.inbox.get()
                try:
                    res = call(tool, args)
                except EpisodeDone:
                    self.outbox.put("episode ended")
                    raise
                self.outbox.put(result_to_text(res))

    out.mkdir(parents=True, exist_ok=True)
    if video:
        runner.HumanoidEnv = VideoEnv
    episodes: dict[str, RemoteAgent] = {}
    lock = threading.Lock()

    def episode(eid: str, agent: RemoteAgent, task_id: str, seed: int) -> None:
        row = runner.run_episode(agent, TASKS_BY_ID[task_id], seed, out, jitter)
        row["episode"] = eid
        with lock, open(out / "results.jsonl", "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        env = recorders.pop(threading.get_ident(), None)
        if env is not None and env.frames:
            h, w, _ = env.frames[0].shape
            p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                  "-s", f"{w}x{h}", "-r", "20", "-i", "-", "-pix_fmt", "yuv420p",
                                  str(out / f"{task_id}_s{seed}.mp4")], stdin=subprocess.PIPE)
            for fr in env.frames:
                p.stdin.write(fr.tobytes())
            p.stdin.close()
            p.wait()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:
            pass

        def do_POST(self) -> None:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if req["op"] == "start":
                eid = f"{req['task']}_s{req['seed']}"
                agent = RemoteAgent()
                episodes[eid] = agent
                threading.Thread(target=episode, args=(eid, agent, req["task"], int(req["seed"])), daemon=True).start()
                reply = {"episode": eid, "prompt": TASKS_BY_ID[req["task"]].prompt}
            elif req["op"] == "call":
                agent = episodes.get(req["episode"])
                if agent is None:
                    reply = {"error": "unknown or finished episode"}
                else:
                    agent.inbox.put((req["tool"], req["args"]))
                    reply = {"result": agent.outbox.get()}
                    if reply["result"] == "episode ended":
                        episodes.pop(req["episode"], None)
            else:  # summary
                rows = [json.loads(l) for l in open(out / "results.jsonl")] if (out / "results.jsonl").exists() else []
                (out / "summary.md").write_text(runner.summarize(rows) + "\n")
                reply = {"summary": runner.summarize(rows)}
            body = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"serving on 127.0.0.1:{port}, writing {out}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def _post(port: int, payload: dict) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{port}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--out", default="runs/subagent")
    s.add_argument("--video", action="store_true")
    s.add_argument("--jitter", type=float, default=0.03)
    st = sub.add_parser("start")
    st.add_argument("task")
    st.add_argument("seed", type=int)
    c = sub.add_parser("call")
    c.add_argument("episode")
    c.add_argument("tool")
    c.add_argument("args", nargs="?", default="{}")
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "serve":
        serve(a.port, Path(a.out), a.video, a.jitter)
    elif a.cmd == "start":
        print(json.dumps(_post(a.port, {"op": "start", "task": a.task, "seed": a.seed})))
    elif a.cmd == "call":
        try:
            args = json.loads(a.args)
        except json.JSONDecodeError as e:
            sys.exit(f"args must be a JSON object: {e}")
        r = _post(a.port, {"op": "call", "episode": a.episode, "tool": a.tool, "args": args})
        print(r.get("result", r.get("error")))
    else:
        print(_post(a.port, {"op": "summary"})["summary"])


if __name__ == "__main__":
    main()
