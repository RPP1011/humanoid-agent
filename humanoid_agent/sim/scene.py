"""Scene construction: G1 (with hands) + tables, objects, cameras, named locations.

Built with MjSpec so tasks can define their own layouts without hand-editing XML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "unitree_g1"
G1_SCENE = ASSET_DIR / "scene_with_hands.xml"


@dataclass
class Table:
    name: str
    pos: tuple[float, float]          # x, y of table center
    size: tuple[float, float] = (0.3, 0.4)  # half extents x, y
    height: float = 0.75


@dataclass
class Obj:
    name: str
    pos: tuple[float, float, float]
    half_size: float = 0.025
    rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)
    category: str = "cube"


@dataclass
class SceneSpec:
    tables: list[Table] = field(default_factory=list)
    objects: list[Obj] = field(default_factory=list)
    # Named navigation targets the planner can refer to (x, y, yaw).
    locations: dict[str, tuple[float, float, float]] = field(default_factory=dict)


def default_scene() -> SceneSpec:
    return SceneSpec(
        tables=[
            Table("table_a", (1.2, 0.0)),
            Table("table_b", (0.0, 1.8)),
        ],
        objects=[
            Obj("red_cube", (1.0, -0.1, 0.80), rgba=(0.85, 0.15, 0.15, 1)),
            Obj("blue_cube", (1.0, 0.15, 0.80), rgba=(0.15, 0.3, 0.85, 1)),
            Obj("green_cube", (0.0, 1.6, 0.80), rgba=(0.15, 0.7, 0.25, 1)),
        ],
        locations={
            "start": (0.0, 0.0, 0.0),
            "table_a": (0.68, 0.0, 0.0),      # standing in front of table_a, facing +x
            "table_b": (0.0, 1.28, 1.5708),   # standing in front of table_b, facing +y
        },
    )


def build_model(scene: SceneSpec) -> mujoco.MjModel:
    if not G1_SCENE.exists():
        raise FileNotFoundError(
            f"G1 assets not found at {ASSET_DIR}. Run scripts/fetch_assets.sh first."
        )
    spec = mujoco.MjSpec.from_file(str(G1_SCENE))

    # Ego camera on the torso (approximate head position), looking along +x.
    torso = spec.body("torso_link")
    torso.add_camera(name="ego", pos=[0.08, 0.0, 0.45], xyaxes=[0, -1, 0, 0.5, 0, 0.87], fovy=75)

    # Overview camera for debugging / eval videos.
    spec.worldbody.add_camera(name="overview", pos=[-1.8, -1.8, 2.6], xyaxes=[0.707, -0.707, 0, 0.45, 0.45, 0.77])

    for t in scene.tables:
        b = spec.worldbody.add_body(name=t.name, pos=[t.pos[0], t.pos[1], 0])
        b.add_geom(
            name=f"{t.name}_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[t.size[0], t.size[1], 0.02],
            pos=[0, 0, t.height - 0.02],
            rgba=[0.55, 0.4, 0.25, 1],
        )
        for sx in (-1, 1):
            for sy in (-1, 1):
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.02, 0.02, (t.height - 0.04) / 2],
                    pos=[sx * (t.size[0] - 0.03), sy * (t.size[1] - 0.03), (t.height - 0.04) / 2],
                    rgba=[0.4, 0.3, 0.2, 1],
                )

    for o in scene.objects:
        b = spec.worldbody.add_body(name=o.name, pos=list(o.pos))
        b.add_freejoint(name=f"{o.name}_free")
        b.add_geom(
            name=f"{o.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[o.half_size] * 3,
            rgba=list(o.rgba),
            mass=0.05,
            friction=[1.5, 0.02, 0.001],
        )

    return spec.compile()
