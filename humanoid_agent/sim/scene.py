"""Scene construction: G1 (with hands) + tables, objects, cameras, named locations.

Built with MjSpec so tasks can define their own layouts without hand-editing XML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).resolve().parents[2] / "assets" / "unitree_g1"
G1_SCENE = ASSET_DIR / "scene_with_hands.xml"
# Collision bits. The robot keeps MuJoCo's default (1); world geoms (floor, tables,
# bags, objects) also carry bit 2, so a just-released object can be set to bit 2
# only: it still lands on things but ignores the hand it is leaving (env.detach).
WORLD_BITS = 3


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
class BeanBag:
    """Soft, lumpy cushion resting on a table: a bed of 3 cm cells, each on a sprung,
    damped vertical slide joint with friction loss.

    APPROXIMATION of a bean bag (see sim/AGENTS.md for why not a MuJoCo flex): it is
    locally compliant (a load sinks the cells under it, an off-centre load sinks one
    side more and tilts what is on it) and somewhat plastic (friction loss keeps a
    dent), but the filling never flows sideways or bulges, and cells only move in z.
    """
    name: str
    pos: tuple[float, float]           # x, y of the centre
    base_z: float = 0.75               # what it rests on (a table top)
    radius: float = 0.17
    height: float = 0.09               # crown height above base_z
    pitch: float = 0.03                # cell size = surface resolution
    stiffness: float = 100.0           # N/m per cell
    damping: float = 1.0               # N s/m per cell
    frictionloss: float = 0.05         # N per cell: dents stay put under small loads
    lumps: float = 0.008               # amplitude (m) of seeded random bumps on the crown
    lump_seed: int = 0
    rgba: tuple[float, float, float, float] = (0.55, 0.25, 0.6, 1.0)

    def cells(self) -> list[tuple[float, float, float]]:
        """(dx, dy, top height above base_z) of every cell, lumps included."""
        rng = np.random.default_rng(self.lump_seed)
        bumps = [(rng.uniform(-0.7, 0.7) * self.radius, rng.uniform(-0.7, 0.7) * self.radius,
                  rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0) * self.lumps) for _ in range(3)]
        n = int(self.radius / self.pitch)
        out = []
        for i in range(-n, n + 1):
            for j in range(-n, n + 1):
                dx, dy = i * self.pitch, j * self.pitch
                r = np.hypot(dx, dy) / self.radius
                if r > 0.95:
                    continue
                # Flat-ish crown with rounded shoulders, like a filled bag sagging on a table.
                h = self.height * np.sqrt(1 - r ** 4)
                h += sum(a * np.exp(-((dx - bx) ** 2 + (dy - by) ** 2) / (2 * 0.05 ** 2)) for bx, by, a in bumps)
                if h > 0.03:
                    out.append((dx, dy, float(h)))
        return out


@dataclass
class SceneSpec:
    tables: list[Table] = field(default_factory=list)
    objects: list[Obj] = field(default_factory=list)
    bags: list[BeanBag] = field(default_factory=list)
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
    # Default pyramidal cones with impratio=1 let resting contacts creep: a cube tower
    # on a 4 deg slope slid apart layer by layer in a few seconds even on a rigid support.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    floor = spec.geom("floor")
    floor.contype = floor.conaffinity = WORLD_BITS

    for t in scene.tables:
        b = spec.worldbody.add_body(name=t.name, pos=[t.pos[0], t.pos[1], 0])
        b.add_geom(
            name=f"{t.name}_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[t.size[0], t.size[1], 0.02],
            pos=[0, 0, t.height - 0.02],
            rgba=[0.55, 0.4, 0.25, 1],
            contype=WORLD_BITS, conaffinity=WORLD_BITS,
        )
        for sx in (-1, 1):
            for sy in (-1, 1):
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.02, 0.02, (t.height - 0.04) / 2],
                    pos=[sx * (t.size[0] - 0.03), sy * (t.size[1] - 0.03), (t.height - 0.04) / 2],
                    rgba=[0.4, 0.3, 0.2, 1], contype=WORLD_BITS, conaffinity=WORLD_BITS,
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
            contype=WORLD_BITS, conaffinity=WORLD_BITS,
        )

    for g in scene.bags:
        # Close-up for videos/debugging: from the front-right, looking down at the crown.
        target = np.array([g.pos[0], g.pos[1], g.base_z + g.height])
        eye = target + np.array([0.55, -0.45, 0.35])
        fwd = (target - eye) / np.linalg.norm(target - eye)
        right = np.cross(fwd, [0, 0, 1])
        right /= np.linalg.norm(right)
        spec.worldbody.add_camera(name=f"{g.name}_cam", pos=list(eye), xyaxes=list(right) + list(np.cross(right, fwd)),
                                  fovy=50)
        half = g.pitch / 2 - 0.001  # 2 mm gaps: cells never touch each other
        for k, (dx, dy, h) in enumerate(g.cells()):
            top = g.base_z + h
            c = spec.worldbody.add_body(name=f"{g.name}_cell{k}", pos=[g.pos[0] + dx, g.pos[1] + dy, top - 0.01],
                                        gravcomp=1.0)
            # Lower limit keeps the 2 cm plate above the table it rests on.
            c.add_joint(name=f"{g.name}_cell{k}_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                        stiffness=g.stiffness, damping=g.damping, frictionloss=g.frictionloss, armature=0.001,
                        limited=True, range=[-(h - 0.021), 0.005])
            c.add_geom(name=f"{g.name}_cell{k}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, 0.01],
                       mass=0.005, friction=[1.2, 0.02, 0.001], rgba=list(g.rgba),
                       contype=WORLD_BITS, conaffinity=WORLD_BITS)
            c.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half, half, (h - 0.02) / 2], group=1,
                       pos=[0, 0, -0.01 - (h - 0.02) / 2], contype=0, conaffinity=0, density=0,
                       rgba=[v * 0.8 for v in g.rgba[:3]] + [1.0])

    return spec.compile()
