"""Scene construction: G1 (with hands) + tables, objects, cameras, named locations.

Built with MjSpec so tasks can define their own layouts without hand-editing XML.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .person import Person

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
    shape: str = "box"                 # "box" (cube of half_size) or "cylinder" (half_size = half height)
    radius: float = 0.0                # cylinder radius


@dataclass
class Fridge:
    """Box fridge with interior shelves and one hinged door (articulated, joint `<name>_door`).

    Local frame: +x is the outward normal of the front face, so a robot working at
    the fridge faces yaw + pi. The hinge is on the robot's right (local +y); the door
    opens by swinging its free edge out toward the robot and over to that side.
    """
    name: str
    pos: tuple[float, float]               # x, y of footprint centre
    yaw: float = 0.0                       # world heading of the front face; multiples of pi/2 only
    size: tuple[float, float, float] = (0.30, 0.30, 1.50)  # half depth, half width, full height
    shelves: tuple[float, ...] = (0.70, 1.05)  # shelf top heights; reachable band for a standing G1
    shelf_names: tuple[str, ...] = ("bottom_shelf", "top_shelf")
    open_angle: float = 1.92               # rad; ~110 deg puts the door clear beside the front
    wall: float = 0.02
    # Shelves stop short of the front so a hand at carry height in the doorway clears them.
    shelf_inset: float = 0.05
    door_thick: float = 0.03

    def to_world(self, lx: float, ly: float) -> tuple[float, float]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return self.pos[0] + c * lx - s * ly, self.pos[1] + s * lx + c * ly

    def _aabb_half(self, hx: float, hy: float) -> tuple[float, float]:
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        return c * hx + s * hy, s * hx + c * hy

    @property
    def footprint(self) -> Table:
        """Axis-aligned footprint of the carcass (door excluded), as an obstacle record."""
        return Table(self.name, self.pos, self._aabb_half(self.size[0] + self.door_thick, self.size[1]),
                     self.size[2])

    def shelf_tables(self) -> list[Table]:
        """Each shelf as a Table-like support record named `<fridge>_<shelf>`."""
        d, w, t = self.size[0], self.size[1], self.wall
        hx, cx = self.shelf_half_depth, t / 2 - self.shelf_inset / 2
        return [Table(f"{self.name}_{n}", self.to_world(cx, 0.0), self._aabb_half(hx, w - t), h)
                for n, h in zip(self.shelf_names, self.shelves)]

    @property
    def shelf_half_depth(self) -> float:
        return self.size[0] - self.wall / 2 - self.shelf_inset / 2

    @property
    def hinge_local(self) -> tuple[float, float]:
        return self.size[0] + self.door_thick / 2, self.size[1]

    def door_footprint(self, angle: float) -> Table:
        """World AABB swept by the door panel at `angle` (for routing around an open door)."""
        hx, hy = self.hinge_local
        w = 2 * self.size[1]
        pts = [(hx, hy), (hx + w * math.sin(angle), hy - w * math.cos(angle))]
        xs, ys = zip(*(self.to_world(*p) for p in pts))
        return Table(f"{self.name}_door", ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2),
                     ((max(xs) - min(xs)) / 2 + 0.02, (max(ys) - min(ys)) / 2 + 0.02), self.size[2])

    def stance(self, standoff: float) -> tuple[float, float, float]:
        """Standing pose `standoff` metres in front of the front face, facing the fridge."""
        x, y = self.to_world(self.size[0] + self.door_thick + standoff, 0.0)
        return x, y, math.atan2(math.sin(self.yaw + math.pi), math.cos(self.yaw + math.pi))


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
    fridges: list[Fridge] = field(default_factory=list)
    people: list[Person] = field(default_factory=list)  # scripted pedestrians (sim/person.py)
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

    for f in scene.fridges:
        _add_fridge(spec, f)

    for o in scene.objects:
        b = spec.worldbody.add_body(name=o.name, pos=list(o.pos))
        b.add_freejoint(name=f"{o.name}_free")
        cyl = o.shape == "cylinder"
        b.add_geom(
            name=f"{o.name}_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER if cyl else mujoco.mjtGeom.mjGEOM_BOX,
            size=[o.radius, o.half_size, 0] if cyl else [o.half_size] * 3,
            rgba=list(o.rgba),
            mass=0.05,
            friction=[1.5, 0.02, 0.001],
            contype=WORLD_BITS, conaffinity=WORLD_BITS,
        )

    for p in scene.people:
        _add_person(spec, p)
    if scene.people:
        # Top-down view for videos: the overview camera is low enough that a person
        # crossing the room hides the robot.
        spec.worldbody.add_camera(name="topdown", pos=[0.1, 0.1, 4.2], xyaxes=[1, 0, 0, 0, 1, 0], fovy=60)

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


def _add_person(spec: mujoco.MjSpec, p: Person) -> None:
    # Mocap body, moved by env each control step. Visual only (group 2, no contact bits):
    # robot-person contact is a distance test in env, and group 2 keeps the person out of
    # surface_below's ray casts.
    b = spec.worldbody.add_body(name=p.name, mocap=True, pos=[*p.path[0], 0])
    vis = dict(contype=0, conaffinity=0, group=2)
    cap = mujoco.mjtGeom.mjGEOM_CAPSULE
    skin, shirt, pants = [0.85, 0.65, 0.5, 1], [0.2, 0.55, 0.45, 1], [0.2, 0.22, 0.3, 1]
    for y in (-0.09, 0.09):
        b.add_geom(type=cap, fromto=[0, y, 0.06, 0, y, 0.85], size=[0.065, 0, 0], rgba=pants, **vis)
    b.add_geom(name=f"{p.name}_torso", type=cap, fromto=[0, 0, 0.95, 0, 0, 1.35], size=[0.16, 0, 0],
               rgba=shirt, **vis)
    for y in (-0.22, 0.22):
        b.add_geom(type=cap, fromto=[0, y, 1.42, 0.03, y, 0.9], size=[0.045, 0, 0], rgba=shirt, **vis)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, pos=[0.0, 0, 1.62], size=[0.11, 0, 0], rgba=skin, **vis)
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.1, 0, 1.64], size=[0.02, 0.05, 0.015],
               rgba=[0.1, 0.1, 0.1, 1], **vis)  # "eyes": shows which way it faces


FRIDGE_WHITE = [0.93, 0.94, 0.95, 1]
_WORLD = dict(contype=WORLD_BITS, conaffinity=WORLD_BITS)


def _add_fridge(spec: mujoco.MjSpec, f: Fridge) -> None:
    box = mujoco.mjtGeom.mjGEOM_BOX
    d, w, h, t = f.size[0], f.size[1], f.size[2], f.wall
    b = spec.worldbody.add_body(name=f.name, pos=[f.pos[0], f.pos[1], 0],
                                quat=[math.cos(f.yaw / 2), 0, 0, math.sin(f.yaw / 2)])
    low = f.shelves[0] - t  # solid compressor/crisper block below the lowest shelf
    for name, size, pos, rgba in [
        ("back", [t / 2, w, h / 2], [-d + t / 2, 0, h / 2], FRIDGE_WHITE),
        ("left", [d, t / 2, h / 2], [0, -w + t / 2, h / 2], FRIDGE_WHITE),
        ("right", [d, t / 2, h / 2], [0, w - t / 2, h / 2], FRIDGE_WHITE),
        ("roof", [d, w, t / 2], [0, 0, h - t / 2], FRIDGE_WHITE),
        ("base", [d, w, low / 2], [0, 0, low / 2], [0.85, 0.86, 0.88, 1]),
    ]:
        b.add_geom(name=f"{f.name}_{name}", type=box, size=size, pos=pos, rgba=rgba, **_WORLD)
    for n, z in zip(f.shelf_names, f.shelves):
        b.add_geom(name=f"{f.name}_{n}", type=box, size=[f.shelf_half_depth, w - t, t / 2],
                   pos=[t / 2 - f.shelf_inset / 2, 0, z - t / 2],
                   rgba=[0.75, 0.85, 0.9, 1], **_WORLD)
    # Close-up for videos: from the robot's front-left (opposite the hinge), so the open
    # door does not hide the shelves.
    target = np.array([*f.to_world(f.size[0] - 0.1, 0.0), 0.95])
    eye = np.array([*f.to_world(f.size[0] + 1.3, -0.9), 1.8])
    fwd = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    spec.worldbody.add_camera(name=f"{f.name}_cam", pos=list(eye), xyaxes=list(right) + list(np.cross(right, fwd)),
                              fovy=55)
    hx, hy = f.hinge_local
    door = b.add_body(name=f"{f.name}_door", pos=[hx, hy, 0])
    door.add_joint(name=f"{f.name}_door", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                   range=[0, f.open_angle + 0.1], damping=1.0)
    door.add_geom(name=f"{f.name}_door_panel", type=box, size=[f.door_thick / 2, w, h / 2 - 0.005],
                  pos=[0, -w, h / 2], rgba=FRIDGE_WHITE, mass=5.0, **_WORLD)
    # Vertical bar handle on the free edge, at a height the standing G1 can reach.
    door.add_geom(name=f"{f.name}_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.012, 0.12, 0],
                  pos=[f.door_thick / 2 + 0.035, -2 * w + 0.06, 1.0], rgba=[0.6, 0.6, 0.62, 1], mass=0.1, **_WORLD)
    door.add_site(name=f"{f.name}_handle", pos=[f.door_thick / 2 + 0.035, -2 * w + 0.06, 1.0])


def kitchen_scene() -> SceneSpec:
    """Minimal kitchen: fridge (soda, juice, yogurt on two shelves), dining table, counter."""
    fridge = Fridge("fridge", (1.6, 0.0), yaw=math.pi)  # front faces -x, toward the start pose
    top, bottom = (z + 0.001 for z in fridge.shelves[::-1])
    return SceneSpec(
        tables=[
            Table("kitchen_table", (0.0, 1.55), size=(0.4, 0.3)),
            Table("counter", (0.0, -1.45), size=(0.5, 0.25), height=0.85),
        ],
        fridges=[fridge],
        objects=[
            Obj("soda_can", (1.43, 0.12, bottom + 0.05), half_size=0.05, shape="cylinder", radius=0.03,
                rgba=(0.8, 0.1, 0.1, 1), category="drink"),
            Obj("juice_box", (1.43, -0.12, bottom + 0.035), half_size=0.035,
                rgba=(0.95, 0.6, 0.1, 1), category="drink"),
            Obj("yogurt", (1.45, -0.14, top + 0.03), half_size=0.03, shape="cylinder", radius=0.035,
                rgba=(0.95, 0.95, 0.85, 1), category="food"),
            Obj("milk", (0.15, -1.33, 0.85 + 0.035), half_size=0.035, rgba=(0.9, 0.92, 1.0, 1), category="drink"),
        ],
        locations={
            "start": (0.0, 0.0, 0.0),
            "fridge": fridge.stance(0.16),
            "kitchen_table": (0.0, 1.03, 1.5708),
            "counter": (0.0, -0.98, -1.5708),
        },
    )
