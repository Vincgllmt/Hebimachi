"""Géométrie de la table de billard, en pixels d'écran.

La caméra est fixe et vue de dessus : la table se calibre une fois pour toutes par jeu, dans
data/games/<id>/billiards.json. Les bandes sont repérées par leur nez (le bord du tapis que
touche une bille) ; une bille ne peut donc pas avoir son centre à moins d'un rayon de ces lignes.

Chaque poche est repérée par ses deux mâchoires (bouts de bande). Une bille entre si, là où elle
atteindrait la ligne des bandes, il n'y a pas de bande (ouverture de la poche) et si son centre
passe à au moins `jaw_margin` rayon des deux mâchoires.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

Point = tuple[float, float]
GRAZE = 0.85   # au-delà de cette part d'un diamètre d'écart latéral, le contact est « de justesse »


@dataclass(frozen=True)
class Pocket:
    name: str
    a: Point          # mâchoire
    b: Point          # autre mâchoire
    side: bool        # poche du milieu (refuse les angles trop rasants)

    @property
    def center(self) -> Point:
        return ((self.a[0] + self.b[0]) / 2, (self.a[1] + self.b[1]) / 2)


@dataclass(frozen=True)
class Table:
    left: float
    top: float
    right: float
    bottom: float
    radius: float = 16.5
    corner_mouth: float = 48      # distance du coin à chaque mâchoire d'une poche de coin
    side_mouth: float = 42        # demi-largeur d'une poche du milieu
    jaw_margin: float = 0.6       # en rayons : écart minimal du centre aux mâchoires
    side_max_angle: float = 55    # en degrés depuis la perpendiculaire, poches du milieu
    screen: tuple[int, int] = (1920, 1080)
    hud_tray: tuple[int, int, int, int] | None = None       # x0, y0, x1, y1
    hud_next: tuple[int, int, int] | None = None            # x, y, rayon de l'icône « NEXT »
    pockets: tuple[Pocket, ...] = field(init=False)

    def __post_init__(self):
        l, t, r, b, c, s = self.left, self.top, self.right, self.bottom, self.corner_mouth, self.side_mouth
        mid = (l + r) / 2
        pockets = (
            Pocket("haut gauche", (l + c, t), (l, t + c), False),
            Pocket("haut milieu", (mid - s, t), (mid + s, t), True),
            Pocket("haut droite", (r - c, t), (r, t + c), False),
            Pocket("bas gauche", (l + c, b), (l, b - c), False),
            Pocket("bas milieu", (mid - s, b), (mid + s, b), True),
            Pocket("bas droite", (r - c, b), (r, b - c), False),
        )
        object.__setattr__(self, "pockets", pockets)

    @classmethod
    def load(cls, path: Path) -> "Table":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        cu, hud = d["cushions"], d.get("hud", {})
        return cls(cu["left"], cu["top"], cu["right"], cu["bottom"],
                   radius=d.get("ball_radius", 16.5), corner_mouth=d.get("corner_mouth", 48),
                   side_mouth=d.get("side_mouth", 42), jaw_margin=d.get("jaw_margin", 0.6),
                   side_max_angle=d.get("side_max_angle", 55), screen=tuple(d.get("screen", (1920, 1080))),
                   hud_tray=tuple(hud["tray"]) if "tray" in hud else None,
                   hud_next=tuple(hud["next_icon"]) if "next_icon" in hud else None)

    def inside(self, p: Point, margin: float = 0) -> bool:
        return (self.left + margin <= p[0] <= self.right - margin
                and self.top + margin <= p[1] <= self.bottom - margin)


@dataclass
class Travel:
    """Trajet d'une bille en ligne droite jusqu'au premier obstacle."""
    start: Point
    end: Point
    pocket: Pocket | None = None   # poche atteinte
    blocker: int | None = None     # indice de la bille rencontrée en premier
    graze: bool = False            # contact de justesse : à la précision de la lecture près
    jaw: Pocket | None = None      # poche manquée de peu : la bille heurte une mâchoire
    too_steep: bool = False        # poche du milieu abordée trop en biais


def _unit(v: Point) -> Point:
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else (0.0, 0.0)


def _cushion_hit(table: Table, p: Point, u: Point) -> float:
    """Distance parcourue avant que le centre ne touche une bande (rectangle réduit d'un rayon)."""
    R = table.radius
    ts = []
    if u[0] > 1e-9:
        ts.append((table.right - R - p[0]) / u[0])
    elif u[0] < -1e-9:
        ts.append((table.left + R - p[0]) / u[0])
    if u[1] > 1e-9:
        ts.append((table.bottom - R - p[1]) / u[1])
    elif u[1] < -1e-9:
        ts.append((table.top + R - p[1]) / u[1])
    return max(0.0, min(ts)) if ts else 0.0


def _mouth_at(table: Table, e: Point) -> Pocket | None:
    """Poche dont l'ouverture se trouve là où la bille atteint la ligne des bandes, s'il y en a une."""
    R, c, s = table.radius, table.corner_mouth, table.side_mouth
    mid = (table.left + table.right) / 2
    top, bottom = abs(e[1] - (table.top + R)) < 1e-6, abs(e[1] - (table.bottom - R)) < 1e-6
    left, right = abs(e[0] - (table.left + R)) < 1e-6, abs(e[0] - (table.right - R)) < 1e-6
    by_name = {p.name: p for p in table.pockets}
    for row, on_row in (("haut", top), ("bas", bottom)):
        if on_row:
            if e[0] < table.left + c:
                return by_name[f"{row} gauche"]
            if e[0] > table.right - c:
                return by_name[f"{row} droite"]
            if abs(e[0] - mid) < s:
                return by_name[f"{row} milieu"]
    for col, on_col in (("gauche", left), ("droite", right)):
        if on_col:
            if e[1] < table.top + c:
                return by_name[f"haut {col}"]
            if e[1] > table.bottom - c:
                return by_name[f"bas {col}"]
    return None


def _ray_distance(p: Point, u: Point, q: Point) -> float:
    """Distance de q à la demi-droite p + t·u."""
    wx, wy = q[0] - p[0], q[1] - p[1]
    along = wx * u[0] + wy * u[1]
    if along <= 0:
        return math.hypot(wx, wy)
    return abs(wx * u[1] - wy * u[0])


def travel(table: Table, start: Point, direction: Point, others: list[Point] = (),
           ignore: tuple[int, ...] = ()) -> Travel:
    """Suit une bille de `start` dans `direction` : poche, bande ou autre bille, le premier venu."""
    u = _unit(direction)
    R = table.radius
    t_end = _cushion_hit(table, start, u)
    pocket = jaw = None
    too_steep = False
    pk = _mouth_at(table, (start[0] + u[0] * t_end, start[1] + u[1] * t_end))
    if pk is not None:
        # pas de bande à cet endroit : la bille entre si elle passe assez loin des deux mâchoires
        clear = min(_ray_distance(start, u, pk.a), _ray_distance(start, u, pk.b))
        steep = pk.side and abs(u[1]) < math.cos(math.radians(table.side_max_angle))
        if clear >= table.jaw_margin * R and not steep:
            pocket = pk
        else:
            jaw, too_steep = pk, steep
    blocker, graze = None, False
    for i, q in enumerate(others):
        if i in ignore:
            continue
        wx, wy = q[0] - start[0], q[1] - start[1]
        along = wx * u[0] + wy * u[1]
        if along <= 0:
            continue
        perp = abs(wx * u[1] - wy * u[0])
        if perp >= 2 * R:
            continue
        t_contact = along - math.sqrt((2 * R) ** 2 - perp ** 2)
        if t_contact < t_end:
            t_end, blocker, pocket, jaw, too_steep = max(t_contact, 0.0), i, None, None, False
            graze = perp > GRAZE * 2 * R
    end = (start[0] + u[0] * t_end, start[1] + u[1] * t_end)
    return Travel(start, end, pocket=pocket, blocker=blocker, graze=graze, jaw=jaw, too_steep=too_steep)
