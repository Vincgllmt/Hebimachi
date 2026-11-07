"""Simulation simplifiée d'un coup, pour une direction de frappe donnée.

Modèle volontairement géométrique :

- chaque bille va en ligne droite ; la bande la renvoie comme un miroir ;
- au choc, la bille frappée part sur la ligne des centres, celle qui frappe sur la tangente
  (règle des 90°, juste pour une bille qui glisse : la blanche avec du coulé ou du rétro s'en écarte,
  et une bille qui roule s'en écarte peu à peu vers ~30°) ;
- l'élan est une distance restant à parcourir : au choc, il se partage en cos² (frappée) et sin²
  (frappeuse) de l'angle de coupe ; chaque rebond en coûte une part.

Les billes mises en mouvement sont suivies l'une après l'autre, face aux billes encore immobiles :
deux billes en mouvement ne se rencontrent pas. Suffisant pour juger un coup à deux ou trois billes.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .table import Point, Table, travel

DEFAULT_POWER = 3000     # élan de la blanche, en pixels (≈ 2 longueurs de table)
CUSHION_KEEP = 0.7       # part de l'élan gardée après un rebond
MIN_TRAVEL = 25          # en dessous, on considère la bille arrêtée
MAX_MOVES = 12


@dataclass
class Segment:
    ball: int
    a: Point
    b: Point
    fate: str                     # pocket, ball, cushion, jaw, stop
    pocket: str | None = None
    hit: int | None = None        # bille touchée au bout du segment


@dataclass
class Outcome:
    first: int | None = None                                  # première bille touchée par la blanche
    segments: list[Segment] = field(default_factory=list)
    pocketed: dict[int, str] = field(default_factory=dict)    # bille -> poche
    contacts: list[tuple[int, int]] = field(default_factory=list)   # (frappeuse, frappée), dans l'ordre
    banked: set[int] = field(default_factory=set)             # billes passées par la bande

    def struck_by(self, ball: int) -> int | None:
        return next((s for s, t in self.contacts if t == ball), None)


def _unit(v: Point) -> Point:
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else (0.0, 0.0)


def simulate(table: Table, positions: list[Point], cue: int, direction: Point,
             start: Point | None = None, power: float = DEFAULT_POWER, max_bounces: int = 1) -> Outcome:
    """Frappe la blanche `cue` (depuis `start`, par défaut sa position) dans `direction`."""
    still = set(range(len(positions))) - {cue}
    return _run(table, positions, cue, still, [(cue, start or positions[cue], _unit(direction), power, 0)],
                Outcome(), max_bounces)


def from_contact(table: Table, positions: list[Point], cue: int, target: int, ghost: Point, aim: Point,
                 power: float = DEFAULT_POWER, max_bounces: int = 1) -> Outcome:
    """Suite d'un coup à partir du choc : la blanche est au fantôme `ghost`, arrivée selon `aim`.

    La bille visée part du fantôme vers son centre, quelle que soit la direction d'arrivée : c'est
    ce qui rend le verdict fiable même après un rebond de la blanche.
    """
    out = Outcome(first=target, contacts=[(cue, target)])
    still = set(range(len(positions))) - {cue, target}
    left = max(power - math.dist(positions[cue], ghost), MIN_TRAVEL * 4)
    seed = _split(cue, ghost, _unit(aim), target, positions[target], left, 0)
    return _run(table, positions, cue, still, seed, out, max_bounces)


def _split(ball: int, at: Point, u: Point, hit: int, hit_pos: Point, left: float, bounces: int) -> list:
    """Partage de l'élan au choc : la frappée sur la ligne des centres, la frappeuse sur la tangente."""
    n = _unit((hit_pos[0] - at[0], hit_pos[1] - at[1]))
    cos = max(0.0, u[0] * n[0] + u[1] * n[1])
    moves = []
    if left * cos * cos > MIN_TRAVEL:
        moves.append((hit, hit_pos, n, left * cos * cos, 0))
    tangent = (u[0] - cos * n[0], u[1] - cos * n[1])
    if left * (1 - cos * cos) > MIN_TRAVEL and math.hypot(*tangent) > 1e-6:
        moves.append((ball, at, _unit(tangent), left * (1 - cos * cos), bounces))
    return moves


def _run(table: Table, positions: list[Point], cue: int, still: set[int], seed: list, out: Outcome,
         max_bounces: int) -> Outcome:
    R = table.radius
    moves = deque(seed)
    count = 0
    while moves and count < MAX_MOVES:
        count += 1
        ball, pos, u, budget, bounces = moves.popleft()
        ids = sorted(still)
        path = travel(table, pos, u, [positions[i] for i in ids])
        length = math.dist(pos, path.end)
        if length > budget:
            stop = (pos[0] + u[0] * budget, pos[1] + u[1] * budget)
            out.segments.append(Segment(ball, pos, stop, "stop"))
            continue
        left = budget - length
        if path.pocket:
            out.segments.append(Segment(ball, pos, path.end, "pocket", pocket=path.pocket.name))
            out.pocketed[ball] = path.pocket.name
        elif path.blocker is not None:
            j = ids[path.blocker]
            out.segments.append(Segment(ball, pos, path.end, "ball", hit=j))
            out.contacts.append((ball, j))
            if ball == cue and out.first is None:
                out.first = j
            still.discard(j)
            moves.extend(_split(ball, path.end, u, j, positions[j], left, bounces))
        elif path.jaw:
            out.segments.append(Segment(ball, pos, path.end, "jaw"))
        else:
            out.segments.append(Segment(ball, pos, path.end, "cushion"))
            if bounces < max_bounces and left * CUSHION_KEEP > MIN_TRAVEL:
                e = path.end
                wall_x = abs(e[0] - (table.left + R)) < 1e-6 or abs(e[0] - (table.right - R)) < 1e-6
                wall_y = abs(e[1] - (table.top + R)) < 1e-6 or abs(e[1] - (table.bottom - R)) < 1e-6
                out.banked.add(ball)
                moves.append((ball, e, (-u[0] if wall_x else u[0], -u[1] if wall_y else u[1]),
                              left * CUSHION_KEEP, bounces + 1))
    return out
