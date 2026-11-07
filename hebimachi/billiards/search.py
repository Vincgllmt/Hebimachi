"""Recherche de coups : direct, combinaison et carambolage.

Le joueur ne règle qu'une chose, la direction de la visée. On balaie donc les directions qui
touchent une bille autorisée, par petits pas, et on simule chacune. Les directions voisines qui
donnent le même enchaînement forment un coup ; la largeur de cette plage est la marge d'erreur,
exprimée en pixels de déplacement de la bille fantôme. Une combinaison a naturellement une marge
plus étroite qu'un coup direct : l'erreur d'angle s'amplifie à chaque choc.

Types de coups (un coup peut en cumuler plusieurs) :
- direct : la bille touchée par la blanche entre sans toucher d'autre bille ;
- combinaison : une bille entre après avoir été frappée par une autre bille que la blanche ;
- carambolage : une bille entre après avoir ricoché sur une autre bille (« kiss »),
  ou après avoir été touchée par la blanche ayant déjà ricoché sur une première bille.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .sim import DEFAULT_POWER, Outcome, simulate
from .table import Point, Table
from .vision import Scene

STEP_DEG = 0.02
KINDS = ("direct", "combinaison", "carambolage", "bande")


@dataclass
class Plan:
    angle: float              # direction de frappe de la blanche, en radians (écran : y vers le bas)
    outcome: Outcome
    ghost: Point              # position de la blanche au premier choc : le fantôme à reproduire
    margin_px: float          # demi-largeur de la plage qui donne le même coup, au niveau du fantôme
    kinds: tuple[str, ...]
    text: str
    banks: int = 0            # bandes de la blanche avant la première bille

    @property
    def score(self) -> float:
        # plusieurs billes empochées : un peu mieux ; bande avant la bille : un peu moins sûr
        return self.margin_px * (1 + 0.3 * (len(self.outcome.pocketed) - 1)) * 0.8 ** self.banks


def legal_first(scene: Scene) -> set[int]:
    """Billes que la blanche peut toucher en premier : la NEXT si le HUD la donne, sinon toutes.

    En rotation, NEXT est la plus petite bille restante : si une pleine de sa couleur est encore sur
    la table, c'est elle (la rayée de même couleur vaut 8 de plus).
    """
    others = {i for i in range(len(scene.balls)) if i != scene.cue}
    if not scene.next_color:
        return others
    same = [i for i in others if scene.balls[i].color == scene.next_color]
    solids = [i for i in same if not scene.balls[i].stripe]
    return set(solids or same) or others


def name(scene: Scene, i: int) -> str:
    """Nom court ; si deux billes se lisent pareil (rouge/bordeaux), on précise où elle est."""
    if i == scene.cue:
        return "la blanche"
    b = scene.balls[i]
    nums = "/".join(map(str, b.numbers)) or "?"
    twins = [j for j, o in enumerate(scene.balls) if j not in (i, scene.cue) and o.numbers == b.numbers]
    if not twins:
        return f"le {nums}"
    o = scene.balls[twins[0]]
    where = (("de gauche" if b.x < o.x else "de droite") if abs(b.x - o.x) >= abs(b.y - o.y)
             else ("du haut" if b.y < o.y else "du bas"))
    return f"le {nums} {where}"


def _cue_kicked(out: Outcome, cue: int) -> bool:
    """La blanche est passée par la bande avant de toucher sa première bille."""
    first = next((seg for seg in out.segments if seg.ball == cue), None)
    return first is not None and first.fate == "cushion"


def kinds_of(out: Outcome, cue: int) -> tuple[str, ...]:
    kinds = ["bande"] if _cue_kicked(out, cue) else []
    cue_contacts = [t for s, t in out.contacts if s == cue]
    for ball in out.pocketed:
        if ball == cue:
            continue
        hitter = out.struck_by(ball)
        kissed = any(s == ball for s, _ in out.contacts)       # a ricoché sur une bille avant d'entrer
        if hitter is not None and hitter != cue:
            kinds.append("combinaison")
        if kissed or (hitter == cue and cue_contacts and cue_contacts[0] != ball):
            kinds.append("carambolage")
        if ball in out.banked:
            kinds.append("bande")
        if hitter == cue and not kissed and ball not in out.banked and cue_contacts[:1] == [ball]:
            kinds.append("direct")
    return tuple(k for k in KINDS if k in kinds)


def describe(out: Outcome, scene: Scene) -> str:
    """Récit du coup, jusqu'à la dernière bille empochée : bandes utiles, chocs, poches."""
    parts: list[tuple[int, str, str]] = []
    for seg in out.segments:
        who = name(scene, seg.ball)
        if seg.fate == "ball":
            parts.append((seg.ball, "ball", f"{who} touche {name(scene, seg.hit)}"))
        elif seg.fate == "pocket":
            parts.append((seg.ball, "pocket", f"{who} entre dans la poche {seg.pocket}"))
        elif seg.fate == "cushion":
            parts.append((seg.ball, "cushion", f"{who} rebondit sur la bande"))
    last = max((i + 1 for i, p in enumerate(parts) if p[1] == "pocket"), default=len(parts))
    parts = parts[:last]
    # une bande n'est utile au récit que si la même bille fait encore quelque chose ensuite
    kept = [p for i, p in enumerate(parts)
            if p[1] != "cushion" or any(q[0] == p[0] and q[1] != "cushion" for q in parts[i + 1:])]
    text = ", ".join(p[2] for p in kept) if kept else "aucun contact"
    return text[0].upper() + text[1:] + "."


def _ghost(out: Outcome, cue: int, start: Point) -> tuple[Point, int]:
    """Position de la blanche au premier choc avec une bille, et nombre de bandes avant."""
    banks = 0
    for seg in out.segments:
        if seg.ball == cue:
            if seg.fate == "ball":
                return seg.b, banks
            if seg.fate == "cushion":
                banks += 1
    return start, banks


def plan_from(table: Table, scene: Scene, a0: float, a1: float, power: float, max_banks: int = 1) -> Plan:
    """Coup au milieu de la plage [a0, a1] ; la marge est le déplacement du fantôme d'un bord à l'autre."""
    cue = scene.cue
    positions = [b.pos for b in scene.balls]

    def run(a):
        return simulate(table, positions, cue, (math.cos(a), math.sin(a)), power=power, max_bounces=max_banks)

    mid = (a0 + a1) / 2
    out = run(mid)
    ghost, banks = _ghost(out, cue, positions[cue])
    g0, g1 = _ghost(run(a0), cue, positions[cue])[0], _ghost(run(a1), cue, positions[cue])[0]
    return Plan(mid, out, ghost, math.dist(g0, g1) / 2, kinds_of(out, cue), describe(out, scene), banks)


def _sweep(table: Table, scene: Scene, angles, power: float, max_banks: int):
    """Simule chaque angle et regroupe les angles voisins de même résultat : [(a0, a1, outcome)]."""
    cue = scene.cue
    positions = [b.pos for b in scene.balls]
    runs: list[list] = []
    for a in angles:
        out = simulate(table, positions, cue, (math.cos(a), math.sin(a)), power=power, max_bounces=max_banks)
        sig = (out.first, tuple(sorted(out.pocketed.items())))
        if runs and runs[-1][0] == sig:
            runs[-1][2] = a
        else:
            runs.append([sig, a, a, out])
    return [(a0, a1, out) for _, a0, a1, out in runs]


def find_shots(scene: Scene, table: Table, legal: set[int] | None = None, power: float = DEFAULT_POWER,
               step_deg: float = STEP_DEG, kicks: bool = True, kick_step_deg: float = 0.05,
               max_banks: int = 1) -> list[Plan]:
    """Coups qui empochent au moins une bille sans empocher la blanche, du plus sûr au moins sûr.

    Avec `kicks`, on cherche aussi les coups où la blanche passe par une bande avant la première
    bille : seule façon d'atteindre une bille cachée.
    """
    if scene.cue is None:
        return []
    legal = legal_first(scene) if legal is None else legal
    cue = scene.cue
    positions = [b.pos for b in scene.balls]
    c = positions[cue]
    step = math.radians(step_deg)
    found: list[tuple[float, float, Outcome]] = []
    for target in sorted(legal):
        t = positions[target]
        d = math.dist(c, t)
        if d <= 2 * table.radius:
            continue
        center = math.atan2(t[1] - c[1], t[0] - c[0])
        n = int(math.asin(min(1.0, 2 * table.radius / d)) / step)
        found += [(a0, a1, o) for a0, a1, o in
                  _sweep(table, scene, (center + k * step for k in range(-n, n + 1)), power, max_banks)]
    if kicks:
        ks = math.radians(kick_step_deg)
        for a0, a1, o in _sweep(table, scene, (k * ks for k in range(int(2 * math.pi / ks))), power, max_banks + 1):
            first_cue = next((seg for seg in o.segments if seg.ball == cue), None)
            if first_cue is not None and first_cue.fate == "cushion":
                found.append((a0, a1, o))

    best: dict[tuple, Plan] = {}
    for a0, a1, out in found:
        if out.first not in legal or cue in out.pocketed or not out.pocketed:
            continue
        plan = plan_from(table, scene, a0, a1, power, max_banks + 1)
        if plan.outcome.first not in legal or cue in plan.outcome.pocketed or not plan.outcome.pocketed:
            continue
        key = (plan.outcome.first, tuple(sorted(plan.outcome.pocketed.items())), plan.kinds)
        if key not in best or plan.score > best[key].score:
            best[key] = plan
    return sorted(best.values(), key=lambda p: -p.score)
