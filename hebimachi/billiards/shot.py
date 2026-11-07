"""Verdict sur la visée affichée par le jeu.

Le jeu trace la trajectoire de la blanche jusqu'à la bille fantôme : on n'a pas à la calculer. On
simule la suite à partir du choc (voir `sim`) : la bille visée part sur la droite qui va du centre
du fantôme à son propre centre ; la blanche part sur la tangente, ce qui n'est juste que sans effet.
Combinaisons et carambolages sont donc vus aussi sur la visée en cours.

Le frottement entre billes (« throw ») dévie un peu la bille visée sur les coupes : non modélisé.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .search import Plan, describe, kinds_of, legal_first, name
from .sim import DEFAULT_POWER, Outcome, from_contact
from .table import Point, Table
from .vision import Scene

CONTACT_TOLERANCE = 1.3   # en diamètres : distance fantôme-bille pour dire qu'ils se touchent
THIN_CUT = 70             # en degrés : au-delà, coupe très fine, peu fiable
SCRATCH_MIN_CUT = 15      # en degrés : en deçà, la blanche garde trop peu de vitesse pour qu'on la suive


@dataclass
class Advice:
    verdict: str                          # empochee, ratee, bande, pas_de_visee
    target: int | None = None             # indice de la bille visée dans scene.balls
    aim: Point | None = None              # direction d'arrivée de la blanche au fantôme
    outcome: Outcome | None = None
    cut: float | None = None              # angle de coupe, en degrés
    foul: str | None = None
    kinds: tuple[str, ...] = ()
    lines: list[str] = field(default_factory=list)


def _unit(v: Point) -> Point:
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n) if n else (0.0, 0.0)


def incoming(scene: Scene, table: Table) -> Point | None:
    """Direction de la blanche juste avant le choc, d'après les derniers pointillés."""
    g = scene.ghost
    near = sorted((d for d in scene.dots if math.dist(d, g) < table.radius * 8), key=lambda d: math.dist(d, g))
    if near:
        far = near[min(len(near), 4) - 1]
        return _unit((g[0] - far[0], g[1] - far[1]))
    if scene.cue is not None:
        c = scene.balls[scene.cue].pos
        return _unit((g[0] - c[0], g[1] - c[1]))
    return None


def _contact(scene: Scene, table: Table) -> int | None:
    best = None
    for i, b in enumerate(scene.balls):
        if i == scene.cue:
            continue
        d = math.dist(b.pos, scene.ghost)
        if d <= 2 * table.radius * CONTACT_TOLERANCE and (best is None or d < best[0]):
            best = (d, i)
    return best[1] if best else None


def analyze(scene: Scene, table: Table, power: float = DEFAULT_POWER, legal: set[int] | None = None) -> Advice:
    """Verdict sur la visée en cours. `legal` : billes autorisées en premier (None = pas de contrôle)."""
    if scene.ghost is None:
        return Advice("pas_de_visee", lines=["Pas de visée à l'écran (bille fantôme introuvable)."])
    aim = incoming(scene, table)
    target = _contact(scene, table)
    if target is None:
        return Advice("bande", aim=aim, lines=["La blanche ne touche aucune bille : elle finit sur la bande."])

    g = scene.ghost
    ball = scene.balls[target]
    u = _unit((ball.x - g[0], ball.y - g[1]))
    if aim is None:
        aim = u
    cue = scene.cue if scene.cue is not None else -1
    positions = [b.pos for b in scene.balls]
    if cue < 0:   # blanche non lue : on la place au fantôme, elle ne sert qu'à l'élan
        positions.append(g)
        cue = len(positions) - 1
    out = from_contact(table, positions, cue, target, g, aim, power=power)
    cos = max(-1.0, min(1.0, aim[0] * u[0] + aim[1] * u[1]))
    cut = math.degrees(math.acos(cos))

    scratch = cue in out.pocketed
    pocketed = [b for b in out.pocketed if b != cue]
    adv = Advice("empochee" if pocketed and not scratch else "ratee", target=target, aim=aim,
                 outcome=out, cut=cut, kinds=kinds_of(out, cue))
    adv.lines.append(f"Bille visée : {name(scene, target)} ({ball.label()}).")
    adv.lines.append(describe(out, scene))
    if not pocketed:
        adv.lines.append(_why_missed(out, scene, target))
    if cut > THIN_CUT:
        adv.lines.append(f"Coupe très fine ({cut:.0f}°) : la moindre erreur fait rater.")
    if scratch:
        msg = f"la blanche file dans la poche {out.pocketed[cue]} (faute)"
        adv.lines.append(f"Attention : sans effet, {msg}. Mettre du rétro ou du coulé."
                         if cut >= SCRATCH_MIN_CUT else f"Risque : {msg}, si elle garde de l'élan.")

    if legal is None and scene.next_color:
        legal = legal_first(scene)
    if legal is not None and target not in legal:
        adv.foul = f"Faute : la première bille touchée doit être la NEXT, pas {name(scene, target)}."
        adv.lines.insert(0, adv.foul)
    return adv


def _why_missed(out: Outcome, scene: Scene, target: int) -> str:
    seg = next((s for s in out.segments if s.ball == target), None)
    if seg is None:
        return "La bille visée bouge à peine."
    if seg.fate == "ball":
        return f"La bille visée est arrêtée par {name(scene, seg.hit)}."
    if seg.fate == "jaw":
        return "La bille visée heurte une mâchoire de poche."
    if seg.fate == "stop":
        return "La bille visée n'a pas assez d'élan pour aller jusqu'à une poche."
    return "La bille visée finit sur la bande, sans poche."


def aim_feedback(scene: Scene, plan: Plan, table: Table) -> str | None:
    """Écart entre le fantôme du jeu et celui du coup choisi, en consigne lisible."""
    if scene.ghost is None:
        return None
    g, p = scene.ghost, plan.ghost
    d = math.dist(g, p)
    if d > 4 * table.radius:
        return "Le fantôme du jeu est loin du coup choisi : visez d'abord " + name(scene, plan.outcome.first) + "."
    if d <= max(plan.margin_px, 0.5):
        return f"✔ Visée bonne (écart {d:.1f} px, marge {plan.margin_px:.1f} px). Tirez."
    dx, dy = p[0] - g[0], p[1] - g[1]
    where = []
    if abs(dx) >= 0.5:
        where.append(f"{abs(dx):.0f} px vers la {'droite' if dx > 0 else 'gauche'}")
    if abs(dy) >= 0.5:
        where.append(f"{abs(dy):.0f} px vers le {'bas' if dy > 0 else 'haut'}")
    return "Décalez le fantôme de " + " et ".join(where) + f" (marge {plan.margin_px:.1f} px)."
