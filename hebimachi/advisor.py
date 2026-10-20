"""Combine efficacité et défense pour classer les défausses."""
from __future__ import annotations

from dataclasses import dataclass

from .defense import Opponent, danger_table
from .efficiency import Analysis, Candidate, analyze

SAFE_THRESHOLD = 3.0  # % de danger toléré quand on est iishanten face à un riichi


@dataclass
class Advice:
    mode: str              # Attaque / Push / Prudence / Défense
    explanation: str
    analysis: Analysis
    ranked: list[Candidate]
    danger: dict[int, tuple[float, str]]  # vide s'il n'y a pas de riichi


def advise(hand: list[int], visible: list[int], opponents: list[Opponent],
           melds: int = 0, draws_left: int | None = None) -> Advice:
    analysis = analyze(hand, visible, melds, draws_left)
    visible_all = [visible[t] + hand.count(t) for t in range(34)]
    threats = [o for o in opponents if o.riichi]
    cands = analysis.candidates

    if not threats or cands[0].discard is None:
        danger = danger_table(opponents, visible_all) if threats else {}
        return Advice("Attaque", "Aucun riichi adverse : maximiser l'efficacité.",
                      analysis, cands, danger)

    danger = danger_table(opponents, visible_all)
    best_s = min(c.shanten for c in cands)

    def d(c: Candidate) -> float:
        return danger[c.discard][0]

    if best_s == 0 and max(c.total for c in cands if c.shanten == 0) >= 4:
        mode = "Push"
        expl = ("Tenpai avec une attente correcte : on pousse en pondérant les ukeire "
                "par la probabilité de ne pas être pris.")
        ranked = sorted(cands, key=lambda c: (c.shanten, -c.total * (1 - d(c) / 100)))
    elif best_s <= 1:
        mode = "Prudence"
        expl = (f"Iishanten/tenpai faible face au riichi : on garde l'avancement seulement "
                f"avec une tuile sûre (danger ≤ {SAFE_THRESHOLD:g} %), sinon on défend.")

        def key(c: Candidate):
            ok = c.shanten == best_s and d(c) <= SAFE_THRESHOLD
            return (0, c.shanten, -c.total) if ok else (1, d(c), c.shanten, -c.total)
        ranked = sorted(cands, key=key)
    else:
        mode = "Défense"
        expl = "Main trop lointaine face au riichi : on joue les tuiles les plus sûres."
        ranked = sorted(cands, key=lambda c: (d(c), c.shanten, -c.total))

    return Advice(mode, expl, analysis, ranked, danger)
