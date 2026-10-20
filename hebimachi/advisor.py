"""Combine efficacité et défense pour classer les défausses."""
from __future__ import annotations

from dataclasses import dataclass

from .defense import Opponent, danger_table
from .efficiency import Analysis, Candidate, analyze
from .tiles import display_name, to_counts

SAFE_THRESHOLD = 3.0  # % de danger toléré quand on est iishanten face à un riichi


@dataclass
class Advice:
    mode: str              # Attaque / Push / Prudence / Défense
    explanation: str
    analysis: Analysis
    ranked: list[Candidate]
    danger: dict[int, tuple[float, str]]  # vide s'il n'y a pas de riichi
    warning: str = ""


def open_yaku(counts: list[int]) -> list[str]:
    """Yaku courants d'une main ouverte complète (main + appels).

    Volontairement simple : tanyao, yakuhai (tout brelan d'honneur, les vents comptés
    comme s'ils valaient un yaku), honitsu, chinitsu, toitoi. Sanshoku, ittsu et
    chanta ne sont pas encore reconnus.
    """
    yaku = []
    tiles = [t for t in range(34) if counts[t]]
    if all(t < 27 and t % 9 not in (0, 8) for t in tiles):
        yaku.append("tanyao")
    if any(counts[t] >= 3 for t in range(27, 34)):
        yaku.append("yakuhai")
    suits = {t // 9 for t in tiles if t < 27}
    if len(suits) == 1:
        yaku.append("chinitsu" if all(t < 27 for t in tiles) else "honitsu")
    if all(c in (0, 2, 3, 4) for c in counts) and sum(c == 2 for c in counts) == 1:
        yaku.append("toitoi")
    return yaku


def yaku_warning(hand: list[int], calls: list[int], best: Candidate) -> str:
    """Prévient quand une main ouverte en tenpai ne peut gagner sur aucune (ou certaines) attente(s)."""
    if not calls or best.shanten != 0 or not best.ukeire:
        return ""
    counts = to_counts(hand + calls)
    if best.discard is not None:
        counts[best.discard] -= 1
    good, bad = [], []
    for t in best.ukeire:
        counts[t] += 1
        (good if open_yaku(counts) else bad).append(t)
        counts[t] -= 1
    if not good:
        return ("Main ouverte SANS YAKU : vous ne pourrez pas gagner (tenpai formel seulement). "
                "Cherchez un yaku (tanyao, brelan d'honneur, une seule couleur…) ou défendez.")
    if bad:
        return (f"Sans yaku sur {', '.join(display_name(t) for t in bad)} : ne gagnez que sur "
                f"{', '.join(display_name(t) for t in good)}.")
    return ""


def advise(hand: list[int], visible: list[int], opponents: list[Opponent],
           melds: int = 0, draws_left: int | None = None, calls: list[int] | None = None) -> Advice:
    adv = _advise(hand, visible, opponents, melds, draws_left)
    adv.warning = yaku_warning(hand, calls or [], adv.ranked[0])
    return adv


def _advise(hand: list[int], visible: list[int], opponents: list[Opponent],
            melds: int, draws_left: int | None) -> Advice:
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
