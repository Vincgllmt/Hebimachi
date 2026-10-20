"""Estimation du danger d'une défausse face à un joueur en riichi.

Les taux sont des ordres de grandeur de deal-in contre un riichi en milieu de
partie, issus de statistiques publiques (parties Tenhou). Ce sont des
estimations, pas des valeurs exactes : elles servent à classer les tuiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tiles import NUM_KINDS

# Danger en % selon le rang de la tuile.
NON_SUJI = {1: 5.5, 2: 7.0, 3: 8.5, 4: 11.5, 5: 12.0, 6: 11.5, 7: 8.5, 8: 7.0, 9: 5.5}
SUJI = {1: 1.8, 2: 3.0, 3: 4.5, 7: 4.5, 8: 3.0, 9: 1.8}
HALF_SUJI_456 = 7.0
DOUBLE_SUJI_456 = 2.5
# Honneurs selon le nombre d'exemplaires visibles (y compris dans notre main).
HONOR = {0: 4.5, 1: 2.5, 2: 1.0, 3: 0.2, 4: 0.0}


@dataclass
class Opponent:
    name: str
    discards: list[int] = field(default_factory=list)
    riichi: bool = False

    @property
    def genbutsu(self) -> set[int]:
        return set(self.discards)


def tile_danger(tile: int, opp: Opponent, visible_all: list[int]) -> tuple[float, str]:
    """Danger (%) d'une tuile contre `opp`, avec sa justification.

    `visible_all` : tous les exemplaires vus, y compris dans notre main.
    """
    if tile in opp.genbutsu:
        return 0.0, "genbutsu"
    if tile >= 27:
        seen = min(visible_all[tile], 4)
        return HONOR[seen], f"honneur, {seen} vu(s)"

    base = tile - tile % 9
    rank = tile % 9 + 1

    def covered(side: int) -> bool:
        """Côté couvert : suji en genbutsu, ou kabe (voisin entièrement visible)."""
        suji = rank + 3 * side
        n1, n2 = rank + side, rank + 2 * side
        if 1 <= suji <= 9 and base + suji - 1 in opp.genbutsu:
            return True
        return any(1 <= n <= 9 and visible_all[base + n - 1] >= 4 for n in (n1, n2))

    if rank in (4, 5, 6):
        sides = covered(-1) + covered(+1)
        danger, why = [(NON_SUJI[rank], "non-suji"),
                       (HALF_SUJI_456, "demi-suji"),
                       (DOUBLE_SUJI_456, "double suji")][sides]
    else:
        side = +1 if rank <= 3 else -1
        if covered(side):
            danger, why = SUJI[rank], "suji/kabe"
        else:
            danger, why = NON_SUJI[rank], "non-suji"

    # Peu d'exemplaires restants : shanpon et tanki deviennent improbables.
    if visible_all[tile] >= 3:
        danger *= 0.6
        why += ", 3 vus"
    return danger, why


def danger_table(opponents: list[Opponent], visible_all: list[int]) -> dict[int, tuple[float, str]]:
    """Danger combiné contre tous les joueurs en riichi : 1 - Π(1 - d_i)."""
    threats = [o for o in opponents if o.riichi]
    table: dict[int, tuple[float, str]] = {}
    for t in range(NUM_KINDS):
        safe = 1.0
        reasons = []
        for o in threats:
            d, why = tile_danger(t, o, visible_all)
            safe *= 1 - d / 100
            reasons.append(f"{o.name}: {why}")
        table[t] = ((1 - safe) * 100, " | ".join(reasons))
    return table
