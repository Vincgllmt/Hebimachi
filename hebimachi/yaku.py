"""Détection des yaku d'une main complète.

Couvre les yaku courants : riichi, menzen tsumo, tanyao, yakuhai, pinfu, iipeiko,
sanshoku, ittsu, toitoi, chanta/junchan, honitsu/chinitsu, chiitoitsu, kokushi.
Ne compte pas les han : il s'agit de savoir si l'on PEUT gagner.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .shanten import shanten_kokushi
from .tiles import to_counts

Set = tuple[str, int]  # ("seq", première tuile) ou ("tri", tuile)


@dataclass
class Context:
    round_wind: int = 27          # 27 = Est … 30 = Nord
    seat_wind: int = 28
    riichi: bool = False          # nous avons déclaré riichi
    tsumo: bool = False           # victoire sur notre pioche (sinon ron)
    calls: list[list[int]] = field(default_factory=list)  # nos appels, un groupe par appel

    @property
    def closed(self) -> bool:
        return not self.calls

    def value_honors(self) -> set[int]:
        return {31, 32, 33, self.round_wind, self.seat_wind}


def call_set(group: list[int]) -> Set:
    if len(set(group)) == 1:
        return ("tri", group[0])  # pon ou kan
    return ("seq", min(group))


def _sets(c: list[int], need: int):
    i = next((k for k in range(34) if c[k]), None)
    if i is None:
        if need == 0:
            yield []
        return
    if need == 0:
        return
    if c[i] >= 3:
        c[i] -= 3
        for rest in _sets(c, need - 1):
            yield [("tri", i)] + rest
        c[i] += 3
    if i < 27 and i % 9 <= 6 and c[i + 1] and c[i + 2]:
        for k in (i, i + 1, i + 2):
            c[k] -= 1
        for rest in _sets(c, need - 1):
            yield [("seq", i)] + rest
        for k in (i, i + 1, i + 2):
            c[k] += 1


def decompositions(counts: list[int], n_sets: int) -> list[tuple[int, list[Set]]]:
    """Toutes les façons de lire la partie cachée en `n_sets` groupes + une paire."""
    c = list(counts)
    out = []
    for p in range(34):
        if c[p] >= 2:
            c[p] -= 2
            out += [(p, sets) for sets in _sets(c, n_sets)]
            c[p] += 2
    return out


def _tiles_of(s: Set) -> list[int]:
    kind, t = s
    return [t, t + 1, t + 2] if kind == "seq" else [t]


def _is_terminal_or_honor(t: int) -> bool:
    return t >= 27 or t % 9 in (0, 8)


def find_yaku(hand: list[int], win_tile: int, ctx: Context) -> list[str]:
    """Yaku d'une main complète. `hand` contient la tuile gagnante, sans les appels."""
    counts = to_counts(hand)
    all_tiles = hand + [t for g in ctx.calls for t in g]
    base = []
    if ctx.riichi:
        base.append("riichi")
    if ctx.closed and ctx.tsumo:
        base.append("menzen tsumo")
    suits = {t // 9 for t in all_tiles if t < 27}
    if len(suits) == 1:
        base.append("chinitsu" if all(t < 27 for t in all_tiles) else "honitsu")
    if all(not _is_terminal_or_honor(t) for t in all_tiles):
        base.append("tanyao")

    if ctx.closed and shanten_kokushi(counts) == -1:
        return ["kokushi musou (yakuman)"]

    best: list[str] = []
    call_sets = [call_set(g) for g in ctx.calls]
    for pair, sets in decompositions(counts, 4 - len(ctx.calls)):
        yaku = _standard_yaku(pair, sets, call_sets, win_tile, ctx)
        if len(yaku) > len(best):
            best = yaku
    if ctx.closed and sum(1 for x in counts if x == 2) == 7:
        chiitoi = ["chiitoitsu"]
        if len(chiitoi) > len(best):
            best = chiitoi
    return base + best


def _standard_yaku(pair: int, closed_sets: list[Set], call_sets: list[Set],
                   win: int, ctx: Context) -> list[str]:
    sets = closed_sets + call_sets
    seqs = [t for k, t in sets if k == "seq"]
    tris = [t for k, t in sets if k == "tri"]
    values = ctx.value_honors()
    yaku = []
    for t in tris:
        if t in (31, 32, 33):
            yaku.append("yakuhai")
        if t == ctx.round_wind:
            yaku.append("yakuhai (vent de table)")
        if t == ctx.seat_wind:
            yaku.append("yakuhai (vent de siège)")
    if ctx.closed and not tris and pair not in values:
        # pinfu : la tuile gagnante doit compléter une attente ryanmen
        if any((win == s and s % 9 != 6) or (win == s + 2 and s % 9 != 0)
               for k, s in closed_sets if k == "seq"):
            yaku.append("pinfu")
    if ctx.closed and any(seqs.count(s) >= 2 for s in set(seqs)):
        yaku.append("iipeiko")
    if any(all(n + 9 * suit in seqs for suit in range(3)) for n in range(7)):
        yaku.append("sanshoku")
    if any(all(9 * suit + n in seqs for n in (0, 3, 6)) for suit in range(3)):
        yaku.append("ittsu")
    if len(tris) == 4:
        yaku.append("toitoi")
    groups = [_tiles_of(s) for s in sets] + [[pair]]
    if seqs and all(any(_is_terminal_or_honor(t) for t in g) for g in groups):
        flat = [t for g in groups for t in g]
        yaku.append("junchan" if all(t < 27 for t in flat) else "chanta")
    return yaku
