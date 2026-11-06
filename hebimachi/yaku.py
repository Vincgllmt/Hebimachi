"""Détection des yaku et calcul de la valeur d'une main complète (han, fu, points).

`evaluate` essaie toutes les lectures de la main, et pour chacune toutes les places
possibles de la tuile gagnante, puis garde la plus chère. `find_yaku` n'en donne que
les noms, pour savoir si l'on PEUT gagner.

Non comptés car inconnus au moment de décider : ura dora, ippatsu, haitei, rinshan,
chankan, double riichi. Une valeur annoncée est donc un minimum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .shanten import shanten_kokushi
from .tiles import to_counts

Set = tuple[str, int]  # ("seq", première tuile) ou ("tri", tuile)
Yaku = tuple[str, int]  # (nom, han)

YAKUMAN = 13
DRAGONS = (31, 32, 33)
WINDS = (27, 28, 29, 30)
GREEN = {19, 20, 21, 23, 25, 32}  # 2 3 4 6 8 sou et Vert
LIMITS = ((13, "yakuman", 8000), (11, "sanbaiman", 6000), (8, "baiman", 4000),
          (6, "haneman", 3000), (5, "mangan", 2000))


@dataclass
class Context:
    round_wind: int = 27          # 27 = Est … 30 = Nord
    seat_wind: int = 28
    riichi: bool = False          # nous avons déclaré riichi
    tsumo: bool = False           # victoire sur notre pioche (sinon ron)
    calls: list[list[int]] = field(default_factory=list)  # nos appels, un groupe par appel
    dora: list[int] = field(default_factory=list)  # tuiles dora (pas les indicateurs)
    aka: list[int] = field(default_factory=list)   # tuiles dont nous tenons le cinq rouge
    goal: object = None           # objectif de victoire (goals.Goal), None = gagner tout court

    @property
    def closed(self) -> bool:
        return not self.calls

    def value_honors(self) -> set[int]:
        return {31, 32, 33, self.round_wind, self.seat_wind}


@dataclass
class Score:
    yaku: list[Yaku]
    dora: int       # dora et cinq rouges, comptés seulement s'il y a un yaku
    han: int
    fu: int
    limit: str      # "" sous le mangan, sinon mangan … yakuman
    base: int       # points de base : le mangan commence à 2000
    ron: int        # points gagnés sur ron (non-donneur)
    ron_dealer: int

    @property
    def names(self) -> list[str]:
        return [n for n, _ in self.yaku]

    def label(self) -> str:
        """« mangan (5 han) » ou « 3 han 40 fu »."""
        return f"{self.limit} ({self.han} han)" if self.limit else f"{self.han} han {self.fu} fu"


def next_dora(indicator: int) -> int:
    """Tuile dora désignée par un indicateur."""
    if indicator < 27:
        return indicator - indicator % 9 + (indicator % 9 + 1) % 9
    if indicator < 31:
        return 27 + (indicator - 27 + 1) % 4
    return 31 + (indicator - 31 + 1) % 3


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


def limit_of(han: int, fu: int) -> tuple[str, int]:
    """(nom de la limite, points de base)."""
    for threshold, name, base in LIMITS:
        if han >= threshold:
            return name, base
    base = fu * 2 ** (han + 2)
    return ("mangan", 2000) if base >= 2000 else ("", base)


def _round100(x: int) -> int:
    return -(-x // 100) * 100


def _score(yaku: list[Yaku], dora: int, fu: int) -> Score:
    han = sum(h for _, h in yaku)
    if han >= YAKUMAN:  # les yakuman ne se cumulent pas avec les dora
        dora = 0
    total = han + dora
    limit, base = limit_of(total, fu)
    return Score(yaku, dora, total, fu, limit, base, _round100(base * 4), _round100(base * 6))


def _hand_yaku(all_tiles: list[int], ctx: Context) -> list[Yaku]:
    """Yaku qui ne dépendent pas de la lecture de la main."""
    closed = ctx.closed
    yaku: list[Yaku] = []
    if ctx.riichi:
        yaku.append(("riichi", 1))
    if closed and ctx.tsumo:
        yaku.append(("menzen tsumo", 1))
    suits = {t // 9 for t in all_tiles if t < 27}
    if len(suits) == 1:
        if all(t < 27 for t in all_tiles):
            yaku.append(("chinitsu", 6 if closed else 5))
        else:
            yaku.append(("honitsu", 3 if closed else 2))
    if all(not _is_terminal_or_honor(t) for t in all_tiles):
        yaku.append(("tanyao", 1))
    return yaku


def _yakuman(all_tiles: list[int], counts: list[int], ctx: Context) -> list[Yaku]:
    """Yakuman de forme : valables quelle que soit la lecture de la main."""
    yaku: list[Yaku] = []
    if all(t >= 27 for t in all_tiles):
        yaku.append(("tsuuiisou (yakuman)", YAKUMAN))
    if all(t < 27 and t % 9 in (0, 8) for t in all_tiles):
        yaku.append(("chinroutou (yakuman)", YAKUMAN))
    if all(t in GREEN for t in all_tiles):
        yaku.append(("ryuuiisou (yakuman)", YAKUMAN))
    if ctx.closed and all(t < 27 for t in all_tiles) and len({t // 9 for t in all_tiles}) == 1:
        base = all_tiles[0] // 9 * 9
        need = (3, 1, 1, 1, 1, 1, 1, 1, 3)
        if all(counts[base + i] >= need[i] for i in range(9)):
            yaku.append(("chuuren poutou (yakuman)", YAKUMAN))
    return yaku


def _keep_yakuman(yaku: list[Yaku]) -> list[Yaku]:
    """Un yakuman efface les yaku ordinaires."""
    return [y for y in yaku if y[1] >= YAKUMAN] or yaku


def evaluate(hand: list[int], win_tile: int, ctx: Context) -> Score | None:
    """Valeur d'une main complète (`hand` contient la tuile gagnante, sans les appels).

    Renvoie None si la main n'est pas complète ou n'a aucun yaku.
    """
    counts = to_counts(hand)
    all_tiles = hand + [t for g in ctx.calls for t in g]
    all_counts = to_counts(all_tiles)
    dora = sum(all_counts[t] for t in ctx.dora) + sum(1 for t in ctx.aka if all_counts[t])

    if ctx.closed and shanten_kokushi(counts) == -1:
        return _score([("kokushi musou (yakuman)", YAKUMAN)], 0, 0)

    base = _hand_yaku(all_tiles, ctx)
    shape = _yakuman(all_tiles, counts, ctx)
    best: Score | None = None

    def consider(yaku: list[Yaku], fu: int) -> None:
        nonlocal best
        if not yaku:
            return
        score = _score(_keep_yakuman(yaku), dora, fu)
        if best is None or (score.base, score.han, score.fu) > (best.base, best.han, best.fu):
            best = score

    for pair, sets in decompositions(counts, 4 - len(ctx.calls)):
        for slot in _win_slots(pair, sets, win_tile):
            yaku, fu = _reading(pair, sets, slot, win_tile, ctx)
            consider(base + yaku + shape, fu)
    if ctx.closed and sum(1 for x in counts if x == 2) == 7:
        pairs = base + [("chiitoitsu", 2)] + shape
        if all(_is_terminal_or_honor(t) for t in all_tiles):
            pairs.append(("honroutou", 2))
        consider(pairs, 25)
    return best


def find_yaku(hand: list[int], win_tile: int, ctx: Context) -> list[str]:
    """Noms des yaku d'une main complète. `hand` contient la tuile gagnante, sans les appels."""
    score = evaluate(hand, win_tile, ctx)
    return score.names if score else []


def _win_slots(pair: int, sets: list[Set], win: int) -> list[int]:
    """Places possibles de la tuile gagnante : -1 = la paire, sinon l'indice du groupe."""
    slots = [i for i, s in enumerate(sets) if win in _tiles_of(s)]
    return slots + [-1] if pair == win else slots


def _is_ryanmen(start: int, win: int) -> bool:
    return (win == start and start % 9 != 6) or (win == start + 2 and start % 9 != 0)


def _reading(pair: int, closed_sets: list[Set], slot: int, win: int,
             ctx: Context) -> tuple[list[Yaku], int]:
    """Yaku et fu d'une lecture : `slot` dit quel groupe la tuile gagnante complète."""
    closed = ctx.closed
    sets = closed_sets + [call_set(g) for g in ctx.calls]
    seqs = [t for k, t in sets if k == "seq"]
    tris = [t for k, t in sets if k == "tri"]
    kans = sum(1 for g in ctx.calls if len(g) == 4)
    values = ctx.value_honors()
    # Un brelan complété par un ron compte comme ouvert (fu et sanankou).
    ron_tri = not ctx.tsumo and slot >= 0 and closed_sets[slot][0] == "tri"
    concealed = sum(1 for k, _ in closed_sets if k == "tri") - ron_tri
    yaku: list[Yaku] = []

    def add(name: str, closed_han: int, open_han: int | None = None) -> None:
        yaku.append((name, closed_han if closed or open_han is None else open_han))

    for t in tris:
        if t in DRAGONS:
            add("yakuhai", 1)
        if t == ctx.round_wind:
            add("yakuhai (vent de table)", 1)
        if t == ctx.seat_wind:
            add("yakuhai (vent de siège)", 1)
    pinfu = (closed and not tris and pair not in values and slot >= 0
             and _is_ryanmen(closed_sets[slot][1], win))
    if pinfu:
        add("pinfu", 1)
    if closed:
        doubles = sum(seqs.count(s) // 2 for s in set(seqs))
        if doubles >= 2:
            add("ryanpeikou", 3)
        elif doubles == 1:
            add("iipeiko", 1)
    if any(all(n + 9 * suit in seqs for suit in range(3)) for n in range(7)):
        add("sanshoku", 2, 1)
    if any(all(n + 9 * suit in tris for suit in range(3)) for n in range(9)):
        add("sanshoku doukou", 2)
    if any(all(9 * suit + n in seqs for n in (0, 3, 6)) for suit in range(3)):
        add("ittsu", 2, 1)
    if len(tris) == 4:
        add("toitoi", 2)
    if concealed == 3:
        add("sanankou", 2)
    if kans == 3:
        add("sankantsu", 2)
    groups = [_tiles_of(s) for s in sets] + [[pair]]
    flat = [t for g in groups for t in g]
    if all(_is_terminal_or_honor(t) for t in flat):
        if not seqs:
            add("honroutou", 2)
    if seqs and all(any(_is_terminal_or_honor(t) for t in g) for g in groups):
        if all(t < 27 for t in flat):
            add("junchan", 3, 2)
        else:
            add("chanta", 2, 1)
    dragons = sum(1 for t in tris if t in DRAGONS)
    if dragons == 2 and pair in DRAGONS:
        add("shousangen", 2)
    # Yakuman liés à la lecture.
    if dragons == 3:
        add("daisangen (yakuman)", YAKUMAN)
    if concealed == 4:
        add("suuankou (yakuman)", YAKUMAN)
    if kans == 4:
        add("suukantsu (yakuman)", YAKUMAN)
    winds = sum(1 for t in tris if t in WINDS)
    if winds == 4:
        add("daisuushii (yakuman)", YAKUMAN)
    elif winds == 3 and pair in WINDS:
        add("shousuushii (yakuman)", YAKUMAN)
    return yaku, _fu(pair, closed_sets, slot, win, ctx, pinfu, ron_tri)


def _fu(pair: int, closed_sets: list[Set], slot: int, win: int, ctx: Context,
        pinfu: bool, ron_tri: bool) -> int:
    closed = ctx.closed
    if pinfu:
        return 20 if ctx.tsumo else 30
    fu = 20 + (10 if closed and not ctx.tsumo else 0) + (2 if ctx.tsumo else 0)
    if pair in DRAGONS:
        fu += 2
    fu += 2 * (pair == ctx.round_wind) + 2 * (pair == ctx.seat_wind)
    for i, (kind, t) in enumerate(closed_sets):
        if kind == "tri":
            is_open = ron_tri and i == slot
            fu += (4 if _is_terminal_or_honor(t) else 2) * (1 if is_open else 2)
    for g in ctx.calls:
        kind, t = call_set(g)
        if kind == "tri":
            fu += (4 if _is_terminal_or_honor(t) else 2) * (4 if len(g) == 4 else 1)
    if slot == -1 or (closed_sets[slot][0] == "seq" and not _is_ryanmen(closed_sets[slot][1], win)):
        fu += 2  # attente tanki, kanchan ou penchan
    if fu == 20 and not closed:
        return 30  # main ouverte sans aucun fu : forfait de 30
    return -(-fu // 10) * 10
