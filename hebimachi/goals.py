"""Objectifs de victoire : gagner en mangan, en haneman, ou avec un full straight (ittsu).

Viser une valeur, ce n'est plus seulement minimiser le shanten : une main à 2 han rapide ne
sert à rien. On énumère donc des « plans » (combinaisons de yaku compatibles), on ne garde
que ceux qui atteignent la valeur voulue, et chaque défausse est jugée sur son plan le plus
proche : un shanten calculé uniquement avec les tuiles que ce plan peut utiliser.

Le plan estime les han sans compter sur la chance : ni menzen tsumo, ni pinfu, ni ura dora.
La valeur réelle d'une victoire, elle, est calculée exactement par `yaku.evaluate`.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .shanten import shanten_chiitoi, shanten_regular
from .tiles import NUM_KINDS, to_counts
from .yaku import Context, Score, Set, call_set

MANGAN_HAN, HANEMAN_HAN = 5, 6
MANGAN_BASE, HANEMAN_BASE = 2000, 3000
TERMINAL_RUNS = (0, 3, 6)


@dataclass(frozen=True)
class Goal:
    limit: str = ""        # "", "mangan" ou "haneman" : valeur minimale de la victoire
    ittsu: bool = False    # la victoire doit contenir un full straight

    @property
    def active(self) -> bool:
        return bool(self.limit or self.ittsu)

    @property
    def min_han(self) -> int:
        return {"mangan": MANGAN_HAN, "haneman": HANEMAN_HAN}.get(self.limit, 1)

    def label(self) -> str:
        parts = [self.limit.capitalize()] if self.limit else []
        if self.ittsu:
            parts.append("full straight")
        return " + ".join(parts)

    def satisfied(self, score: Score) -> bool:
        base = {"mangan": MANGAN_BASE, "haneman": HANEMAN_BASE}.get(self.limit, 0)
        return score.base >= base and (not self.ittsu or "ittsu" in score.names)


@dataclass(frozen=True)
class Plan:
    name: str
    han: int                          # hors dora
    allowed: tuple[bool, ...]         # tuiles utilisables
    runs: tuple[int, ...] = ()        # suites obligatoires (première tuile)
    triplets: tuple[int, ...] = ()    # brelans obligatoires
    form: str = "standard"            # standard, toitoi ou chiitoi


@dataclass
class GoalOption:
    """Où en est la main vis-à-vis de l'objectif, par son meilleur plan."""
    shanten: int
    plan: str
    han: int                          # estimation du plan, dora comprises
    ukeire: dict[int, int]            # tuiles qui rapprochent de l'objectif

    @property
    def total(self) -> int:
        return sum(self.ukeire.values())


def _allow(pred) -> tuple[bool, ...]:
    return tuple(pred(t) for t in range(NUM_KINDS))


def _tile_filters(closed: bool) -> list[tuple[str, int, tuple[bool, ...], int | None]]:
    """(nom, han, tuiles permises, couleur imposée) ; le nom vide n'impose rien."""
    filters = [("", 0, _allow(lambda t: True), None),
               ("tanyao", 1, _allow(lambda t: t < 27 and t % 9 not in (0, 8)), None)]
    for suit, letter in enumerate("mps"):
        filters.append((f"honitsu {letter}", 3 if closed else 2,
                        _allow(lambda t, s=suit: t >= 27 or t // 9 == s), suit))
        filters.append((f"chinitsu {letter}", 6 if closed else 5,
                        _allow(lambda t, s=suit: t < 27 and t // 9 == s), suit))
    return filters


def _forms(closed: bool, goal: Goal) -> list[tuple[str, int, tuple[int, ...], str]]:
    """(nom, han, suites imposées, forme)."""
    ittsu = [(f"full straight {letter}", 2 if closed else 1,
              tuple(9 * suit + n for n in TERMINAL_RUNS), "standard")
             for suit, letter in enumerate("mps")]
    if goal.ittsu:
        return ittsu
    forms = [("", 0, (), "standard"), ("toitoi", 2, (), "toitoi")] + ittsu
    forms += [(f"sanshoku {n + 1}{n + 2}{n + 3}", 2 if closed else 1,
               tuple(n + 9 * suit for suit in range(3)), "standard") for n in range(7)]
    if closed:
        forms.append(("chiitoitsu", 2, (), "chiitoi"))
    return forms


def _compatible(suit: int | None, allowed: tuple[bool, ...], runs: tuple[int, ...]) -> bool:
    tiles = [t + i for t in runs for i in range(3)]
    return all(allowed[t] for t in tiles) and (suit is None or all(t // 9 == suit for t in tiles))


def build_plans(goal: Goal, ctx: Context, counts: list[int]) -> list[Plan]:
    """Plans réalistes pour cette main : yakuhai seulement sur les honneurs déjà en paire."""
    closed = ctx.closed
    sets = [call_set(g) for g in ctx.calls]
    worth = {t: 1 + (t == ctx.round_wind == ctx.seat_wind) for t in ctx.value_honors()}
    held = [t for t in sorted(worth) if counts[t] >= 2 or ("tri", t) in sets]
    plans = []
    for fname, fhan, allowed, suit in _tile_filters(closed):
        for name, han, runs, form in _forms(closed, goal):
            if not _compatible(suit, allowed, runs):
                continue
            usable = [t for t in held if allowed[t]] if form != "chiitoi" else []
            for size in range(len(usable) + 1):
                for honors in combinations(usable, size):
                    total = fhan + han + sum(worth[t] for t in honors)
                    if closed:
                        total += 1  # riichi
                    elif total == 0:
                        continue    # main ouverte sans yaku : elle ne peut pas gagner
                    label = [p for p in (fname, name) if p] + (["yakuhai"] if honors else [])
                    plans.append(Plan(" + ".join(label + (["riichi"] if closed else [])),
                                      total, allowed, runs, honors, form))
    return plans


def _dora_in(plan: Plan, counts: list[int], ctx: Context) -> int:
    """Dora et cinq rouges que ce plan garde : ceux qu'il ne peut pas utiliser seront jetés."""
    called = to_counts([t for g in ctx.calls for t in g])
    n = sum(called[t] + (counts[t] if plan.allowed[t] else 0) for t in ctx.dora)
    return n + sum(1 for t in ctx.aka if called[t] or (counts[t] and plan.allowed[t]))


def _toitoi(c: list[int], remaining: list[int], cap: int) -> int | None:
    """Shanten vers `cap` brelans + une paire, sur les tuiles encore disponibles."""
    kinds = sorted((n for n in range(NUM_KINDS) if c[n] + remaining[n] >= 2),
                   key=lambda n: -min(c[n], 3))
    triples = [n for n in kinds if c[n] + remaining[n] >= 3][:cap]
    pair = next((n for n in kinds if n not in triples), None)
    if len(triples) < cap or pair is None:
        return None
    return sum(3 - min(c[n], 3) for n in triples) + 2 - min(c[pair], 2) - 1


def plan_shanten(plan: Plan, counts: list[int], sets: list[Set], remaining: list[int]) -> int | None:
    """Shanten vers ce plan, ou None s'il est devenu impossible."""
    c = [n if ok else 0 for n, ok in zip(counts, plan.allowed)]
    melds = len(sets)
    if plan.form == "chiitoi":
        return shanten_chiitoi(c)
    if plan.form == "toitoi":
        if any(k != "tri" for k, _ in sets):
            return None
        return _toitoi(c, remaining, 4 - melds)
    called = list(sets)
    cost = extra = 0
    for t in plan.triplets:
        if ("tri", t) in called:
            called.remove(("tri", t))
            continue
        have = min(c[t], 3)
        if 3 - have > remaining[t]:
            return None
        c[t] -= have
        cost += 3 - have
        extra += 1
    for start in plan.runs:
        if ("seq", start) in called:
            called.remove(("seq", start))
            continue
        for t in (start, start + 1, start + 2):
            if c[t]:
                c[t] -= 1
            elif remaining[t]:
                cost += 1
            else:
                return None
        extra += 1
    if melds + extra > 4:
        return None
    return cost + shanten_regular(c, melds + extra)


def _closest(plans: list[Plan], counts: list[int], ctx: Context, remaining: list[int],
             goal: Goal) -> tuple[int, list[Plan]] | None:
    """Plus petit shanten parmi les plans qui atteignent l'objectif, et les plans à égalité."""
    sets = [call_set(g) for g in ctx.calls]
    called = {t for g in ctx.calls for t in g}
    best, tied = None, []
    for plan in plans:
        if plan.han + _dora_in(plan, counts, ctx) < goal.min_han:
            continue
        if not all(plan.allowed[t] for t in called):
            continue
        s = plan_shanten(plan, counts, sets, remaining)
        if s is None or (best is not None and s > best):
            continue
        if best is None or s < best:
            best, tied = s, []
        tied.append(plan)
    return None if best is None else (best, tied)


def _option(found: tuple[int, list[Plan]], counts: list[int], ctx: Context,
            remaining: list[int]) -> GoalOption:
    s, plans = found
    sets = [call_set(g) for g in ctx.calls]
    accepted: dict[int, int] = {}
    for t in range(NUM_KINDS):
        if not remaining[t] or counts[t] >= 4:
            continue
        counts[t] += 1
        for plan in plans:
            after = plan_shanten(plan, counts, sets, remaining)
            if after is not None and after < s:
                accepted[t] = remaining[t]
                break
        counts[t] -= 1
    top = max(plans, key=lambda p: p.han + _dora_in(p, counts, ctx))
    return GoalOption(s, top.name, top.han + _dora_in(top, counts, ctx), accepted)


def goal_status(hand: list[int], ctx: Context, remaining: list[int]) -> GoalOption | None:
    """Distance de la main (sans défausse) à l'objectif ; None s'il est hors d'atteinte."""
    counts = to_counts(hand)
    found = _closest(build_plans(ctx.goal, ctx, counts), counts, ctx, remaining, ctx.goal)
    return _option(found, counts, ctx, remaining) if found else None


def rank_discards(hand: list[int], ctx: Context, remaining: list[int],
                  forbidden: set[int] = frozenset()) -> dict[int, GoalOption]:
    """Option de chaque défausse qui garde l'objectif possible.

    Les tuiles acceptées ne sont calculées que pour les défausses les plus proches du but :
    les autres sont de toute façon classées derrière.
    """
    counts = to_counts(hand)
    plans = build_plans(ctx.goal, ctx, counts)
    found = {}
    for d in sorted(set(hand) - set(forbidden)):
        counts[d] -= 1
        result = _closest(plans, counts, ctx, remaining, ctx.goal)
        counts[d] += 1
        if result is not None:
            found[d] = result
    if not found:
        return {}
    closest = min(s for s, _ in found.values())
    options = {}
    for d, result in found.items():
        counts[d] -= 1
        if result[0] == closest:
            options[d] = _option(result, counts, ctx, remaining)
        else:
            top = result[1][0]
            options[d] = GoalOption(result[0], top.name, top.han + _dora_in(top, counts, ctx), {})
        counts[d] += 1
    return options
