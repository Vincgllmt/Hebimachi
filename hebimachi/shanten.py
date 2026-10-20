"""Calcul du shanten (nombre de tuiles manquantes avant tenpai).

-1 = main complète, 0 = tenpai, 1 = iishanten, etc.

La forme standard est décomposée couleur par couleur (programmation dynamique
mémoïsée) en ensembles (mentsu), formes d'attente (taatsu) et paire, puis les
couleurs sont combinées. On garde uniquement les décompositions Pareto-optimales.
"""
from __future__ import annotations

from functools import lru_cache

from .tiles import TERMINALS_HONORS

Option = tuple[int, int, int]  # (mentsu, taatsu, paire)


def _prune(options: set[Option]) -> tuple[Option, ...]:
    """Supprime les décompositions dominées (à paire égale)."""
    return tuple(
        o for o in options
        if not any(q != o and q[2] == o[2] and q[0] >= o[0] and q[1] >= o[1] for q in options)
    )


@lru_cache(maxsize=None)
def _decompose(counts: tuple[int, ...], honors: bool) -> tuple[Option, ...]:
    n = len(counts)
    i = 0
    while i < n and counts[i] == 0:
        i += 1
    if i == n:
        return ((0, 0, 0),)

    results: set[Option] = set()

    def add(removed: tuple[int, ...], dm: int, dt: int, dp: int) -> None:
        rest = list(counts)
        for j in removed:
            rest[j] -= 1
        for m, t, p in _decompose(tuple(rest), honors):
            if p + dp <= 1:
                results.add((min(m + dm, 4), min(t + dt, 4), p + dp))

    c = counts
    if c[i] >= 3:
        add((i, i, i), 1, 0, 0)
    if not honors and i + 2 < n and c[i + 1] and c[i + 2]:
        add((i, i + 1, i + 2), 1, 0, 0)
    if c[i] >= 2:
        add((i, i), 0, 0, 1)  # paire
        add((i, i), 0, 1, 0)  # paire utilisée comme attente (shanpon)
    if not honors:
        if i + 1 < n and c[i + 1]:
            add((i, i + 1), 0, 1, 0)  # ryanmen / penchan
        if i + 2 < n and c[i + 2]:
            add((i, i + 2), 0, 1, 0)  # kanchan
    add((i,), 0, 0, 0)  # tuile isolée
    return _prune(results)


def _combine(a: tuple[Option, ...], b: tuple[Option, ...]) -> tuple[Option, ...]:
    res: set[Option] = set()
    for m1, t1, p1 in a:
        for m2, t2, p2 in b:
            if p1 + p2 <= 1:
                res.add((min(m1 + m2, 4), min(t1 + t2, 4), p1 + p2))
    return _prune(res)


def shanten_regular(counts: tuple[int, ...] | list[int], melds: int = 0) -> int:
    c = tuple(counts)
    opts = _decompose(c[0:9], False)
    opts = _combine(opts, _decompose(c[9:18], False))
    opts = _combine(opts, _decompose(c[18:27], False))
    opts = _combine(opts, _decompose(c[27:34], True))
    cap = 4 - melds
    best = 8
    for m, t, p in opts:
        mm = min(m, cap)
        tt = min(t, cap - mm)
        best = min(best, 2 * cap - 2 * mm - tt - p)
    return best


def shanten_chiitoi(counts: tuple[int, ...] | list[int]) -> int:
    pairs = sum(1 for x in counts if x >= 2)
    kinds = sum(1 for x in counts if x >= 1)
    return 6 - pairs + max(0, 7 - kinds)


def shanten_kokushi(counts: tuple[int, ...] | list[int]) -> int:
    kinds = sum(1 for i in TERMINALS_HONORS if counts[i])
    pair = any(counts[i] >= 2 for i in TERMINALS_HONORS)
    return 13 - kinds - (1 if pair else 0)


@lru_cache(maxsize=200_000)
def _shanten_cached(counts: tuple[int, ...], melds: int) -> int:
    s = shanten_regular(counts, melds)
    if melds == 0:
        s = min(s, shanten_chiitoi(counts), shanten_kokushi(counts))
    return s


def shanten(counts: tuple[int, ...] | list[int], melds: int = 0) -> int:
    """Shanten minimal toutes formes confondues (chiitoi/kokushi seulement main fermée)."""
    return _shanten_cached(tuple(counts), melds)
