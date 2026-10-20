"""Efficacité de main : ukeire, probabilités de pioche et simulation Monte Carlo."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import comb

from .shanten import shanten
from .tiles import NUM_KINDS, to_counts

LIVE_WALL = 70  # pioches disponibles après la distribution (136 - 52 - 14 du mur mort)


@dataclass
class Candidate:
    discard: int | None          # tuile défaussée (None si main de 13 tuiles)
    shanten: int                 # shanten après la défausse
    ukeire: dict[int, int]       # tuile acceptée -> exemplaires encore invisibles
    total: int                   # somme des ukeire
    p_next: float                # P(la prochaine pioche fait avancer la main)
    p_within: float              # P(avancer au moins une fois d'ici la fin du mur)
    p_tenpai: float | None = None  # P(tenpai d'ici la fin) estimée par Monte Carlo


@dataclass
class Analysis:
    shanten: int
    candidates: list[Candidate]
    unseen: int                  # tuiles invisibles pour nous (mur + mains adverses)
    draws_left: int
    remaining: list[int] = field(default_factory=list)


def remaining_counts(hand_counts: list[int], visible: list[int]) -> list[int]:
    """Exemplaires non vus de chaque tuile (4 - main - tuiles visibles)."""
    return [max(0, 4 - hand_counts[t] - visible[t]) for t in range(NUM_KINDS)]


def estimate_draws_left(total_discards: int) -> int:
    return max(0, (LIVE_WALL - total_discards) // 4)


def p_at_least_one(good: int, pool: int, draws: int) -> float:
    """Probabilité hypergéométrique de piocher au moins une tuile utile."""
    if good <= 0 or draws <= 0 or pool <= 0:
        return 0.0
    draws = min(draws, pool)
    if pool - good < draws:
        return 1.0
    return 1.0 - comb(pool - good, draws) / comb(pool, draws)


def ukeire(counts: list[int], melds: int, remaining: list[int]) -> tuple[int, dict[int, int]]:
    """Shanten d'une main de 13 tuiles et tuiles qui le font baisser."""
    s = shanten(counts, melds)
    accepted: dict[int, int] = {}
    for t in range(NUM_KINDS):
        if remaining[t] == 0 or counts[t] >= 4:
            continue
        counts[t] += 1
        if shanten(counts, melds) < s:
            accepted[t] = remaining[t]
        counts[t] -= 1
    return s, accepted


def analyze(hand: list[int], visible: list[int], melds: int = 0,
            draws_left: int | None = None) -> Analysis:
    """Analyse une main de 14 tuiles (3n+2) ou de 13 tuiles (3n+1).

    `visible` : compte des tuiles visibles hors de notre main (défausses, dora, appels).
    """
    counts = to_counts(hand)
    if any(c > 4 for c in counts):
        raise ValueError("Plus de 4 exemplaires d'une même tuile")
    size = len(hand) + 3 * melds
    if draws_left is None:
        draws_left = estimate_draws_left(sum(visible))

    base_remaining = remaining_counts(counts, visible)
    unseen = sum(base_remaining)
    current = shanten(counts, melds)
    candidates: list[Candidate] = []

    if size % 3 == 2:
        for d in range(NUM_KINDS):
            if counts[d] == 0:
                continue
            counts[d] -= 1
            # la tuile défaussée reste visible : base_remaining l'exclut déjà
            s, acc = ukeire(counts, melds, base_remaining)
            counts[d] += 1
            candidates.append(_candidate(d, s, acc, unseen, draws_left))
    elif size % 3 == 1:
        s, acc = ukeire(counts, melds, base_remaining)
        candidates.append(_candidate(None, s, acc, unseen, draws_left))
    else:
        raise ValueError(f"Main de taille invalide ({len(hand)} tuiles, {melds} appels)")

    candidates.sort(key=lambda c: (c.shanten, -c.total))
    return Analysis(current, candidates, unseen, draws_left, base_remaining)


def _candidate(d: int | None, s: int, acc: dict[int, int], unseen: int, draws: int) -> Candidate:
    total = sum(acc.values())
    return Candidate(d, s, acc, total,
                     total / unseen if unseen else 0.0,
                     p_at_least_one(total, unseen, draws))


def simulate_tenpai(counts13: list[int], melds: int, remaining: list[int], draws: int,
                    n_sims: int = 400, seed: int | None = None) -> float:
    """Estime P(atteindre le tenpai en `draws` pioches) par Monte Carlo.

    Politique simple : on garde une pioche si elle fait baisser le shanten (en
    défaussant la tuile qui conserve le meilleur ukeire immédiat parmi celles qui
    préservent le nouveau shanten), sinon on la rejette. Les pioches viennent du
    pool des tuiles invisibles, ce qui suppose que les mains adverses y sont mêlées.
    """
    if shanten(counts13, melds) <= 0:
        return 1.0
    pool = [t for t in range(NUM_KINDS) for _ in range(remaining[t])]
    if not pool or draws <= 0:
        return 0.0
    rng = random.Random(seed)
    success = 0
    for _ in range(n_sims):
        hand = list(counts13)
        s = shanten(hand, melds)
        for t in rng.sample(pool, min(draws, len(pool))):
            hand[t] += 1
            new_s = shanten(hand, melds)
            if new_s < s:
                hand[_best_discard(hand, melds, new_s, remaining)] -= 1
                s = new_s
                if s <= 0:
                    success += 1
                    break
            else:
                hand[t] -= 1
    return success / n_sims


def _best_discard(hand14: list[int], melds: int, target: int, remaining: list[int]) -> int:
    best, best_score = -1, -1
    for d in range(NUM_KINDS):
        if hand14[d] == 0:
            continue
        hand14[d] -= 1
        if shanten(hand14, melds) == target:
            score = sum(remaining[t] for t in range(NUM_KINDS)
                        if hand14[t] < 4 and _improves(hand14, melds, t, target))
            if score > best_score:
                best, best_score = d, score
        hand14[d] += 1
    return best


def _improves(hand: list[int], melds: int, t: int, s: int) -> bool:
    hand[t] += 1
    ok = shanten(hand, melds) < s
    hand[t] -= 1
    return ok
