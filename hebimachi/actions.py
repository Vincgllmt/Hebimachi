"""Décisions d'action : ron, tsumo, riichi, pon, chi, kan.

Chaque fonction renvoie une `Decision` (ou None si l'action n'est pas possible),
avec une recommandation et une raison courte, pour répondre au prompt du jeu.
"""
from __future__ import annotations

from dataclasses import dataclass

from .efficiency import analyze
from .shanten import shanten
from .tiles import display_name, to_counts
from .yaku import Context, call_set, find_yaku


@dataclass
class Decision:
    action: str          # RON, TSUMO, RIICHI, PON, CHI, KAN
    recommend: bool
    reason: str
    discard: int | None = None   # défausse à faire après l'appel (ou avec le riichi)
    used: tuple[int, ...] = ()   # tuiles de la main utilisées par l'appel

    def headline(self, tile: int | None = None) -> str:
        name = f" {display_name(tile)}" if tile is not None else ""
        return f"{self.action}{name} ?  →  {'OUI' if self.recommend else 'NON'}"


def waits(hand13: list[int], melds: int) -> list[int]:
    counts = to_counts(hand13)
    out = []
    for t in range(34):
        if counts[t] < 4:
            counts[t] += 1
            if shanten(counts, melds) == -1:
                out.append(t)
            counts[t] -= 1
    return out


# ---------- victoire ----------
def check_ron(hand13: list[int], tile: int, ctx: Context, my_discards: list[int]) -> Decision | None:
    melds = len(ctx.calls)
    if tile not in waits(hand13, melds):
        return None
    ron_ctx = Context(ctx.round_wind, ctx.seat_wind, ctx.riichi, False, ctx.calls)
    yaku = find_yaku(hand13 + [tile], tile, ron_ctx)
    furiten = [w for w in waits(hand13, melds) if w in my_discards]
    if furiten:
        return Decision("RON", False, f"Furiten : vous avez déjà jeté {display_name(furiten[0])}, "
                                      "le ron est interdit (seul le tsumo est permis).")
    if not yaku:
        return Decision("RON", False, "Main complète mais SANS YAKU : le ron est impossible.")
    return Decision("RON", True, f"Victoire ! Yaku : {', '.join(yaku)}.")


def check_tsumo(hand14: list[int], drawn: int, ctx: Context) -> Decision | None:
    if shanten(to_counts(hand14), len(ctx.calls)) != -1:
        return None
    tsumo_ctx = Context(ctx.round_wind, ctx.seat_wind, ctx.riichi, True, ctx.calls)
    yaku = find_yaku(hand14, drawn, tsumo_ctx)
    if not yaku:
        return Decision("TSUMO", False, "Main complète mais SANS YAKU : impossible de déclarer tsumo.")
    return Decision("TSUMO", True, f"Victoire ! Yaku : {', '.join(yaku)}.")


# ---------- riichi ----------
def check_riichi(hand14: list[int], visible: list[int], ctx: Context, draws_left: int,
                 threats: int = 0) -> Decision | None:
    """Riichi possible si main fermée, tenpai après une défausse, et au moins 4 tuiles au mur."""
    if not ctx.closed or ctx.riichi:
        return None
    a = analyze(hand14, visible, 0, draws_left)
    tenpai = [c for c in a.candidates if c.shanten == 0]
    if not tenpai:
        return None
    best = max(tenpai, key=lambda c: c.total)
    if draws_left < 1:
        return Decision("RIICHI", False, "Plus assez de tuiles au mur pour déclarer riichi.", best.discard)
    if best.total == 0:
        return Decision("RIICHI", False, "Attente morte : toutes vos tuiles gagnantes sont visibles.",
                        best.discard)
    hand13 = list(hand14)
    hand13.remove(best.discard)
    has_yaku = any(find_yaku(hand13 + [w], w, Context(ctx.round_wind, ctx.seat_wind, False, False))
                   for w in best.ukeire)
    wait_txt = ", ".join(display_name(t) for t in sorted(best.ukeire))
    if threats and best.total <= 4 and has_yaku:
        return Decision("RIICHI", False, f"Attente faible ({best.total} tuiles : {wait_txt}) face à un "
                                         "riichi : restez dama, vous avez déjà un yaku.", best.discard)
    reason = f"Tenpai fermé, attente {wait_txt} ({best.total} tuiles)."
    reason += " Le riichi ajoute de la valeur." if has_yaku else " Sans riichi, pas de yaku pour gagner en ron."
    return Decision("RIICHI", True, reason, best.discard)


# ---------- appels ----------
def _yaku_route(hand_counts: list[int], calls: list[list[int]], ctx: Context) -> str | None:
    """Un yaku reste-t-il atteignable avec ces appels ? Renvoie lequel, ou None."""
    values = ctx.value_honors()
    sets = [call_set(g) for g in calls]
    if any(k == "tri" and t in values for k, t in sets):
        return "yakuhai"
    call_tiles = [t for g in calls for t in g]
    hand_tiles = [t for t in range(34) for _ in range(hand_counts[t])]
    if all(t < 27 and t % 9 not in (0, 8) for t in call_tiles):
        if sum(1 for t in hand_tiles if t >= 27 or t % 9 in (0, 8)) <= 2:
            return "tanyao"
    suits = {t // 9 for t in call_tiles if t < 27}
    if len(suits) <= 1:
        suit = next(iter(suits), None)
        off = sum(1 for t in hand_tiles if t < 27 and (suit is None or t // 9 != suit))
        if off <= 2:
            return "honitsu"
    pairs = [t for t in values if hand_counts[t] >= 2]
    if pairs:
        return f"yakuhai, risqué : il faudra encore appeler le {display_name(pairs[0])}"
    if all(k == "tri" for k, _ in sets) and sum(1 for c in hand_counts if c >= 2) >= 3:
        return "toitoi"
    return None


def _options(counts: list[int], tile: int, can_chi: bool) -> list[tuple[str, tuple[int, ...]]]:
    opts = []
    if counts[tile] >= 2:
        opts.append(("PON", (tile, tile)))
    if counts[tile] >= 3:
        opts.append(("KAN", (tile, tile, tile)))
    if can_chi and tile < 27:
        n = tile % 9
        for a, b in ((-2, -1), (-1, 1), (1, 2)):
            if 0 <= n + a <= 8 and 0 <= n + b <= 8 and counts[tile + a] and counts[tile + b]:
                opts.append(("CHI", (tile + a, tile + b)))
    return opts


MIN_LIVE_WAIT = 4  # en dessous, une attente ouverte ne vaut pas de casser la main fermée


def _winning_waits(rest: list[int], best, calls: list[list[int]], ctx: Context) -> dict[int, int | None]:
    """Attentes après l'appel et la défausse : tuiles restantes si elles donnent un yaku, sinon None."""
    hand = list(rest)
    hand.remove(best.discard)
    open_ctx = Context(ctx.round_wind, ctx.seat_wind, False, False, calls)
    return {t: (n if find_yaku(hand + [t], t, open_ctx) else None) for t, n in best.ukeire.items()}


def check_calls(hand13: list[int], tile: int, from_kamicha: bool, ctx: Context,
                visible: list[int], draws_left: int, threats: int = 0) -> list[Decision]:
    """Évalue chaque appel possible sur la tuile jetée, comparé au fait de passer."""
    if ctx.riichi:
        return []
    counts = to_counts(hand13)
    melds = len(ctx.calls)
    skip = analyze(hand13, visible, melds, draws_left).candidates[0]
    decisions = []
    for kind, used in _options(counts, tile, from_kamicha):
        rest = list(hand13)
        for t in used:
            rest.remove(t)
        new_calls = ctx.calls + [sorted(list(used) + [tile])]
        vis = list(visible)
        for t in used:
            vis[t] += 1
        if kind == "KAN":
            s_after = shanten(to_counts(rest), melds + 1)
            ok = not ctx.closed and threats == 0 and s_after <= skip.shanten
            reason = ("Kan ouvert : ne fait pas avancer la main et offre un dora aux adversaires."
                      if not ok else "Main déjà ouverte, pas de danger : le kan ajoute un dora.")
            decisions.append(Decision(kind, ok, reason, None, used))
            continue
        # Kuikae : interdit de rejeter la tuile prise (ni l'autre bout de la suite).
        forbidden = {tile}
        if kind == "CHI":
            lo, hi = min(used + (tile,)), max(used + (tile,))
            if tile == lo and hi % 9 < 8:
                forbidden.add(hi + 1)
            if tile == hi and lo % 9 > 0:
                forbidden.add(lo - 1)
        cands = [c for c in analyze(rest, vis, melds + 1, draws_left).candidates
                 if c.discard not in forbidden]
        if not cands:
            continue
        best = cands[0]
        after = to_counts(rest)
        after[best.discard] -= 1
        route = _yaku_route(after, new_calls, ctx)
        faster = best.shanten < skip.shanten
        gain = (f"{skip.shanten}-shanten → {best.shanten}-shanten" if faster
                else f"reste à {best.shanten}-shanten")
        live = _winning_waits(rest, best, new_calls, ctx) if best.shanten == 0 else None
        if threats and best.shanten >= 1:
            rec, reason = False, "Riichi adverse : appeler vous prive de tuiles sûres sans vous mettre tenpai."
        elif live is not None:
            # Tenpai après l'appel : on vérifie le yaku attente par attente, sans deviner.
            good = {t: n for t, n in live.items() if n is not None}
            count = sum(good.values())
            names = ", ".join(display_name(t) for t in sorted(good)) or "aucune"
            dead = [display_name(t) for t, n in live.items() if n is None]
            if not good:
                rec, reason = False, "Tenpai après l'appel, mais aucune attente n'a de yaku : impossible de gagner."
            elif count < MIN_LIVE_WAIT:
                rec, reason = False, (f"Tenpai, mais seul {names} donne un yaku ({count} tuile(s) restante(s)) : "
                                      "attente trop faible, gardez la main fermée.")
            else:
                rec, reason = True, f"Tenpai ({gain}), attente avec yaku : {names} ({count} tuiles)."
                if dead:
                    reason += f" Sans yaku sur {', '.join(dead)}."
                if ctx.closed:
                    reason += " Vous renoncez au riichi."
        elif route is None:
            rec, reason = False, "Après l'appel, aucun yaku possible : vous ne pourriez pas gagner."
        elif route == "yakuhai" and best.shanten <= skip.shanten:
            rec, reason = True, f"Brelan d'honneur = yaku garanti ({gain})."
        elif faster:
            rec, reason = True, f"Accélère la main ({gain}), yaku visé : {route}."
            if ctx.closed and best.shanten == 0:
                reason += " Vous renoncez au riichi."
        else:
            rec, reason = False, f"N'accélère pas la main ({gain}) : gardez-la fermée."
        if rec:
            reason += f" Puis jetez {display_name(best.discard)}."
        decisions.append(Decision(kind, rec, reason, best.discard, used))
    decisions.sort(key=lambda d: not d.recommend)
    return decisions


def check_closed_kan(hand14: list[int], ctx: Context, threats: int = 0) -> Decision | None:
    """Ankan (4 identiques en main) pendant notre tour."""
    counts = to_counts(hand14)
    quads = [t for t in range(34) if counts[t] == 4]
    if not quads:
        return None
    t = quads[0]
    melds = len(ctx.calls)
    best_now = 8
    for d in set(hand14):
        counts[d] -= 1
        best_now = min(best_now, shanten(counts, melds))
        counts[d] += 1
    rest = [x for x in hand14 if x != t]
    s_after = shanten(to_counts(rest), melds + 1)
    if ctx.riichi:
        # En riichi, la 4e tuile vient d'être piochée : on compare avec l'attente d'avant.
        w_now = waits(rest + [t] * 3, melds)
        w_after = waits(rest, melds + 1)
        ok = bool(w_after) and set(w_after) == set(w_now)
        return Decision("KAN", ok, "En riichi : kan permis seulement s'il ne change pas l'attente."
                        + ("" if ok else " Ici il la change : ne le faites pas."))
    if threats:
        return Decision("KAN", False, "Riichi adverse : un kan offre un dora de plus à l'adversaire.")
    ok = s_after <= best_now
    return Decision("KAN", ok, "Kan fermé : la main reste fermée et vous gagnez un dora."
                    if ok else "Le kan casserait la forme de votre main.")


# ---------- prompt du jeu : la tuile proposée n'est pas encore visible ----------
def prompt_candidates(hand13: list[int], word: str, ctx: Context, visible: list[int]) -> list[int]:
    """Tuiles qui ont pu déclencher ce prompt (encore disponibles, compatibles avec la main)."""
    counts = to_counts(hand13)
    left = [4 - counts[t] - visible[t] for t in range(34)]
    if word == "Chi":
        tiles = [t for t in range(27) if any(k == "CHI" for k, _ in _options(counts, t, True))]
    elif word == "Pon":
        tiles = [t for t in range(34) if counts[t] >= 2]
    elif word == "Kan":
        tiles = [t for t in range(34) if counts[t] >= 3]
    elif word == "Ron":
        tiles = waits(hand13, len(ctx.calls))
    else:
        return []
    return [t for t in tiles if left[t] > 0]


def evaluate_prompt(hand13: list[int], word: str, ctx: Context, visible: list[int], draws_left: int,
                    threats: int, my_discards: list[int]) -> list[tuple[int, Decision]]:
    """Décision pour chaque tuile candidate d'un prompt Chi / Pon / Kan / Ron."""
    out = []
    for t in prompt_candidates(hand13, word, ctx, visible):
        if word == "Ron":
            d = check_ron(hand13, t, ctx, my_discards)
        else:
            options = [d for d in check_calls(hand13, t, word == "Chi", ctx, visible, draws_left, threats)
                       if d.action == word.upper()]
            d = options[0] if options else None
        if d is not None:
            out.append((t, d))
    return out
