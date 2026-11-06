"""Calcul de la valeur des mains et objectifs de victoire (mangan, haneman, full straight)."""
import time

import pytest

from hebimachi.actions import check_calls, check_riichi, check_ron, check_tsumo
from hebimachi.advisor import advise
from hebimachi.goals import Goal, goal_status, rank_discards
from hebimachi.tiles import parse, to_counts
from hebimachi.yaku import Context, evaluate, limit_of, next_dora

T = lambda s: parse(s)[0]  # noqa: E731
NONE = [0] * 34
MANGAN, HANEMAN, ITTSU = Goal("mangan"), Goal("haneman"), Goal(ittsu=True)


def full(hand, visible=NONE):
    return [4 - to_counts(hand)[t] - visible[t] for t in range(34)]


def score(hand, win, **ctx):
    return evaluate(parse(hand), T(win), Context(**ctx))


@pytest.mark.parametrize("hand, win, ctx, han, fu, ron", [
    ("234m567m234p678s55s", "4p", {}, 2, 30, 2000),                 # tanyao pinfu
    ("234m567m234p678s55s", "4p", {"tsumo": True}, 3, 20, 2600),    # pinfu tsumo : 20 fu
    ("1155m2299p3366s77z", "7z", {}, 2, 25, 1600),                  # chiitoitsu : 25 fu
    ("111m222p333s789s55m", "5m", {}, 2, 50, 3200),                 # sanankou, attente tanki
    ("123456789m234p55s", "9m", {"riichi": True}, 4, 30, 7700),     # riichi pinfu ittsu
])
def test_han_fu_and_points(hand, win, ctx, han, fu, ron):
    s = score(hand, win, **ctx)
    assert (s.han, s.fu, s.ron) == (han, fu, ron)


def test_most_valuable_reading_wins():
    s = score("111222333m789p11z", "1z")  # lu aussi 123 123 123 : iipeiko + chanta
    assert set(s.names) == {"iipeiko", "chanta"} and s.han == 3


def test_ron_on_shanpon_breaks_concealed_triplet():
    assert score("111m222p333s789s55m", "1m") is None         # 2 brelans cachés seulement
    assert "sanankou" in score("111m222p333s789s55m", "1m", tsumo=True).names


def test_open_hand_loses_han():
    closed = score("123456789m123p55s", "9m", riichi=False)
    opened = evaluate(parse("456789m123p55s"), T("9m"), Context(calls=[parse("123m")]))
    assert dict(closed.yaku)["ittsu"] == 2 and dict(opened.yaku)["ittsu"] == 1


def test_open_no_fu_hand_is_30_fu():
    s = evaluate(parse("234m567p55s678s"), T("8s"), Context(calls=[parse("345s")]))
    assert s.names == ["tanyao"] and s.fu == 30


def test_limits():
    assert limit_of(5, 30) == ("mangan", 2000)
    assert limit_of(4, 40) == ("mangan", 2000)   # 40 * 2^6 = 2560
    assert limit_of(4, 30) == ("", 1920)
    assert limit_of(3, 70) == ("mangan", 2000)
    assert limit_of(6, 30) == ("haneman", 3000)
    assert limit_of(8, 30)[0] == "baiman"
    assert limit_of(11, 30)[0] == "sanbaiman"
    assert limit_of(13, 30)[0] == "yakuman"


def test_yakuman_hands():
    assert score("19m19p19s12345677z", "7z").limit == "yakuman"
    assert score("11122233344455m", "5m", tsumo=True).names == ["suuankou (yakuman)"]
    assert score("11123456789999m", "5m").names == ["chuuren poutou (yakuman)"]
    assert "daisangen (yakuman)" in score("555666777z123m11s", "1s").names


def test_dora_and_red_fives_count_only_with_a_yaku():
    s = score("234m567m234p678s55s", "4p", dora=[T("5s")], aka=[T("5s")])
    assert s.dora == 3 and s.han == 5 and s.limit == "mangan"
    no_yaku = evaluate(parse("123789m55s888s"), T("3m"),
                       Context(calls=[parse("567p")], dora=[T("5s")]))
    assert no_yaku is None


def test_dora_indicator_wraps_around():
    assert next_dora(T("9m")) == T("1m")
    assert next_dora(T("4m")) == T("5m")
    assert next_dora(T("4z")) == T("1z")   # Nord -> Est
    assert next_dora(T("7z")) == T("5z")   # Rouge -> Blanc


def test_goal_satisfied():
    mangan = score("123456789m234p55s", "9m", riichi=True, dora=[T("5s")])  # 4 + 2 = 6 han
    assert MANGAN.satisfied(mangan) and HANEMAN.satisfied(mangan) and ITTSU.satisfied(mangan)
    cheap = score("234m567m234p678s55s", "4p")
    assert not MANGAN.satisfied(cheap) and not ITTSU.satisfied(cheap)
    assert Goal("mangan", True).label() == "Mangan + full straight"
    assert not Goal().active


# ---------- gagner seulement si l'objectif est atteint ----------
def test_cheap_ron_is_refused_with_goal():
    hand = parse("234m567m234p67s55s")  # attente 5-8s, tanyao pinfu
    assert check_ron(hand, T("8s"), Context(), []).recommend
    d = check_ron(hand, T("8s"), Context(goal=MANGAN), [])
    assert not d.recommend and "Mangan" in d.reason


def test_valuable_ron_is_taken_with_goal():
    hand = parse("234m567m234p67s55s")
    ctx = Context(goal=MANGAN, dora=[T("5s")], riichi=True)  # riichi tanyao pinfu + 2 dora
    d = check_ron(hand, T("8s"), ctx, [])
    assert d.recommend and "mangan" in d.reason


def test_tsumo_follows_goal_and_warns_about_furiten():
    d = check_tsumo(parse("123m456p789s11z222z"), T("2z"), Context(goal=HANEMAN, riichi=True))
    assert not d.recommend and "furiten" in d.reason
    d = check_ron(parse("123456789m23p55s"), T("4p"), Context(goal=ITTSU), [])
    assert d.recommend and "ittsu" in d.reason


# ---------- orienter les défausses ----------
def test_full_straight_discard_keeps_the_straight():
    # Sans objectif, 1m est la tuile la moins utile ; pour le full straight elle est indispensable.
    hand = parse("12356789m1p55s77z3s")
    ctx = Context(goal=ITTSU)
    adv = advise(hand, NONE, [], ctx=ctx)
    best = adv.ranked[0]
    assert best.goal.plan.startswith("full straight m")
    assert best.discard not in parse("12356789m")
    assert T("4m") in best.goal.ukeire
    assert "full straight" in adv.goal_note


def test_full_straight_is_dead_when_a_tile_is_gone():
    hand = parse("12356789m1p55s77z3s")
    visible = to_counts(parse("4444m"))
    options = rank_discards(hand, Context(goal=ITTSU), full(hand, visible))
    assert all(not o.plan.startswith("full straight m") for o in options.values())


def test_mangan_prefers_flush_over_speed():
    # Sans objectif, on jette 9s puis on garde tout ; pour 5 han la couleur et les honneurs comptent.
    hand = parse("1234567m789p55z1z9s")
    adv = advise(hand, NONE, [], ctx=Context(goal=MANGAN))
    assert adv.ranked[0].discard not in parse("1234567m5z")
    assert "m" in adv.ranked[0].goal.plan and adv.ranked[0].goal.han >= 5


def test_dora_stay_in_the_plan():
    hand = parse("234m567p345s5m6p77z1z")
    no_dora = rank_discards(hand, Context(goal=MANGAN), full(hand))
    dora = rank_discards(hand, Context(goal=MANGAN, dora=parse("77z5m")), full(hand))
    assert min(o.shanten for o in dora.values()) < min(o.shanten for o in no_dora.values())


def test_goal_does_not_override_defense():
    from hebimachi.defense import Opponent
    hand = parse("147m258p369s1234z5z")
    opp = Opponent("Kamicha", parse("3z"), riichi=True)
    adv = advise(hand, to_counts(parse("3z")), [opp], ctx=Context(goal=MANGAN))
    assert adv.mode == "Défense" and adv.ranked[0].discard == T("3z")
    assert "défense" in adv.goal_note


def test_impossible_goal_falls_back_to_standard_advice():
    # Appels dans deux couleurs, sans honneur : ni couleur unique ni full straight possible.
    ctx = Context(goal=ITTSU, calls=[parse("234m"), parse("345p"), parse("456s")])
    hand = parse("11s22z3z")
    adv = advise(hand, NONE, [], melds=3, ctx=ctx)
    assert "hors d'atteinte" in adv.goal_note
    assert adv.ranked[0].discard is not None


# ---------- riichi et appels ----------
def test_riichi_only_when_it_reaches_the_goal():
    # 4 han dama (tanyao, sanshoku, pinfu), le riichi fait le 5e.
    hand = parse("234m234p234s67s55m1z")
    ctx = Context(goal=MANGAN)
    d = check_riichi(hand, NONE, ctx, draws_left=10)
    assert d.recommend and "Mangan" in d.reason


def test_riichi_refused_when_goal_stays_out_of_reach():
    hand = parse("123m456p789s45m55z7z")
    d = check_riichi(hand, NONE, Context(goal=HANEMAN), draws_left=10)
    assert not d.recommend and "aucune attente" in d.reason


def test_dama_when_goal_is_already_met():
    hand = parse("222333444m67s55m1z")  # chinitsu impossible ici, mais sanankou + tanyao + dora
    ctx = Context(goal=MANGAN, dora=[T("2m"), T("5m")])
    d = check_riichi(hand, NONE, ctx, draws_left=10)
    assert not d.recommend and "dama" in d.reason


def test_call_that_advances_the_flush_is_accepted():
    # Déjà ouverte en man : seul le chinitsu (5 han ouvert) peut encore faire un mangan.
    ctx = Context(goal=MANGAN, calls=[parse("123m")])
    hand = parse("44567m889m5p9s")
    decisions = check_calls(hand, T("6m"), True, ctx, NONE, 12)
    chi = [d for d in decisions if d.action == "CHI"]
    assert any(d.recommend for d in chi)
    best = next(d for d in chi if d.recommend)
    assert "chinitsu m" in best.reason and best.discard in parse("5p9s")


def test_call_that_kills_the_goal_is_refused():
    # Main déjà en attente du 4m : le pon de la paire de 5s ne fait rien gagner au full straight.
    hand = parse("12356789m234p55s")
    decisions = check_calls(hand, T("5s"), False, Context(goal=ITTSU), NONE, 12)
    pon = next(d for d in decisions if d.action == "PON")
    assert not pon.recommend and "objectif" in pon.reason.lower()
    # Un pon de couleur sans rapport avec trois brelans déjà appelés : plus de place pour les suites.
    ctx = Context(goal=ITTSU, calls=[parse("111p"), parse("222s")])
    decisions = check_calls(parse("123456m55s"), T("5s"), False, ctx, NONE, 12)
    pon = next(d for d in decisions if d.action == "PON")
    assert not pon.recommend and "hors d'atteinte" in pon.reason


def test_goal_status_for_a_waiting_hand():
    hand = parse("12356789m234p55s")
    status = goal_status(hand, Context(goal=ITTSU), full(hand))
    assert status.shanten == 0 and T("4m") in status.ukeire


def test_ranking_is_fast_enough_for_auto_capture():
    hand = parse("1234567m789p55z1z")
    start = time.perf_counter()
    for _ in range(3):
        rank_discards(hand, Context(goal=MANGAN, dora=parse("5z")), full(hand))
    assert (time.perf_counter() - start) / 3 < 0.5
