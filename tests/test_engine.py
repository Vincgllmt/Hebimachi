from hebimachi.advisor import advise
from hebimachi.defense import Opponent, tile_danger
from hebimachi.efficiency import analyze, p_at_least_one, simulate_tenpai, remaining_counts
from hebimachi.shanten import shanten
from hebimachi.tiles import format_labels, parse, parse_labels, to_counts


def sh(text, melds=0):
    return shanten(to_counts(parse(text)), melds)


def test_parse_and_red_fives():
    assert parse("0m") == parse("5m")
    assert parse_labels("10p7z") == ["1p", "0p", "7z"]
    assert format_labels(["1m", "2m", "0p", "5z"]) == "12m0p5z"


def test_shanten_basic():
    assert sh("123m456p789s11z222z") == -1           # main complète
    assert sh("123m456p789s11z22z") == 0              # tenpai shanpon
    assert sh("1112345678999m") == 0                  # chuuren (9 attentes)
    assert sh("123m456p789s1z2z3z4z") == 2            # ni paire ni attente
    assert sh("19m19p19s1234567z") == 0               # kokushi 13 attentes
    assert sh("1122m3344p5566s7z") == 0               # chiitoitsu tenpai
    assert sh("147m258p369s1234z") == 6


def test_shanten_with_melds():
    # 2 appels + 7 tuiles : 123m 45p 11z -> tenpai ryanmen
    assert sh("123m45p11z", melds=2) == 0


def test_ukeire_counts():
    a = analyze(parse("123m456p789s23m11z"), [0] * 34)   # 13 tuiles, attente 1-4m
    c = a.candidates[0]
    assert c.shanten == 0
    assert set(c.ukeire) == set(parse("14m"))
    assert c.total == 3 + 4  # un 1m est déjà en main


def test_best_discard_is_isolated_tile():
    a = analyze(parse("123m456p789s23m11z7z"), [0] * 34)
    assert a.candidates[0].discard == parse("7z")[0]


def test_probabilities():
    assert p_at_least_one(0, 100, 10) == 0
    assert abs(p_at_least_one(4, 100, 1) - 0.04) < 1e-9
    assert p_at_least_one(95, 100, 10) == 1.0


def test_monte_carlo_tenpai():
    counts = to_counts(parse("123m456p789s2m5p1z"))  # iishanten
    rem = remaining_counts(counts, [0] * 34)
    p = simulate_tenpai(counts, 0, rem, draws=15, n_sims=200, seed=1)
    assert 0.5 < p <= 1.0


def test_defense_suji_and_genbutsu():
    opp = Opponent("Toimen", parse("4m9p"), riichi=True)
    vis = [0] * 34
    assert tile_danger(parse("4m")[0], opp, vis)[0] == 0           # genbutsu
    assert tile_danger(parse("1m")[0], opp, vis)[1] == "suji/kabe"  # suji de 4m
    assert tile_danger(parse("7m")[0], opp, vis)[1] == "suji/kabe"  # suji de 4m
    assert tile_danger(parse("5p")[0], opp, vis)[1] == "non-suji"


def test_advisor_defends_far_hand():
    hand = parse("147m258p369s1234z5z")  # 14 tuiles, très loin
    opp = Opponent("Kamicha", parse("3z"), riichi=True)
    adv = advise(hand, to_counts(parse("3z")), [opp])
    assert adv.mode == "Défense"
    assert adv.ranked[0].discard == parse("3z")[0]  # genbutsu


def test_open_hand_without_yaku_is_flagged():
    # chi 567p déjà fait ; en jetant 7s on attend 3m, sans aucun yaku
    hand = parse("12789m05888s7s")
    adv = advise(hand, [0] * 34, [], melds=1, calls=parse("067p"))
    assert adv.ranked[0].discard == parse("7s")[0]
    assert "SANS YAKU" in adv.warning


def test_open_tanyao_is_fine():
    # chi 456p ; en jetant le 1m pioché on attend 5s/8s, tout en tanyao
    adv = advise(parse("234m66p345s67s1m"), [0] * 34, [], melds=1, calls=parse("456p"))
    assert adv.ranked[0].discard == parse("1m")[0]
    assert adv.warning == ""
