from hebimachi.actions import check_calls, check_closed_kan, check_riichi, check_ron, check_tsumo
from hebimachi.tiles import parse
from hebimachi.yaku import Context, find_yaku

T = lambda s: parse(s)[0]  # noqa: E731
NONE = [0] * 34


def test_yaku_detection():
    ctx = Context()
    assert "pinfu" in find_yaku(parse("234m567m234p678s55s"), T("4p"), ctx)
    assert "tanyao" in find_yaku(parse("234m567m234p678s55s"), T("4p"), ctx)
    assert "iipeiko" in find_yaku(parse("112233m456p789s11z"), T("3m"), ctx)
    assert "ittsu" in find_yaku(parse("123456789m234p55s"), T("9m"), ctx)
    assert "sanshoku" in find_yaku(parse("123m123p123s789s55z"), T("5z"), ctx)
    assert "chiitoitsu" in find_yaku(parse("1155m2299p3366s77z"), T("7z"), ctx)
    # aucun yaku : main ouverte (chi 567p), sans tanyao ni honneur
    open_ctx = Context(calls=[parse("567p")])
    assert find_yaku(parse("123789m55s888s"), T("3m"), open_ctx) == []


def test_ron_and_furiten():
    ctx = Context(riichi=True)
    hand = parse("123m456p789s1122z")  # attente shanpon 1z/2z
    assert check_ron(hand, T("1z"), ctx, my_discards=[]).recommend
    furiten = check_ron(hand, T("1z"), ctx, my_discards=parse("2z"))
    assert not furiten.recommend and "Furiten" in furiten.reason
    assert check_ron(hand, T("5m"), ctx, []) is None


def test_ron_without_yaku_refused():
    ctx = Context(calls=[parse("567p")])
    d = check_ron(parse("12789m05888s"), T("3m"), ctx, [])
    assert not d.recommend and "YAKU" in d.reason


def test_closed_tsumo_always_has_yaku():
    d = check_tsumo(parse("123m456p789s11z222z"), T("2z"), Context())
    assert d.recommend and "menzen tsumo" in d.reason


def test_riichi_recommended_without_yaku():
    # jeter le Rouge pioché -> tenpai 3m-6m ; paire de Blanc donc pas de pinfu : aucun yaku sans riichi
    d = check_riichi(parse("123m456p789s45m55z7z"), NONE, Context(), draws_left=10)
    assert d.recommend and d.discard == T("7z") and "pas de yaku" in d.reason


def test_pon_dragon_is_yes_and_useless_chi_is_no():
    hand = parse("77z123m456p78s159m")  # paire de Rouge
    ctx = Context()
    decisions = check_calls(hand, T("7z"), from_kamicha=False, ctx=ctx, visible=NONE, draws_left=12)
    pon = next(d for d in decisions if d.action == "PON")
    assert pon.recommend
    # chi qui ne fait pas avancer une main déjà correcte et fermée
    hand2 = parse("123m456p789s23m11z")  # tenpai 1m-4m
    decisions = check_calls(hand2, T("4m"), from_kamicha=True, ctx=ctx, visible=NONE, draws_left=12)
    assert all(not d.recommend for d in decisions if d.action == "CHI")


def test_call_refused_without_yaku_route():
    hand = parse("19m19p19s1236789m")[:13]
    decisions = check_calls(hand, T("2m"), True, Context(), NONE, 12)
    assert all(not d.recommend for d in decisions)


def test_closed_kan():
    d = check_closed_kan(parse("1111m456p789s23m55z"), Context())
    assert d is not None and d.recommend
