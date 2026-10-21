"""Lecture de vraies captures de Like a Dragon: Ishin! (zones et modèles de data/)."""
from pathlib import Path

import cv2
import pytest

from hebimachi.capture import Config, calls_region, crop
from hebimachi.recognition import TileMatcher
from hebimachi.tiles import format_labels

ROOT = Path(__file__).resolve().parent.parent
SCREENS = Path(__file__).parent / "screens"
GAME = ROOT / "data" / "games" / "ishin"  # captures de Like a Dragon: Ishin!

EXPECTED = {
    "milieu_de_donne.png": {
        "hand": "4467888m40p55661z", "dora": "2p", "riichi": None,
        "me": "17z91p67s2z8s", "toimen": "2s514z6p2m4p2s",
        "kamicha": "1z5s2p9m2z52m17p", "shimocha": "0m5s6z3p3z4m5p2z",
    },
    "appel_et_riichi.png": {
        "hand": "12789m058887s", "dora": "6p", "riichi": ("kamicha", 4),
        "me": "4z7p9s4m9s66m6z", "toimen": "49p92m127s5z",
        "kamicha": "1s6p5z7s7p2s2m84p", "shimocha": "5m137z1127s",
    },
    "deux_rangees.png": {
        "hand": "12340s40p7m", "dora": "3p", "riichi": None,
        "me": "2z7s7z647p1z9s2p3819980m", "toimen": "9m617z1s1p1z9p6s9p5s1m98s324p",
        "kamicha": "3z8p6z1p4z8m1z94p42z2m8s8m9p9m5p", "shimocha": "527z2m3s34z6m2s2m1s1m66z13m",
    },
}


@pytest.fixture(scope="module")
def setup():
    config = Config.load(GAME / "config.json")
    matcher = TileMatcher(GAME / "templates")
    if not config.hand or matcher.missing():
        pytest.skip("zones ou modèles non configurés")
    return config, matcher


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_screen(setup, name):
    config, matcher = setup
    exp = EXPECTED[name]
    img = cv2.imread(str(SCREENS / name))

    calls = matcher.read_tiles(crop(img, calls_region(config)), max_long=0.9 * 92)
    left = calls_region(config)[0] + min((d.x for d in calls), default=10_000)
    hand = [d for d in matcher.detect(crop(img, config.hand)) if config.hand[0] + d.x + d.w <= left]
    assert format_labels([d.label for d in hand]) == exp["hand"]
    assert format_labels([d.label for d in matcher.read_tiles(crop(img, config.dora))]) == exp["dora"]

    for seat, region in config.rivers.items():
        labels, riichi = matcher.read_river(crop(img, region), seat)
        assert format_labels(labels) == exp[seat], seat
        assert (riichi is not None) == (exp["riichi"] is not None and exp["riichi"][0] == seat), seat
        if riichi is not None:
            assert riichi == exp["riichi"][1]


PROMPTS = {
    "milieu_de_donne.png": [], "appel_et_riichi.png": [], "deux_rangees.png": [], "prompt_chi.png": ["Chi"],
    "prompt_pon.png": ["Pon"], "prompt_chi_2.png": ["Chi"], "prompt_ron.png": ["Ron"],
    "prompt_riichi.png": ["Riichi"], "prompt_kan_pon.png": ["Pon", "Kan"], "trois_appels.png": [],
}


@pytest.mark.parametrize("name", sorted(PROMPTS))
def test_prompt(setup, name):
    from hebimachi.recognition import PromptReader

    config, matcher = setup
    reader = PromptReader(GAME / "templates")
    shown, actions = reader.read(crop(cv2.imread(str(SCREENS / name)), config.extra["prompt"]))
    assert actions == PROMPTS[name]
    assert shown == bool(PROMPTS[name])


def test_ron_not_read_as_pon_without_ron_template(setup, tmp_path):
    """« Ron » et « Pon » ne diffèrent que d'une lettre : sans modèle Ron, mot inconnu plutôt que Pon."""
    import shutil

    from hebimachi.recognition import PromptReader

    config, matcher = setup
    for p in (GAME / "templates").glob("prompt_*.png"):
        if p.stem != "prompt_Ron":
            shutil.copy(p, tmp_path)
    reader = PromptReader(tmp_path)
    shown, actions = reader.read(crop(cv2.imread(str(SCREENS / "prompt_ron.png")), config.extra["prompt"]))
    assert shown and actions == []


def test_chi_prompt_advice(setup):
    """Pendant le prompt, la tuile jetée est invisible : on évalue toutes les tuiles possibles."""
    from hebimachi.actions import evaluate_prompt
    from hebimachi.tiles import parse
    from hebimachi.yaku import Context

    config, matcher = setup
    img = cv2.imread(str(SCREENS / "prompt_chi.png"))
    hand = [d.label for d in matcher.detect(crop(img, config.hand))]
    assert format_labels(hand) == "35m24577799s89p7z"
    results = evaluate_prompt(parse(format_labels(hand)), "Chi", Context(27, 30), [0] * 34, 12, 0, [])
    assert {t for t, _ in results} == set(parse("4m7p3s6s8s"))
    assert not any(d.recommend for _, d in results)


def test_pon_prompt_advice(setup):
    from hebimachi.actions import evaluate_prompt
    from hebimachi.tiles import parse
    from hebimachi.yaku import Context

    config, matcher = setup
    img = cv2.imread(str(SCREENS / "prompt_pon.png"))
    hand = format_labels([d.label for d in matcher.detect(crop(img, config.hand))])
    assert hand == "44789m405s07p455z"
    results = dict(evaluate_prompt(parse(hand), "Pon", Context(27, 29), [0] * 34, 12, 0, []))
    haku = results[parse("5z")[0]]
    assert haku.recommend and "yaku garanti" in haku.reason


def test_chi_to_weak_yaku_wait_is_refused(setup):
    """Chi 678p -> tenpai 4m/Blanc, mais seul le Blanc (2 restants) a un yaku : on refuse."""
    from hebimachi.actions import evaluate_prompt
    from hebimachi.tiles import parse
    from hebimachi.yaku import Context

    config, matcher = setup
    img = cv2.imread(str(SCREENS / "prompt_chi_2.png"))
    hand = format_labels([d.label for d in matcher.detect(crop(img, config.hand))])
    assert hand == "44789m406s788p55z"
    results = dict(evaluate_prompt(parse(hand), "Chi", Context(27, 29), [0] * 34, 10, 0, []))
    assert not results[parse("6p")[0]].recommend
    assert "seul Blanc" in results[parse("6p")[0]].reason


def test_three_calls(setup):
    """Partie du 21/10 : le 3e appel (chi 345m) repousse le 1er hors de la zone des appels réglée."""
    from hebimachi.recognition import group_calls

    config, matcher = setup
    img = cv2.imread(str(SCREENS / "trois_appels.png"))
    region = calls_region(config)
    calls = matcher.read_tiles(crop(img, region), max_long=0.9 * 92)
    for d in calls:
        d.x += region[0]
    groups = group_calls(calls)
    assert [format_labels([d.label for d in g]) for g in groups] == ["345m", "978p", "666z"]
    hand = [d for d in matcher.detect(crop(img, config.hand))
            if config.hand[0] + d.x + d.w <= min(d.x for d in calls)]
    assert format_labels([d.label for d in hand]) == "4m1335p"
