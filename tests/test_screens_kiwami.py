"""Yakuza Kiwami : apprentissage des tuiles à partir des modèles d'Ishin, sur de vraies captures."""
from pathlib import Path

import cv2
import pytest

from hebimachi.capture import crop
from hebimachi.recognition import TileMatcher
from hebimachi.tiles import format_labels, parse_labels

ROOT = Path(__file__).resolve().parent.parent
SCREENS = Path(__file__).parent / "screens" / "kiwami"
ISHIN = ROOT / "data" / "games" / "ishin" / "templates"

# Zones de l'écran 1920×1080 (les config.json ne sont pas versionnées).
HAND, DORA = (370, 860, 1080, 170), (620, 460, 300, 100)
RIVERS = {"me": (540, 685, 720, 195), "toimen": (530, 75, 720, 225),
          "kamicha": (215, 180, 230, 700), "shimocha": (1340, 170, 220, 700)}

EXPECTED = {
    "debut_de_donne.png": {
        "hand": "124699m56s37p157z4m",  # ordre affiché, pioche (4m) en dernier
        "dora": "1z",
        "rivers": {"me": "8s", "toimen": "4z1m", "kamicha": "42z", "shimocha": "8s3z"},
        "riichi": None,
        "weak": {"8s", "4z", "2z", "3z"},
    },
    "riichi_gauche.png": {  # Rina (kamicha) en riichi : 9p couché, collé à ses autres défausses
        "hand": "1244699m56s37p57z6m",
        "dora": "1z",
        "rivers": {"me": "8s11z", "toimen": "4z1m67z", "kamicha": "42z2s9p", "shimocha": "8s3z1p9s"},
        "riichi": ("kamicha", 3),
        "weak": {"8s", "1z", "4z", "2z", "3z", "6z", "2s", "9p", "1p", "9s"},
    },
}


@pytest.fixture
def matcher(tmp_path):
    if not (ISHIN / "1m.png").exists():
        pytest.skip("modèles d'Ishin absents")
    return TileMatcher(tmp_path, reference=ISHIN)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_learns_from_ishin(matcher, name):
    exp = EXPECTED[name]
    img = cv2.imread(str(SCREENS / name))
    hand = crop(img, HAND)

    learned = matcher.auto_learn(hand)
    # Chaque tuile de la main, dragon blanc compris (face vide, repérée à part).
    assert sorted(learned) == sorted(set(parse_labels(exp["hand"])))
    assert format_labels([d.label for d in matcher.detect(hand)]) == exp["hand"]

    assert format_labels([d.label for d in matcher.read_tiles(crop(img, DORA))]) == exp["dora"]
    for seat, tiles in exp["rivers"].items():
        labels, riichi = matcher.read_river(crop(img, RIVERS[seat]), seat)
        assert format_labels(labels) == tiles, seat
        assert riichi == (exp["riichi"][1] if exp["riichi"] and exp["riichi"][0] == seat else None), seat
    # Appris sur des rivières : à remplacer dès que ces tuiles passeront dans la main.
    assert matcher.weak == exp["weak"]


def test_hand_replaces_weak_model(matcher):
    img = cv2.imread(str(SCREENS / "debut_de_donne.png"))
    matcher.auto_learn(crop(img, HAND))
    matcher.read_river(crop(img, RIVERS["shimocha"]), "shimocha")
    assert "8s" in matcher.weak
    # La tuile est ensuite apprise dans la main : le modèle faible est remplacé.
    tw, th = matcher.tile_size
    matcher.learn_from_strip(crop(img, HAND)[:th, :tw], ["8s"])
    assert "8s" not in matcher.weak


def test_default_zones_from_game_json():
    from hebimachi.capture import Config
    from hebimachi.games import GAMES_DIR, load_game

    game = load_game(GAMES_DIR / "kiwami")
    config = Config()
    assert config.apply_defaults(game.zones, 1920, 1080)
    assert config.hand == (150, 860, 1300, 170) and config.dora == DORA
    assert config.rivers == RIVERS
    assert not config.apply_defaults(game.zones, 1920, 1080)  # rien ne manque plus
