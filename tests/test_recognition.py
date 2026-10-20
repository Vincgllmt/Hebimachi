import numpy as np

from hebimachi.recognition import TileMatcher
from hebimachi.tiles import ALL_LABELS


def _fake_tiles(rng, w=40, h=56):
    """Une 'tuile' synthétique distincte par étiquette."""
    return {lb: rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for lb in ALL_LABELS}


def test_learn_then_detect_with_drawn_tile_gap(tmp_path):
    rng = np.random.default_rng(0)
    fake = _fake_tiles(rng)
    hand = ["1m", "2m", "3m", "0p", "5p", "7s", "8s", "9s", "1z", "1z", "5z", "6z", "7z"]
    strip = np.hstack([fake[lb] for lb in hand])

    matcher = TileMatcher(tmp_path)
    matcher.learn_from_strip(strip, hand)
    # Apprendre le reste de l'alphabet pour que les modèles concurrents existent.
    others = [lb for lb in ALL_LABELS if lb not in hand]
    matcher.learn_from_strip(np.hstack([fake[lb] for lb in others]), others)
    assert not matcher.missing()

    drawn = "4m"
    scene = np.full((80, 40 * 15 + 60, 3), 30, np.uint8)
    scene[10:66, 10:10 + strip.shape[1]] = strip
    scene[14:70, 10 + strip.shape[1] + 20:10 + strip.shape[1] + 60] = fake[drawn]  # piochée, décalée

    labels = [d.label for d in matcher.detect(scene)]
    assert labels == hand + [drawn]


def _light_tile(rng, w=46, h=64):
    """Face claire peu saturée avec un motif sombre, comme une vraie tuile."""
    face = np.full((h, w, 3), 225, np.uint8)
    for _ in range(6):
        x, y = rng.integers(4, w - 12), rng.integers(4, h - 12)
        face[y:y + 8, x:x + 8] = rng.integers(0, 120, 3, dtype=np.uint8)
    return face


def test_river_reading_and_sideways_riichi(tmp_path):
    rng = np.random.default_rng(1)
    faces = {lb: _light_tile(rng) for lb in ALL_LABELS}
    matcher = TileMatcher(tmp_path)
    matcher.learn_from_strip(np.hstack([faces[lb] for lb in ALL_LABELS]), list(ALL_LABELS), inset=0)

    river = ["1z", "9m", "5p", "2s", "7z"]
    scene = np.zeros((90, 400, 3), np.uint8)
    scene[:] = (60, 150, 30)  # tapis vert (BGR)
    x = 10
    for i, lb in enumerate(river):
        tile = faces[lb] if i != 3 else np.rot90(faces[lb]).copy()  # 4e tuile couchée = riichi
        h, w = tile.shape[:2]
        scene[80 - h:80, x:x + w] = tile
        x += w + 6

    labels, riichi = matcher.read_river(scene, "me")
    assert labels == river
    assert riichi == 3
