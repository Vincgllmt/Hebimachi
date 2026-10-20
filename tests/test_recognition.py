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
