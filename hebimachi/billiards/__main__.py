"""Prototype : analyse des captures du billard.

    python -m hebimachi.billiards capture.png [autres.png ...] [--game kiwami] [--out dossier]

Affiche le verdict et écrit une copie annotée de chaque capture (`<nom>_analyse.png`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from ..games import GAMES_DIR
from .render import draw
from .shot import analyze
from .table import Table
from .vision import normalize, read_scene


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hebimachi.billiards")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--game", default="kiwami")
    ap.add_argument("--out", type=Path, default=None, help="dossier des images annotées (défaut : à côté)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    table = Table.load(GAMES_DIR / args.game / "billiards.json")
    for path in args.images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path} : illisible")
            continue
        image = normalize(image, table)
        scene = read_scene(image, table)
        advice = analyze(scene, table)
        print(f"{path.name} : {len(scene.balls)} billes"
              + (f", NEXT = {scene.next_color}" if scene.next_color else (", HUD visible" if scene.hud else "")))
        for line in advice.lines:
            print("  " + line)
        out_dir = args.out or path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{path.stem}_analyse.png"), draw(image, table, scene, advice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
