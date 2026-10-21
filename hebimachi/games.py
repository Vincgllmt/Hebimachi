"""Jeux pris en charge : chaque jeu a son dossier data/games/<id>/.

Un dossier de jeu contient :
- game.json : nom affiché et particularités du jeu (voir `Game`) ;
- config.json : zones de l'écran (propres à chaque machine, non versionnées) ;
- templates/ : modèles de tuiles, lettres de vent et mots des boutons, appris sur ce jeu.

Pour ajouter un jeu, créer son dossier avec un game.json ; il apparaît dans la liste.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES_DIR = ROOT / "data" / "games"
SETTINGS_PATH = ROOT / "data" / "settings.json"  # dernier jeu choisi
DEFAULT_GAME = "ishin"


@dataclass
class Game:
    id: str
    name: str
    dir: Path
    riichi_hint: str = ""  # comment déclarer le riichi quand le jeu ne le propose pas tout seul

    @property
    def config_path(self) -> Path:
        return self.dir / "config.json"

    @property
    def templates(self) -> Path:
        return self.dir / "templates"


def load_game(path: Path) -> Game:
    data = json.loads((path / "game.json").read_text(encoding="utf-8"))
    return Game(path.name, data.get("name", path.name), path, data.get("riichi_hint", ""))


def list_games() -> list[Game]:
    return [load_game(p) for p in sorted(GAMES_DIR.iterdir()) if (p / "game.json").exists()]


def last_game_id() -> str:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8")).get("game", DEFAULT_GAME)
    except (OSError, ValueError):
        return DEFAULT_GAME


def save_last_game(game_id: str) -> None:
    SETTINGS_PATH.write_text(json.dumps({"game": game_id}), encoding="utf-8")
