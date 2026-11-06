"""Lecture du plateau de shogi à l'écran.

Le plateau est repéré par ses quatre coins puis redressé, ce qui marche aussi bien en vue de
dessus qu'en vue inclinée. Chaque case est ensuite comparée à des modèles appris sur le jeu :
la position de départ contient toutes les pièces non promues des deux camps, un seul écran
suffit donc à les apprendre ; les pièces promues s'apprennent quand elles apparaissent.

Convention : une pièce lue est positive si elle appartient au joueur du bas de l'écran.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .position import K, LETTERS, PB, PL, PN, PP, PR, PS, START_SFEN, Position

CELL = 64                  # taille d'une case redressée, en pixels
INNER = 0.70               # part centrale de la case gardée comme modèle
INK_AREA = 0.6             # part centrale où l'on cherche de l'encre (évite les lignes du plateau)
DEFAULT_INK = 0.04
MIN_SCORE = 0.55           # en dessous, la pièce n'est pas reconnue
ALL_TYPES = tuple(LETTERS) + (PP, PL, PN, PS, PB, PR)
Corners = list[tuple[int, int]]  # haut-gauche, haut-droit, bas-droit, bas-gauche


def type_name(t: int) -> str:
    return LETTERS[t] if t <= K else "+" + LETTERS[t - 8]


def rectify(image: np.ndarray, corners: Corners) -> np.ndarray:
    """Plateau redressé, de 9 × CELL pixels de côté."""
    side = 9 * CELL
    src = np.array(corners, np.float32)
    dst = np.array([(0, 0), (side, 0), (side, side), (0, side)], np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (side, side), flags=cv2.INTER_AREA)


def cell_image(board: np.ndarray, index: int) -> np.ndarray:
    r, c = divmod(index, 9)
    return board[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL]


def _gray(cell: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) if cell.ndim == 3 else cell


def _inner(gray: np.ndarray, share: float) -> np.ndarray:
    m = int(CELL * (1 - share) / 2)
    return gray[m:CELL - m, m:CELL - m]


def ink(cell: np.ndarray) -> float:
    """Densité de contours au centre de la case : forte sur un kanji, faible sur le bois nu."""
    gray = _inner(_gray(cell), INK_AREA)
    return float((cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 150) > 0).mean())


class BoardReader:
    def __init__(self, template_dir: Path):
        self.dir = Path(template_dir) / "shogi"
        self.templates: dict[tuple[int, int], np.ndarray] = {}  # (sens, type) -> modèle ; sens +1 = du bas
        self.ink_threshold = DEFAULT_INK
        self.reload()

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def _path(self, sign: int, t: int) -> Path:
        return self.dir / f"{'up' if sign > 0 else 'down'}_{type_name(t)}.png"

    def reload(self) -> None:
        self.templates = {}
        for sign in (1, -1):
            for t in ALL_TYPES:
                p = self._path(sign, t)
                if p.exists():
                    img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.templates[sign, t] = img
        if self.meta_path.exists():
            self.ink_threshold = json.loads(self.meta_path.read_text()).get("ink", DEFAULT_INK)

    @property
    def ready(self) -> bool:
        return any(t == K for _, t in self.templates)

    def missing(self) -> list[str]:
        """Types de pièces sans aucun modèle (dans un sens ou dans l'autre)."""
        return [type_name(t) for t in ALL_TYPES if (1, t) not in self.templates and (-1, t) not in self.templates]

    # ---------- apprentissage ----------
    def learn_cell(self, cell: np.ndarray, piece: int) -> None:
        """Enregistre le modèle d'une case ; `piece` est positive pour le joueur du bas."""
        self.dir.mkdir(parents=True, exist_ok=True)
        sign = 1 if piece > 0 else -1
        ok, buf = cv2.imencode(".png", _inner(_gray(cell), INNER))
        if ok:
            buf.tofile(self._path(sign, abs(piece)))
        self.reload()

    def learn_initial(self, board: np.ndarray) -> int:
        """Apprend depuis la position de départ à l'écran ; renvoie le nombre de modèles."""
        start = Position(START_SFEN).b
        empties, pieces = [], []
        seen = set()
        for i, p in enumerate(start):
            cell = cell_image(board, i)
            (pieces if p else empties).append(ink(cell))
            if p and p not in seen:
                seen.add(p)
                self.learn_cell(cell, p)
        self.ink_threshold = (max(empties) + min(pieces)) / 2
        self.meta_path.write_text(json.dumps({"ink": self.ink_threshold}))
        self.reload()
        return len(seen)

    # ---------- lecture ----------
    def _candidates(self):
        for (sign, t), tpl in self.templates.items():
            yield sign, t, tpl
            if (-sign, t) not in self.templates:  # sens jamais vu : le modèle retourné en tient lieu
                yield -sign, t, cv2.rotate(tpl, cv2.ROTATE_180)

    def classify(self, cell: np.ndarray) -> tuple[int, float]:
        """(pièce, score) d'une case non vide ; pièce = 0 si rien ne ressemble assez."""
        gray = _gray(cell)
        best, best_score = 0, -1.0
        for sign, t, tpl in self._candidates():
            score = float(cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best, best_score = sign * t, score
        return (best, best_score) if best_score >= MIN_SCORE else (0, best_score)

    def read(self, board: np.ndarray) -> tuple[list[int], list[int]]:
        """(81 cases vues de l'écran, indices des cases occupées mais non reconnues)."""
        cells, unknown = [], []
        for i in range(81):
            cell = cell_image(board, i)
            if ink(cell) <= self.ink_threshold:
                cells.append(0)
                continue
            piece, _ = self.classify(cell)
            if not piece:
                unknown.append(i)
            cells.append(piece)
        return cells, unknown


def screen_to_board(cells: list[int], my_side: int) -> list[int]:
    """Cases vues de l'écran -> plateau SFEN, sachant que le joueur du bas tient `my_side`."""
    if my_side == 0:
        return list(cells)
    return [-p for p in reversed(cells)]


def board_to_screen_index(sq: int, my_side: int) -> int:
    return sq if my_side == 0 else 80 - sq
