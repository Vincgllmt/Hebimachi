"""Suivi d'une partie : position courante, coups joués, synchronisation avec l'écran."""
from __future__ import annotations

from dataclasses import dataclass, field

from .position import (B, BLACK, G, HAND_ORDER, L, N, P, R, S, START_SFEN, Position, drop_piece,
                       infer_moves, is_drop, is_promotion, move_from, move_text, move_to,
                       pieces_in_play, square_name)

FULL_SET = {P: 18, L: 4, N: 4, S: 4, G: 4, B: 2, R: 2}
PIECE_NAMES = {1: "Pion", 2: "Lance", 3: "Cavalier", 4: "Argent", 5: "Or", 6: "Fou", 7: "Tour", 8: "Roi",
               9: "Tokin", 10: "Lance promue", 11: "Cavalier promu", 12: "Argent promu",
               14: "Cheval-dragon", 15: "Roi-dragon"}


def describe(pos: Position, m: int) -> str:
    """Coup en français, ex. « Pion 7g → 7f » ou « Parachutez Or en 5e ». `pos` est la position avant."""
    to = square_name(move_to(m))
    if is_drop(m):
        return f"Parachutez {PIECE_NAMES[drop_piece(m)]} en {to}"
    name = PIECE_NAMES[abs(pos.b[move_from(m)])]
    capture = pos.b[move_to(m)]
    text = f"{name} {square_name(move_from(m))} → {to}"
    if capture:
        text += f" (prend {PIECE_NAMES[abs(capture)]})"
    return text + (", promotion" if is_promotion(m) else "")


@dataclass
class Sync:
    """Résultat de la comparaison entre la position suivie et l'écran."""
    state: str                         # same, moved, new_game ou lost
    moves: list[str] = field(default_factory=list)  # coups reconnus, en notation japonaise


class Session:
    def __init__(self, my_side: int = BLACK):
        self.my_side = my_side
        self.pos = Position()
        self.played: list[int] = []

    @property
    def my_turn(self) -> bool:
        return self.pos.side == self.my_side

    def reset(self, sfen: str = START_SFEN) -> None:
        self.pos = Position(sfen)
        self.played = []

    def play(self, m: int) -> None:
        self.pos.do(m)
        self.played.append(m)

    def undo(self) -> bool:
        if not self.played:
            return False
        self.played.pop()
        self.pos.undo()
        return True

    def sync(self, board: list[int], unknown: frozenset[int] = frozenset()) -> Sync:
        """Rattrape l'écran : retrouve les coups joués, donc aussi les captures et les pièces en main.

        Les cases `unknown` (pièce non reconnue) sont déduites des coups légaux : une fois la
        synchronisation faite, `pos.b` dit quelle pièce s'y trouve.
        """
        moves = infer_moves(self.pos, board, unknown=unknown)
        if moves is None:
            if not unknown and board == Position().b:
                self.reset()
                return Sync("new_game")
            return Sync("lost")
        if not moves:
            return Sync("same")
        names = []
        for m in moves:
            names.append(move_text(self.pos, m))
            self.play(m)
        return Sync("moved", names)

    def adopt(self, board: list[int]) -> None:
        """Repart de l'écran quand le suivi est perdu (ou pour un problème de tsume).

        Ce qui manque au plateau est en main. On garde ce que l'on savait déjà, et le reste
        va à l'adversaire : c'est la supposition prudente, et la convention des tsume.
        """
        on_board = pieces_in_play(board)
        hands = [[0] * 8, [0] * 8]
        for t in HAND_ORDER:
            left = FULL_SET[t] - on_board.get(t, 0)
            for color in (self.my_side, 1 - self.my_side):
                kept = min(left, self.pos.hand[color][t])
                hands[color][t] = kept
                left -= kept
            hands[1 - self.my_side][t] += max(0, left)
        self.pos = Position()
        self.pos.load(board, hands, self.my_side)
        self.played = []

    def set_piece(self, sq: int, piece: int) -> None:
        """Correction à la main d'une case ; l'historique repart de cette position."""
        board = list(self.pos.b)
        board[sq] = piece
        self._reload(board, self.pos.hand)

    def change_hand(self, color: int, t: int, delta: int) -> None:
        hands = [list(self.pos.hand[0]), list(self.pos.hand[1])]
        hands[color][t] = max(0, hands[color][t] + delta)
        self._reload(self.pos.b, hands)

    def _reload(self, board: list[int], hands: list[list[int]]) -> None:
        side = self.pos.side
        self.pos = Position()
        self.pos.load(board, hands, side)
        self.played = []


def side_name(side: int) -> str:
    return "Sente" if side == BLACK else "Gote"
