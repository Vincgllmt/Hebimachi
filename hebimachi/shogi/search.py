"""Recherche du meilleur coup : solveur de mat, puis alpha-bêta à approfondissement itératif.

Le mat d'abord : un mat forcé est la façon la plus sûre de gagner, et le solveur (échecs
seulement côté attaquant) le voit bien plus loin que l'alpha-bêta. En Python pur, la recherche
reste modeste (quelques demi-coups) ; un moteur USI externe (voir usi.py) joue bien plus fort.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from .position import BLACK, DROP_BASE, G, K, P, PROMOTE_FLAG, S, STEP, VALUE, Position

INF = 1_000_000
MATE = 100_000
MATE_BOUND = MATE - 1000
EXACT, LOWER, UPPER = 0, 1, 2
MAX_QDEPTH = 6
MATE_SHARE = 0.3              # part du temps donnée au solveur de mat
MATE_DEPTHS = (1, 3, 5, 7, 9, 11)
CONTEMPT = 30                 # on veut gagner : une répétition compte comme un léger recul
FUTILITY_MARGIN = (0, 200, 450)      # par profondeur restante : gain maximal d'un coup calme
QUIET_LIMIT = (0, 10, 22, 45)        # coups calmes essayés au plus, par profondeur restante


def _ring(sq: int) -> tuple[int, ...]:
    r, c = divmod(sq, 9)
    return tuple(rr * 9 + cc for rr in range(max(0, r - 2), min(9, r + 3))
                 for cc in range(max(0, c - 2), min(9, c + 3)) if (rr, cc) != (r, c))


RING = [_ring(sq) for sq in range(81)]
GUARDS = {G, S, 9, 10, 11, 12}       # or, argent et pièces promues « comme l'or »


class Timeout(Exception):
    pass


@dataclass
class Line:
    move: int
    score: int                 # centipions pour le joueur au trait
    pv: list[int] = field(default_factory=list)
    mate: int | None = None    # demi-coups avant le mat : positif = on mate, négatif = on est maté
    depth: int = 0


def _king_danger(pos: Position, color: int) -> int:
    """Pénalité pour le roi de `color` : attaquants proches, défenseurs absents, pièces en main adverses."""
    k = pos.ksq[color]
    if k < 0:
        return 0
    b = pos.b
    sign = 1 - 2 * color
    danger = 0
    for sq in RING[k]:
        p = b[sq] * sign
        if p > 0:
            if p in GUARDS:
                danger -= 14
        elif p < 0:
            danger += 28 if -p != P else 8
    enemy_hand = pos.hand[1 - color]
    danger += 5 * sum(enemy_hand) + 20 * (enemy_hand[7] + enemy_hand[6] + enemy_hand[5])
    open_near = sum(1 for sq in STEP[K][k] if not b[sq])
    return danger + open_near * (6 + 2 * sum(enemy_hand))


def evaluate(pos: Position) -> int:
    """Évaluation pour le joueur au trait."""
    score = pos.mat - _king_danger(pos, BLACK) + _king_danger(pos, 1)
    return score if pos.side == BLACK else -score


def _piece_value(p: int) -> int:
    return VALUE.get(abs(p), 0)


class Searcher:
    def __init__(self, pos: Position, deadline: float, stop: threading.Event | None = None):
        self.pos = pos
        self.deadline = deadline
        self.stop = stop
        self.nodes = 0
        self.tt: dict[int, tuple[int, int, int, int]] = {}
        self.killers = [[0, 0] for _ in range(128)]
        self.history: dict[int, int] = {}
        self.path: set[int] = set()
        self.root_side = pos.side

    def _tick(self) -> None:
        self.nodes += 1
        if not self.nodes & 511 and (time.monotonic() > self.deadline
                                     or (self.stop is not None and self.stop.is_set())):
            raise Timeout

    def _ordered(self, moves: list[int], tt_move: int, ply: int) -> list[int]:
        b = self.pos.b
        k1, k2 = self.killers[ply]
        history = self.history
        scored = []
        for m in moves:
            if m == tt_move:
                s = 10_000_000
            else:
                to = (m >> 7) & 127
                cap = b[to]
                if cap:
                    frm = m & 127
                    attacker = 0 if frm >= DROP_BASE else _piece_value(b[frm])
                    s = 1_000_000 + 10 * _piece_value(cap) - attacker // 10
                elif m & PROMOTE_FLAG:
                    s = 900_000
                elif m == k1 or m == k2:
                    s = 800_000
                else:
                    s = history.get(m, 0)
            scored.append((s, m))
        scored.sort(reverse=True)
        return [m for _, m in scored]

    def quiesce(self, alpha: int, beta: int, ply: int, depth: int) -> int:
        self._tick()
        pos = self.pos
        stand = evaluate(pos)
        if stand >= beta or depth >= MAX_QDEPTH:
            return stand
        alpha = max(alpha, stand)
        b, side = pos.b, pos.side
        captures = pos.capture_moves()
        captures.sort(key=lambda m: -_piece_value(b[(m >> 7) & 127]))
        for m in captures:
            pos.do(m)
            if pos.in_check(side):
                pos.undo()
                continue
            try:
                score = -self.quiesce(-beta, -alpha, ply + 1, depth + 1)
            finally:
                pos.undo()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def negamax(self, depth: int, alpha: int, beta: int, ply: int, allow_null: bool = True) -> int:
        pos = self.pos
        if pos.hash in self.path:
            return -CONTEMPT if pos.side == self.root_side else CONTEMPT
        in_check = pos.in_check()
        if in_check and ply < 60:
            depth += 1
        if depth <= 0:
            return self.quiesce(alpha, beta, ply, 0)
        self._tick()
        key = pos.hash
        entry = self.tt.get(key)
        tt_move = 0
        if entry is not None:
            e_depth, e_score, e_flag, tt_move = entry
            if e_depth >= depth and ply > 0:
                if e_flag == EXACT or (e_flag == LOWER and e_score >= beta) or (e_flag == UPPER and e_score <= alpha):
                    return e_score

        if allow_null and depth >= 3 and not in_check and beta < MATE_BOUND and evaluate(pos) >= beta:
            pos.null_move()
            try:
                score = -self.negamax(depth - 3, -beta, -beta + 1, ply + 1, False)
            finally:
                pos.undo_null()
            if score >= beta:
                return score

        side = pos.side
        best, best_move, legal, quiets = -INF, 0, 0, 0
        start_alpha = alpha
        # Près des feuilles, un coup calme ne rattrapera pas un gros retard : on ne le joue pas.
        prunable = not in_check and depth < len(QUIET_LIMIT) and abs(alpha) < MATE_BOUND
        hopeless = prunable and depth < len(FUTILITY_MARGIN) and evaluate(pos) + FUTILITY_MARGIN[depth] <= alpha
        self.path.add(key)
        try:
            for m in self._ordered(pos.pseudo_moves(False), tt_move, min(ply, 127)):
                quiet = not pos.b[(m >> 7) & 127] and not m & PROMOTE_FLAG
                pos.do(m)
                if pos.in_check(side) or pos.is_pawn_drop_mate(m):
                    pos.undo()
                    continue
                legal += 1
                if quiet:
                    quiets += 1
                    if legal > 1 and prunable and (hopeless or quiets > QUIET_LIMIT[depth]) and not pos.in_check():
                        pos.undo()
                        continue
                try:
                    if legal == 1:
                        score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
                    else:
                        reduce = 0
                        if quiet and depth >= 3 and legal > 4 and not in_check:
                            reduce = 2 if legal > 16 and depth >= 5 else 1
                        score = -self.negamax(depth - 1 - reduce, -alpha - 1, -alpha, ply + 1)
                        if alpha < score and (reduce or score < beta):
                            score = -self.negamax(depth - 1, -beta, -alpha, ply + 1)
                finally:
                    pos.undo()
                if score > best:
                    best, best_move = score, m
                alpha = max(alpha, score)
                if alpha >= beta:
                    if quiet:
                        killers = self.killers[min(ply, 127)]
                        if killers[0] != m:
                            killers[0], killers[1] = m, killers[0]
                        self.history[m] = self.history.get(m, 0) + depth * depth
                    break
        finally:
            self.path.discard(key)
        if legal == 0:
            return -MATE + ply  # sans coup légal, on a perdu (même sans échec)
        flag = LOWER if best >= beta else UPPER if best <= start_alpha else EXACT
        self.tt[key] = (depth, best, flag, best_move)
        return best

    def pv_after(self, first: int, limit: int = 12) -> list[int]:
        """Variante principale lue dans la table, en vérifiant la légalité de chaque coup."""
        pos = self.pos
        pv = [first]
        pos.do(first)
        played, seen = 1, {pos.hash}
        while len(pv) < limit:
            entry = self.tt.get(pos.hash)
            if entry is None or not entry[3] or entry[3] not in pos.legal_moves():
                break
            pv.append(entry[3])
            pos.do(entry[3])
            played += 1
            if pos.hash in seen:
                break
            seen.add(pos.hash)
        for _ in range(played):
            pos.undo()
        return pv


# ---------- solveur de mat ----------
class MateSolver:
    """Cherche un mat par échecs successifs : l'attaquant ne joue que des échecs."""

    def __init__(self, pos: Position, deadline: float, stop: threading.Event | None = None):
        self.pos = pos
        self.deadline = deadline
        self.stop = stop
        self.nodes = 0
        self.tt: dict[tuple[int, int], list[int] | None] = {}

    def _tick(self) -> None:
        self.nodes += 1
        if not self.nodes & 127 and (time.monotonic() > self.deadline
                                     or (self.stop is not None and self.stop.is_set())):
            raise Timeout

    def _checks(self) -> list[int]:
        pos = self.pos
        side = pos.side
        king = pos.ksq[1 - side]
        kr, kc = divmod(king, 9)
        found = []
        for m in pos.pseudo_moves():
            pos.do(m)
            if pos.in_check() and not pos.in_check(side) and not pos.is_pawn_drop_mate(m):
                to = (m >> 7) & 127
                near = max(abs(to // 9 - kr), abs(to % 9 - kc))
                found.append((near, -_piece_value(pos.stack[-1][1]), m))
            pos.undo()
        found.sort()
        return [m for _, _, m in found]

    def attack(self, depth: int) -> list[int] | None:
        """Séquence de mat en `depth` demi-coups au plus (impair), ou None."""
        self._tick()
        pos = self.pos
        key = (pos.hash, depth)
        if key in self.tt:
            return self.tt[key]
        self.tt[key] = None  # coupe aussi les boucles de répétition
        for m in self._checks():
            pos.do(m)
            try:
                reply = self.defend(depth - 1)
            finally:
                pos.undo()
            if reply is not None:
                self.tt[key] = [m] + reply
                return self.tt[key]
        return None

    def defend(self, depth: int) -> list[int] | None:
        """Après un échec : renvoie la plus longue défense si toutes sont matées, sinon None."""
        self._tick()
        pos = self.pos
        evasions = pos.legal_moves()
        if not evasions:
            return []
        if depth <= 0:
            return None
        longest: list[int] | None = None
        for m in evasions:
            pos.do(m)
            try:
                line = self.attack(depth - 1)
            finally:
                pos.undo()
            if line is None:
                return None
            if longest is None or len(line) + 1 > len(longest):
                longest = [m] + line
        return longest


def find_mate(pos: Position, seconds: float = 1.0, max_plies: int = MATE_DEPTHS[-1],
              stop: threading.Event | None = None) -> list[int] | None:
    """Plus court mat forcé trouvé dans le temps imparti, sous forme de variante."""
    if pos.ksq[1 - pos.side] < 0 or pos.in_check():
        return None
    work = pos.copy()
    solver = MateSolver(work, time.monotonic() + seconds, stop)
    try:
        for depth in MATE_DEPTHS:
            if depth > max_plies:
                break
            line = solver.attack(depth)
            if line is not None:
                return line
    except Timeout:
        while len(work.stack) > len(pos.stack):
            work.undo()
    return None


def _line_of(move: int, score: int, searcher: Searcher, depth: int) -> Line:
    mate = None
    if score > MATE_BOUND:
        mate = MATE - score
    elif score < -MATE_BOUND:
        mate = -(MATE + score)
    return Line(move, score, searcher.pv_after(move), mate, depth)


def analyze(pos: Position, seconds: float = 3.0, multipv: int = 3,
            stop: threading.Event | None = None, on_progress=None) -> list[Line]:
    """Meilleurs coups de la position, du meilleur au moins bon. Vide si aucun coup légal."""
    work = pos.copy()
    work.stack = []
    root = work.legal_moves()
    if not root:
        return []
    start = time.monotonic()
    mate = find_mate(work, seconds * MATE_SHARE, stop=stop)
    if mate:
        line = Line(mate[0], MATE - len(mate), mate, len(mate), len(mate))
        if on_progress:
            on_progress([line])
        return [line]

    searcher = Searcher(work, start + seconds, stop)
    results: list[Line] = []
    searcher.path.add(work.hash)
    ordered = searcher._ordered(root, 0, 0)
    base = len(work.stack)
    for depth in range(1, 40):
        scores: dict[int, int] = {}
        try:
            for m in ordered:
                alpha = -INF
                if len(scores) >= multipv:
                    alpha = sorted(scores.values(), reverse=True)[multipv - 1]
                work.do(m)
                try:
                    score = -searcher.negamax(depth - 1, -INF, -alpha, 1)
                finally:
                    work.undo()
                scores[m] = score
        except Timeout:
            while len(work.stack) > base:
                work.undo()
            break
        ordered = sorted(ordered, key=lambda m: -scores[m])
        results = [_line_of(m, scores[m], searcher, depth) for m in ordered[:multipv]]
        if on_progress:
            on_progress(results)
        if time.monotonic() - start > seconds * 0.6 or abs(scores[ordered[0]]) > MATE_BOUND:
            break
    if not results:  # même la profondeur 1 n'a pas pu finir
        results = [Line(m, 0, [m], None, 0) for m in ordered[:multipv]]
    return results
