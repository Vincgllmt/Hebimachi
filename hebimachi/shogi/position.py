"""Règles du shogi : position, coups légaux, notation SFEN/USI.

Plateau : liste de 81 cases, de 9a (index 0) à 1i (index 80), dans l'ordre de lecture du
SFEN. Sente (0) joue vers le haut avec des pièces positives, gote (1) avec des pièces
négatives. Un coup est un entier (voir `make_move`), pour que la recherche reste rapide.
"""
from __future__ import annotations

import random

BLACK, WHITE = 0, 1
P, L, N, S, G, B, R, K = 1, 2, 3, 4, 5, 6, 7, 8
PROMOTION = 8                       # pièce promue = pièce + 8
PP, PL, PN, PS, PB, PR = 9, 10, 11, 12, 14, 15
LETTERS = {P: "P", L: "L", N: "N", S: "S", G: "G", B: "B", R: "R", K: "K"}
FROM_LETTER = {v: k for k, v in LETTERS.items()}
KANJI = {P: "歩", L: "香", N: "桂", S: "銀", G: "金", B: "角", R: "飛", K: "王",
         PP: "と", PL: "杏", PN: "圭", PS: "全", PB: "馬", PR: "龍"}
RANK_KANJI = "一二三四五六七八九"
HAND_ORDER = (R, B, G, S, N, L, P)
START_SFEN = "lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1"
DROP_BASE = 81                      # origine d'un parachutage : 81 + type de pièce
PROMOTE_FLAG = 1 << 14

# Valeurs en centipions (ordres de grandeur classiques du shogi informatique).
VALUE = {P: 90, L: 315, N: 405, S: 495, G: 540, B: 855, R: 990, K: 0,
         PP: 540, PL: 540, PN: 540, PS: 540, PB: 945, PR: 1395}
HAND_VALUE = {t: VALUE[t] * 115 // 100 for t in (P, L, N, S, G, B, R)}

_ORTHO = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAG = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_GOLD = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0))
_STEPS = {P: ((-1, 0),), N: ((-2, -1), (-2, 1)), S: ((-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 1)),
          G: _GOLD, K: _ORTHO + _DIAG, PP: _GOLD, PL: _GOLD, PN: _GOLD, PS: _GOLD,
          PB: _ORTHO, PR: _DIAG}
_RAYS = {L: ((-1, 0),), B: _DIAG, R: _ORTHO, PB: _DIAG, PR: _ORTHO}

# Tables indexées par la pièce signée : l'index négatif de Python tombe en fin de liste.
STEP: list = [()] * 31
RAY: list = [()] * 31
VAL: list = [[0] * 81 for _ in range(31)]   # valeur + position, du point de vue de sente
NEAR: tuple = ([], [])    # NEAR[couleur][case] : (origine, pièces qui attaquent la case d'un pas)
LINES: tuple = ([], [])   # LINES[couleur][case] : (cases en ligne, pièces à longue portée)
ZOBRIST: list = []
ZOBRIST_HAND: list = []
ZOBRIST_SIDE = 0


def _inside(r: int, c: int) -> bool:
    return 0 <= r < 9 and 0 <= c < 9


def _placement(t: int, sq: int) -> int:
    """Bonus de position pour sente : pions avancés, roi à l'abri, pièces mineures au centre."""
    row, col = divmod(sq, 9)
    centre = 4 - abs(col - 4)
    if t == P:
        return (6 - row) * 3
    if t == K:
        return (row - 8) * 25 - centre * 4
    if t in (S, G):
        return centre * 3 + (8 - row)
    if t in (PP, PL, PN, PS):
        return (8 - row) * 4
    return 0


def _build_tables() -> None:
    global ZOBRIST_SIDE
    types = list(_STEPS.keys() | _RAYS.keys())
    for t in types:
        for sign in (1, -1):
            steps, rays = [], []
            for sq in range(81):
                r, c = divmod(sq, 9)
                steps.append(tuple((r + sign * dr) * 9 + c + sign * dc
                                   for dr, dc in _STEPS.get(t, ()) if _inside(r + sign * dr, c + sign * dc)))
                lines = []
                for dr, dc in _RAYS.get(t, ()):
                    line, rr, cc = [], r + sign * dr, c + sign * dc
                    while _inside(rr, cc):
                        line.append(rr * 9 + cc)
                        rr, cc = rr + sign * dr, cc + sign * dc
                    if line:
                        lines.append(tuple(line))
                rays.append(tuple(lines))
            STEP[sign * t], RAY[sign * t] = steps, rays
        for sq in range(81):
            VAL[t][sq] = VALUE[t] + _placement(t, sq)
            VAL[-t][sq] = -VALUE[t] - _placement(t, 80 - sq)
    for color, sign in ((BLACK, 1), (WHITE, -1)):
        for target in range(81):
            near: dict[int, set[int]] = {}
            for t in types:
                for src in range(81):
                    if target in STEP[sign * t][src]:
                        near.setdefault(src, set()).add(sign * t)
            NEAR[color].append(tuple((src, frozenset(ps)) for src, ps in near.items()))
            r, c = divmod(target, 9)
            lines = []
            for dr, dc in _ORTHO + _DIAG:
                line, rr, cc = [], r + dr, c + dc
                while _inside(rr, cc):
                    line.append(rr * 9 + cc)
                    rr, cc = rr + dr, cc + dc
                # Une pièce placée dans cette direction attaque la cible en allant à l'opposé.
                ps = frozenset(sign * t for t, dirs in _RAYS.items()
                               if (-sign * dr, -sign * dc) in dirs)
                if line and ps:
                    lines.append((tuple(line), ps))
            LINES[color].append(tuple(lines))
    rng = random.Random(20261106)
    ZOBRIST.extend([rng.getrandbits(64) for _ in range(31)] for _ in range(81))
    ZOBRIST_HAND.extend([[rng.getrandbits(64) for _ in range(20)] for _ in range(8)] for _ in range(2))
    ZOBRIST_SIDE = rng.getrandbits(64)


_build_tables()


def make_move(frm: int, to: int, promote: bool = False) -> int:
    return frm | to << 7 | (PROMOTE_FLAG if promote else 0)


def make_drop(piece: int, to: int) -> int:
    return (DROP_BASE + piece) | to << 7


def move_from(m: int) -> int:
    return m & 127


def move_to(m: int) -> int:
    return (m >> 7) & 127


def is_drop(m: int) -> bool:
    return m & 127 >= DROP_BASE


def is_promotion(m: int) -> bool:
    return bool(m & PROMOTE_FLAG)


def drop_piece(m: int) -> int:
    return (m & 127) - DROP_BASE


def square_name(sq: int) -> str:
    return f"{9 - sq % 9}{'abcdefghi'[sq // 9]}"


def square_of(name: str) -> int:
    file, rank = int(name[0]), "abcdefghi".index(name[1])
    if not 1 <= file <= 9:
        raise ValueError(f"Case invalide : {name}")
    return rank * 9 + 9 - file


def move_to_usi(m: int) -> str:
    if is_drop(m):
        return f"{LETTERS[drop_piece(m)]}*{square_name(move_to(m))}"
    return square_name(move_from(m)) + square_name(move_to(m)) + ("+" if is_promotion(m) else "")


def move_from_usi(text: str) -> int:
    if text[1] == "*":
        return make_drop(FROM_LETTER[text[0]], square_of(text[2:4]))
    return make_move(square_of(text[0:2]), square_of(text[2:4]), text.endswith("+"))


class Position:
    def __init__(self, sfen: str = START_SFEN):
        self.b = [0] * 81
        self.hand = [[0] * 8, [0] * 8]
        self.side = BLACK
        self.ksq = [-1, -1]
        self.hash = 0
        self.mat = 0
        self.stack: list[tuple[int, int, int, int]] = []
        self.ply = 1
        self.set_sfen(sfen)

    # ---------- SFEN ----------
    def set_sfen(self, sfen: str) -> None:
        parts = sfen.split()
        if len(parts) < 3:
            raise ValueError("SFEN incomplet : plateau, trait et pièces en main sont nécessaires")
        rows = parts[0].split("/")
        if len(rows) != 9:
            raise ValueError("SFEN : il faut 9 rangées")
        board = []
        for row in rows:
            cells: list[int] = []
            promoted = False
            for ch in row:
                if ch == "+":
                    promoted = True
                elif ch.isdigit():
                    cells += [0] * int(ch)
                else:
                    t = FROM_LETTER.get(ch.upper())
                    if t is None:
                        raise ValueError(f"SFEN : pièce inconnue {ch!r}")
                    t += PROMOTION if promoted else 0
                    cells.append(t if ch.isupper() else -t)
                    promoted = False
            if len(cells) != 9:
                raise ValueError(f"SFEN : rangée de {len(cells)} cases")
            board += cells
        hands = [[0] * 8, [0] * 8]
        if parts[2] != "-":
            count = ""
            for ch in parts[2]:
                if ch.isdigit():
                    count += ch
                    continue
                t = FROM_LETTER.get(ch.upper())
                if t is None or t == K:
                    raise ValueError(f"SFEN : pièce en main inconnue {ch!r}")
                hands[BLACK if ch.isupper() else WHITE][t] += int(count or 1)
                count = ""
        side = {"b": BLACK, "w": WHITE}.get(parts[1])
        if side is None:
            raise ValueError("SFEN : le trait doit être b ou w")
        self.load(board, hands, side)
        self.ply = int(parts[3]) if len(parts) > 3 else 1

    def load(self, board: list[int], hands: list[list[int]], side: int) -> None:
        self.b = list(board)
        self.hand = [list(hands[0]), list(hands[1])]
        self.side = side
        self.stack = []
        self.ksq = [self.b.index(K) if K in self.b else -1, self.b.index(-K) if -K in self.b else -1]
        self.hash = ZOBRIST_SIDE if side else 0
        self.mat = 0
        for sq, p in enumerate(self.b):
            if p:
                self.hash ^= ZOBRIST[sq][p]
                self.mat += VAL[p][sq]
        for color in (BLACK, WHITE):
            for t in HAND_ORDER:
                self.hash ^= ZOBRIST_HAND[color][t][self.hand[color][t]]
                self.mat += (1 - 2 * color) * HAND_VALUE[t] * self.hand[color][t]

    def sfen(self) -> str:
        rows = []
        for r in range(9):
            row, empty = "", 0
            for p in self.b[r * 9:r * 9 + 9]:
                if not p:
                    empty += 1
                    continue
                if empty:
                    row += str(empty)
                    empty = 0
                t = abs(p)
                letter = LETTERS[t - PROMOTION if t > K else t]
                row += ("+" if t > K else "") + (letter if p > 0 else letter.lower())
            rows.append(row + (str(empty) if empty else ""))
        hand = ""
        for color in (BLACK, WHITE):
            for t in HAND_ORDER:
                n = self.hand[color][t]
                if n:
                    letter = LETTERS[t] if color == BLACK else LETTERS[t].lower()
                    hand += (str(n) if n > 1 else "") + letter
        return f"{'/'.join(rows)} {'bw'[self.side]} {hand or '-'} {self.ply}"

    def copy(self) -> "Position":
        other = Position.__new__(Position)
        other.b, other.side, other.hash, other.mat = list(self.b), self.side, self.hash, self.mat
        other.hand = [list(self.hand[0]), list(self.hand[1])]
        other.ksq, other.stack, other.ply = list(self.ksq), list(self.stack), self.ply
        return other

    # ---------- jouer ----------
    def do(self, m: int) -> None:
        b, side = self.b, self.side
        sign = 1 - 2 * side
        frm, to = m & 127, (m >> 7) & 127
        h, mat, cap = self.hash, self.mat, 0
        if frm >= DROP_BASE:
            t = frm - DROP_BASE
            hand = self.hand[side]
            h ^= ZOBRIST_HAND[side][t][hand[t]]
            hand[t] -= 1
            h ^= ZOBRIST_HAND[side][t][hand[t]]
            p = sign * t
            mat += VAL[p][to] - sign * HAND_VALUE[t]
        else:
            p = b[frm]
            b[frm] = 0
            h ^= ZOBRIST[frm][p]
            mat -= VAL[p][frm]
            cap = b[to]
            if cap:
                h ^= ZOBRIST[to][cap]
                mat -= VAL[cap][to]
                t = abs(cap)
                if t > K:
                    t -= PROMOTION
                if t != K:
                    hand = self.hand[side]
                    h ^= ZOBRIST_HAND[side][t][hand[t]]
                    hand[t] += 1
                    h ^= ZOBRIST_HAND[side][t][hand[t]]
                    mat += sign * HAND_VALUE[t]
            if m & PROMOTE_FLAG:
                p += sign * PROMOTION
            elif p == sign * K:
                self.ksq[side] = to
            mat += VAL[p][to]
        b[to] = p
        h ^= ZOBRIST[to][p] ^ ZOBRIST_SIDE
        self.stack.append((m, cap, self.hash, self.mat))
        self.hash, self.mat = h, mat
        self.side = 1 - side

    def undo(self) -> None:
        m, cap, self.hash, self.mat = self.stack.pop()
        side = self.side = 1 - self.side
        b = self.b
        sign = 1 - 2 * side
        frm, to = m & 127, (m >> 7) & 127
        if frm >= DROP_BASE:
            self.hand[side][frm - DROP_BASE] += 1
            b[to] = 0
            return
        p = b[to]
        if m & PROMOTE_FLAG:
            p -= sign * PROMOTION
        elif p == sign * K:
            self.ksq[side] = frm
        b[frm] = p
        b[to] = cap
        if cap:
            t = abs(cap)
            if t > K:
                t -= PROMOTION
            if t != K:
                self.hand[side][t] -= 1

    def null_move(self) -> None:
        self.stack.append((0, 0, self.hash, self.mat))
        self.hash ^= ZOBRIST_SIDE
        self.side = 1 - self.side

    def undo_null(self) -> None:
        _, _, self.hash, self.mat = self.stack.pop()
        self.side = 1 - self.side

    # ---------- attaques ----------
    def attacked(self, sq: int, by: int) -> bool:
        b = self.b
        for src, pieces in NEAR[by][sq]:
            if b[src] in pieces:
                return True
        for line, pieces in LINES[by][sq]:
            for s in line:
                v = b[s]
                if v:
                    if v in pieces:
                        return True
                    break
        return False

    def in_check(self, side: int | None = None) -> bool:
        side = self.side if side is None else side
        k = self.ksq[side]
        return k >= 0 and self.attacked(k, 1 - side)

    # ---------- génération ----------
    def pseudo_moves(self, all_promotions: bool = True) -> list[int]:
        """Coups sans tester l'échec au roi ni le mat par parachutage de pion.

        Sans `all_promotions`, on écarte le refus de promouvoir un pion, un fou ou une tour,
        qui n'est presque jamais le bon coup (la recherche y gagne du temps).
        """
        b, side = self.b, self.side
        sign = 1 - 2 * side
        moves: list[int] = []
        add = moves.append
        zone_lo, zone_hi = (0, 27) if side == BLACK else (54, 81)
        for sq in range(81):
            p = b[sq]
            if p * sign <= 0:
                continue
            t = p * sign
            promotable = t <= R and t != G
            from_zone = zone_lo <= sq < zone_hi
            targets = [to for to in STEP[p][sq] if b[to] * sign <= 0]
            for ray in RAY[p][sq]:
                for to in ray:
                    v = b[to]
                    if v * sign > 0:
                        break
                    targets.append(to)
                    if v:
                        break
            base = sq
            for to in targets:
                m = base | to << 7
                if promotable and (from_zone or zone_lo <= to < zone_hi):
                    add(m | PROMOTE_FLAG)
                    rel = to // 9 if side == BLACK else 8 - to // 9
                    if t == P or t == L:
                        forced = rel == 0
                    else:
                        forced = t == N and rel <= 1
                    if not forced and (all_promotions or t in (L, N, S)):
                        add(m)
                else:
                    add(m)
        hand = self.hand[side]
        if any(hand):
            empties = [sq for sq in range(81) if not b[sq]]
            pawn = sign * P
            pawn_files = {sq % 9 for sq in range(81) if b[sq] == pawn} if hand[P] else ()
            for t in HAND_ORDER:
                if not hand[t]:
                    continue
                base = DROP_BASE + t
                for to in empties:
                    if t <= N:
                        rel = to // 9 if side == BLACK else 8 - to // 9
                        if rel < (2 if t == N else 1) or (t == P and to % 9 in pawn_files):
                            continue
                    add(base | to << 7)
        return moves

    def capture_moves(self) -> list[int]:
        """Captures seules, sans parcourir les parachutages : pour la fin de la recherche."""
        b, side = self.b, self.side
        sign = 1 - 2 * side
        zone_lo, zone_hi = (0, 27) if side == BLACK else (54, 81)
        moves: list[int] = []
        for sq in range(81):
            p = b[sq]
            if p * sign <= 0:
                continue
            t = p * sign
            promotable = t <= R and t != G
            targets = [to for to in STEP[p][sq] if b[to] * sign < 0]
            for ray in RAY[p][sq]:
                for to in ray:
                    v = b[to]
                    if v:
                        if v * sign < 0:
                            targets.append(to)
                        break
            for to in targets:
                m = sq | to << 7
                if promotable and (zone_lo <= sq < zone_hi or zone_lo <= to < zone_hi):
                    m |= PROMOTE_FLAG
                moves.append(m)
        return moves

    def is_pawn_drop_mate(self, m: int) -> bool:
        """À appeler après `do(m)` : le coup joué était-il un mat par parachutage de pion ?"""
        return m & 127 == DROP_BASE + P and self.in_check() and not self.has_legal_move()

    def has_legal_move(self) -> bool:
        side = self.side
        for m in self.pseudo_moves():
            self.do(m)
            ok = not self.in_check(side)
            self.undo()
            if ok:
                return True
        return False

    def legal_moves(self) -> list[int]:
        side = self.side
        out = []
        for m in self.pseudo_moves():
            self.do(m)
            if not self.in_check(side) and not self.is_pawn_drop_mate(m):
                out.append(m)
            self.undo()
        return out

    def perft(self, depth: int) -> int:
        if depth == 0:
            return 1
        moves = self.legal_moves()
        if depth == 1:
            return len(moves)
        total = 0
        for m in moves:
            self.do(m)
            total += self.perft(depth - 1)
            self.undo()
        return total


def pieces_in_play(board: list[int]) -> dict[int, int]:
    """Nombre de pièces par type (promues ramenées à leur type de base) sur le plateau."""
    out: dict[int, int] = {}
    for p in board:
        if p:
            t = abs(p)
            t = t - PROMOTION if t > K else t
            out[t] = out.get(t, 0) + 1
    return out


def move_text(pos: Position, m: int) -> str:
    """Notation japonaise lisible, ex. ▲7六歩, ▽5五角打, ▲2三歩成. `pos` est la position avant le coup."""
    to = move_to(m)
    mark = "▲" if pos.side == BLACK else "▽"
    dest = f"{9 - to % 9}{RANK_KANJI[to // 9]}"
    if is_drop(m):
        return f"{mark}{dest}{KANJI[drop_piece(m)]}打"
    t = abs(pos.b[move_from(m)])
    return f"{mark}{dest}{KANJI[t]}{'成' if is_promotion(m) else ''}"


def infer_moves(pos: Position, board: list[int], max_plies: int = 2,
                unknown: frozenset[int] = frozenset()) -> list[int] | None:
    """Coups qui mènent de `pos` au plateau lu à l'écran (liste vide s'il n'a pas changé).

    C'est ce qui évite de lire les pièces en main : en suivant les coups depuis une position
    connue, on sait ce que chacun a capturé. Les cases `unknown` sont occupées par une pièce
    non reconnue : n'importe quelle pièce y convient. Renvoie None si aucune suite ne convient.
    """
    def matches(cells: list[int]) -> bool:
        if not unknown:
            return cells == board
        return all(cells[i] != 0 if i in unknown else cells[i] == board[i] for i in range(81))

    if matches(pos.b):
        return []
    work = pos.copy()

    def search(depth: int) -> list[int] | None:
        for m in work.legal_moves():
            work.do(m)
            found = [] if matches(work.b) else (search(depth - 1) if depth > 1 else None)
            work.undo()
            if found is not None:
                return [m] + found
        return None

    return search(max_plies)
