"""Règles du shogi (validées par perft), recherche du meilleur coup et client USI."""
import sys
import time
from pathlib import Path

import pytest

from hebimachi.shogi.position import (BLACK, G, K, P, R, START_SFEN, WHITE, Position, infer_moves,
                                      make_drop, move_from_usi, move_text, move_to_usi, pieces_in_play,
                                      square_of)
from hebimachi.shogi.search import analyze, evaluate, find_mate
from hebimachi.shogi.usi import UsiEngine, UsiError, parse_info

MATSURI = "l6nl/5+P1gk/2np1S3/p1p4Pp/3P2Sp1/1PPb2P1P/P5GS1/R8/LN4bKL w RGgsn5p 1"
MAX_MOVES = "R8/2K1S1SSk/4B4/9/9/9/9/9/1L1L1L3 b RBGSNLP3g3n17p 1"


def legal_usi(sfen):
    return {move_to_usi(m) for m in Position(sfen).legal_moves()}


# ---------- règles ----------
@pytest.mark.parametrize("depth, nodes", [(1, 30), (2, 900), (3, 25470)])
def test_perft_start(depth, nodes):
    assert Position().perft(depth) == nodes


@pytest.mark.parametrize("depth, nodes", [(1, 207), (2, 28684)])
def test_perft_with_drops_and_promotions(depth, nodes):
    assert Position(MATSURI).perft(depth) == nodes


def test_position_with_most_legal_moves():
    assert Position(MAX_MOVES).perft(1) == 593


@pytest.mark.parametrize("sfen", [START_SFEN, MATSURI, MAX_MOVES])
def test_sfen_round_trip(sfen):
    assert Position(sfen).sfen() == sfen


def test_do_undo_restores_everything():
    pos = Position(MATSURI)
    before = (list(pos.b), [list(h) for h in pos.hand], pos.hash, pos.mat, pos.side, list(pos.ksq))
    for m in pos.legal_moves():
        pos.do(m)
        pos.undo()
        assert before == (pos.b, pos.hand, pos.hash, pos.mat, pos.side, pos.ksq)


def test_incremental_hash_and_material_match_a_fresh_position():
    pos = Position(MATSURI)
    for m in pos.legal_moves():
        pos.do(m)
        fresh = Position(pos.sfen())
        assert (pos.hash, pos.mat) == (fresh.hash, fresh.mat), move_to_usi(m)
        pos.undo()


def test_nifu_forbids_second_pawn_on_a_file():
    moves = legal_usi("4k4/9/9/9/9/9/4P4/9/4K4 b P 1")
    assert "P*5e" not in moves and "P*4e" in moves


def test_pieces_without_a_future_cannot_be_dropped():
    moves = legal_usi("4k4/9/9/9/9/9/9/9/4K4 b PLN 1")
    assert "P*1a" not in moves and "L*1a" not in moves
    assert "N*1a" not in moves and "N*1b" not in moves and "N*1c" in moves


def test_pawn_drop_mate_is_forbidden_but_pawn_drop_check_is_not():
    # Le roi en 1a n'a plus de fuite : le pion en 1b materait, c'est interdit.
    assert "P*1b" not in legal_usi("7nk/9/7G1/9/9/9/9/9/4K4 b P 1")
    # Sans le cavalier, 2a reste libre : le même échec est permis.
    assert "P*1b" in legal_usi("8k/9/9/9/9/9/9/9/4K4 b P 1")


def test_forced_promotion_on_last_rank():
    moves = legal_usi("4k4/P8/9/9/9/9/9/9/4K4 b - 1")
    assert "9b9a+" in moves and "9b9a" not in moves
    moves = legal_usi("4k4/9/N8/9/9/9/9/9/4K4 b - 1")
    assert "9c8a+" in moves and "9c8a" not in moves


def test_pinned_piece_cannot_move_away():
    moves = legal_usi("4r3k/9/9/9/9/9/9/4G4/4K4 b - 1")
    assert "5h4h" not in moves and "5h5g" in moves


def test_captured_piece_goes_to_hand_unpromoted():
    pos = Position("4k4/9/9/9/4+r4/4P4/9/9/4K4 b - 1")
    pos.do(move_from_usi("5f5e"))
    assert pos.hand[BLACK][R] == 1


def test_tsume_position_without_attacker_king():
    assert len(Position("8k/9/9/9/9/9/9/9/9 b G 1").legal_moves()) > 0


def test_move_text():
    pos = Position()
    assert move_text(pos, move_from_usi("7g7f")) == "▲7六歩"
    assert move_text(Position("8k/9/9/9/9/9/9/9/8K b G 1"), make_drop(G, square_of("5e"))) == "▲5五金打"
    assert move_text(Position("8k/9/9/1P7/9/9/9/9/8K b - 1"), move_from_usi("8d8c+")) == "▲8三歩成"


def test_infer_moves_recovers_captures_for_both_sides():
    pos = Position("4k4/9/9/4p4/4P4/9/9/9/4K4 b - 1")
    after = pos.copy()
    after.do(move_from_usi("5e5d"))
    after.do(move_from_usi("5a5b"))
    assert infer_moves(pos, after.b) == [move_from_usi("5e5d"), move_from_usi("5a5b")]
    assert infer_moves(pos, pos.b) == []
    unreachable = list(pos.b)
    unreachable[0] = K
    assert infer_moves(pos, unreachable) is None


def test_pieces_in_play_counts_promoted_as_base():
    assert pieces_in_play(Position("4k4/9/9/9/4+P4/9/9/9/4K4 b - 1").b) == {K: 2, P: 1}


# ---------- recherche ----------
def test_evaluation_is_symmetric():
    assert evaluate(Position()) == evaluate(Position(START_SFEN.replace(" b ", " w ")))


def test_finds_mate_in_one():
    lines = analyze(Position("7nk/9/7G1/9/9/9/9/9/4K4 b G 1"), seconds=1)
    assert lines[0].mate == 1
    pos = Position("7nk/9/7G1/9/9/9/9/9/4K4 b G 1")
    pos.do(lines[0].move)
    assert pos.in_check() and not pos.legal_moves()


def forced_mate(pos, plies):
    """Vérification indépendante du solveur : le camp au trait force-t-il le mat, toutes défenses vues ?"""
    if plies <= 0:
        return False
    for m in pos.legal_moves():
        pos.do(m)
        replies = pos.legal_moves()
        won = pos.in_check() and all(_loses(pos, r, plies - 2) for r in replies)
        pos.undo()
        if won:
            return True
    return False


def _loses(pos, reply, plies):
    pos.do(reply)
    lost = forced_mate(pos, plies)
    pos.undo()
    return lost


@pytest.mark.parametrize("sfen, plies", [
    ("7nl/7Gk/9/7P1/9/9/9/9/4K4 b G 1", 1),
    ("7nk/9/8P/9/9/9/9/9/4K4 b 2G 1", 3),
    ("8k/9/7P1/9/9/9/9/9/4K4 b 2GS 1", 5),
])
def test_mate_solver_finds_forced_mates(sfen, plies):
    line = find_mate(Position(sfen), seconds=5)
    assert line is not None and len(line) <= plies
    pos = Position(sfen)
    for m in line:
        assert m in pos.legal_moves()
        pos.do(m)
    assert pos.in_check() and not pos.legal_moves()
    assert forced_mate(Position(sfen), len(line))


def test_no_mate_in_the_opening():
    assert find_mate(Position(), seconds=0.2) is None


def test_takes_a_free_rook():
    lines = analyze(Position("4k4/9/9/4r4/4P4/9/9/9/4K4 b - 1"), seconds=1)
    assert move_to_usi(lines[0].move) == "5e5d"


def test_does_not_lose_a_piece_to_a_defended_pawn():
    lines = analyze(Position("4k4/9/4p4/4p4/9/4R4/9/9/4K4 b - 1"), seconds=1)
    assert move_to_usi(lines[0].move) != "5f5d"


def test_escapes_a_threat_on_its_rook():
    # Le fou adverse attaque la tour : elle doit partir (ou prendre), pas rester en prise.
    sfen = "4k4/9/9/9/9/2b6/9/4R4/8K b - 1"
    lines = analyze(Position(sfen), seconds=1)
    pos = Position(sfen)
    pos.do(lines[0].move)
    assert lines[0].score > 0 and not pos.attacked(next(i for i, p in enumerate(pos.b) if p == R), WHITE)


def test_analysis_respects_the_time_limit_and_returns_ranked_moves():
    start = time.monotonic()
    lines = analyze(Position(), seconds=1, multipv=3)
    assert time.monotonic() - start < 2.5
    assert len(lines) == 3 and lines[0].score >= lines[1].score >= lines[2].score
    assert len({line.move for line in lines}) == 3
    assert lines[0].depth >= 3


def test_white_to_move_is_handled():
    lines = analyze(Position("4k4/9/9/4p4/4R4/9/9/9/4K4 w - 1"), seconds=1)
    assert move_to_usi(lines[0].move) == "5d5e"
    assert Position("4k4/9/9/4p4/4R4/9/9/9/4K4 w - 1").side == WHITE


def test_no_legal_move_returns_nothing():
    assert analyze(Position("8k/7GG/9/9/7R1/9/9/9/8K w - 1"), seconds=0.2) == []


# ---------- USI ----------
FAKE = [sys.executable, str(Path(__file__).parent / "fake_usi.py")]


def test_parse_info_scores():
    rank, line = parse_info("info depth 12 seldepth 18 multipv 2 score cp -35 nodes 1 pv 7g7f 3c3d")
    assert (rank, line.score, line.depth, len(line.pv)) == (2, -35, 12, 2)
    assert parse_info("info depth 3 multipv 1 score mate 5 pv 7g7f")[1].mate == 5
    assert parse_info("info depth 3 score mate -4 pv 7g7f")[1].mate == -4
    assert parse_info("info depth 3 score mate + pv 7g7f 3c3d 2g2f")[1].mate == 3
    assert parse_info("info string hello") is None


def test_usi_engine_round_trip():
    engine = UsiEngine(FAKE)
    try:
        assert engine.name == "FauxMoteur 1.0"
        lines = engine.analyze(START_SFEN, seconds=0.1, multipv=2)
        assert move_to_usi(lines[0].move) == "7g7f" and lines[0].score == 55
        assert lines[1].mate == -3
    finally:
        engine.close()


def test_usi_missing_engine_is_a_clear_error():
    with pytest.raises(UsiError):
        UsiEngine("C:/nexiste/pas/moteur.exe")
