"""Lecture du plateau sur des images de synthèse (rendu proche d'un plateau de jeu).

Ces tests valident la chaîne redressement -> apprentissage -> lecture -> suivi des coups.
Ils ne remplacent pas une capture du vrai jeu.
"""
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from hebimachi.shogi.position import BLACK, KANJI, PP, R, START_SFEN, WHITE, Position, move_from_usi
from hebimachi.shogi.session import Session, describe
from hebimachi.shogi.vision import (BoardReader, board_to_screen_index, cell_image, ink, rectify,
                                    screen_to_board, type_name)

FONT_PATH = Path("C:/Windows/Fonts/msgothic.ttc")
pytestmark = pytest.mark.skipif(not FONT_PATH.exists(), reason="police japonaise absente")
CORNERS = [(200, 120), (920, 120), (920, 840), (200, 840)]


def render(board, cell=80, noise=4, seed=1):
    """Plateau de bois avec pièces pentagonales ; `board` est vu de l'écran (positif = en bas)."""
    rng = np.random.default_rng(seed)
    font = ImageFont.truetype(str(FONT_PATH), int(cell * 0.62))
    side = cell * 9
    wood = np.full((side, side, 3), (120, 180, 220), np.uint8)
    wood = np.clip(wood + rng.normal(0, noise, wood.shape), 0, 255).astype(np.uint8)
    img = Image.fromarray(wood)
    draw = ImageDraw.Draw(img)
    for i in range(10):
        draw.line([(i * cell, 0), (i * cell, side)], fill=(40, 40, 40), width=2)
        draw.line([(0, i * cell), (side, i * cell)], fill=(40, 40, 40), width=2)
    for i, p in enumerate(board):
        if not p:
            continue
        tile = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
        d = ImageDraw.Draw(tile)
        m = cell * 0.1
        d.polygon([(cell / 2, m / 2), (cell - m, cell * 0.25), (cell - m / 2, cell - m / 2),
                   (m / 2, cell - m / 2), (m, cell * 0.25)], fill=(150, 215, 240, 255), outline=(30, 60, 90, 255))
        ink_color = (20, 20, 170, 255) if abs(p) > 8 else (15, 15, 15, 255)
        d.text((cell / 2, cell * 0.56), KANJI[abs(p)], font=font, fill=ink_color, anchor="mm")
        if p < 0:
            tile = tile.rotate(180)
        r, c = divmod(i, 9)
        img.paste(tile, (c * cell, r * cell), tile)
    return np.array(img)


def on_screen(board_img, offset=(200, 120), size=(1080, 1920)):
    """Pose le plateau sur un écran 1080p, avec du décor autour."""
    screen = np.full((*size, 3), (40, 90, 40), np.uint8)
    h, w = board_img.shape[:2]
    screen[offset[1]:offset[1] + h, offset[0]:offset[0] + w] = board_img
    return screen


@pytest.fixture()
def reader(tmp_path):
    r = BoardReader(tmp_path)
    r.learn_initial(rectify(on_screen(render(Position().b)), CORNERS))
    return r


def read_board(reader, cells, **kw):
    return reader.read(rectify(on_screen(render(cells, **kw)), CORNERS))


def test_start_position_teaches_every_unpromoted_piece(reader):
    assert reader.ready
    assert set(reader.missing()) == {"+P", "+L", "+N", "+S", "+B", "+R"}


def test_reads_the_start_position(reader):
    cells, unknown = read_board(reader, Position().b, seed=2)
    assert unknown == [] and cells == Position().b


def test_reads_a_midgame_position_with_other_noise(reader):
    board = Position("ln1g3nl/1r1s1kg2/p2pppspp/2p3p2/1p5P1/2P1P4/PPSP1PP1P/2G1K2R1/LN3GSNL b Bb 1").b
    cells, unknown = read_board(reader, board, seed=7, noise=7)
    assert unknown == [] and cells == board


def test_ink_separates_pieces_from_empty_squares(reader):
    board = rectify(on_screen(render(Position().b)), CORNERS)
    assert ink(cell_image(board, 40)) < reader.ink_threshold < ink(cell_image(board, 0))


def test_reads_a_tilted_board(tmp_path):
    """Vue inclinée : le plateau en trapèze est redressé par ses quatre coins."""
    straight = on_screen(render(Position().b))
    src = np.array(CORNERS, np.float32)
    dst = np.array([(280, 160), (840, 160), (980, 820), (140, 820)], np.float32)
    warp = lambda img: cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (1920, 1080))  # noqa: E731
    reader = BoardReader(tmp_path)
    corners = [tuple(map(int, p)) for p in dst]
    reader.learn_initial(rectify(warp(straight), corners))
    board = Position("lnsgkgsnl/1r5b1/ppppppppp/9/9/2P6/PP1PPPPPP/1B5R1/LNSGKGSNL w - 2").b
    cells, unknown = reader.read(rectify(warp(on_screen(render(board, seed=3))), corners))
    assert unknown == [] and cells == board


def test_unlearned_promoted_piece_is_flagged_then_learned(reader):
    board = Position("4k4/9/9/9/4+P4/9/9/9/4K4 b - 1").b
    img = rectify(on_screen(render(board)), CORNERS)
    cells, unknown = reader.read(img)
    assert unknown == [40] or cells[40] != PP   # jamais une lecture sûre d'une pièce inconnue
    reader.learn_cell(cell_image(img, 40), PP)
    cells, unknown = reader.read(img)
    assert unknown == [] and cells[40] == PP


def test_opponent_piece_uses_a_rotated_model_when_never_seen(reader):
    board = Position("4k4/9/9/9/4+P4/9/9/9/4K4 b - 1").b
    img = rectify(on_screen(render(board)), CORNERS)
    reader.learn_cell(cell_image(img, 40), PP)
    flipped = rectify(on_screen(render(Position("4k4/9/9/9/4+p4/9/9/9/4K4 b - 1").b)), CORNERS)
    cells, unknown = reader.read(flipped)
    assert unknown == [] and cells[40] == -PP


def test_models_survive_a_restart(reader, tmp_path):
    again = BoardReader(tmp_path)
    assert again.ready and again.ink_threshold == reader.ink_threshold
    assert type_name(PP) == "+P"


def test_gote_view_is_converted_to_sfen_board():
    start = Position().b
    assert screen_to_board(start, BLACK) == start
    assert screen_to_board(start, WHITE) == start  # la position de départ est symétrique
    pos = Position("4k4/9/9/9/9/9/9/9/4K3R b - 1")
    screen = [-p for p in reversed(pos.b)]
    assert screen_to_board(screen, WHITE) == pos.b
    assert board_to_screen_index(0, WHITE) == 80 and board_to_screen_index(5, BLACK) == 5


# ---------- suivi de partie ----------
def test_full_loop_from_screen_to_advice(reader):
    """Écran -> plateau -> coups retrouvés -> pièces en main connues sans les lire."""
    session = Session(BLACK)
    pos = Position()
    for usi in ("7g7f", "3c3d", "8h2b+", "3a2b"):   # échange des fous, dont un promu jamais vu
        pos.do(move_from_usi(usi))
        cells, unknown = read_board(reader, pos.b)
        result = session.sync(screen_to_board(cells, BLACK), frozenset(unknown))
        assert result.state == "moved"
    assert session.pos.sfen().split()[:3] == pos.sfen().split()[:3]
    assert session.pos.hand[BLACK][6] == 1 and session.pos.hand[WHITE][6] == 1


def test_unknown_piece_is_deduced_from_the_legal_move():
    """La promotion du fou se déduit : c'est le seul coup légal qui explique les cases connues."""
    session = Session(BLACK)
    for usi in ("7g7f", "3c3d"):
        session.play(move_from_usi(usi))
    target = Position(session.pos.sfen())
    target.do(move_from_usi("8h2b+"))
    board = list(target.b)
    board[16] = 0  # 2b : occupée, mais illisible
    assert session.sync(board, frozenset({16})).state == "moved"
    assert session.pos.b[16] == 14


def test_unknown_square_cannot_be_empty_after_the_move():
    session = Session(BLACK)
    board = list(Position().b)
    board[40] = 0
    assert session.sync(board, frozenset({40})).state == "lost"


def test_sync_handles_two_plies_between_captures():
    session = Session(BLACK)
    pos = Position()
    pos.do(move_from_usi("7g7f"))
    pos.do(move_from_usi("3c3d"))
    result = session.sync(pos.b)
    assert result.state == "moved" and len(result.moves) == 2 and session.my_turn


def test_sync_keeps_the_position_on_an_impossible_board():
    session = Session(BLACK)
    board = list(Position().b)
    board[40] = R
    before = session.pos.sfen()
    assert session.sync(board).state == "lost" and session.pos.sfen() == before


def test_sync_detects_a_new_game():
    session = Session(BLACK)
    for usi in ("7g7f", "3c3d", "2g2f", "8c8d", "2f2e", "8d8e"):
        session.play(move_from_usi(usi))
    assert session.sync(Position().b).state == "new_game"
    assert session.pos.sfen() == START_SFEN


def test_adopt_keeps_known_hands_and_gives_the_rest_to_the_opponent():
    session = Session(BLACK)
    session.reset("4k4/9/9/9/9/9/9/9/4K4 b 2G 1")
    session.adopt(Position("4k4/9/9/9/9/9/9/9/4K4 b - 1").b)
    assert session.pos.hand[BLACK][5] == 2 and session.pos.hand[WHITE][5] == 2
    assert session.pos.hand[WHITE][1] == 18 and session.my_turn


def test_manual_corrections():
    session = Session(BLACK)
    session.set_piece(40, 1)
    session.change_hand(BLACK, 5, 2)
    session.change_hand(BLACK, 5, -1)
    assert session.pos.b[40] == 1 and session.pos.hand[BLACK][5] == 1
    assert session.pos.sfen() == Position(session.pos.sfen()).sfen()


def test_undo_and_describe():
    session = Session(BLACK)
    m = move_from_usi("7g7f")
    assert describe(session.pos, m) == "Pion 7g → 7f"
    session.play(m)
    assert session.undo() and not session.undo()
    pos = Position("4k4/9/9/4r4/4P4/9/9/9/4K4 b G 1")
    assert describe(pos, move_from_usi("5e5d")) == "Pion 5e → 5d (prend Tour)"
    assert describe(pos, move_from_usi("G*1a")) == "Parachutez Or en 1a"
