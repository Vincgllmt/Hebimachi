"""Billard : géométrie des poches, puis lecture de vraies captures de Yakuza Kiwami (mode rotation)."""
import math
from collections import Counter
from pathlib import Path

import cv2
import pytest

from hebimachi.billiards.search import find_shots, legal_first
from hebimachi.billiards.shot import aim_feedback, analyze
from hebimachi.billiards.sim import simulate
from hebimachi.billiards.table import Table, travel
from hebimachi.billiards.vision import read_scene

ROOT = Path(__file__).resolve().parent.parent
SCREENS = Path(__file__).parent / "screens" / "billard"
TABLE = Table.load(ROOT / "data" / "games" / "kiwami" / "billiards.json")
R = TABLE.radius


# --- géométrie --------------------------------------------------------------------------------

@pytest.mark.parametrize("name, corner", [
    ("haut gauche", (TABLE.left, TABLE.top)), ("haut droite", (TABLE.right, TABLE.top)),
    ("bas gauche", (TABLE.left, TABLE.bottom)), ("bas droite", (TABLE.right, TABLE.bottom)),
])
def test_ball_sent_at_a_corner_drops_in_it(name, corner):
    start = (960, 540)
    path = travel(TABLE, start, (corner[0] - start[0], corner[1] - start[1]))
    assert path.pocket and path.pocket.name == name


def test_ball_sent_straight_at_a_side_pocket_drops_in_it():
    path = travel(TABLE, (960, 600), (0, 1))
    assert path.pocket and path.pocket.name == "bas milieu"
    assert path.end[1] == pytest.approx(TABLE.bottom - R)


def test_side_pocket_refuses_a_grazing_angle():
    # arrive sur la poche du milieu à 70° de la perpendiculaire
    a = math.radians(70)
    start = (960 - 300 * math.sin(a), TABLE.bottom - R - 300 * math.cos(a))
    path = travel(TABLE, start, (math.sin(a), math.cos(a)))
    assert path.pocket is None and path.too_steep


def test_ball_beside_a_pocket_hits_the_cushion():
    path = travel(TABLE, (600, 540), (0, 1))
    assert path.pocket is None and path.jaw is None
    assert path.end == pytest.approx((600, TABLE.bottom - R))


def test_ball_in_the_way_blocks_the_path():
    start = (960, 400)
    path = travel(TABLE, start, (0, 1), others=[(965, 700)])
    assert path.blocker == 0 and path.pocket is None and not path.graze
    assert math.dist(path.end, (965, 700)) == pytest.approx(2 * R)


def test_ball_barely_in_the_way_is_a_graze():
    path = travel(TABLE, (960, 400), (0, 1), others=[(960 + 2 * R * 0.95, 700)])
    assert path.blocker == 0 and path.graze


def test_ball_far_enough_does_not_block():
    path = travel(TABLE, (960, 400), (0, 1), others=[(960 + 2 * R + 1, 700)])
    assert path.blocker is None and path.pocket


# --- captures du jeu ----------------------------------------------------------------------------

# Les trois captures montrent la même position (mode rotation, 1 à 4 et 8 déjà empochés).
BALLS = Counter({("rouge", False): 1, ("vert", False): 1, ("orange", False): 1, ("jaune", True): 1,
                 ("orange", True): 1, ("rouge", True): 2, ("violet", True): 1, ("bleu", True): 1,
                 ("vert", True): 1, ("blanche", False): 1})
CUE = (852, 430)


def scene_of(name):
    image = cv2.imread(str(SCREENS / name))
    assert image is not None, name
    return read_scene(image, TABLE)


@pytest.mark.parametrize("name", ["rebond_vers_7.png", "fantome_contre_6.png", "hud_rotation.png"])
def test_reads_every_ball(name):
    scene = scene_of(name)
    assert Counter((b.color, b.stripe) for b in scene.balls) == BALLS
    assert scene.cue is not None and math.dist(scene.balls[scene.cue].pos, CUE) < 1.5


def test_ball_centers_are_stable_across_captures():
    """La position n'a pas bougé entre les captures : la lecture doit être la même au pixel près."""
    scenes = [scene_of(n) for n in ("rebond_vers_7.png", "fantome_contre_6.png", "hud_rotation.png")]
    ref = sorted(b.pos for b in scenes[0].balls)
    for s in scenes[1:]:
        for p, q in zip(ref, sorted(b.pos for b in s.balls)):
            assert math.dist(p, q) <= 1.0


def test_ghost_against_the_6_aims_it_into_the_7():
    scene = scene_of("fantome_contre_6.png")
    assert scene.ghost and math.dist(scene.ghost, (946, 616)) < 2
    advice = analyze(scene, TABLE)
    target = scene.balls[advice.target]
    assert (target.color, target.stripe) == ("vert", False)
    assert advice.verdict == "ratee"
    hit = [t for s, t in advice.outcome.contacts if s == advice.target]
    assert hit and scene.balls[hit[0]].color == "rouge"


def test_bank_shot_reaches_the_7():
    scene = scene_of("rebond_vers_7.png")
    advice = analyze(scene, TABLE)
    assert (scene.balls[advice.target].color, scene.balls[advice.target].stripe) == ("rouge", False)
    # la blanche arrive de la droite, après le rebond : direction lue sur les derniers pointillés
    assert advice.aim[0] < -0.9
    assert advice.verdict != "empochee"


def test_hud_gives_next_ball_and_aim_on_cushion_touches_nothing():
    scene = scene_of("hud_rotation.png")
    assert scene.hud and scene.next_color == "orange"
    assert analyze(scene, TABLE).verdict == "bande"


def test_without_hud_no_foul_check():
    scene = scene_of("fantome_contre_6.png")
    assert not scene.hud and scene.next_color is None
    assert analyze(scene, TABLE).foul is None


def test_hitting_another_ball_than_next_is_a_foul():
    scene = scene_of("fantome_contre_6.png")
    scene.next_color = "orange"   # NEXT = 5, lu sur hud_rotation.png pour la même position
    advice = analyze(scene, TABLE)
    assert advice.foul and advice.lines[0].startswith("Faute")


def test_ball_too_close_to_a_jaw_hits_it():
    path = travel(TABLE, (925, 600), (0, 1))   # mâchoire gauche de la poche du bas en x = 918
    assert path.pocket is None and path.jaw and path.jaw.name == "bas milieu"


# --- simulation et recherche de coups -----------------------------------------------------------

def test_full_hit_sends_the_ball_straight_and_stops_the_cue():
    # blanche, bille visée et poche du bas milieu alignées
    pos = [(960, 400), (960, 600)]
    out = simulate(TABLE, pos, 0, (0, 1))
    assert out.first == 1 and out.pocketed == {1: "bas milieu"}
    assert not any(s.ball == 0 for s in out.segments[1:])   # la blanche ne repart pas


def test_combination_is_found_and_named():
    # blanche -> A -> B -> poche du bas milieu, tout aligné
    pos = [(960, 300), (960, 450), (960, 650)]
    out = simulate(TABLE, pos, 0, (0, 1))
    assert out.contacts == [(0, 1), (1, 2)] and out.pocketed == {2: "bas milieu"}


def test_rotation_next_is_the_solid_of_its_color():
    scene = scene_of("hud_rotation.png")
    legal = legal_first(scene)
    assert [(scene.balls[i].color, scene.balls[i].stripe) for i in legal] == [("orange", False)]


def test_hidden_next_is_reached_off_a_cushion():
    """Le 10 cache le 5 : seul un coup avec la blanche par la bande le touche en premier."""
    scene = scene_of("hud_rotation.png")
    plans = find_shots(scene, TABLE)
    assert plans and all("bande" in p.kinds for p in plans)
    assert all(scene.balls[p.outcome.first].color == "orange" for p in plans)


def test_search_finds_combinations_and_caroms():
    scene = scene_of("fantome_contre_6.png")
    plans = find_shots(scene, TABLE)
    kinds = {k for p in plans for k in p.kinds}
    assert {"direct", "combinaison", "carambolage"} <= kinds
    assert plans == sorted(plans, key=lambda p: -p.score)
    combo = next(p for p in plans if "combinaison" in p.kinds)
    assert combo.text.count("touche") >= 2 and "entre dans la poche" in combo.text


def test_aim_feedback_says_where_to_move_the_ghost():
    scene = scene_of("fantome_contre_6.png")
    plan = next(p for p in find_shots(scene, TABLE) if p.outcome.first == find_target(scene))
    scene.ghost = plan.ghost
    assert aim_feedback(scene, plan, TABLE).startswith("✔")
    scene.ghost = (plan.ghost[0] - 6, plan.ghost[1])
    assert "droite" in aim_feedback(scene, plan, TABLE)


def find_target(scene):
    """Le 6 (vert plein), visé sur fantome_contre_6.png."""
    return next(i for i, b in enumerate(scene.balls) if (b.color, b.stripe) == ("vert", False))
