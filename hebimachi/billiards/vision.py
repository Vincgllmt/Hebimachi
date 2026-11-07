"""Lecture d'une capture du billard : billes, bille fantôme, pointillés de visée et HUD.

La table est vue de dessus par une caméra fixe : pas de redressement. On estime le tapis sans les
billes par un filtre médian très large (l'éclairage varie lentement), et tout ce qui s'en écarte
est un objet : bille, pointillé ou texte du HUD. Les billes sont cherchées par `HoughCircles`.

- La bille fantôme, dessinée par le jeu au bout des pointillés, est un disque clair et uni.
- La blanche est claire mais ombrée ; on la confirme par les pointillés qui en partent.
- Les autres billes sont reconnues par leur couleur et pleines ou rayées. Les numéros ne sont pas
  lus : le rouge (3, 11) et le bordeaux (7, 15) sont trop proches pour être séparés.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .table import Point, Table

FG_DIFF = 25          # écart au tapis à partir duquel un pixel est un objet
BG_SCALE = 4          # le fond est estimé sur l'image réduite d'autant
BG_KERNEL = 51        # médian sur l'image réduite : ~200 px à l'écran
MIN_FILL = 0.85       # part du disque central qui doit être un objet pour garder un cercle
GHOST_MAX_SAT = 45
GHOST_MAX_STD = 6
STRIPE_MAX_COLOR = 0.7

NUMBERS = {
    "jaune": (1, 9), "bleu": (2, 10), "rouge": (3, 7, 11, 15), "violet": (4, 12),
    "orange": (5, 13), "vert": (6, 14), "noir": (8,), "blanche": (0,),
}
HUE_CLASSES = ((-15, 12, "rouge"), (13, 19, "orange"), (20, 38, "jaune"), (39, 90, "vert"),
               (91, 130, "bleu"), (131, 165, "violet"))


@dataclass
class Ball:
    x: float
    y: float
    r: float
    color: str = "?"
    stripe: bool = False

    @property
    def pos(self) -> Point:
        return (self.x, self.y)

    @property
    def numbers(self) -> tuple[int, ...]:
        """Numéros possibles pour cette couleur et ce motif."""
        nums = NUMBERS.get(self.color, ())
        if self.color in ("noir", "blanche"):
            return nums
        return tuple(n for n in nums if (n > 8) == self.stripe)

    def label(self) -> str:
        if self.color in ("blanche", "noir", "?"):
            return {"blanche": "la blanche", "noir": "le 8", "?": "une bille"}[self.color]
        nums = "/".join(map(str, self.numbers))
        return f"{self.color} {'rayée' if self.stripe else 'pleine'} ({nums})"


@dataclass
class Scene:
    balls: list[Ball] = field(default_factory=list)   # billes réelles, blanche comprise
    cue: int | None = None                             # indice de la blanche dans `balls`
    ghost: Point | None = None
    dots: list[Point] = field(default_factory=list)    # pointillés de visée
    hud: bool = False                                  # HUD du jeu visible
    next_color: str | None = None                      # couleur de la bille « NEXT »


def normalize(image: np.ndarray, table: Table) -> np.ndarray:
    """Ramène la capture à la résolution de calibration (une découpe à quelques pixels près est complétée)."""
    w, h = table.screen
    ih, iw = image.shape[:2]
    if (iw, ih) == (w, h):
        return image
    if 0 <= w - iw <= 4 and 0 <= h - ih <= 4:
        return cv2.copyMakeBorder(image, 0, h - ih, 0, w - iw, cv2.BORDER_REPLICATE)
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def _roi(table: Table) -> tuple[int, int, int, int]:
    return (int(table.left) + 2, int(table.top) + 2, int(table.right) - 2, int(table.bottom) - 2)


def foreground(image: np.ndarray, table: Table) -> np.ndarray:
    """Écart de chaque pixel du tapis à l'estimation du tapis vide (0 hors du tapis)."""
    x0, y0, x1, y1 = _roi(table)
    roi = image[y0:y1, x0:x1]
    small = cv2.resize(roi, None, fx=1 / BG_SCALE, fy=1 / BG_SCALE, interpolation=cv2.INTER_AREA)
    bg = cv2.resize(cv2.medianBlur(small, BG_KERNEL), (roi.shape[1], roi.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    diff = np.zeros(image.shape[:2], np.uint8)
    diff[y0:y1, x0:x1] = cv2.absdiff(roi, bg).max(axis=2)
    return diff


_ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
_COS, _SIN = np.cos(_ANGLES).astype(np.float32), np.sin(_ANGLES).astype(np.float32)


def gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = image.astype(np.float32)
    return cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)


def refine_center(grad: tuple[np.ndarray, np.ndarray], x: float, y: float, r: float,
                  reach: float = 8) -> Point:
    """Centre du cercle de rayon `r` qui longe le mieux un contour, autour de (x, y).

    Le rayon est fixé : un rayon libre s'accrocherait à la frontière blanc/couleur des rayées.
    Score = médiane, sur le cercle, du gradient radial (en valeur absolue, sommé sur les canaux).
    """
    gx, gy = grad

    def best(cx: float, cy: float, half: float, step: float) -> Point:
        offs = np.arange(-half, half + step / 2, step, dtype=np.float32)
        dx, dy = np.meshgrid(offs, offs)
        cxs, cys = (cx + dx.ravel())[:, None], (cy + dy.ravel())[:, None]
        px, py = cxs + r * _COS, cys + r * _SIN
        vx = cv2.remap(gx, px, py, cv2.INTER_LINEAR)
        vy = cv2.remap(gy, px, py, cv2.INTER_LINEAR)
        radial = np.abs(vx * _COS[None, :, None] + vy * _SIN[None, :, None]).sum(axis=2)
        i = int(np.argmax(np.median(radial, axis=1)))
        return float(cxs[i, 0]), float(cys[i, 0])

    cx, cy = best(x, y, reach, 1.0)
    return best(cx, cy, 1.0, 0.25)


def _disk(shape, x: float, y: float, r: float) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (round(x), round(y)), max(1, round(r)), 255, -1)
    return m > 0


def classify(hsv_pixels: np.ndarray) -> tuple[str, bool]:
    """Couleur et motif (rayée ?) d'une bille, d'après ses pixels HSV."""
    s, v = hsv_pixels[:, 1].astype(int), hsv_pixels[:, 2].astype(int)
    colored = (s > 110) & (v > 60)
    dark = v < 60
    if dark.mean() > 0.35:
        return "noir", False
    if colored.mean() < 0.03:
        return "blanche", False
    hue = hsv_pixels[colored, 0].astype(int)
    hue = np.where(hue > 165, hue - 180, hue)
    h = float(np.median(hue))
    color = next((name for lo, hi, name in HUE_CLASSES if lo <= h <= hi), "?")
    return color, bool(colored.mean() < STRIPE_MAX_COLOR)


def _dots(diff: np.ndarray, hsv: np.ndarray, circles: list[tuple[float, float, float]]) -> list[Point]:
    mask = (diff > FG_DIFF).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    dots = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (12 <= area <= 90 and w <= 12 and h <= 12):
            continue
        cx, cy = cent[i]
        if any(math.hypot(cx - bx, cy - by) < br + 4 for bx, by, br in circles):
            continue
        px = hsv[lab == i]
        if px[:, 1].mean() < 90 and px[:, 2].mean() > 150:
            dots.append((float(cx), float(cy)))
    return dots


def read_hud(image: np.ndarray, hsv: np.ndarray, table: Table) -> tuple[bool, str | None]:
    if not table.hud_tray:
        return False, None
    x0, y0, x1, y1 = table.hud_tray
    tray = hsv[y0:y1, x0:x1]
    purple = (tray[..., 0] >= 130) & (tray[..., 0] <= 165) & (tray[..., 1] > 80)
    if purple.mean() < 0.3:
        return False, None
    if not table.hud_next:
        return True, None
    x, y, r = table.hud_next
    px = hsv[_disk(hsv.shape, x, y, r * 0.9)]
    color, _ = classify(px)
    return True, (None if color in ("blanche", "?") else color)


def read_scene(image: np.ndarray, table: Table) -> Scene:
    image = normalize(image, table)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    diff = foreground(image, table)
    R = table.radius
    found = cv2.HoughCircles(cv2.GaussianBlur(diff, (5, 5), 1.5), cv2.HOUGH_GRADIENT, dp=1,
                             minDist=R * 1.3, param1=80, param2=14,
                             minRadius=round(R * 0.8), maxRadius=round(R * 1.15))
    grad = gradients(image)
    circles = []
    for x, y, r in (found[0] if found is not None else []):
        inner = _disk(diff.shape, x, y, r * 0.8)
        if (diff[inner] > FG_DIFF).mean() < MIN_FILL:
            continue
        x, y = refine_center(grad, float(x), float(y), R)
        if all(math.hypot(x - cx, y - cy) > R for cx, cy, _ in circles):
            circles.append((x, y, R))

    scene = Scene()
    scene.dots = _dots(diff, hsv, circles)
    whites, ghosts = [], []
    for x, y, r in circles:
        px = hsv[_disk(hsv.shape, x, y, r * 0.8)]
        if px[:, 1].mean() < GHOST_MAX_SAT and px[:, 2].std() < GHOST_MAX_STD:
            ghosts.append((x, y))
            continue
        color, stripe = classify(hsv[_disk(hsv.shape, x, y, r * 0.85)])
        scene.balls.append(Ball(x, y, r, color, stripe))
        if color == "blanche":
            whites.append(len(scene.balls) - 1)
    if ghosts:
        # le fantôme est au bout des pointillés
        scene.ghost = min(ghosts, key=lambda g: not _near_dot(scene.dots, g, R))
    if whites:
        # plusieurs candidates (une rayée vue par le dessus paraît blanche) : celle d'où partent les pointillés
        scene.cue = min(whites, key=lambda i: (not _near_dot(scene.dots, scene.balls[i].pos, R),
                                               -_whiteness(hsv, scene.balls[i])))
    scene.hud, scene.next_color = read_hud(image, hsv, table)
    return scene


def _near_dot(dots: list[Point], p: Point, R: float) -> bool:
    return any(math.dist(d, p) < R * 3 for d in dots)


def _whiteness(hsv: np.ndarray, ball: Ball) -> float:
    px = hsv[_disk(hsv.shape, ball.x, ball.y, ball.r * 0.85)]
    return float(((px[:, 1] < 110) & (px[:, 2] > 150)).mean())
