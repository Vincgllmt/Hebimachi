"""Reconnaissance des tuiles par template matching (OpenCV).

Les modèles sont appris une fois à partir de captures du jeu (Like a Dragon:
Ishin!, vue de dessus), voir `learn_from_strip`. La détection balaie toute la
zone avec chaque modèle puis garde les meilleurs pics sans chevauchement, ce qui
gère l'écart de la tuile piochée, les tuiles surélevées et les rivières sur
plusieurs rangées.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .tiles import ALL_LABELS

# Rotation qui remet chaque rivière « à l'endroit », vue depuis son joueur.
# Après rotation, l'ordre des défausses se lit de gauche à droite, rangée par rangée.
RIVER_ROTATIONS = {
    "me": None,
    "shimocha": cv2.ROTATE_90_CLOCKWISE,         # joueur de droite
    "toimen": cv2.ROTATE_180,                    # joueur d'en face
    "kamicha": cv2.ROTATE_90_COUNTERCLOCKWISE,   # joueur de gauche
}
DEFAULT_SCALES = tuple(s / 100 for s in range(40, 101, 4))


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float
    sideways: bool = False  # tuile couchée (déclaration de riichi)


def _overlap(a: Detection, b: Detection) -> float:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    return ix * iy / min(a.w * a.h, b.w * b.h)


def order_rows(dets: list[Detection]) -> list[Detection]:
    """Trie par rangée (de haut en bas) puis de gauche à droite."""
    rows: list[list[Detection]] = []
    for d in sorted(dets, key=lambda d: d.y + d.h / 2):
        cy = d.y + d.h / 2
        if rows and abs(cy - (rows[-1][0].y + rows[-1][0].h / 2)) < min(d.h, rows[-1][0].h) / 2:
            rows[-1].append(d)
        else:
            rows.append([d])
    return [d for row in rows for d in sorted(row, key=lambda d: d.x)]


class TileMatcher:
    def __init__(self, template_dir: Path):
        self.dir = Path(template_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.templates: dict[str, np.ndarray] = {}
        self.scale_cache: dict[str, float] = {}
        self.reload()

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def tile_size(self) -> tuple[int, int] | None:
        if self.meta_path.exists():
            w, h = json.loads(self.meta_path.read_text())["tile_size"]
            return w, h
        return None

    def reload(self) -> None:
        self.templates = {}
        for label in ALL_LABELS:
            p = self.dir / f"{label}.png"
            if p.exists():
                img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates[label] = img

    def missing(self) -> list[str]:
        return [lb for lb in ALL_LABELS if lb not in self.templates]

    # ---------- apprentissage ----------
    def _save(self, label: str, img: np.ndarray, overwrite: bool) -> None:
        if not overwrite and label in self.templates:
            return
        ok, buf = cv2.imencode(".png", img)
        if ok:
            buf.tofile(self.dir / f"{label}.png")  # tofile gère les chemins non ASCII

    def learn_from_strip(self, strip: np.ndarray, labels: list[str], inset: float = 0.04,
                         overwrite: bool = True) -> None:
        """Découpe une rangée de tuiles contiguës en parts égales et enregistre les modèles."""
        h, w = strip.shape[:2]
        n = len(labels)
        if n == 0:
            raise ValueError("Aucune étiquette")
        size = self.tile_size
        if size is None:
            size = (int(w / n * (1 - 2 * inset)), int(h * (1 - 2 * inset)))
            self.meta_path.write_text(json.dumps({"tile_size": size}))
        tw, th = size
        for i, label in enumerate(labels):
            # Fenêtre de taille fixe centrée sur la case, sans rééchantillonner si possible.
            cx, cy = w * (i + 0.5) / n, h / 2
            x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
            crop = strip[max(0, y0):y0 + th, max(0, x0):x0 + tw]
            if crop.shape[1] != tw or crop.shape[0] != th:
                crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            self._save(label, crop, overwrite)
        self.reload()

    def learn_tile(self, tile: np.ndarray, label: str, overwrite: bool = False) -> None:
        """Modèle à partir d'une tuile isolée (ex. une tuile de rivière, plus petite)."""
        size = self.tile_size
        if size is None:
            raise ValueError("Apprenez d'abord des tuiles de la main (taille de référence)")
        self._save(label, cv2.resize(tile, size, interpolation=cv2.INTER_CUBIC), overwrite)
        self.reload()

    # ---------- détection ----------
    def detect(self, image: np.ndarray, scale: float = 1.0, threshold: float = 0.72,
               sideways: bool = False) -> list[Detection]:
        """Tuiles détectées, de gauche à droite. `sideways` cherche aussi les tuiles couchées."""
        if not self.templates:
            return []
        labels = list(self.templates)
        tpls = [self.templates[lb] for lb in labels]
        if scale != 1.0:
            tpls = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tpls]
        sets = [(tpls, False)]
        if sideways:
            sets.append(([cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE) for t in tpls], True))
            sets.append(([cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE) for t in tpls], True))

        cands: list[Detection] = []
        for group, side in sets:
            th, tw = group[0].shape[:2]
            if image.shape[0] < th or image.shape[1] < tw:
                continue
            res = np.stack([cv2.matchTemplate(image, t, cv2.TM_CCOEFF_NORMED) for t in group])
            best, arg = res.max(axis=0), res.argmax(axis=0)
            # Maxima locaux au-dessus du seuil : quelques dizaines de candidats au plus.
            k = np.ones((max(3, th // 2) | 1, max(3, tw // 2) | 1), np.uint8)
            peaks = (best >= threshold) & (best >= cv2.dilate(best, k))
            for y, x in zip(*np.nonzero(peaks)):
                cands.append(Detection(int(x), int(y), tw, th, labels[arg[y, x]],
                                       float(best[y, x]), side))

        kept: list[Detection] = []
        for c in sorted(cands, key=lambda d: -d.score):
            if all(_overlap(c, k) < 0.35 for k in kept):
                kept.append(c)
        return sorted(kept, key=lambda d: d.x)

    def detect_multiscale(self, image: np.ndarray, scales=DEFAULT_SCALES, threshold: float = 0.72,
                          sideways: bool = False, key: str | None = None) -> list[Detection]:
        """Cherche la meilleure échelle (zones où les tuiles sont plus petites que dans la main).

        Avec `key`, l'échelle trouvée est mémorisée et réessayée en premier ensuite.
        """
        if key in self.scale_cache:
            dets = self.detect(image, self.scale_cache[key], threshold, sideways)
            if dets:
                return dets
        best: list[Detection] = []
        best_total, best_scale = 0.0, None
        for s in scales:
            dets = self.detect(image, s, threshold)  # pas de tuiles couchées pour choisir l'échelle
            total = sum(d.score for d in dets)
            if total > best_total:
                best, best_total, best_scale = dets, total, s
        if best_scale is None:
            return []
        if key is not None:
            self.scale_cache[key] = best_scale
        return self.detect(image, best_scale, threshold, sideways) if sideways else best

    # ---------- tuiles isolées sur le tapis (rivières, dora) ----------
    def classify(self, image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[str, float]:
        """Meilleur modèle pour la face de tuile délimitée par `box` (x, y, w, h)."""
        x, y, w, h = box
        pad = max(2, int(0.08 * max(w, h)))
        roi = image[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        best_label, best_score = "", -1.0
        for label, tpl in self.templates.items():
            t = cv2.resize(tpl, (int(w * 0.94), int(h * 0.94)), interpolation=cv2.INTER_AREA)
            if roi.shape[0] < t.shape[0] or roi.shape[1] < t.shape[1]:
                continue
            score = float(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max())
            if score > best_score:
                best_label, best_score = label, score
        return best_label, best_score

    def read_tiles(self, image: np.ndarray, threshold: float = 0.6) -> list[Detection]:
        """Tuiles posées sur le tapis, dans l'ordre de lecture ; les tuiles couchées sont redressées."""
        dets = []
        for x, y, w, h in tile_boxes(image):
            if w > h:  # tuile couchée : on essaie les deux sens de rotation
                sub = image[y:y + h, x:x + w]
                options = []
                for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                    r = cv2.copyMakeBorder(cv2.rotate(sub, rot), 4, 4, 4, 4, cv2.BORDER_REPLICATE)
                    options.append(self.classify(r, (4, 4, h, w)))
                label, score = max(options, key=lambda o: o[1])
            else:
                label, score = self.classify(image, (x, y, w, h))
            if score >= threshold:
                dets.append(Detection(x, y, w, h, label, score, sideways=w > h))
        return order_rows(dets)

    def read_river(self, image: np.ndarray, seat: str, threshold: float = 0.6) -> tuple[list[str], int | None]:
        """Défausses d'un joueur dans l'ordre, et index de la tuile couchée (riichi) s'il y en a une."""
        rot = RIVER_ROTATIONS[seat]
        upright = cv2.rotate(image, rot) if rot is not None else image
        dets = self.read_tiles(upright, threshold)
        riichi = next((i for i, d in enumerate(dets) if d.sideways), None)
        return [d.label for d in dets], riichi


def tile_boxes(image: np.ndarray, min_area: int = 600) -> list[tuple[int, int, int, int]]:
    """Faces de tuiles (claires et peu saturées) séparées par le tapis vert."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 160)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][4] >= min_area]
    if not boxes:
        return []
    # On écarte ce qui n'a pas la taille d'une tuile (textes, reflets).
    long_side = float(np.median([max(w, h) for _, _, w, h in boxes]))
    return [b for b in boxes
            if 0.75 * long_side <= max(b[2], b[3]) <= 1.3 * long_side
            and 1.15 <= max(b[2], b[3]) / min(b[2], b[3]) <= 1.8]
