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


def group_calls(dets: list[Detection]) -> list[list[Detection]]:
    """Appels séparés par un écart net entre tuiles (un kan compte 4 tuiles, pas 3)."""
    groups: list[list[Detection]] = []
    for d in sorted(dets, key=lambda d: d.x):
        prev = groups[-1][-1] if groups else None
        if prev is not None and d.x - (prev.x + prev.w) < 0.4 * min(d.w, d.h):
            groups[-1].append(d)
        else:
            groups.append([d])
    return groups


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

    def learn_tile(self, tile: np.ndarray, label: str, overwrite: bool = False,
                   inset: float = 0.04) -> None:
        """Modèle à partir d'une tuile isolée (ex. une tuile de rivière, plus petite)."""
        size = self.tile_size
        if size is None:
            raise ValueError("Apprenez d'abord des tuiles de la main (taille de référence)")
        # Même cadrage que les modèles de la main : on rogne la face comme `learn_from_strip`.
        h, w = tile.shape[:2]
        dy, dx = int(round(h * inset)), int(round(w * inset))
        tile = tile[dy:h - dy, dx:w - dx]
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

    def read_tiles(self, image: np.ndarray, threshold: float = 0.6,
                   max_long: float | None = None) -> list[Detection]:
        """Tuiles posées sur le tapis, dans l'ordre de lecture ; les tuiles couchées sont redressées."""
        dets = []
        for x, y, w, h in tile_boxes(image, max_long=max_long):
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


def tile_boxes(image: np.ndarray, min_area: int = 600,
               max_long: float | None = None) -> list[tuple[int, int, int, int]]:
    """Faces de tuiles (peu saturées) séparées par le tapis vert.

    Les faces claires sont cherchées d'abord ; les tuiles grisées par le jeu (plus
    sombres) sont ajoutées ensuite, si elles ne contiennent presque pas de face claire.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_sat = hsv[:, :, 1] < 70
    bright = (low_sat & (hsv[:, :, 2] > 160)).astype(np.uint8)
    grey = ((hsv[:, :, 1] < 60) & (hsv[:, :, 2] > 25)).astype(np.uint8)
    boxes = []
    for mask, greyed in ((bright, False), (grey, True)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5) if greyed else (3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i][4] < min_area or (greyed and bright[labels == i].mean() > 0.3):
                continue
            boxes.append(tuple(int(v) for v in stats[i][:4]))
    if max_long is not None:  # ex. appels : plus petits que les tuiles de la main
        boxes = [b for b in boxes if max(b[2], b[3]) <= max_long]
    if not boxes:
        return []
    # On écarte ce qui n'a pas la taille d'une tuile (textes, reflets).
    long_side = float(np.median([max(w, h) for _, _, w, h in boxes]))
    return [b for b in boxes
            if 0.75 * long_side <= max(b[2], b[3]) <= 1.3 * long_side
            and 1.15 <= max(b[2], b[3]) / min(b[2], b[3]) <= 1.8]


# ---------- vents : lettre de notre siège, manche en cours ----------
SEAT_LETTERS = {"E": 27, "S": 28, "O": 29, "N": 30}  # « O » = Ouest dans la version française


def letter_mask(image: np.ndarray, size=(32, 40)) -> np.ndarray | None:
    """Silhouette de la lettre beige d'un panneau de joueur, normalisée."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] >= 8) & (hsv[:, :, 0] <= 35) & (hsv[:, :, 1] < 120)
            & (hsv[:, :, 2] > 170)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    x, y, w, h = stats[i][:4]
    if h < 15:
        return None
    return cv2.resize((labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255, size,
                      interpolation=cv2.INTER_AREA)


class WindReader:
    """Lit notre vent de siège (lettre du panneau) et la manche (« Est » ou non)."""

    def __init__(self, template_dir: Path):
        self.dir = Path(template_dir)

    def _load(self, name: str) -> np.ndarray | None:
        p = self.dir / f"{name}.png"
        return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE) if p.exists() else None

    def learn_seat(self, panel: np.ndarray, letter: str) -> None:
        m = letter_mask(panel)
        if m is None:
            raise ValueError("Lettre introuvable dans la zone")
        cv2.imencode(".png", m)[1].tofile(self.dir / f"seat_{letter}.png")

    def learn_round(self, text: np.ndarray) -> None:
        cv2.imencode(".png", cv2.cvtColor(text, cv2.COLOR_BGR2GRAY))[1].tofile(self.dir / "round_E.png")

    def read_seat(self, panel: np.ndarray) -> int | None:
        m = letter_mask(panel)
        if m is None:
            return None
        scores = {}
        for letter, wind in SEAT_LETTERS.items():
            tpl = self._load(f"seat_{letter}")
            if tpl is not None:
                scores[wind] = float(cv2.matchTemplate(m, tpl, cv2.TM_CCOEFF_NORMED).max())
        if not scores:
            return None
        wind, score = max(scores.items(), key=lambda kv: kv[1])
        return wind if score > 0.6 else None

    def read_round(self, region: np.ndarray) -> int | None:
        """27 (Est) si le mot « Est » est trouvé, sinon 28 (Sud) ; None sans modèle."""
        tpl = self._load("round_E")
        if tpl is None:
            return None
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
            return None
        return 27 if cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED).max() > 0.75 else 28


# ---------- prompt du jeu (boutons Chi / Pon / Kan / Ron / Riichi / Tsumo / Annuler) ----------
PROMPT_ACTIONS = ("Chi", "Pon", "Kan", "Ron", "Riichi", "Tsumo")


class PromptReader:
    """Reconnaît les mots des boutons d'action affichés par le jeu.

    Chaque mot appris est un modèle en niveaux de gris (data/games/<jeu>/templates/prompt_<Mot>.png).
    Si un bouton est affiché (« Annuler » visible) sans mot connu, la zone est enregistrée
    dans templates/prompt_inconnu/ (dossier du jeu) pour pouvoir apprendre ce nouveau mot ensuite.
    """

    # Les mots appris matchent à ~1.0 ; « Pon » atteint encore 0.86 sur « Ron » (une lettre d'écart).
    def __init__(self, template_dir: Path, threshold: float = 0.93):
        self.dir = Path(template_dir)
        self.threshold = threshold
        self._saved = False
        self.templates: dict[str, np.ndarray] = {}
        for p in self.dir.glob("prompt_*.png"):
            img = cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates[p.stem.removeprefix("prompt_")] = img

    def learn(self, word_img: np.ndarray, word: str) -> None:
        gray = cv2.cvtColor(word_img, cv2.COLOR_BGR2GRAY)
        cv2.imencode(".png", gray)[1].tofile(self.dir / f"prompt_{word}.png")
        self.templates[word] = gray

    def read(self, region: np.ndarray) -> tuple[bool, list[str]]:
        """(un prompt est-il affiché ?, actions proposées) dans la zone des boutons."""
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        hits: list[Detection] = []
        for word, tpl in self.templates.items():
            if gray.shape[0] < tpl.shape[0] or gray.shape[1] < tpl.shape[1]:
                continue
            _, score, _, (x, y) = cv2.minMaxLoc(cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED))
            if score >= self.threshold:
                hits.append(Detection(x, y, tpl.shape[1], tpl.shape[0], word, score))
        # Deux mots au même endroit (mots proches) : seul le meilleur score est le bon.
        found: list[str] = []
        kept: list[Detection] = []
        for d in sorted(hits, key=lambda d: -d.score):
            if all(_overlap(d, k) < 0.5 for k in kept):
                kept.append(d)
                found.append(d.label)
        actions = [w for w in PROMPT_ACTIONS if w in found]
        shown = bool(actions) or "Annuler" in found
        if shown and not actions and not self._saved:
            # Une seule image par apparition du prompt, pas une à chaque capture.
            unknown = self.dir / "prompt_inconnu"
            unknown.mkdir(exist_ok=True)
            cv2.imencode(".png", region)[1].tofile(unknown / f"{len(list(unknown.glob('*.png'))):03d}.png")
        self._saved = shown and not actions
        return shown, actions
