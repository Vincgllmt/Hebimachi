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
    """Modèles de tuiles d'un jeu.

    `reference` : dossier de modèles d'un autre jeu au dessin proche (ex. Ishin pour Kiwami).
    Les tuiles qu'il reconnaît sans ambiguïté sont apprises toutes seules (`auto_learn`).
    """

    def __init__(self, template_dir: Path, reference: Path | None = None):
        self.dir = Path(template_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.templates: dict[str, np.ndarray] = {}
        self.scale_cache: dict[str, float] = {}
        self.reference = (TileMatcher(reference) if reference is not None
                          and Path(reference).resolve() != self.dir.resolve() else None)
        self.reload()

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    def _meta(self) -> dict:
        try:
            return json.loads(self.meta_path.read_text())
        except (OSError, ValueError):
            return {}

    def _write_meta(self, **changes) -> None:
        self.meta_path.write_text(json.dumps({**self._meta(), **changes}))

    @property
    def tile_size(self) -> tuple[int, int] | None:
        size = self._meta().get("tile_size")
        return (size[0], size[1]) if size else None

    @property
    def weak(self) -> set[str]:
        """Modèles appris sur une tuile de rivière (petite, redimensionnée) : à remplacer dès
        que la tuile passe dans la main."""
        return set(self._meta().get("weak", []))

    def _set_weak(self, label: str, weak: bool) -> None:
        labels = self.weak
        if (label in labels) != weak:
            self._write_meta(weak=sorted(labels ^ {label}))

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
            self._write_meta(tile_size=size)
        tw, th = size
        for i, label in enumerate(labels):
            # Fenêtre de taille fixe centrée sur la case, sans rééchantillonner si possible.
            crop = self._crop_center(strip, w * (i + 0.5) / n, h / 2)
            self._save(label, crop, overwrite or label in self.weak)
            self._set_weak(label, False)
        self.reload()

    def _crop_center(self, image: np.ndarray, cx: float, cy: float) -> np.ndarray:
        """Fenêtre de la taille des modèles centrée sur (cx, cy), redimensionnée si elle déborde."""
        tw, th = self.tile_size
        x0, y0 = int(round(cx - tw / 2)), int(round(cy - th / 2))
        crop = image[max(0, y0):y0 + th, max(0, x0):x0 + tw]
        if crop.shape[1] != tw or crop.shape[0] != th:
            crop = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
        return crop

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

    # ---------- apprentissage automatique (modèles d'un jeu proche) ----------
    @staticmethod
    def _ranking(roi: np.ndarray, templates: dict[str, np.ndarray],
                 size: tuple[int, int]) -> list[tuple[float, str]]:
        """Scores de chaque modèle, ramené à `size`, sur la zone `roi` ; meilleur en premier."""
        out = []
        for label, tpl in templates.items():
            t = cv2.resize(tpl, size, interpolation=cv2.INTER_AREA)
            if roi.shape[0] >= t.shape[0] and roi.shape[1] >= t.shape[1]:
                out.append((float(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED).max()), label))
        return sorted(out, reverse=True)

    def _sure(self, roi: np.ndarray, size: tuple[int, int], min_score: float,
              margin: float) -> str | None:
        """Étiquette de la tuile selon le jeu de référence, si elle est sans ambiguïté et
        qu'aucun modèle déjà appris (fiable) ne la reconnaît comme une autre tuile."""
        ranked = self._ranking(roi, self.reference.templates, size)
        if not ranked or ranked[0][0] < min_score:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < margin:
            return None
        label = ranked[0][1]
        trusted = {lb: t for lb, t in self.templates.items() if lb not in self.weak and lb != label}
        own = self._ranking(roi, trusted, size)
        if own and own[0][0] >= 0.9:
            return None
        return label

    def auto_learn(self, hand: np.ndarray, min_score: float = 0.75, margin: float = 0.1) -> list[str]:
        """Apprend les modèles manquants (ou faibles) à partir de la zone de la main.

        Les tuiles sont repérées avec les modèles du jeu de référence ; une tuile n'est apprise
        que si la référence la reconnaît nettement. Le dragon blanc, face vide que le template
        matching ne distingue pas, est repéré comme une case claire et unie entre deux tuiles.
        """
        ref = self.reference
        todo = set(self.missing()) | self.weak
        if ref is None or not ref.templates or not todo:
            return []
        rw, rh = next(iter(ref.templates.values())).shape[1::-1]
        if self.tile_size is None:
            dets = ref.detect_multiscale(hand, scales=tuple(s / 100 for s in range(50, 151, 5)),
                                         threshold=min_score)
            if len(dets) < 3:
                return []
            w = int(np.median([d.w for d in dets]))
            self._write_meta(tile_size=(w, int(round(w * rh / rw))))
        tw, th = self.tile_size
        dets = ref.detect(hand, tw / rw, threshold=min_score)
        learned: list[str] = []
        for d in sorted(dets, key=lambda d: -d.score):
            if d.label not in todo or d.label in learned:
                continue
            pad = max(2, d.w // 10)
            roi = hand[max(0, d.y - pad):d.y + d.h + pad, max(0, d.x - pad):d.x + d.w + pad]
            if self._sure(roi, (d.w, d.h), min_score, margin) != d.label:
                continue
            self._save(d.label, self._crop_center(hand, d.x + d.w / 2, d.y + d.h / 2), True)
            self._set_weak(d.label, False)
            learned.append(d.label)
        if "5z" in todo:
            known = dets + (self.detect(hand, threshold=0.85) if self.templates else [])
            blank = self._find_blank(hand, known)
            if blank is not None:
                self._save("5z", self._crop_center(hand, *blank), True)
                self._set_weak("5z", False)
                learned.append("5z")
        if learned:
            self.reload()
        return learned

    def _find_blank(self, hand: np.ndarray, dets: list[Detection]) -> tuple[float, float] | None:
        """Centre d'une case vide (face claire et unie) entre deux tuiles reconnues de la main."""
        if len(dets) < 2:
            return None
        dets = sorted(dets, key=lambda d: d.x)
        pitch = float(np.median([b.x - a.x for a, b in zip(dets, dets[1:])]))
        tw, th = self.tile_size
        hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
        for a, b in zip(dets, dets[1:]):
            if not 1.6 * pitch < b.x - a.x < 2.4 * pitch:
                continue  # pas exactement une case libre entre les deux
            cx, cy = (a.x + b.x + a.w) / 2, a.y + a.h / 2
            x0, x1 = int(cx - 0.3 * tw), int(cx + 0.3 * tw)
            y0, y1 = int(cy - 0.3 * th), int(cy + 0.3 * th)
            face = hsv[max(0, y0):y1, max(0, x0):x1]
            if (face.size and face[:, :, 2].mean() > 170 and face[:, :, 1].mean() < 70
                    and face[:, :, 2].std() < 12):
                return cx, cy
        return None

    def _reference_label(self, tile: np.ndarray, min_score: float = 0.6,
                         margin: float = 0.15) -> str | None:
        """Étiquette d'une tuile isolée debout selon la référence, si elle est sans ambiguïté."""
        if self.reference is None or self.tile_size is None:
            return None
        h, w = tile.shape[:2]
        pad = max(2, int(0.08 * max(w, h)))
        roi = cv2.copyMakeBorder(tile, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        return self._sure(roi, (int(w * 0.94), int(h * 0.94)), min_score, margin)

    def _learn_weak(self, tile: np.ndarray, label: str) -> bool:
        """Enregistre un modèle faible, sauf si un modèle fiable existe déjà pour cette tuile."""
        if label in self.templates and label not in self.weak:
            return False
        self.learn_tile(tile, label, overwrite=True)
        self._set_weak(label, True)
        return True

    def learn_from_reference(self, tile: np.ndarray) -> str | None:
        """Tuile isolée (rivière, dora) inconnue : apprise si la référence la reconnaît nettement.

        Le modèle est marqué faible : il sera remplacé quand la tuile apparaîtra dans la main.
        """
        label = self._reference_label(tile)
        return label if label is not None and self._learn_weak(tile, label) else None

    def _classify_sideways(self, sub: np.ndarray) -> tuple[str, float]:
        """Meilleur modèle pour une tuile couchée, redressée dans un sens ou dans l'autre."""
        h, w = sub.shape[:2]
        options = []
        for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
            r = cv2.copyMakeBorder(cv2.rotate(sub, rot), 4, 4, 4, 4, cv2.BORDER_REPLICATE)
            options.append(self.classify(r, (4, 4, h, w)))
        return max(options, key=lambda o: o[1])

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
        boxes = tile_boxes(image, max_long=max_long)
        # Taille habituelle d'une tuile de cette zone : seule une tuile entière est apprise,
        # jamais un morceau de tuile mal découpé.
        long_side = float(np.median([max(b[2], b[3]) for b in boxes])) if boxes else 0.0
        short_side = float(np.median([min(b[2], b[3]) for b in boxes])) if boxes else 0.0
        for x, y, w, h in boxes:
            whole = (abs(max(w, h) - long_side) <= 0.1 * long_side
                     and abs(min(w, h) - short_side) <= 0.1 * short_side)
            can_learn = self.reference is not None and whole
            if w > h:  # tuile couchée (riichi) : on essaie les deux sens de rotation
                sub = image[y:y + h, x:x + w]
                label, score = self._classify_sideways(sub)
                if score < threshold and can_learn:
                    # Apprise seulement si les deux sens ne désignent pas deux tuiles différentes.
                    found = {}
                    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                        lb = self._reference_label(cv2.rotate(sub, rot))
                        if lb is not None:
                            found.setdefault(lb, rot)
                    if len(found) == 1:
                        (lb, rot), = found.items()
                        if self._learn_weak(cv2.rotate(sub, rot), lb):
                            label, score = self._classify_sideways(sub)
            else:
                label, score = self.classify(image, (x, y, w, h))
                if score < threshold and can_learn:
                    if self.learn_from_reference(image[y:y + h, x:x + w]) is not None:
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


def _tile_shaped(box: tuple[int, int, int, int]) -> bool:
    w, h = box[2], box[3]
    return min(w, h) >= 8 and 1.15 <= max(w, h) / min(w, h) <= 1.8


def _tighten(box: tuple[int, int, int, int], clean: np.ndarray) -> tuple[int, int, int, int]:
    """Réduit le cadre aux rangées et colonnes qui contiennent vraiment de la face."""
    x, y, w, h = box
    face = clean[y:y + h, x:x + w]
    rows = np.nonzero(face.mean(axis=1) >= 0.15)[0]
    cols = np.nonzero(face.mean(axis=0) >= 0.15)[0]
    if not len(rows) or not len(cols):
        return x, y, 0, 0
    return (x + int(cols[0]), y + int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))


def _split_seams(box: tuple[int, int, int, int], clean: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Coupe un bloc de tuiles accolées (rivières de Kiwami) sur la jointure.

    La jointure est une ligne qui traverse toute la face sans un pixel de face nette, au milieu
    du bloc ; le dessin d'une tuile, lui, ne touche jamais les bords de la face.
    """
    x, y, w, h = box
    face = clean[y:y + h, x:x + w]
    if face.mean() < 0.5:
        return [box]  # face terne (tuile grisée, ombre) : la jointure n'y est pas lisible
    for axis, length in ((0, w), (1, h)):  # colonnes, puis rangées
        if length < 1.3 * (h if axis == 0 else w):
            continue  # trop étroit pour contenir deux tuiles côte à côte
        lines = face.mean(axis=axis)  # part de face nette sur chaque colonne (ou rangée)
        lo, hi = int(0.3 * length), int(0.7 * length)
        if hi <= lo:
            continue
        cut = lo + int(np.argmin(lines[lo:hi]))
        if lines[cut] > 0.1:
            continue
        seam = face[:, cut] if axis == 0 else face[cut, :]
        k = max(2, int(0.15 * len(seam)))
        if np.concatenate([seam[:k], seam[-k:]]).mean() > 0.2:
            continue  # la ligne s'arrête avant les marges de la face : c'est le dessin d'une tuile
        a, b = ((x, y, cut, h), (x + cut + 1, y, w - cut - 1, h)) if axis == 0 else \
               ((x, y, w, cut), (x, y + cut + 1, w, h - cut - 1))
        # Chaque part est resserrée sur sa face : une tuile couchée (riichi) est moins haute
        # que ses voisines debout.
        a, b = _tighten(a, clean), _tighten(b, clean)
        # Une part peut encore contenir plusieurs tuiles (découpée ensuite) : on n'écarte que les
        # morceaux trop fins ou trop allongés pour être une tuile, debout ou couchée.
        if not all(min(p[2], p[3]) >= 8 and max(p[2], p[3]) / min(p[2], p[3]) <= 2.2 for p in (a, b)):
            continue  # pas deux tuiles : on laisse le bloc tel quel
        return _split_seams(a, clean) + _split_seams(b, clean)
    return [box]


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
    clean = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 195)  # face nette, sans bord ni ombre
    boxes = []
    for mask, greyed in ((bright, False), (grey, True)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5) if greyed else (3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        for i in range(1, n):
            if stats[i][4] < min_area or (greyed and bright[labels == i].mean() > 0.3):
                continue
            box = tuple(int(v) for v in stats[i][:4])
            boxes.extend(_split_seams(box, clean) if not greyed else [box])
    if max_long is not None:  # ex. appels : plus petits que les tuiles de la main
        boxes = [b for b in boxes if max(b[2], b[3]) <= max_long]
    if not boxes:
        return []
    # On écarte ce qui n'a pas la taille d'une tuile (textes, reflets).
    long_side = float(np.median([max(w, h) for _, _, w, h in boxes]))
    return [b for b in boxes
            if 0.75 * long_side <= max(b[2], b[3]) <= 1.3 * long_side
            and _tile_shaped(b)]


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
