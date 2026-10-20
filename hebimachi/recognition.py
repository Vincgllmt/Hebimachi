"""Reconnaissance des tuiles par template matching (OpenCV).

Les modèles sont appris une fois à partir de captures du client (Mahjong Soul),
voir `learn_from_strip`. La détection balaie toute la zone avec chaque modèle,
ce qui gère naturellement l'écart de la tuile piochée et les tuiles survolées.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .tiles import ALL_LABELS


@dataclass
class Detection:
    x: int
    y: int
    label: str
    score: float


class TileMatcher:
    def __init__(self, template_dir: Path):
        self.dir = Path(template_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.templates: dict[str, np.ndarray] = {}
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

    def learn_from_strip(self, strip: np.ndarray, labels: list[str], inset: float = 0.04) -> None:
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
            ok, buf = cv2.imencode(".png", crop)
            if ok:
                buf.tofile(self.dir / f"{label}.png")  # tofile gère les chemins non ASCII
        self.reload()

    def detect(self, image: np.ndarray, scale: float = 1.0, threshold: float = 0.72) -> list[Detection]:
        if not self.templates:
            return []
        labels = list(self.templates)
        tpls = [self.templates[lb] for lb in labels]
        if scale != 1.0:
            tpls = [cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for t in tpls]
        th, tw = tpls[0].shape[:2]
        if image.shape[0] < th or image.shape[1] < tw:
            return []

        # Pour chaque colonne x : meilleur score (sur y) de chaque modèle.
        cols, rows = [], []
        for t in tpls:
            res = cv2.matchTemplate(image, t, cv2.TM_CCOEFF_NORMED)
            cols.append(res.max(axis=0))
            rows.append(res.argmax(axis=0))
        scores = np.stack(cols)
        best_label = scores.argmax(axis=0)
        best_score = scores.max(axis=0)

        # Suppression des non-maxima : une tuile par fenêtre de ~largeur de tuile.
        min_dist = int(tw * 0.7)
        accepted: list[int] = []
        for x in np.argsort(-best_score):
            if best_score[x] < threshold:
                break
            if all(abs(int(x) - a) >= min_dist for a in accepted):
                accepted.append(int(x))
        return [Detection(x, int(rows[best_label[x]][x]), labels[best_label[x]], float(best_score[x]))
                for x in sorted(accepted)]

    def detect_multiscale(self, image: np.ndarray, scales=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                          threshold: float = 0.72) -> list[Detection]:
        """Pour les zones où les tuiles sont plus petites (indicateurs dora)."""
        best: list[Detection] = []
        best_total = 0.0
        for s in scales:
            dets = self.detect(image, s, threshold)
            total = sum(d.score for d in dets)
            if total > best_total:
                best, best_total = dets, total
        return best
