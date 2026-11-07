"""Dessine la lecture, le verdict de la visée en cours et le coup conseillé sur la capture."""
from __future__ import annotations

import unicodedata

import cv2
import numpy as np

from .search import Plan
from .shot import Advice
from .sim import Outcome
from .table import Table
from .vision import Scene

GREEN, RED, ORANGE, CYAN, WHITE, BLUE, MAGENTA = ((60, 220, 60), (60, 60, 230), (0, 160, 255), (255, 220, 0),
                                                  (255, 255, 255), (255, 120, 40), (230, 80, 230))


def _pt(p) -> tuple[int, int]:
    return round(p[0]), round(p[1])


def draw(image: np.ndarray, table: Table, scene: Scene, advice: Advice | None = None,
         plan: Plan | None = None, text: bool = True) -> np.ndarray:
    out = image.copy()
    R = table.radius
    for pk in table.pockets:
        cv2.line(out, _pt(pk.a), _pt(pk.b), BLUE, 2)
    for i, b in enumerate(scene.balls):
        color = WHITE if i == scene.cue else CYAN
        cv2.circle(out, _pt(b.pos), round(R), color, 1, cv2.LINE_AA)
        if i != scene.cue:
            cv2.putText(out, "/".join(map(str, b.numbers)) or "?", (round(b.x - R), round(b.y - R - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
    for d in scene.dots:
        cv2.circle(out, _pt(d), 2, ORANGE, -1)
    if scene.ghost:
        cv2.circle(out, _pt(scene.ghost), round(R), ORANGE, 2, cv2.LINE_AA)
    if plan is not None:
        _outcome(out, plan.outcome, scene.cue, R)
        cv2.circle(out, _pt(plan.ghost), round(R), MAGENTA, 2, cv2.LINE_AA)
        cv2.drawMarker(out, _pt(plan.ghost), MAGENTA, cv2.MARKER_CROSS, 10, 1)
    elif advice is not None and advice.outcome is not None:
        _outcome(out, advice.outcome, scene.cue, R)
    if text and advice is not None:
        y = 40
        for line in advice.lines:
            _label(out, line, y)
            y += 30
    return out


def _outcome(img, out: Outcome, cue: int | None, R: float):
    for seg in out.segments:
        if seg.ball == cue:
            _dashed(img, seg.a, seg.b, RED if seg.fate == "pocket" else WHITE)
            continue
        color = GREEN if seg.fate == "pocket" else (ORANGE if seg.fate == "ball" else RED)
        cv2.line(img, _pt(seg.a), _pt(seg.b), color, 2, cv2.LINE_AA)
        cv2.circle(img, _pt(seg.b), round(R), color, 1 if seg.fate == "ball" else 2, cv2.LINE_AA)


def _label(img, line: str, y: int):
    text = _ascii(line)
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    cv2.rectangle(img, (232, y - h - 6), (248 + w, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (240, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 1, cv2.LINE_AA)


def _dashed(img, a, b, color, step=12):
    n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // step))
    for k in range(0, n, 2):
        p = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
        q = (a[0] + (b[0] - a[0]) * (k + 1) / n, a[1] + (b[1] - a[1]) * (k + 1) / n)
        cv2.line(img, _pt(p), _pt(q), color, 1, cv2.LINE_AA)


def _ascii(text: str) -> str:
    """OpenCV n'écrit que l'ASCII."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
