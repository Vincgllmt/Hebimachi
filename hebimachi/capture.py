"""Capture d'écran et configuration des zones (main, dora)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mss
import numpy as np

Region = tuple[int, int, int, int]  # x, y, largeur, hauteur (relatif à l'écran capturé)


def enable_dpi_awareness() -> None:
    """Sous Windows, aligne les coordonnées tkinter sur les pixels physiques de mss."""
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        pass


@dataclass
class Config:
    monitor: int = 1
    hand: Region | None = None
    dora: Region | None = None
    calls: Region | None = None      # zone de nos appels (chi/pon/kan), en bas à droite
    rivers: dict[str, Region] = field(default_factory=dict)  # me / shimocha / toimen / kamicha
    threshold: float = 0.72
    extra: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Config":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for k in ("hand", "dora", "calls"):
                if data.get(k):
                    data[k] = tuple(data[k])
            data["rivers"] = {k: tuple(v) for k, v in data.get("rivers", {}).items()}
            return cls(**data)
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def list_monitors() -> list[str]:
    """Libellés des écrans physiques (l'index 0 de mss est le bureau virtuel entier)."""
    with mss.mss() as sct:
        return [f"Écran {i} — {m['width']}×{m['height']} @ ({m['left']}, {m['top']})"
                for i, m in enumerate(sct.monitors) if i > 0]


def grab_screen(monitor: int = 1) -> np.ndarray:
    """Capture complète d'un écran, en BGR."""
    with mss.mss() as sct:
        mon = sct.monitors[min(monitor, len(sct.monitors) - 1)]
        img = np.asarray(sct.grab(mon))
    return np.ascontiguousarray(img[:, :, :3])


def crop(image: np.ndarray, region: Region | None) -> np.ndarray | None:
    if region is None:
        return None
    x, y, w, h = region
    out = image[max(0, y):y + h, max(0, x):x + w]
    return out if out.size else None
