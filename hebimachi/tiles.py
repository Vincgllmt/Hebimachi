"""Représentation des tuiles.

Indices 0..33 :
  0-8   man   (1m..9m)
  9-17  pin   (1p..9p)
  18-26 sou   (1s..9s)
  27-33 honneurs (1z=Est, 2z=Sud, 3z=Ouest, 4z=Nord, 5z=Blanc, 6z=Vert, 7z=Rouge)

Notation texte : "123m406p55z" ; le chiffre 0 désigne un 5 rouge (aka).
"""
from __future__ import annotations

SUIT_CHARS = "mpsz"
NUM_KINDS = 34
HONOR_NAMES = ["Est", "Sud", "Ouest", "Nord", "Blanc", "Vert", "Rouge"]
TERMINALS_HONORS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)

# Toutes les étiquettes de modèles reconnaissables (34 tuiles + 3 cinq rouges).
ALL_LABELS = (
    [f"{n}{s}" for s in "mps" for n in range(1, 10)]
    + [f"{n}z" for n in range(1, 8)]
    + ["0m", "0p", "0s"]
)


def parse_labels(text: str) -> list[str]:
    """'1230m 55z' -> ['1m', '2m', '3m', '0m', '5z', '5z'] (conserve les cinq rouges)."""
    labels: list[str] = []
    pending: list[str] = []
    for ch in text.replace(" ", "").replace(",", ""):
        if ch.isdigit():
            pending.append(ch)
        elif ch in SUIT_CHARS:
            if not pending:
                raise ValueError(f"Couleur '{ch}' sans chiffre")
            for d in pending:
                if ch == "z" and not 1 <= int(d) <= 7:
                    raise ValueError(f"Honneur invalide : {d}z")
                labels.append(d + ch)
            pending = []
        else:
            raise ValueError(f"Caractère inattendu : {ch!r}")
    if pending:
        raise ValueError("Chiffres sans couleur à la fin")
    return labels


def label_to_index(label: str) -> int:
    num, suit = int(label[0]), label[1]
    if num == 0:
        num = 5
    return SUIT_CHARS.index(suit) * 9 + num - 1


def parse(text: str) -> list[int]:
    """'123m0p' -> [0, 1, 2, 13]"""
    return [label_to_index(lb) for lb in parse_labels(text)]


def to_counts(tiles: list[int]) -> list[int]:
    counts = [0] * NUM_KINDS
    for t in tiles:
        counts[t] += 1
    return counts


def name(idx: int) -> str:
    return f"{idx % 9 + 1}{SUIT_CHARS[idx // 9]}"


def display_name(idx: int) -> str:
    """Nom lisible : '5p', ou 'Est' pour les honneurs."""
    if idx >= 27:
        return HONOR_NAMES[idx - 27]
    return name(idx)


def format_tiles(tiles: list[int]) -> str:
    """[0, 1, 2, 31] -> '123m5z' (triée)."""
    out = []
    for s, suit in enumerate(SUIT_CHARS):
        nums = sorted(t % 9 + 1 for t in tiles if t // 9 == s)
        if nums:
            out.append("".join(map(str, nums)) + suit)
    return "".join(out)


def format_labels(labels: list[str]) -> str:
    """Formate des étiquettes en conservant l'ordre et les cinq rouges."""
    out, current, suit = [], "", None
    for lb in labels:
        if lb[1] != suit and suit is not None:
            out.append(current + suit)
            current = ""
        suit = lb[1]
        current += lb[0]
    if suit is not None:
        out.append(current + suit)
    return "".join(out)


def is_number(idx: int) -> bool:
    return idx < 27
