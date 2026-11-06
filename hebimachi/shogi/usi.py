"""Client USI : fait analyser la position par un moteur externe (YaneuraOu, Fairy-Stockfish…).

Bien plus fort que la recherche intégrée. Le moteur reste lancé entre deux analyses.
"""
from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

from .position import move_from_usi
from .search import MATE, Line

HANDSHAKE_TIMEOUT = 15.0
MARGIN = 5.0  # secondes accordées au moteur au-delà du temps demandé


class UsiError(Exception):
    pass


def parse_info(line: str) -> tuple[int, Line] | None:
    """`info … multipv 2 score cp 34 pv 7g7f …` -> (2, Line) ; None si la ligne n'a pas de pv."""
    tokens = line.split()
    if "pv" not in tokens or "score" not in tokens:
        return None
    rank, depth = 1, 0
    if "multipv" in tokens:
        rank = int(tokens[tokens.index("multipv") + 1])
    if "depth" in tokens:
        depth = int(tokens[tokens.index("depth") + 1])
    i = tokens.index("score")
    kind, raw = tokens[i + 1], tokens[i + 2]
    pv = [move_from_usi(t) for t in tokens[tokens.index("pv") + 1:]]
    if not pv:
        return None
    if kind == "cp":
        return rank, Line(pv[0], int(raw), pv, None, depth)
    if raw in ("+", "-"):   # mat sans distance connue
        plies = len(pv) if raw == "+" else -len(pv)
    else:
        plies = int(raw)
    score = MATE - abs(plies)
    return rank, Line(pv[0], score if plies > 0 else -score, pv, plies, depth)


class UsiEngine:
    def __init__(self, command: list[str] | str, options: dict[str, str] | None = None):
        cmd = [command] if isinstance(command, str) else list(command)
        self.name = Path(cmd[-1]).stem
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                # Les fichiers d'évaluation sont cherchés à côté de l'exécutable.
                cwd=str(Path(cmd[-1]).parent) if Path(cmd[-1]).parent.is_dir() else None,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            raise UsiError(f"Impossible de lancer le moteur : {exc}") from exc
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.multipv_supported = False
        self._send("usi")
        for line in self._until("usiok", HANDSHAKE_TIMEOUT):
            if line.startswith("id name "):
                self.name = line[8:].strip()
            elif "name MultiPV" in line:
                self.multipv_supported = True
        for key, value in (options or {}).items():
            self._send(f"setoption name {key} value {value}")
        self._ready()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def _send(self, text: str) -> None:
        try:
            self.proc.stdin.write(text + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise UsiError("Le moteur ne répond plus") from exc

    def _until(self, word: str, timeout: float):
        """Lignes du moteur jusqu'à celle qui commence par `word` (incluse)."""
        limit = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.01, limit - time.monotonic()))
            except queue.Empty:
                raise UsiError(f"Pas de réponse du moteur ({word})") from None
            if line is None:
                raise UsiError("Le moteur s'est arrêté")
            yield line
            if line.startswith(word):
                return

    def _ready(self) -> None:
        self._send("isready")
        for _ in self._until("readyok", HANDSHAKE_TIMEOUT * 4):  # chargement de l'évaluation
            pass

    def analyze(self, sfen: str, seconds: float = 3.0, multipv: int = 3,
                stop: threading.Event | None = None, on_progress=None) -> list[Line]:
        if self.multipv_supported:
            self._send(f"setoption name MultiPV value {multipv}")
        self._send("usinewgame")
        self._ready()
        self._send(f"position sfen {sfen}")
        self._send(f"go byoyomi {int(seconds * 1000)}")
        best: dict[int, Line] = {}
        stopped = False
        limit = time.monotonic() + seconds + MARGIN
        while True:
            if stop is not None and stop.is_set() and not stopped:
                self._send("stop")
                stopped = True
            try:
                line = self.lines.get(timeout=0.05)
            except queue.Empty:
                if time.monotonic() > limit:
                    self._send("stop")
                    limit += MARGIN
                    if stopped:
                        raise UsiError("Le moteur ne rend pas de coup") from None
                    stopped = True
                continue
            if line is None:
                raise UsiError("Le moteur s'est arrêté")
            if line.startswith("info"):
                parsed = parse_info(line)
                if parsed:
                    best[parsed[0]] = parsed[1]
                    if on_progress:
                        on_progress([best[k] for k in sorted(best)])
            elif line.startswith("bestmove"):
                move = line.split()[1]
                if move in ("resign", "win", "none"):
                    return [best[k] for k in sorted(best)]
                chosen = move_from_usi(move)
                lines = [best[k] for k in sorted(best)]
                if not lines or lines[0].move != chosen:
                    lines = [Line(chosen, 0, [chosen])] + [x for x in lines if x.move != chosen]
                return lines

    def close(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=2)
        except (UsiError, subprocess.TimeoutExpired):
            self.proc.kill()
