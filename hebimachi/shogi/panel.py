"""Onglet Shogi de la fenêtre : plateau, conseil et réglages."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import search
from .position import (BLACK, DROP_BASE, HAND_ORDER, K, KANJI, RANK_KANJI, WHITE, drop_piece, is_drop,
                       make_drop, make_move, move_from, move_text, move_to)
from .session import PIECE_NAMES, Session, describe, side_name
from .usi import UsiEngine, UsiError
from .vision import BoardReader, cell_image, rectify, screen_to_board, type_name

SQUARE = 54
MARGIN = 24
HAND_ROW = 46
AUTO_INTERVAL_MS = 1200
POLL_MS = 60   # le fil d'analyse dépose ses résultats dans une file, lue depuis le fil de tkinter
WOOD, LINE, BEST, SECOND, FROM_COLOR, SELECT = "#e3bd7b", "#5a3d18", "#2fbf71", "#f0a020", "#b8e6c8", "#6ab0ff"
PANEL_BG = "#1f4d36"
EDIT_TYPES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15)


class CornerSelector(tk.Toplevel):
    """Quatre clics sur les coins du plateau de jeu (lignes extérieures de la grille)."""
    STEPS = ("HAUT-GAUCHE", "HAUT-DROIT", "BAS-DROIT", "BAS-GAUCHE")

    def __init__(self, master, image: np.ndarray):
        super().__init__(master)
        self.title("Coins du plateau")
        self.result: list[tuple[int, int]] | None = None
        self.points: list[tuple[int, int]] = []
        h, w = image.shape[:2]
        self.scale = min(1500 / w, 820 / h, 1.0)
        small = cv2.resize(image, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.hint = tk.StringVar()
        ttk.Label(self, textvariable=self.hint, padding=6).pack()
        self.canvas = tk.Canvas(self, width=self.photo.width(), height=self.photo.height(),
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<Button-1>", self._click)
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<BackSpace>", lambda _e: self._back())
        self._hint()
        self.grab_set()
        self.focus_force()

    def _hint(self):
        self.hint.set(f"Cliquez le coin {self.STEPS[len(self.points)]} de la grille "
                      "(Retour arrière : annuler, Échap : abandonner)")

    def _back(self):
        if self.points:
            self.points.pop()
            self.canvas.delete(f"p{len(self.points)}")
            self._hint()

    def _click(self, e):
        tag = f"p{len(self.points)}"
        self.canvas.create_oval(e.x - 5, e.y - 5, e.x + 5, e.y + 5, outline="#ff3355", width=2, tags=tag)
        self.points.append((int(e.x / self.scale), int(e.y / self.scale)))
        if len(self.points) == 4:
            self.result = self.points
            self.destroy()
        else:
            self._hint()


class ShogiPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.session = Session()
        self.reader: BoardReader | None = None
        self.engine: UsiEngine | None = None
        self.engine_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.results: queue.Queue = queue.Queue()
        self.generation = 0
        self.lines: list[search.Line] = []
        self.selected: int | None = None      # case de départ ou DROP_BASE + type
        self.last_board: np.ndarray | None = None
        self.auto = tk.BooleanVar(value=False)
        self.side_var = tk.StringVar(value=side_name(BLACK))
        self.time_var = tk.StringVar(value="3")
        self.sfen_var = tk.StringVar()
        self.engine_var = tk.StringVar(value="Moteur : intégré")
        self.title_var = tk.StringVar(value="Prêt")
        self.detail_var = tk.StringVar(value="Calibrez le plateau, ou jouez les coups à la souris.")
        self.status_var = tk.StringVar()
        self._build()
        self.reload()
        self._poll()

    # ---------- construction ----------
    def _build(self):
        bar = ttk.Frame(self, padding=(6, 6, 6, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="📷 Lire l'écran", command=self.capture).pack(side="left")
        ttk.Checkbutton(bar, text="Auto", variable=self.auto, command=self._auto_loop).pack(side="left", padx=6)
        ttk.Button(bar, text="Coins du plateau…", command=self.calibrate).pack(side="left")
        ttk.Button(bar, text="Apprendre les pièces", command=self.learn_start).pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(bar, text="Je joue").pack(side="left", padx=(0, 3))
        box = ttk.Combobox(bar, textvariable=self.side_var, values=[side_name(BLACK), side_name(WHITE)],
                           state="readonly", width=7)
        box.pack(side="left")
        box.bind("<<ComboboxSelected>>", lambda _e: self.set_side(box.current()))
        ttk.Label(bar, text="Temps (s)").pack(side="left", padx=(10, 3))
        spin = ttk.Spinbox(bar, from_=1, to=60, textvariable=self.time_var, width=4, command=self._save)
        spin.pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(bar, textvariable=self.engine_var).pack(side="left")
        ttk.Button(bar, text="Choisir un moteur USI…", command=self.choose_engine).pack(side="left", padx=4)
        ttk.Button(bar, text="Intégré", command=self.use_builtin).pack(side="left")

        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        size = MARGIN + 9 * SQUARE + MARGIN
        self.canvas = tk.Canvas(body, width=size, height=size + 2 * HAND_ROW, bg="#caa468", highlightthickness=0)
        self.canvas.pack(side="left", anchor="n")
        self.canvas.bind("<Button-1>", self._left_click)
        self.canvas.bind("<Button-3>", self._right_click)

        side = ttk.Frame(body, padding=(10, 0, 0, 0))
        side.pack(side="left", fill="both", expand=True)
        panel = tk.Frame(side, bg=PANEL_BG, padx=12, pady=8)
        panel.pack(fill="x")
        tk.Label(panel, textvariable=self.title_var, bg=PANEL_BG, fg="white", font=("Segoe UI", 20, "bold"),
                 anchor="w", justify="left", wraplength=560).pack(fill="x")
        tk.Label(panel, textvariable=self.detail_var, bg=PANEL_BG, fg="#cfe8d6", font=("Segoe UI", 12),
                 anchor="w", justify="left", wraplength=560).pack(fill="x")

        cols = ("rank", "move", "eval", "line")
        self.tree = ttk.Treeview(side, columns=cols, show="headings", height=4)
        for c, head, width in zip(cols, ("#", "Coup", "Évaluation", "Suite prévue"), (30, 190, 90, 280)):
            self.tree.heading(c, text=head)
            self.tree.column(c, width=width, anchor="center" if c in ("rank", "eval") else "w")
        self.tree.pack(fill="x", pady=6)

        row = ttk.Frame(side)
        row.pack(fill="x")
        ttk.Button(row, text="Analyser", command=self.analyze).pack(side="left")
        ttk.Button(row, text="Jouer le conseil", command=self.play_best).pack(side="left", padx=4)
        ttk.Button(row, text="Annuler un coup", command=self.undo).pack(side="left")
        ttk.Button(row, text="Nouvelle partie", command=self.new_game).pack(side="left", padx=4)
        ttk.Button(row, text="Reprendre depuis l'écran", command=self.adopt).pack(side="left")

        sfen = ttk.LabelFrame(side, text="Position SFEN (pour coller un problème de tsume)", padding=6)
        sfen.pack(fill="x", pady=8)
        entry = ttk.Entry(sfen, textvariable=self.sfen_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda _e: self.load_sfen())
        ttk.Button(sfen, text="Charger", command=self.load_sfen).pack(side="left", padx=(4, 0))

        ttk.Label(side, wraplength=560, foreground="#555", text=(
            "Clic gauche : jouer un coup (pièce puis destination, ou pièce en main puis case). "
            "Clic droit : corriger une case ou une pièce en main. Les pièces promues s'apprennent seules "
            "pendant le suivi ; une correction à la main apprend aussi le modèle de la case.")).pack(anchor="w")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", padding=4).pack(fill="x", side="bottom")

    # ---------- configuration ----------
    def reload(self):
        """Modèles et réglages du jeu choisi."""
        self.cancel()
        self.reader = BoardReader(self.app.game.templates)
        extra = self.app.config.extra
        side = extra.get("shogi_side", BLACK)
        self.session = Session(side)
        self.side_var.set(side_name(side))
        self.time_var.set(str(extra.get("shogi_time", 3)))
        self.set_engine(extra.get("shogi_engine"))
        self.lines = []
        self.refresh()

    def _save(self):
        extra = self.app.config.extra
        extra["shogi_side"] = self.session.my_side
        extra["shogi_time"] = self._seconds()
        self.app.config.save(self.app.game.config_path)

    def _seconds(self) -> float:
        try:
            return min(60.0, max(0.5, float(self.time_var.get())))
        except ValueError:
            return 3.0

    def set_side(self, side: int):
        self.session.my_side = side
        self._save()
        self.lines = []
        self.refresh()
        self.analyze_if_my_turn()

    def set_engine(self, path: str | None):
        with self.engine_lock:
            if self.engine is not None:
                self.engine.close()
                self.engine = None
            if path:
                try:
                    self.engine = UsiEngine(path)
                    self.engine_var.set(f"Moteur : {self.engine.name}")
                    return
                except UsiError as exc:
                    self.status_var.set(f"{exc} — retour au moteur intégré.")
            self.engine_var.set("Moteur : intégré")

    def choose_engine(self):
        path = filedialog.askopenfilename(title="Moteur USI", filetypes=[("Programmes", "*.exe"), ("Tous", "*.*")])
        if not path:
            return
        self.cancel()
        self.set_engine(path)
        if self.engine is not None:
            self.app.config.extra["shogi_engine"] = path
            self.app.config.save(self.app.game.config_path)

    def use_builtin(self):
        self.cancel()
        self.set_engine(None)
        self.app.config.extra.pop("shogi_engine", None)
        self.app.config.save(self.app.game.config_path)

    def close(self):
        self.cancel()
        self.set_engine(None)

    # ---------- vision ----------
    def _board_image(self) -> np.ndarray | None:
        corners = self.app.config.extra.get("shogi_board")
        if not corners:
            self.status_var.set("Indiquez d'abord les coins du plateau (« Coins du plateau… »).")
            return None
        image = self.app.current_image(force_grab=True)
        self.last_board = rectify(image, [tuple(p) for p in corners])
        return self.last_board

    def calibrate(self):
        image = self.app.current_image(force_grab=True)
        selector = CornerSelector(self.winfo_toplevel(), image)
        self.wait_window(selector)
        if selector.result:
            self.app.config.extra["shogi_board"] = selector.result
            self.app.config.save(self.app.game.config_path)
            self.status_var.set("Plateau repéré. Ouvrez une partie, puis « Apprendre les pièces » sur la position de départ.")

    def learn_start(self):
        board = self._board_image()
        if board is None:
            return
        if not messagebox.askokcancel("Apprendre les pièces", (
                "L'écran doit montrer la position de DÉPART, sans curseur sur une pièce, "
                f"avec vos pièces en bas (vous jouez {side_name(self.session.my_side)}).\n\nContinuer ?")):
            return
        learned = self.reader.learn_initial(board)
        self.status_var.set(f"{learned} modèles appris. Reste à apprendre, quand ils apparaîtront : "
                            f"{' '.join(self.reader.missing()) or 'rien'}.")

    def capture(self):
        if not self.reader.ready:
            self.status_var.set("Aucun modèle de pièce : utilisez « Apprendre les pièces ».")
            return
        board = self._board_image()
        if board is None:
            return
        cells, unknown = self.reader.read(board)
        me = self.session.my_side
        squares = frozenset(self._square_of(i) for i in unknown)
        result = self.session.sync(screen_to_board(cells, me), squares)
        if result.state == "lost":
            self.status_var.set("L'écran ne correspond à aucune suite de coups : lecture ratée, écran masqué "
                                "ou partie reprise en route (« Reprendre depuis l'écran »). Position gardée.")
            return
        learned = self._learn_deduced(board, unknown) if result.state != "new_game" else []
        if result.state == "same":
            self.status_var.set("Position inchangée.")
        else:
            text = "Nouvelle partie détectée." if result.state == "new_game" else "Coups lus : " + " ".join(result.moves)
            if learned:
                text += f" — modèle appris : {' '.join(learned)}"
            self.status_var.set(text)
            self.lines = []
            self.refresh()
            self.analyze_if_my_turn()

    def _square_of(self, index: int) -> int:
        """Case du plateau SFEN pour une case de l'écran (la conversion est son propre inverse)."""
        return index if self.session.my_side == BLACK else 80 - index

    def _learn_deduced(self, board: np.ndarray, unknown: list[int]) -> list[str]:
        """Le suivi sait quelle pièce occupe une case non reconnue : on en fait un modèle."""
        sign = 1 if self.session.my_side == BLACK else -1
        names = []
        for i in unknown:
            piece = self.session.pos.b[self._square_of(i)] * sign
            self.reader.learn_cell(cell_image(board, i), piece)
            names.append(type_name(abs(piece)))
        return names

    def adopt(self):
        board = self._board_image()
        if board is None or not self.reader.ready:
            return
        cells, unknown = self.reader.read(board)
        if unknown:
            self.status_var.set(f"{len(unknown)} case(s) non reconnue(s) : apprenez-les d'abord (clic droit).")
            return
        self.session.adopt(screen_to_board(cells, self.session.my_side))
        self.status_var.set("Position reprise : à vous de jouer. Vérifiez les pièces en main (clic droit pour corriger).")
        self.after_change()

    def _auto_loop(self):
        if not self.auto.get():
            return
        if self.app.current_tab() is self:
            try:
                self.capture()
            except Exception as exc:  # la boucle ne doit pas mourir sur une capture ratée
                self.status_var.set(f"Erreur de capture : {exc}")
        self.after(AUTO_INTERVAL_MS, self._auto_loop)

    # ---------- partie ----------
    def after_change(self):
        self.selected = None
        self.lines = []
        self.refresh()
        self.analyze_if_my_turn()

    def new_game(self):
        self.cancel()
        self.session.reset()
        self.after_change()

    def undo(self):
        self.cancel()
        if self.session.undo():
            self.after_change()

    def load_sfen(self):
        self.cancel()
        try:
            self.session.reset(self.sfen_var.get().strip())
        except (ValueError, IndexError) as exc:
            self.status_var.set(f"SFEN invalide : {exc}")
            return
        self.after_change()

    def play_best(self):
        if self.lines and self.lines[0].move in self.session.pos.legal_moves():
            self.play(self.lines[0].move)

    def play(self, m: int):
        self.cancel()
        self.session.play(m)
        self.after_change()

    # ---------- analyse ----------
    def cancel(self):
        self.generation += 1
        self.stop_event.set()

    def analyze_if_my_turn(self):
        if self.session.my_turn:
            self.analyze()
        else:
            self.cancel()
            self.title_var.set("Au tour de l'adversaire")
            self.detail_var.set("Le conseil s'affichera dès que le coup adverse sera lu ou joué ici.")

    def analyze(self):
        self.cancel()
        self.generation += 1
        gen = self.generation
        stop = self.stop_event = threading.Event()
        pos = self.session.pos.copy()
        seconds = self._seconds()
        self.title_var.set("Analyse en cours…")
        self.detail_var.set(f"Trait à {side_name(pos.side)}, {seconds:g} s de réflexion.")

        def report(lines):
            self.results.put((gen, lines, False))

        def work():
            with self.engine_lock:
                if stop.is_set():
                    return
                try:
                    if self.engine is not None:
                        lines = self.engine.analyze(pos.sfen(), seconds, 3, stop, report)
                    else:
                        lines = search.analyze(pos, seconds, 3, stop, report)
                except UsiError as exc:
                    self.results.put((gen, f"Moteur USI : {exc}. Analyse faite par le moteur intégré.", False))
                    lines = search.analyze(pos, seconds, 3, stop, report)
            self.results.put((gen, lines, True))

        threading.Thread(target=work, daemon=True).start()

    def _poll(self):
        try:
            while True:
                gen, payload, final = self.results.get_nowait()
                if gen == self.generation:
                    if isinstance(payload, str):
                        self.status_var.set(payload)
                    else:
                        self._show_lines(payload, final)
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll)

    def _show_lines(self, lines, final: bool):
        self.lines = lines
        pos = self.session.pos
        self.tree.delete(*self.tree.get_children())
        for i, line in enumerate(lines):
            self.tree.insert("", "end", values=(i + 1, describe(pos, line.move), _eval_text(line),
                                                self._pv_text(line)))
        self.title_var.set(self._title(lines, final))
        self.detail_var.set(self._detail(lines))
        self.refresh(keep_text=True)

    def _pv_text(self, line) -> str:
        work = self.session.pos.copy()
        out = []
        for m in line.pv[:7]:
            out.append(move_text(work, m))
            work.do(m)
        return " ".join(out)

    def _title(self, lines, final: bool) -> str:
        if not lines:
            if self.session.pos.in_check():
                return "Échec et mat : partie perdue"
            return "Plus aucun coup légal"
        best = lines[0]
        text = describe(self.session.pos, best.move)
        if best.mate is not None and best.mate > 0:
            return f"Mat en {best.mate} : {text}"
        return text if final else f"{text} …"

    def _detail(self, lines) -> str:
        if not lines:
            return ""
        best = lines[0]
        source = self.engine.name if self.engine else "moteur intégré"
        if best.mate is not None and best.mate > 0:
            return f"Mat forcé, quelle que soit la défense. Suivez la variante. ({source})"
        if best.mate is not None:
            return f"L'adversaire a un mat en {-best.mate} : ce coup résiste le plus longtemps. ({source})"
        return f"{_verdict(best.score)} ({_pawns(best.score)}), profondeur {best.depth}. ({source})"

    # ---------- plateau ----------
    def _xy(self, sq: int) -> tuple[int, int]:
        """Coin haut-gauche d'une case à l'écran : nos pièces sont en bas."""
        index = sq if self.session.my_side == BLACK else 80 - sq
        r, c = divmod(index, 9)
        return MARGIN + c * SQUARE, HAND_ROW + MARGIN + r * SQUARE

    def _square_at(self, x: int, y: int) -> int | None:
        c, r = (x - MARGIN) // SQUARE, (y - HAND_ROW - MARGIN) // SQUARE
        if not (0 <= c < 9 and 0 <= r < 9):
            return None
        index = r * 9 + c
        return index if self.session.my_side == BLACK else 80 - index

    def _hand_at(self, x: int, y: int) -> tuple[int, int] | None:
        """(couleur, type) de la pièce en main sous le pointeur."""
        size = MARGIN * 2 + 9 * SQUARE + 2 * HAND_ROW
        me = self.session.my_side
        if y < HAND_ROW:
            color = 1 - me
        elif y >= size - HAND_ROW:
            color = me
        else:
            return None
        i = (x - MARGIN) // (SQUARE + 12)
        return (color, HAND_ORDER[i]) if 0 <= i < len(HAND_ORDER) else None

    def refresh(self, keep_text: bool = False):
        cv = self.canvas
        cv.delete("all")
        pos = self.session.pos
        me = self.session.my_side
        cv.create_rectangle(MARGIN, HAND_ROW + MARGIN, MARGIN + 9 * SQUARE, HAND_ROW + MARGIN + 9 * SQUARE, fill=WOOD)
        self._highlight()
        for i in range(10):
            d = i * SQUARE
            cv.create_line(MARGIN + d, HAND_ROW + MARGIN, MARGIN + d, HAND_ROW + MARGIN + 9 * SQUARE, fill=LINE)
            cv.create_line(MARGIN, HAND_ROW + MARGIN + d, MARGIN + 9 * SQUARE, HAND_ROW + MARGIN + d, fill=LINE)
        for i in range(9):
            file = 9 - i if me == BLACK else i + 1
            rank = i if me == BLACK else 8 - i
            cv.create_text(MARGIN + i * SQUARE + SQUARE / 2, HAND_ROW + MARGIN / 2, text=str(file), fill=LINE)
            cv.create_text(MARGIN + 9 * SQUARE + MARGIN / 2, HAND_ROW + MARGIN + i * SQUARE + SQUARE / 2,
                           text=RANK_KANJI[rank], fill=LINE)
        for sq, p in enumerate(pos.b):
            if p:
                x, y = self._xy(sq)
                mine = (p > 0) == (me == BLACK)
                cv.create_text(x + SQUARE / 2, y + SQUARE / 2, text=KANJI[abs(p)], angle=0 if mine else 180,
                               font=("Yu Gothic UI", 24, "bold"), fill="#a11" if abs(p) > K else "#111")
        self._draw_hand(1 - me, HAND_ROW / 2)
        self._draw_hand(me, MARGIN * 2 + 9 * SQUARE + HAND_ROW * 1.5)
        self._arrow()
        self.sfen_var.set(pos.sfen())
        if not keep_text and not self.lines:
            if not pos.legal_moves():
                self.title_var.set("Partie terminée")
                self.detail_var.set("Aucun coup légal pour le joueur au trait.")

    def _draw_hand(self, color: int, y: float):
        hand = self.session.pos.hand[color]
        best = self.lines[0].move if self.lines else None
        self.canvas.create_text(4, y, text="☗" if color == BLACK else "☖", anchor="w", font=("Segoe UI", 14))
        for i, t in enumerate(HAND_ORDER):
            x = MARGIN + i * (SQUARE + 12)
            chosen = self.selected == DROP_BASE + t and color == self.session.pos.side
            advised = (best is not None and is_drop(best) and drop_piece(best) == t
                       and color == self.session.pos.side)
            if advised or chosen:
                self.canvas.create_rectangle(x, y - 18, x + SQUARE + 6, y + 18, outline=BEST if advised else SELECT, width=3)
            self.canvas.create_text(x + 20, y, text=KANJI[t], font=("Yu Gothic UI", 19, "bold"),
                                    fill="#111" if hand[t] else "#a8905f")
            if hand[t]:
                self.canvas.create_text(x + 46, y, text=f"×{hand[t]}", font=("Segoe UI", 11))

    def _highlight(self):
        cv = self.canvas
        if self.selected is not None and self.selected < DROP_BASE:
            x, y = self._xy(self.selected)
            cv.create_rectangle(x, y, x + SQUARE, y + SQUARE, fill=SELECT, outline="")
        if self.lines:
            m = self.lines[0].move
            if not is_drop(m):
                x, y = self._xy(move_from(m))
                cv.create_rectangle(x, y, x + SQUARE, y + SQUARE, fill=FROM_COLOR, outline="")
            x, y = self._xy(move_to(m))
            cv.create_rectangle(x, y, x + SQUARE, y + SQUARE, fill=BEST, outline="")

    def _arrow(self):
        for line, color, width in zip(self.lines[:2], (BEST, SECOND), (5, 2)):
            m = line.move
            if is_drop(m):
                continue
            x0, y0 = self._xy(move_from(m))
            x1, y1 = self._xy(move_to(m))
            h = SQUARE / 2
            self.canvas.create_line(x0 + h, y0 + h, x1 + h, y1 + h, fill=color, width=width, arrow="last",
                                    arrowshape=(14, 18, 6))

    # ---------- souris ----------
    def _left_click(self, e):
        hand = self._hand_at(e.x, e.y)
        if hand is not None:
            color, t = hand
            if color == self.session.pos.side and self.session.pos.hand[color][t]:
                self.selected = DROP_BASE + t
                self.refresh(keep_text=True)
            return
        sq = self._square_at(e.x, e.y)
        if sq is None:
            return
        pos = self.session.pos
        legal = pos.legal_moves()
        if self.selected is None:
            self._select(sq)
            return
        if self.selected >= DROP_BASE:
            m = make_drop(self.selected - DROP_BASE, sq)
            if m in legal:
                self.play(m)
                return
            self.selected = None
            self._select(sq)
            return
        plain, promo = make_move(self.selected, sq), make_move(self.selected, sq, True)
        options = [m for m in (plain, promo) if m in legal]
        if not options:
            self.selected = None
            self._select(sq)
        elif len(options) == 1:
            self.play(options[0])
        else:
            self.play(promo if messagebox.askyesno("Promotion", "Promouvoir cette pièce ?") else plain)

    def _select(self, sq: int):
        p = self.session.pos.b[sq]
        mine = p and (p > 0) == (self.session.pos.side == BLACK)
        self.selected = sq if mine else None
        self.refresh(keep_text=True)

    def _right_click(self, e):
        hand = self._hand_at(e.x, e.y)
        menu = tk.Menu(self, tearoff=0)
        if hand is not None:
            color, t = hand
            name = PIECE_NAMES[t]
            menu.add_command(label=f"Ajouter un {name}", command=lambda: self._edit_hand(color, t, 1))
            menu.add_command(label=f"Retirer un {name}", command=lambda: self._edit_hand(color, t, -1))
        else:
            sq = self._square_at(e.x, e.y)
            if sq is None:
                return
            menu.add_command(label="Case vide", command=lambda: self._edit_square(sq, 0))
            for title, owner in (("À moi", 1), ("À l'adversaire", -1)):
                sub = tk.Menu(menu, tearoff=0)
                for t in EDIT_TYPES:
                    sub.add_command(label=f"{KANJI[t]}  {PIECE_NAMES[t]}",
                                    command=lambda t=t, owner=owner: self._edit_square(sq, owner * t))
                menu.add_cascade(label=title, menu=sub)
        menu.tk_popup(e.x_root, e.y_root)

    def _edit_hand(self, color: int, t: int, delta: int):
        self.cancel()
        self.session.change_hand(color, t, delta)
        self.after_change()

    def _edit_square(self, sq: int, owned: int):
        """`owned` est positif pour nous : on le traduit en couleur, et on apprend le modèle si possible."""
        self.cancel()
        sign = 1 if self.session.my_side == BLACK else -1
        self.session.set_piece(sq, owned * sign)
        if owned and self.last_board is not None:
            index = sq if self.session.my_side == BLACK else 80 - sq
            self.reader.learn_cell(cell_image(self.last_board, index), owned)
            self.status_var.set(f"Modèle appris : {type_name(abs(owned))}.")
        self.after_change()


def _verdict(score: int) -> str:
    if score >= 800:
        return "Position gagnante"
    if score >= 250:
        return "Net avantage"
    if score > -250:
        return "Position équilibrée"
    if score > -800:
        return "Position difficile"
    return "Position très difficile"


def _pawns(score: int) -> str:
    """Score en pions, sans le « -0.0 » d'un arrondi."""
    value = round(score / 100, 1)
    return f"{value:+.1f}" if value else "0.0"


def _eval_text(line) -> str:
    if line.mate is not None:
        return f"Mat en {line.mate}" if line.mate > 0 else f"Maté en {-line.mate}"
    return _pawns(line.score)
