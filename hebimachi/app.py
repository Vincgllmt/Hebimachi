"""Fenêtre principale de Hebimachi (tkinter)."""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .advisor import Advice, advise
from .capture import Config, crop, enable_dpi_awareness, grab_screen, list_monitors
from .defense import Opponent
from .efficiency import estimate_draws_left, simulate_tenpai
from .recognition import TileMatcher
from .tiles import display_name, format_labels, parse, parse_labels, to_counts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIG_PATH = DATA / "config.json"
OPPONENTS = ("Shimocha (droite)", "Toimen (face)", "Kamicha (gauche)")
AUTO_INTERVAL_MS = 1500


def to_photo(img_bgr: np.ndarray, scale: float = 1.0) -> ImageTk.PhotoImage:
    if scale != 1.0:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))


class RegionSelector(tk.Toplevel):
    """Affiche une image et laisse l'utilisateur tracer un rectangle (Échap = ignorer)."""

    def __init__(self, master, image: np.ndarray, title: str, max_size=(1600, 900), min_scale=0.2):
        super().__init__(master)
        self.title(title)
        self.result: tuple[int, int, int, int] | None = None
        h, w = image.shape[:2]
        self.scale = max(min_scale, min(max_size[0] / w, max_size[1] / h, 3.0))
        self.photo = to_photo(image, self.scale)
        ttk.Label(self, text=title + "  —  glissez pour sélectionner, Entrée pour valider, Échap pour ignorer",
                  padding=6).pack()
        self.canvas = tk.Canvas(self, width=self.photo.width(), height=self.photo.height(),
                                cursor="crosshair", highlightthickness=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.rect = None
        self.start = (0, 0)
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.bind("<Return>", lambda e: self.destroy())
        self.bind("<Escape>", lambda e: self._cancel())
        self.grab_set()
        self.focus_force()

    def _press(self, e):
        self.start = (e.x, e.y)
        if self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ff3355", width=2)

    def _drag(self, e):
        x0, y0 = self.start
        self.canvas.coords(self.rect, x0, y0, e.x, e.y)
        xa, xb = sorted((x0, e.x))
        ya, yb = sorted((y0, e.y))
        s = self.scale
        if xb - xa > 3 and yb - ya > 3:
            self.result = (int(xa / s), int(ya / s), int((xb - xa) / s), int((yb - ya) / s))

    def _cancel(self):
        self.result = None
        self.destroy()


class LearnDialog(tk.Toplevel):
    """Apprentissage des modèles : sélectionner une rangée de tuiles contiguës et les nommer."""

    def __init__(self, app: "App"):
        super().__init__(app.root)
        self.app = app
        self.title("Apprendre les tuiles")
        self.strip = None
        ttk.Label(self, wraplength=900, padding=6, text=(
            "1. Encadrez précisément des tuiles CONTIGUËS de votre main (sans la tuile piochée, "
            "qui est séparée).\n2. Tapez ces tuiles de gauche à droite, ex. 1123m406p77z "
            "(0 = cinq rouge).\n3. Enregistrez, puis recommencez avec d'autres mains jusqu'à "
            "ce qu'il ne manque plus rien.")).pack(anchor="w")
        self.canvas = tk.Canvas(self, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(padx=6)
        row = ttk.Frame(self, padding=6)
        row.pack(fill="x")
        ttk.Label(row, text="Tuiles :").pack(side="left")
        self.labels_var = tk.StringVar()
        ttk.Entry(row, textvariable=self.labels_var, width=40).pack(side="left", padx=4)
        ttk.Button(row, text="Enregistrer", command=self.save).pack(side="left")
        ttk.Button(row, text="Nouvelle capture", command=self.refresh).pack(side="left", padx=4)
        self.missing_var = tk.StringVar()
        ttk.Label(self, textvariable=self.missing_var, wraplength=900, padding=6,
                  foreground="#a33").pack(anchor="w")
        self.rect = None
        self.sel = None
        self.canvas.bind("<ButtonPress-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.refresh()

    def refresh(self):
        img = self.app.current_image(force_grab=True)
        self.strip = crop(img, self.app.config.hand) if img is not None else None
        if self.strip is None:
            messagebox.showwarning("Zone manquante", "Définissez d'abord la zone de la main.", parent=self)
            self.destroy()
            return
        h, w = self.strip.shape[:2]
        self.scale = min(1500 / w, 2.0)
        self.photo = to_photo(self.strip, self.scale)
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.sel = None
        self._update_missing()

    def _update_missing(self):
        miss = self.app.matcher.missing()
        self.missing_var.set("Modèles manquants : " + (" ".join(miss) if miss else "aucun ✔"))

    def _press(self, e):
        self.start = (e.x, e.y)
        self.rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#ff3355", width=2)
        self.canvas.delete("sel")
        self.canvas.itemconfig(self.rect, tags="sel")

    def _drag(self, e):
        x0, y0 = self.start
        self.canvas.coords(self.rect, x0, y0, e.x, e.y)
        xa, xb = sorted((x0, e.x))
        ya, yb = sorted((y0, e.y))
        s = self.scale
        self.sel = (int(xa / s), int(ya / s), int(xb / s), int(yb / s))

    def save(self):
        try:
            labels = parse_labels(self.labels_var.get())
        except ValueError as exc:
            messagebox.showerror("Notation", str(exc), parent=self)
            return
        if not self.sel or not labels:
            messagebox.showwarning("Sélection", "Encadrez les tuiles et saisissez-les.", parent=self)
            return
        x0, y0, x1, y1 = self.sel
        self.app.matcher.learn_from_strip(self.strip[y0:y1, x0:x1], labels)
        self.labels_var.set("")
        self.canvas.delete("sel")
        self.sel = None
        self._update_missing()
        self.app.status(f"{len(labels)} modèles enregistrés.")


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Hebimachi — assistant Riichi Mahjong")
        root.geometry("1180x760")
        self.config = Config.load(CONFIG_PATH)
        self.matcher = TileMatcher(DATA / "templates")
        self.loaded_image: np.ndarray | None = None
        self.last_image: np.ndarray | None = None
        self.auto = tk.BooleanVar(value=False)
        self.generation = 0
        self._build()
        if not self.config.hand:
            self.status("Première utilisation : « Zones… » puis « Apprendre les tuiles… ». "
                        "Vous pouvez aussi saisir la main à la main et cliquer sur Analyser.")

    # ---------- construction de l'interface ----------
    def _build(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="📷 Capturer", command=self.capture).pack(side="left")
        ttk.Checkbutton(bar, text="Auto", variable=self.auto, command=self._auto_loop).pack(side="left", padx=6)
        ttk.Button(bar, text="Ouvrir une image…", command=self.open_image).pack(side="left")
        ttk.Button(bar, text="Écran en direct", command=self.use_live).pack(side="left", padx=4)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        monitors = list_monitors()
        self.monitor_var = tk.StringVar(value=monitors[min(self.config.monitor, len(monitors)) - 1])
        box = ttk.Combobox(bar, textvariable=self.monitor_var, values=monitors, state="readonly", width=34)
        box.pack(side="left", padx=(0, 6))
        box.bind("<<ComboboxSelected>>", lambda _e: self.set_monitor(box.current() + 1))
        ttk.Button(bar, text="Zones…", command=self.calibrate).pack(side="left")
        ttk.Button(bar, text="Apprendre les tuiles…", command=lambda: LearnDialog(self)).pack(side="left", padx=4)

        form = ttk.LabelFrame(self.root, text="Situation (corrigez au besoin — notation 123m406p77z)", padding=8)
        form.pack(fill="x", padx=6)
        self.vars: dict[str, tk.StringVar] = {}

        def field(row, label, key, width=44, col=0):
            ttk.Label(form, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=2)
            v = self.vars[key] = tk.StringVar()
            e = ttk.Entry(form, textvariable=v, width=width)
            e.grid(row=row, column=col + 1, sticky="we", pady=2)
            e.bind("<Return>", lambda _e: self.analyze())

        field(0, "Main (13 ou 14 tuiles)", "hand")
        field(0, "Mes appels (groupes séparés par espaces)", "calls", 28, col=2)
        field(1, "Indicateurs dora", "dora")
        field(1, "Mes défausses", "mine", 28, col=2)
        self.riichi: list[tk.BooleanVar] = []
        for i, name in enumerate(OPPONENTS):
            field(2 + i, f"Défausses {name}", f"opp{i}")
            v = tk.BooleanVar()
            ttk.Checkbutton(form, text="Riichi", variable=v, command=self.analyze).grid(
                row=2 + i, column=2, sticky="w", padx=8)
            self.riichi.append(v)
        field(5, "Autres tuiles visibles (appels adverses…)", "other")
        ttk.Button(form, text="Analyser", command=self.analyze).grid(row=5, column=3, sticky="e")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        self.summary = tk.StringVar(value="—")
        ttk.Label(self.root, textvariable=self.summary, padding=(8, 6), font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.explain = tk.StringVar()
        ttk.Label(self.root, textvariable=self.explain, padding=(8, 0), wraplength=1100).pack(anchor="w")

        cols = ("rank", "discard", "shanten", "accept", "n", "pnext", "pwithin", "ptenpai", "danger", "why")
        heads = ("#", "Défausse", "Shanten", "Tuiles acceptées", "Ukeire", "P(pioche)",
                 "P(avance)", "P(tenpai) MC", "Danger", "Défense")
        widths = (30, 70, 60, 260, 55, 75, 75, 90, 65, 330)
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=14)
        for c, h, w in zip(cols, heads, widths):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="w" if c in ("accept", "why") else "center")
        self.tree.tag_configure("best", background="#d9f2dc")
        self.tree.pack(fill="both", expand=True, padx=6, pady=6)

        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", padding=4).pack(fill="x", side="bottom")

    def status(self, text: str):
        self.status_var.set(text)

    # ---------- capture et vision ----------
    def current_image(self, force_grab: bool = False) -> np.ndarray | None:
        if self.loaded_image is not None:
            return self.loaded_image
        if force_grab or self.last_image is None:
            self.last_image = grab_screen(self.config.monitor)
        return self.last_image

    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if path:
            self.loaded_image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
            self.status(f"Image chargée : {path}")
            self.capture()

    def use_live(self):
        self.loaded_image = None
        self.status("Source : écran en direct.")

    def set_monitor(self, index: int):
        if index == self.config.monitor:
            return
        self.config.monitor = index
        # Les zones sont relatives à l'écran : elles ne valent plus rien sur un autre écran.
        self.config.hand = self.config.dora = None
        self.config.save(CONFIG_PATH)
        self.last_image = None
        self.status(f"Écran {index} sélectionné : redéfinissez les zones (« Zones… »).")

    def calibrate(self):
        img = self.current_image(force_grab=True)
        for key, title in (("hand", "Zone de VOTRE MAIN (un peu de marge en hauteur)"),
                           ("dora", "Zone des INDICATEURS DORA")):
            sel = RegionSelector(self.root, img, title)
            self.root.wait_window(sel)
            if sel.result:
                setattr(self.config, key, sel.result)
        self.config.save(CONFIG_PATH)
        self.status(f"Zones enregistrées : main={self.config.hand} dora={self.config.dora}")

    def capture(self):
        if not self.config.hand:
            self.status("Définissez d'abord les zones (bouton « Zones… »).")
            return
        if not self.matcher.templates:
            self.status("Aucun modèle de tuile : utilisez « Apprendre les tuiles… ».")
            return
        img = self.current_image(force_grab=True)
        hand_img = crop(img, self.config.hand)
        dets = self.matcher.detect(hand_img, threshold=self.config.threshold) if hand_img is not None else []
        hand_txt = format_labels([d.label for d in dets])
        changed = hand_txt != self.vars["hand"].get()
        self.vars["hand"].set(hand_txt)
        dora_img = crop(img, self.config.dora)
        if dora_img is not None:
            dora = self.matcher.detect_multiscale(dora_img, threshold=self.config.threshold)
            self.vars["dora"].set(format_labels([d.label for d in dora]))
        low = [d for d in dets if d.score < self.config.threshold + 0.08]
        self.status(f"{len(dets)} tuiles reconnues" + (f" ({len(low)} incertaines)" if low else ""))
        if changed or not self.auto.get():
            self.analyze()

    def _auto_loop(self):
        if self.auto.get():
            try:
                self.capture()
            except Exception as exc:  # la boucle ne doit pas mourir sur une capture ratée
                self.status(f"Erreur de capture : {exc}")
            self.root.after(AUTO_INTERVAL_MS, self._auto_loop)

    # ---------- analyse ----------
    def _tiles(self, key: str) -> list[int]:
        return parse(self.vars[key].get())

    def analyze(self):
        try:
            hand = self._tiles("hand")
            calls_groups = self.vars["calls"].get().split()
            calls = [t for g in calls_groups for t in parse(g)]
            melds = len(calls_groups)
            opponents = [Opponent(OPPONENTS[i].split()[0], self._tiles(f"opp{i}"), self.riichi[i].get())
                         for i in range(3)]
            mine = self._tiles("mine")
            visible_tiles = self._tiles("dora") + mine + calls + self._tiles("other")
            visible_tiles += [t for o in opponents for t in o.discards]
            visible = to_counts(visible_tiles)
            total_discards = len(mine) + sum(len(o.discards) for o in opponents)
            adv = advise(hand, visible, opponents, melds, estimate_draws_left(total_discards))
        except ValueError as exc:
            self.summary.set(f"Erreur : {exc}")
            return
        self._show(adv, melds)

    def _show(self, adv: Advice, melds: int):
        a = adv.analysis
        state = {-1: "Main complète (agari) !", 0: "Tenpai"}.get(a.shanten, f"{a.shanten}-shanten")
        self.summary.set(f"{state}   •   Mode : {adv.mode}   •   Tuiles invisibles : {a.unseen}"
                         f"   •   Pioches restantes estimées : {a.draws_left}")
        self.explain.set(adv.explanation)
        self.tree.delete(*self.tree.get_children())
        rows = {}
        for i, c in enumerate(adv.ranked):
            accept = " ".join(f"{display_name(t)}×{n}" for t, n in sorted(c.ukeire.items()))
            d, why = adv.danger.get(c.discard, (None, "")) if c.discard is not None else (None, "")
            iid = self.tree.insert("", "end", tags=("best",) if i == 0 else (), values=(
                i + 1,
                display_name(c.discard) if c.discard is not None else "—",
                c.shanten, accept, c.total,
                f"{c.p_next:.0%}", f"{c.p_within:.0%}",
                "…" if 1 <= c.shanten <= 4 and i < 6 else "",
                f"{d:.1f} %" if d is not None else "", why))
            rows[iid] = c

        # Monte Carlo en tâche de fond sur les meilleures options.
        self.generation += 1
        gen = self.generation
        hand = to_counts(parse(self.vars["hand"].get()))
        todo = [(iid, c) for iid, c in list(rows.items())[:6] if 1 <= c.shanten <= 4]

        def work():
            for iid, c in todo:
                counts = list(hand)
                if c.discard is not None:
                    counts[c.discard] -= 1
                p = simulate_tenpai(counts, melds, a.remaining, a.draws_left, n_sims=300)
                self.root.after(0, self._set_mc, gen, iid, p)

        if todo:
            threading.Thread(target=work, daemon=True).start()

    def _set_mc(self, gen: int, iid: str, p: float):
        if gen == self.generation and self.tree.exists(iid):
            self.tree.set(iid, "ptenpai", f"{p:.0%}")


def main():
    enable_dpi_awareness()
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
