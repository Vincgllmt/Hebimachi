"""Fenêtre principale de Hebimachi (tkinter)."""
from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .actions import check_calls, check_closed_kan, check_riichi, check_ron, check_tsumo, evaluate_prompt
from .advisor import Advice, advise
from .capture import Config, crop, enable_dpi_awareness, grab_screen, list_monitors
from .defense import Opponent
from .efficiency import estimate_draws_left, simulate_tenpai
from .recognition import PromptReader, TileMatcher, WindReader
from .tiles import display_name, format_labels, label_to_index, parse, parse_labels, to_counts
from .yaku import Context

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CONFIG_PATH = DATA / "config.json"
OPPONENTS = ("Shimocha (droite)", "Toimen (face)", "Kamicha (gauche)")
WIND_NAMES = ["Est", "Sud", "Ouest", "Nord"]
SEAT_NAMES = {"shimocha": "Shimocha", "toimen": "Toimen", "kamicha": "Kamicha"}
RIVER_FIELDS = {"me": "mine", "shimocha": "opp0", "toimen": "opp1", "kamicha": "opp2"}
RIVER_TITLES = {
    "me": "Zone de VOTRE RIVIÈRE (prévoyez la place de 3 rangées)",
    "shimocha": "Rivière de SHIMOCHA (joueur de droite)",
    "toimen": "Rivière de TOIMEN (joueur d'en face)",
    "kamicha": "Rivière de KAMICHA (joueur de gauche)",
}
AUTO_INTERVAL_MS = 1500
HAND_THRESHOLD = 0.78   # sous ce score, l'écran est sans doute flou (menu pause) ou masqué
PANEL_BG = "#1f4d36"
PROMPT_BG = "#8a4a10"  # bandeau orange quand le jeu affiche un prompt d'action
HAND_TILE_W, HAND_TILE_H = 54, 74
CHOICE_COLORS = ("#ff3b3b", "#ffb020", "#ffe066")  # 1er, 2e, 3e choix
CALL_COLOR = "#3ddc84"  # tuiles à utiliser pour un appel


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
        self.winds = WindReader(DATA / "templates")
        self.prompts = PromptReader(DATA / "templates")
        self.prompt: list[str] = []      # actions proposées par le jeu en ce moment
        self.prompt_shown = False
        self.prev_rivers: dict[str, list[str]] = {}
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

        # Bandeau du conseil : ce qu'il faut jouer, en gros, sur l'image de la main.
        panel = tk.Frame(self.root, bg=PANEL_BG, padx=12, pady=8)
        panel.pack(fill="x", padx=6, pady=(0, 6))
        self.advice_var = tk.StringVar(value="En attente d'une main…")
        title = tk.Label(panel, textvariable=self.advice_var, bg=PANEL_BG, fg="white",
                         font=("Segoe UI", 22, "bold"), anchor="w")
        title.pack(fill="x")
        self.advice_detail = tk.StringVar()
        detail = tk.Label(panel, textvariable=self.advice_detail, bg=PANEL_BG, fg="#cfe8d6",
                          font=("Segoe UI", 12), anchor="w", justify="left", wraplength=1120)
        detail.pack(fill="x")
        self.hand_canvas = tk.Canvas(panel, height=HAND_TILE_H + 34, bg=PANEL_BG, highlightthickness=0)
        self.hand_canvas.pack(fill="x", pady=(6, 0))
        self.panel_widgets = [panel, title, detail, self.hand_canvas]
        self.tile_photos: list[ImageTk.PhotoImage] = []

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

        # Tuile que l'on peut appeler (pon/chi/kan/ron), et contexte des vents.
        field(6, "Dernière défausse adverse (appel/ron)", "offer", 10)
        extra = ttk.Frame(form)
        extra.grid(row=6, column=2, columnspan=2, sticky="w")
        self.offer_from = tk.StringVar(value=OPPONENTS[0].split()[0])
        self.seat_var = tk.StringVar(value="Sud")
        self.round_var = tk.StringVar(value="Est")
        self.my_riichi = tk.BooleanVar()
        for label, var, values in (("jetée par", self.offer_from, [o.split()[0] for o in OPPONENTS]),
                                   ("Mon vent", self.seat_var, WIND_NAMES),
                                   ("Vent de table", self.round_var, WIND_NAMES)):
            ttk.Label(extra, text=label).pack(side="left", padx=(8, 2))
            cb = ttk.Combobox(extra, textvariable=var, values=values, state="readonly", width=9)
            cb.pack(side="left")
            cb.bind("<<ComboboxSelected>>", lambda _e: self.analyze())
        ttk.Checkbutton(extra, text="Je suis en riichi", variable=self.my_riichi,
                        command=self.analyze).pack(side="left", padx=8)
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
        self.tree = ttk.Treeview(self.root, columns=cols, show="headings", height=8)
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
        self.config.hand = self.config.dora = self.config.calls = None
        self.config.rivers = {}
        self.config.save(CONFIG_PATH)
        self.last_image = None
        self.status(f"Écran {index} sélectionné : redéfinissez les zones (« Zones… »).")

    def calibrate(self):
        img = self.current_image(force_grab=True)
        for key, title in (("hand", "Zone de VOTRE MAIN (un peu de marge en hauteur)"),
                           ("dora", "Zone des INDICATEURS DORA"),
                           ("calls", "Zone de VOS APPELS (chi/pon/kan) : du milieu de la main jusqu'au bord droit")):
            sel = RegionSelector(self.root, img, title)
            self.root.wait_window(sel)
            if sel.result:
                setattr(self.config, key, sel.result)
        for seat, title in RIVER_TITLES.items():
            sel = RegionSelector(self.root, img, title)
            self.root.wait_window(sel)
            if sel.result:
                self.config.rivers[seat] = sel.result
        self.config.save(CONFIG_PATH)
        self.status(f"Zones enregistrées : main={self.config.hand} dora={self.config.dora} "
                    f"rivières={', '.join(self.config.rivers) or 'aucune'}")

    def _read_calls(self, img: np.ndarray, hand_dets) -> list | None:
        """Tuiles de nos appels, plus petites que celles de la main (ce qui écarte la tuile piochée)."""
        calls_img = crop(img, self.config.calls)
        if calls_img is None or not hand_dets:
            return None
        max_long = 0.9 * max(d.h for d in hand_dets)
        dets = self.matcher.read_tiles(calls_img, max_long=max_long)
        return dets or None

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
        calls = self._read_calls(img, dets)
        if calls is not None:
            # Les tuiles appelées ne font pas partie de la main, même si les zones se chevauchent.
            left = self.config.calls[0] + min(d.x for d in calls)
            dets = [d for d in dets if self.config.hand[0] + d.x + d.w <= left]
        melds = len(calls) // 3 if calls else 0
        worst = min((d.score for d in dets), default=0.0)
        if len(dets) + 3 * melds not in (13, 14) or worst < HAND_THRESHOLD:
            # Écran flou (menu pause), animation ou fenêtre masquée : on garde la dernière analyse.
            self.status(f"Écran illisible ({len(dets)} tuiles, score min {worst:.2f}) — jeu en pause, "
                        "animation ou fenêtre masquée ? Dernière analyse conservée.")
            return
        before = {k: v.get() for k, v in self.vars.items()}
        self.vars["hand"].set(format_labels([d.label for d in dets]))
        groups = [calls[i:i + 3] for i in range(0, 3 * melds, 3)] if calls else []
        self.vars["calls"].set(" ".join(format_labels([d.label for d in g]) for g in groups))
        dora_img = crop(img, self.config.dora)
        if dora_img is not None:
            self.vars["dora"].set(format_labels([d.label for d in self.matcher.read_tiles(dora_img)]))
        riichi_seen = []
        rivers: dict[str, list[str]] = {}
        for seat, region in self.config.rivers.items():
            river_img = crop(img, region)
            if river_img is None:
                continue
            labels, riichi = self.matcher.read_river(river_img, seat)
            rivers[seat] = labels
            self.vars[RIVER_FIELDS[seat]].set(format_labels(labels))
            if seat == "me":
                self.my_riichi.set(riichi is not None)
            else:
                self.riichi[int(RIVER_FIELDS[seat][-1])].set(riichi is not None)
                if riichi is not None:
                    riichi_seen.append(seat)
        self._update_offer(rivers, len(dets) + 3 * melds)
        self._read_winds(img)
        prompt_before = (self.prompt_shown, self.prompt)
        prompt_region = crop(img, self.config.extra.get("prompt"))
        self.prompt_shown, self.prompt = (self.prompts.read(prompt_region) if prompt_region is not None
                                          else (False, []))
        changed = (before != {k: v.get() for k, v in self.vars.items()}
                   or (self.prompt_shown, self.prompt) != prompt_before)
        low = [d for d in dets if d.score < self.config.threshold + 0.08]
        self.status(f"{len(dets)} tuiles reconnues en main" + (f" ({len(low)} incertaines)" if low else "")
                    + (f" — riichi : {', '.join(riichi_seen)}" if riichi_seen else ""))
        if changed or not self.auto.get():
            self.analyze()

    def _update_offer(self, rivers: dict[str, list[str]], hand_size: int):
        """Repère la tuile qu'un adversaire vient de jeter : c'est elle que l'on peut appeler."""
        if hand_size == 14:  # notre tour : plus rien à appeler
            self.vars["offer"].set("")
        else:
            grown = [s for s in ("shimocha", "toimen", "kamicha")
                     if s in rivers and s in self.prev_rivers
                     and len(rivers[s]) == len(self.prev_rivers[s]) + 1]
            if len(grown) == 1:
                seat = grown[0]
                self.vars["offer"].set(rivers[seat][-1])
                self.offer_from.set(SEAT_NAMES[seat])
        self.prev_rivers = rivers

    def _read_winds(self, img: np.ndarray):
        seat_region, round_region = self.config.extra.get("seat"), self.config.extra.get("round")
        seat = self.winds.read_seat(crop(img, seat_region)) if seat_region else None
        rnd = self.winds.read_round(crop(img, round_region)) if round_region else None
        if seat is not None:
            self.seat_var.set(WIND_NAMES[seat - 27])
        if rnd is not None:
            self.round_var.set(WIND_NAMES[rnd - 27])

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
            draws_left = estimate_draws_left(total_discards)
            adv = advise(hand, visible, opponents, melds, draws_left, calls)
            ctx = Context(27 + WIND_NAMES.index(self.round_var.get()), 27 + WIND_NAMES.index(self.seat_var.get()),
                          self.my_riichi.get(), calls=[parse(g) for g in calls_groups])
            actions = self._actions(hand, ctx, visible, mine, draws_left, sum(o.riichi for o in opponents), adv)
        except ValueError as exc:
            self.summary.set(f"Erreur : {exc}")
            self.advice_var.set("Main à vérifier")
            self.advice_detail.set(f"{exc}. Corrigez le champ « Main » (13 ou 14 tuiles, appels compris).")
            return
        self._show(adv, melds, actions)

    def _actions(self, hand, ctx: Context, visible, mine, draws_left, threats, adv: Advice) -> dict:
        """Ron/appels sur la défausse adverse (13 tuiles), ou tsumo/kan/riichi à notre tour (14)."""
        out: dict = {}
        if (len(hand) + 3 * len(ctx.calls)) % 3 == 1:
            words = [w for w in self.prompt if w in ("Ron", "Pon", "Chi", "Kan")]
            if words:
                out["prompt"] = {w: evaluate_prompt(hand, w, ctx, visible, draws_left, threats, mine)
                                 for w in words}
                return out
            offer = parse(self.vars["offer"].get())
            if offer:
                tile = offer[-1]
                out["offer"] = tile
                out["ron"] = check_ron(hand, tile, ctx, mine)
                out["calls"] = check_calls(hand, tile, self.offer_from.get() == SEAT_NAMES["kamicha"],
                                           ctx, visible, draws_left, threats)
            return out
        drawn = label_to_index(parse_labels(self.vars["hand"].get())[-1])
        out["drawn"] = drawn
        out["tsumo"] = check_tsumo(hand, drawn, ctx)
        out["kan"] = check_closed_kan(hand, ctx, threats)
        if adv.mode != "Défense":
            out["riichi"] = check_riichi(hand, visible, ctx, draws_left, threats)
        return out

    # ---------- affichage du conseil ----------
    def _action_text(self, adv: Advice, actions: dict) -> tuple[str, str] | None:
        """Réponse au prompt du jeu (ron, tsumo, pon, chi, kan, riichi), prioritaire sur la défausse."""
        tsumo = actions.get("tsumo")
        if tsumo is not None:
            return ("TSUMO !", tsumo.reason) if tsumo.recommend else None
        if "prompt" in actions:
            return self._prompt_text(actions["prompt"])
        if "offer" in actions:
            tile = actions["offer"]
            ron = actions.get("ron")
            if ron is not None and ron.recommend:
                return f"RON sur {display_name(tile)} !", ron.reason
            calls = actions.get("calls", [])
            notes = [f"{d.action} : {d.reason}" for d in calls]
            if ron is not None:
                notes.insert(0, f"RON : {ron.reason}")
            yes = [d for d in calls if d.recommend]
            if yes:
                d = yes[0]
                used = " + ".join(display_name(t) for t in d.used)
                return f"{d.action} {display_name(tile)}  →  OUI", f"Avec {used}. {d.reason}"
            if notes:
                return f"Passez : n'appelez pas {display_name(tile)}", "\n".join(notes)
        return None

    @staticmethod
    def _prompt_text(prompt: dict[str, list]) -> tuple[str, str]:
        """Prompt Ron/Pon/Chi/Kan : la tuile jetée n'est pas encore visible, on répond pour chaque cas."""
        lines, titles = [], []
        for word in ("Ron", "Pon", "Kan", "Chi"):
            if word not in prompt:
                continue
            results = prompt[word]
            yes = [t for t, d in results if d.recommend]
            if not results:
                titles.append(f"{word.upper()} ?")
            elif len(yes) == len(results):
                titles.append(f"{word.upper()} → OUI")
            elif not yes:
                titles.append(f"{word.upper()} → NON")
            else:
                titles.append(f"{word.upper()} : OUI si {', '.join(display_name(t) for t in yes)}")
            for t, d in results:
                used = f" (avec {' + '.join(display_name(u) for u in d.used)})" if d.used else ""
                lines.append(f"{'✔' if d.recommend else '✘'} {display_name(t)}{used} : {d.reason}")
        return "  •  ".join(titles), "\n".join(lines[:8])

    def _advice_text(self, adv: Advice, actions: dict | None = None) -> tuple[str, str]:
        a, best = adv.analysis, adv.ranked[0]
        actions = actions or {}

        def tiles_list(c) -> str:
            return ", ".join(display_name(t) for t in sorted(c.ukeire))

        action = self._action_text(adv, actions)
        if action:
            return action
        tsumo = actions.get("tsumo")
        if a.shanten == -1:
            reason = tsumo.reason if tsumo else "Déclarez la victoire (si vous avez un yaku)."
            return ("Main complète, mais pas de tsumo possible" if tsumo else "TSUMO ! Main complète"), reason
        if self.my_riichi.get() and "drawn" in actions:
            kan = actions.get("kan")
            detail = "Vous êtes en riichi : la main ne peut plus changer."
            if kan:
                detail += f" KAN fermé : {'OUI' if kan.recommend else 'NON'}. {kan.reason}"
            return f"Riichi : jetez la tuile piochée ({display_name(actions['drawn'])})", detail
        if best.discard is None:  # 13 tuiles : ce n'est pas à nous de jeter
            if best.shanten == 0:
                return "Tenpai — en attente", f"Attente : {tiles_list(best)} ({best.total} tuiles restantes)."
            return "En attente de votre pioche", f"{best.shanten}-shanten, {best.total} tuiles utiles."

        name = display_name(best.discard)
        danger = adv.danger.get(best.discard, (None, ""))[0]
        riichi = actions.get("riichi")
        if riichi is not None and riichi.recommend:
            return (f"Jetez {display_name(riichi.discard)} et déclarez RIICHI",
                    riichi.reason + self._kan_note(actions))
        if best.shanten == 0:
            title = f"Jetez {name}  →  TENPAI"
            detail = f"Attente : {tiles_list(best)} ({best.total} tuiles restantes)."
            if riichi is not None:
                detail += f" Pas de riichi (dama) : {riichi.reason}"
        else:
            title = f"Jetez {name}"
            detail = (f"Vous restez à {best.shanten}-shanten avec {best.total} tuiles utiles "
                      f"({best.p_next:.0%} de chances d'avancer à la prochaine pioche).")
        if adv.mode == "Défense":
            detail = f"Riichi adverse : tuile la plus sûre ({danger:.1f} % de risque). On abandonne la main."
        elif danger is not None:
            detail += f" Risque face au riichi : {danger:.1f} %."
        others = [display_name(c.discard) for c in adv.ranked[1:3]
                  if c.shanten == best.shanten and c.discard is not None]
        if others:
            detail += f"  Autres bons choix : {', '.join(others)}."
        if adv.warning:
            detail = f"⚠ {adv.warning}\n{detail}"
        return title, detail + self._kan_note(actions)

    @staticmethod
    def _kan_note(actions: dict) -> str:
        kan = actions.get("kan")
        if kan is None:
            return ""
        return f"\nKAN fermé possible : {'OUI' if kan.recommend else 'NON'} — {kan.reason}"

    def _hand_marks(self, labels: list[str], adv: Advice, actions: dict) -> dict[int, tuple[str, int, str]]:
        """Position dans la main -> (couleur, épaisseur, légende) des tuiles à mettre en avant."""
        marks: dict[int, tuple[str, int, str]] = {}

        def mark(tile: int, style: tuple[str, int, str], prefer_last: bool = True) -> None:
            pos = [i for i, lb in enumerate(labels) if label_to_index(lb) == tile and i not in marks]
            if pos:
                # On garde les cinq rouges, et on préfère jeter la tuile piochée (la dernière).
                plain = [i for i in pos if labels[i][0] != "0"] or pos
                marks[plain[-1] if prefer_last else plain[0]] = style

        call = next((d for d in actions.get("calls", []) if d.recommend), None)
        riichi = actions.get("riichi")
        if call is not None:
            for t in call.used:
                mark(t, (CALL_COLOR, 5, "APPEL"), prefer_last=False)
        elif self.my_riichi.get() and "drawn" in actions:
            marks[len(labels) - 1] = (CHOICE_COLORS[0], 5, "▲ JETER")
        elif riichi is not None and riichi.recommend:
            mark(riichi.discard, (CHOICE_COLORS[0], 5, "▲ RIICHI"))
        else:
            for rank, c in enumerate(c for c in adv.ranked[:3] if c.discard is not None):
                mark(c.discard, (CHOICE_COLORS[rank], 5 if rank == 0 else 3, "▲ JETER" if rank == 0 else ""))
        return marks

    def _draw_hand(self, adv: Advice, actions: dict | None = None):
        """Main en images : tuile à jeter en rouge (puis orange, jaune), tuiles d'un appel en vert."""
        cv = self.hand_canvas
        cv.delete("all")
        self.tile_photos = []
        try:
            labels = parse_labels(self.vars["hand"].get())
        except ValueError:
            return
        marks = self._hand_marks(labels, adv, actions or {})
        x = 4
        for i, lb in enumerate(labels):
            if i == len(labels) - 1 and len(labels) % 3 == 2:
                x += 14  # écart de la tuile piochée, comme dans le jeu
            y = 18
            if i in marks:
                y = 6 if marks[i][1] == 5 else 12  # le meilleur choix « sort » de la main
            tpl = self.matcher.templates.get(lb)
            if tpl is not None:
                photo = to_photo(cv2.resize(tpl, (HAND_TILE_W, HAND_TILE_H), interpolation=cv2.INTER_AREA))
                self.tile_photos.append(photo)
                cv.create_image(x, y, anchor="nw", image=photo)
            else:
                cv.create_rectangle(x, y, x + HAND_TILE_W, y + HAND_TILE_H, fill="#f4f0e6")
                cv.create_text(x + HAND_TILE_W / 2, y + HAND_TILE_H / 2, text=lb, font=("Segoe UI", 14, "bold"))
            if i in marks:
                color, width, caption = marks[i]
                cv.create_rectangle(x - 3, y - 3, x + HAND_TILE_W + 3, y + HAND_TILE_H + 3,
                                    outline=color, width=width)
                if caption:
                    cv.create_text(x + HAND_TILE_W / 2, y + HAND_TILE_H + 12, text=caption,
                                   fill=color, font=("Segoe UI", 10, "bold"))
            x += HAND_TILE_W + 4

    def _show(self, adv: Advice, melds: int, actions: dict | None = None):
        a = adv.analysis
        title, detail = self._advice_text(adv, actions)
        bg = PROMPT_BG if self.prompt_shown else PANEL_BG
        for w in self.panel_widgets:
            w.configure(bg=bg)
        if self.prompt_shown:
            title = f"⚡ {title}"
        self.advice_var.set(title)
        self.advice_detail.set(detail)
        self._draw_hand(adv, actions)
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
                try:
                    self.root.after(0, self._set_mc, gen, iid, p)
                except RuntimeError:  # fenêtre fermée entre-temps
                    return

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
