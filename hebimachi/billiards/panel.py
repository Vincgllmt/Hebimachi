"""Onglet Billard de la fenêtre : verdict sur la visée du jeu et recherche de coups.

- « Lire l'écran » (ou Auto) : lit la table et juge la visée affichée par le jeu.
- « Chercher un coup » : liste les coups possibles (direct, combinaison, carambolage, bande),
  du plus sûr au moins sûr. En choisir un affiche le fantôme à reproduire (cercle magenta) ; en
  Auto, l'onglet dit ensuite de combien décaler le fantôme du jeu pour le rejoindre.
"""
from __future__ import annotations

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .render import draw
from .search import KINDS, Plan, find_shots, legal_first
from .shot import Advice, aim_feedback, analyze
from .table import Table
from .vision import Scene, normalize, read_scene

AUTO_INTERVAL_MS = 700
POLL_MS = 60
PREVIEW_WIDTH = 860
MOVED_PX = 3          # une bille qui bouge de plus : la table a changé, les coups trouvés sont périmés
PANEL_BG = "#1f4d36"
VERDICT = {   # titre, couleur
    "empochee": ("Rentre", "#3ddc84"),
    "ratee": ("Rate", "#ff6b6b"),
    "bande": ("Aucune bille touchée", "#f0a020"),
    "pas_de_visee": ("Pas de visée", "#cfe8d6"),
}
FILTERS = ("Tous les coups",) + tuple(k.capitalize() for k in KINDS)
FIRST = ("NEXT (lue sur le HUD)", "N'importe quelle bille")


def same_table(a: Scene | None, b: Scene) -> bool:
    if a is None or len(a.balls) != len(b.balls):
        return False
    return all(min(math.dist(p.pos, q.pos) for q in a.balls) <= MOVED_PX for p in b.balls)


class BilliardsPanel(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.table: Table | None = None
        self.auto = tk.BooleanVar(value=False)
        self.filter_var = tk.StringVar(value=FILTERS[0])
        self.first_var = tk.StringVar(value=FIRST[0])
        self.title_var = tk.StringVar(value="Prêt")
        self.detail_var = tk.StringVar(value="Visez dans le jeu (vue de dessus), puis « Lire l'écran » ou cochez Auto. "
                                             "« Chercher un coup » propose des coups, combinaisons et carambolages.")
        self.status_var = tk.StringVar()
        self.results: queue.Queue = queue.Queue()
        self.busy = False
        self.photo: ImageTk.PhotoImage | None = None
        self.image: np.ndarray | None = None
        self.scene: Scene | None = None
        self.advice: Advice | None = None
        self.plans: list[Plan] = []
        self.shown: list[Plan] = []      # après filtre, dans l'ordre de la liste
        self.plan: Plan | None = None    # coup choisi
        self.plans_scene: Scene | None = None
        self._build()
        self.reload()
        self._poll()

    def _build(self):
        bar = ttk.Frame(self, padding=(6, 6, 6, 2))
        bar.pack(fill="x")
        ttk.Button(bar, text="📷 Lire l'écran", command=self.capture).pack(side="left")
        ttk.Checkbutton(bar, text="Auto", variable=self.auto, command=self._auto_loop).pack(side="left", padx=6)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Button(bar, text="🎯 Chercher un coup", command=lambda: self.capture(search=True)).pack(side="left")
        box = ttk.Combobox(bar, textvariable=self.filter_var, values=FILTERS, state="readonly", width=15)
        box.pack(side="left", padx=6)
        box.bind("<<ComboboxSelected>>", lambda _e: self._fill_list())
        ttk.Label(bar, text="Première bille :").pack(side="left", padx=(6, 3))
        first = ttk.Combobox(bar, textvariable=self.first_var, values=FIRST, state="readonly", width=22)
        first.pack(side="left")
        first.bind("<<ComboboxSelected>>", lambda _e: self.capture(search=True))

        panel = tk.Frame(self, bg=PANEL_BG, padx=12, pady=8)
        panel.pack(fill="x", padx=6, pady=(4, 0))
        self.title_label = tk.Label(panel, textvariable=self.title_var, bg=PANEL_BG, fg="white",
                                    font=("Segoe UI", 18, "bold"), anchor="w", justify="left", wraplength=1100)
        self.title_label.pack(fill="x")
        tk.Label(panel, textvariable=self.detail_var, bg=PANEL_BG, fg="#cfe8d6", font=("Segoe UI", 11),
                 anchor="w", justify="left", wraplength=1100).pack(fill="x")

        ttk.Label(self, textvariable=self.status_var, relief="sunken", padding=4).pack(fill="x", side="bottom")
        body = ttk.Frame(self, padding=6)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Label(body, bg="#202020")
        self.canvas.pack(side="left", anchor="n")
        side = ttk.Frame(body, padding=(8, 0, 0, 0))
        side.pack(side="left", fill="both", expand=True)
        ttk.Label(side, text="Coups possibles (du plus sûr au moins sûr) :").pack(anchor="w")
        cols = (("kind", "Type", 150), ("margin", "Marge", 60), ("text", "Déroulé", 420))
        self.tree = ttk.Treeview(side, columns=[c for c, _, _ in cols], show="headings", height=18)
        for c, h, w in cols:
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center" if c == "margin" else "w", stretch=c == "text")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self._select())
        ttk.Label(side, foreground="#666", wraplength=600, justify="left",
                  text="Marge : de combien de pixels le fantôme peut s'écarter du cercle magenta sans changer le "
                       "coup. Modèle sans effet ni frottement entre billes : à confirmer en jeu.").pack(anchor="w", pady=4)

    def reload(self):
        """Calibration de la table pour le jeu choisi."""
        path = self.app.game.dir / "billiards.json"
        self.table = Table.load(path) if path.exists() else None
        self.plans, self.plan, self.plans_scene = [], None, None
        self._fill_list()
        self.status_var.set("" if self.table else
                            f"Billard non calibré pour {self.app.game.name} (pas de billiards.json).")

    # ---------- lecture ----------
    def capture(self, search: bool = False):
        if self.table is None:
            self.status_var.set(f"Billard non calibré pour {self.app.game.name} (pas de billiards.json).")
            return
        if self.busy:
            return
        image = self.app.current_image(force_grab=True)
        if image is None:
            return
        self.busy = True
        if search:
            self.status_var.set("Recherche des coups…")
        any_ball = self.first_var.get() == FIRST[1]
        args = (image, self.table, search, self.plans_scene, any_ball, self.auto.get())
        threading.Thread(target=self._work, args=args, daemon=True).start()

    def _work(self, image: np.ndarray, table: Table, search: bool, known: Scene | None, any_ball: bool,
              auto: bool):
        try:
            image = normalize(image, table)
            scene = read_scene(image, table)
            legal = {i for i in range(len(scene.balls)) if i != scene.cue} if any_ball else legal_first(scene)
            advice = analyze(scene, table, legal=legal)
            plans = None
            # en Auto, on recherche seul dès que la table a changé (nouveau tour)
            if search or (auto and known is not None and not same_table(known, scene)):
                plans = find_shots(scene, table, legal=legal)
            self.results.put((image, scene, advice, plans))
        except Exception as exc:  # affiché dans la barre d'état, la boucle Auto continue
            self.results.put(exc)

    def _poll(self):
        try:
            while True:
                item = self.results.get_nowait()
                self.busy = False
                if isinstance(item, Exception):
                    self.status_var.set(f"Erreur de lecture : {item}")
                    continue
                image, scene, advice, plans = item
                self.image, self.scene, self.advice = image, scene, advice
                if plans is not None:
                    self.plans, self.plans_scene = plans, scene
                    self.plan = None
                    self._fill_list()
                elif self.plans_scene is not None and not same_table(self.plans_scene, scene):
                    self.plans, self.plan, self.plans_scene = [], None, None
                    self._fill_list()
                self.show()
        except queue.Empty:
            pass
        self.after(POLL_MS, self._poll)

    # ---------- affichage ----------
    def _fill_list(self):
        self.tree.delete(*self.tree.get_children())
        wanted = self.filter_var.get()
        kind = None if wanted == FILTERS[0] else wanted.lower()
        self.shown = [p for p in self.plans if kind is None or kind in p.kinds]
        for i, p in enumerate(self.shown):
            self.tree.insert("", "end", iid=str(i),
                             values=(" + ".join(p.kinds), f"{p.margin_px:.1f} px", p.text))
        if self.shown:
            self.tree.selection_set("0")

    def _select(self):
        sel = self.tree.selection()
        self.plan = self.shown[int(sel[0])] if sel else None
        self.show()

    def show(self):
        if self.scene is None or self.image is None:
            return
        scene, advice, plan = self.scene, self.advice, self.plan
        if plan is not None:
            feedback = aim_feedback(scene, plan, self.table)
            self.title_var.set(feedback or "Placez le fantôme du jeu sur le cercle magenta.")
            self.title_label.configure(fg="#3ddc84" if feedback and feedback.startswith("✔") else "white")
            self.detail_var.set(f"Coup choisi ({' + '.join(plan.kinds)}) : {plan.text}\n"
                                f"Visée actuelle : {' '.join(advice.lines) if advice else '—'}")
        elif advice is not None:
            title, color = VERDICT.get(advice.verdict, (advice.verdict, "white"))
            if advice.foul:
                title, color = "Faute — " + title, "#ff6b6b"
            if advice.kinds:
                title += f" ({' + '.join(advice.kinds)})"
            self.title_var.set(title)
            self.title_label.configure(fg=color)
            self.detail_var.set("\n".join(advice.lines))
        hud = (f"NEXT = {scene.next_color}" if scene.next_color
               else "HUD visible, NEXT illisible" if scene.hud else "HUD caché : pas de contrôle de faute")
        cut = f" — coupe {advice.cut:.0f}°" if advice and advice.cut is not None else ""
        found = f" — {len(self.plans)} coups trouvés" if self.plans_scene is not None else ""
        self.status_var.set(f"{len(scene.balls)} billes lues — {hud}{cut}{found}")
        annotated = draw(self.image, self.table, scene, advice, plan=plan, text=False)
        h, w = annotated.shape[:2]
        small = cv2.resize(annotated, (PREVIEW_WIDTH, round(h * PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.canvas.configure(image=self.photo)

    def _auto_loop(self):
        if not self.auto.get():
            return
        if self.app.current_tab() is self:
            self.capture()
        self.after(AUTO_INTERVAL_MS, self._auto_loop)
