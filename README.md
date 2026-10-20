# Hebimachi — assistant probabiliste de Riichi Mahjong

Lit l'écran du mahjong de **Like a Dragon: Ishin!** (vue de dessus 2D), reconnaît vos tuiles et classe les défausses selon
l'efficacité (shanten, ukeire, probabilités) et la défense face aux riichi.

## Lancer

```
pip install -r requirements.txt
python -m hebimachi
```

## Première utilisation

1. **Zones…** : encadrez la zone de votre main (avec un peu de marge en hauteur),
   puis celle des indicateurs dora (Échap pour ignorer).
2. **Apprendre les tuiles…** : encadrez des tuiles contiguës de votre main, tapez-les
   (`1123m406p77z`, `0` = cinq rouge), enregistrez. Recommencez sur plusieurs mains
   jusqu'à ce que les 37 modèles soient appris. À refaire si la résolution ou la
   taille de la fenêtre du jeu change.
3. **Capturer** (ou **Auto**, toutes les 1,5 s) : la main est reconnue puis analysée.

Les défausses adverses et les riichi se saisissent pour l'instant à la main ; on peut
aussi tout saisir sans vision et cliquer sur **Analyser**.

## Colonnes

| Colonne      | Signification                                                                  |
|--------------|--------------------------------------------------------------------------------|
| Shanten      | shanten après la défausse (0 = tenpai)                                         |
| Ukeire       | nombre d'exemplaires encore invisibles qui font avancer la main                |
| P(pioche)    | probabilité que la prochaine pioche fasse avancer                              |
| P(avance)    | probabilité d'avancer au moins une fois avant la fin du mur (hypergéométrique) |
| P(tenpai) MC | probabilité d'atteindre le tenpai, simulation Monte Carlo (300 tirages)        |
| Danger       | % de deal-in estimé contre les riichi (genbutsu, suji, kabe, honneurs vus)     |

Modes : **Attaque** (pas de riichi), **Push** (tenpai avec ≥ 4 ukeire), **Prudence**
(iishanten : on n'avance qu'avec une tuile ≤ 3 % de danger), **Défense** (main lointaine).

## Architecture

```
hebimachi/
  tiles.py        notation et indices des tuiles
  shanten.py      shanten standard / chiitoitsu / kokushi (DP mémoïsée)
  efficiency.py   ukeire, probabilités, Monte Carlo
  defense.py      danger des tuiles face au riichi
  advisor.py      combine efficacité et défense
  recognition.py  template matching OpenCV
  capture.py      capture d'écran (mss) et configuration des zones
  app.py          interface tkinter
```

Tests : `python -m pytest`
