# Hebimachi — assistant aux mini-jeux de la série Yakuza

Lit l'écran des mini-jeux de la série **Yakuza / Like a Dragon** et conseille, sans jamais toucher
au jeu :

- **Mahjong** : reconnaît vos tuiles et classe les défausses selon l'efficacité
  (shanten, ukeire, probabilités) et la défense face aux riichi, avec en option un objectif de
  victoire (mangan, haneman, full straight).
- **Shogi** : suit la partie et indique le meilleur coup, en cherchant d'abord un mat forcé.

D'autres mini-jeux sont étudiés dans [docs/spec-mini-jeux.md](docs/spec-mini-jeux.md).

## Lancer

```
pip install -r requirements.txt
python -m hebimachi
```

## Jeux de la série

Choisissez le jeu dans la liste en haut de la fenêtre. Chaque jeu a son dossier
`data/games/<id>/`, car l'affichage change d'un opus à l'autre : zones de l'écran et modèles
appris lui sont propres, et sont à refaire pour chaque nouveau jeu.

Pour ajouter un jeu, créez son dossier avec un `game.json` (`{"name": "Yakuza Kiwami"}`) : il
apparaît dans la liste, et il ne reste qu'à le calibrer. Un seul jeu est configuré pour l'instant :
*Like a Dragon: Ishin!*, sur lequel la lecture du mahjong est validée.

## Mahjong

### Première utilisation

1. Choisissez l'écran du jeu dans la liste déroulante, puis **Zones…** : encadrez votre
   main, les indicateurs dora, puis les quatre rivières (Échap pour ignorer une zone).
   Laissez de la place aux rangées suivantes des rivières.
2. **Apprendre les tuiles…** : encadrez des tuiles contiguës de votre main, tapez-les
   (`1123m406p77z`, `0` = cinq rouge), enregistrez. Recommencez sur plusieurs mains
   jusqu'à ce que les 37 modèles soient appris. À refaire si la résolution ou la
   taille de la fenêtre du jeu change.
3. **Capturer** (ou **Auto**, toutes les 1,5 s) : la main est reconnue puis analysée.

Les rivières sont lues automatiquement, dans l'ordre des défausses ; une tuile couchée
coche « Riichi » pour ce joueur. Tous les champs restent modifiables à la main, et l'on
peut tout saisir sans vision puis cliquer sur **Analyser**.

### Objectifs de victoire

Trois cases optionnelles et indépendantes, à cocher quand il faut gagner d'une certaine façon
(par exemple pour la liste de complétion) :

| Case          | Une victoire compte si…                                          |
|---------------|------------------------------------------------------------------|
| Mangan        | elle vaut au moins un mangan (5 han, ou 4 han 40 fu, 3 han 70 fu) |
| Haneman       | elle vaut au moins un haneman (6 han) ; prime sur Mangan          |
| Full straight | elle contient l'ittsu ; se combine avec les deux autres           |

Une fois coché, l'objectif change tous les conseils :

- **Défausses** : classées vers le plan de yaku le plus proche qui atteint l'objectif (colonne
  **Objectif** : distance au tenpai et plan visé). Les dora et cinq rouges comptent.
- **Ron et tsumo** : une victoire insuffisante est déconseillée, avec sa valeur exacte.
- **Riichi** : conseillé seulement s'il fait passer le seuil ; dama si l'objectif est déjà atteint.
- **Appels** : acceptés seulement s'ils rapprochent de l'objectif.

Face à un riichi adverse, la défense reste prioritaire. Sont ignorés car inconnus d'avance : ura
dora, ippatsu et autres yaku de circonstance ; la valeur annoncée est donc un minimum. Un kan
fermé saisi dans « Mes appels » compte comme un appel ouvert.

### Colonnes

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

## Shogi

### Première utilisation

1. Onglet **Shogi**, choisissez votre camp (**Je joue**), puis **Coins du plateau…** : cliquez les
   quatre coins de la grille. Une vue inclinée convient aussi.
2. Sur la position de départ, vos pièces en bas : **Apprendre les pièces**. Un seul écran suffit.
3. **Lire l'écran** (ou **Auto**) à chaque coup : le conseil s'affiche quand c'est à vous de jouer.

Hebimachi n'a pas besoin de lire les pièces en main : il retrouve chaque coup joué en comparant le
plateau aux coups légaux, et sait donc ce qui a été capturé. Il faut pour cela suivre la partie
depuis le début ; sinon **Reprendre depuis l'écran** repart du plateau affiché, en donnant les
pièces inconnues à l'adversaire. Vérifiez alors les pièces en main (clic droit pour corriger).
Les pièces promues s'apprennent seules la première fois qu'elles apparaissent.

Sans vision, tout se joue aussi à la souris sur le plateau, et une position se colle en SFEN, par
exemple pour un problème de tsume.

### Force de jeu

Le moteur intégré en Python cherche d'abord un mat forcé, puis quelques coups d'avance (profondeur
de 3 à 8 selon la position, avec 3 s par coup) : il joue correctement, sans plus. Pour le meilleur
conseil possible, **Choisir un moteur USI…** branche un moteur externe, comme YaneuraOu ou
Fairy-Stockfish, à installer soi-même.

## Architecture

```
hebimachi/
  tiles.py        notation et indices des tuiles
  shanten.py      shanten standard / chiitoitsu / kokushi (DP mémoïsée)
  efficiency.py   ukeire, probabilités, Monte Carlo
  defense.py      danger des tuiles face au riichi
  yaku.py         yaku, han, fu et points d'une main complète
  goals.py        objectifs de victoire : plans de yaku et classement des défausses
  actions.py      ron, tsumo, riichi, appels
  advisor.py      combine efficacité, défense et objectif
  recognition.py  template matching OpenCV
  capture.py      capture d'écran (mss) et configuration des zones
  games.py        jeux pris en charge (data/games/<id>/)
  app.py          interface tkinter
  shogi/
    position.py   règles, SFEN/USI, coups légaux (validés par perft)
    search.py     solveur de mat et alpha-bêta
    usi.py        client pour moteur externe
    vision.py     redressement du plateau et lecture des pièces
    session.py    suivi de la partie, synchronisation avec l'écran
    panel.py      onglet Shogi
```

Tests : `python -m pytest`
