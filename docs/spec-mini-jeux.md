# Spec — mini-jeux au-delà de la v2

Étude de faisabilité pour koi-koi, oicho-kabu et billard (combination shot, carom shot), des
mini-jeux que l'on retrouve dans la série Yakuza. Rien de tout cela n'est implémenté : ce document
dit ce qu'il faudrait pour le faire. Premier jeu cible : **Yakuza Kiwami**, le jeu en cours.

Principe commun, inchangé depuis la v1 : Hebimachi **lit l'écran et conseille**. Il ne touche ni
au jeu ni à la manette.

## Verdict

| Jeu        | Faisable ?               | Ce que l'outil apporte                                | Priorité |
|------------|--------------------------|-------------------------------------------------------|----------|
| Koi-koi    | Oui                      | Beaucoup : carte à jouer, koi-koi ou arrêt            | 1        |
| Oicho-kabu | Oui                      | Moyen : tirer ou non, selon les cartes vues           | 2        |
| Billard    | Oui (Kiwami : vue fixe)  | Dire si la visée affichée empoche la bille            | 3        |

Les règles, les barèmes et l'affichage changent d'un jeu de la série à l'autre. Chaque jeu garde
donc ses particularités dans `data/games/<id>/`, et les chiffres de ce document sont à vérifier
sur le jeu visé.

## Socle à construire une seule fois

- **Un onglet par mode.** Le shogi a posé le modèle : un `ttk.Frame` qui reçoit `app` et utilise
  `app.current_image()` et `app.config.extra`. Chaque nouveau mode suit ce patron.
- **`hebimachi/cards/`**, lecteur de cartes commun à koi-koi et oicho-kabu : mêmes hanafuda, même
  principe que `TileMatcher` (modèles appris sur le jeu, seuil de confiance, correction à la main).
- **Saisie manuelle toujours possible**, comme pour le mahjong : l'analyse ne doit jamais dépendre
  de la vision seule.

## Koi-koi — priorité 1

**Pourquoi.** C'est le seul de la liste où de vraies décisions répétées pèsent sur le résultat, et
l'information y est riche : on voit sa main, le tapis et les captures des deux joueurs.

**Règles à gérer.** 48 cartes, 12 mois de 4. À deux : 8 en main, 8 au tapis, 8 à l'adversaire, 24
à la pioche. À chaque tour, jouer une carte, l'apparier au tapis, puis retourner la pioche. Sur un
yaku : arrêter ou annoncer koi-koi.

**Conseils.**
1. Carte à jouer, et carte du tapis à prendre si deux conviennent.
2. Koi-koi ou arrêt : espérance de points en continuant, risque que l'adversaire marque avant.
3. Alerte quand l'adversaire approche d'un yaku, avec les cartes à lui refuser.

**Méthode.** Les 32 cartes non vues se répartissent au hasard entre main adverse et pioche :
Monte Carlo par tirages de ces cartes (*determinization*), chaque coup candidat étant joué jusqu'à
la fin de la manche par une politique simple. C'est la structure déjà utilisée pour `simulate_tenpai`.
Une table de yaku par jeu dans `game.json`, car barèmes et yaku varient d'un jeu à l'autre.

**Vision.** 48 modèles. Zones : main, tapis, captures des deux joueurs. Le tapis peut empiler ou
décaler des cartes : à vérifier sur capture.

**Critères d'acceptation.** Lecture exacte d'au moins 95 % des cartes sur 20 captures réelles ;
conseil en moins d'une seconde ; sur des parties simulées, la politique bat nettement un joueur
qui prend la première carte possible et s'arrête au premier yaku.

**Taille :** grande. Le plus gros du travail est la vision et la table de yaku.

## Oicho-kabu — priorité 2

**Règles de base.** 40 cartes de valeur 1 à 10, 4 de chaque. La main vaut la somme modulo 10 ; 9
(kabu) est le mieux. Les mains spéciales (arashi, shippin, kuppin…) et les règles de tirage du
croupier **varient selon les jeux : à relever dans le jeu visé avant de coder**.

**Conseil.** Tirer ou rester, et sur quelle mise placer ses jetons. Le calcul est exact, sans
simulation : on énumère les cartes restantes après retrait des cartes vues, on applique la règle
du croupier, et on compare P(gagner | tirer) à P(gagner | rester).

**Vision.** Le plus simple du lot : peu de cartes, bien espacées, valeurs à reconnaître seulement.

**Critères d'acceptation.** Probabilités identiques à une énumération brute de référence ; conseil
vérifié sur 10 mains réelles relevées à la main.

**Taille :** moyenne, moins encore une fois `cards/` en place.

## Billard — priorité 3

**Ce que montre Kiwami** (captures `img.png`, `img_1.png`, `screen.png`, 1920×1080, mode rotation) :

- **Vue de dessus, caméra fixe.** On vise dans cette vue ; seul le réglage de la puissance passe
  dans une autre vue, où la visée est figée. Pas de redressement : le tapis (≈ x 220–1700,
  y 170–905), les 6 poches et le rayon des billes (≈ 16 px) se calibrent une fois pour toutes.
- **Le jeu trace la trajectoire de la blanche** en pointillés jusqu'à la première bille touchée, en
  suivant un éventuel rebond sur la bande. Au bout, il dessine une **bille fantôme** blanche
  translucide, contre la bille visée. Il n'indique pas où part la bille visée : c'est le manque
  que l'outil comble.
- **HUD du mode rotation** : billes empochées en haut à gauche, **« NEXT » = bille à toucher en
  premier** (la plus petite restante), scores des deux joueurs en bas. Le HUD masque la poche
  haut-gauche ; sa position est connue par calibration.
- Overlay Discord éventuel à gauche, sur la bande : zone à ignorer.

**Détection.** Billes par `HoughCircles` dans le tapis, identifiées par couleur et par
pleine/rayée (les numéros sont souvent illisibles). Le HUD réduit l'ambiguïté : les billes empochées
n'y sont plus, il reste au plus une pleine et une rayée par couleur. Bille fantôme : au bout des
pointillés, disque clair sans ombre ni dégradé, ce qui la distingue de la blanche.

**Conseil de base.** La bille visée part sur la droite allant du centre du fantôme à son centre. On
la prolonge : poche atteinte sans autre bille sur le chemin, ou non. On le signale aussi quand la
première bille touchée n'est pas la « NEXT » (faute en rotation). Exemple `img_1.png` : le fantôme
touche le 6, qui partirait vers la bande du bas à gauche de la poche du milieu, et le coup est une
faute puisque NEXT = 5. L'effet de « throw » (légère déviation par frottement) reste à mesurer.

**Calculs.** Bille fantôme : pour envoyer une bille `O` vers un point `T`, la blanche doit arriver au
contact en `G = O − 2R · (T − O)/|T − O|`.

- **Combination shot** (la blanche frappe A, A frappe B, B entre) : on calcule la bille fantôme à
  rebours, de B vers A puis de A vers la blanche. Chaque étage multiplie l'erreur d'angle : au-delà
  de deux billes, le coup est à déconseiller. L'outil classe les coups par angle de coupe et par
  chemin libre.
- **Carom shot** (la blanche ou une bille rebondit sur une autre bille avant la cible) : après le
  choc, la blanche part à 90° de la direction de la bille frappée si elle glisse, et plutôt à 30°
  environ si elle roule. Le résultat dépend donc de l'effet, et l'outil doit donner une zone plutôt
  qu'une ligne.

**Plus besoin de surimpression.** Le joueur voit déjà le fantôme dans le jeu. L'outil dit si la
visée actuelle empoche la bille, et peut indiquer où placer le fantôme pour le meilleur coup. La
physique de la blanche après le choc (effet, frottement) n'est utile que pour le placement suivant,
en second temps.

**Plan.** Prototype hors ligne : lire une capture, détecter tapis, billes, fantôme et NEXT, tracer
la trajectoire de la bille visée sur l'image avec un verdict. Ensuite seulement, onglet en direct
et recherche du meilleur coup (combination, carom).

**État : prototype fait** (`hebimachi/billiards/`, `python -m hebimachi.billiards capture.png`,
tests dans `tests/test_billiards.py`). Sur les 3 captures : 11 billes sur 11 lues avec couleur et
motif, centres stables à 1 px près d'une capture à l'autre, fantôme, pointillés et NEXT lus.
Le HUD peut être caché : la lecture des billes n'en dépend pas, seul le contrôle de faute
(NEXT) disparaît. Reste à valider en jeu le critère des 7 coups sur 10, et à régler les
mâchoires (`jaw_margin`) et l'angle limite des poches du milieu sur des coups réels.

**Recherche de coups** (`sim.py`, `search.py`, onglet Billard → « Chercher un coup ») : on balaie
les directions qui touchent une bille autorisée (pas de 0,02°), plus un tour complet avec la blanche
par la bande pour atteindre une NEXT cachée. Chaque direction est simulée : ligne droite, bande en
miroir, choc selon la règle des 90° (frappée sur la ligne des centres, frappeuse sur la tangente),
élan partagé en cos²/sin² et réduit à chaque bande. Les directions voisines de même résultat
forment un coup ; leur plage donne la **marge**, en pixels de déplacement du fantôme. Types :
direct, combinaison (la bille empochée a été frappée par une autre que la blanche), carambolage
(elle a ricoché sur une bille, ou la blanche l'a touchée après avoir dévié sur une première), bande.
Le coup choisi s'affiche avec le fantôme à reproduire ; en Auto, l'onglet dit de combien décaler
le fantôme du jeu. Limites : pas d'effet ni de frottement, billes en mouvement qui ne se croisent
pas, élan de frappe fixe (`DEFAULT_POWER`). La règle des 90° vaut pour une bille qui glisse : à
vérifier en jeu, surtout pour les carambolages de la blanche.

**Critères d'acceptation du prototype.** Billes localisées à moins d'un demi-rayon sur 20 captures ;
coups conseillés réussis sur au moins 7 essais sur 10 en jeu, à la main.

**Taille :** moyenne pour le prototype (risque levé par la vue fixe), grande pour la recherche de
coups.

## Ce qu'il faut pour démarrer

1. Créer le dossier du jeu cible dans `data/games/<id>/`.
2. Pour chaque mini-jeu retenu, 5 à 10 captures en 1920×1080 à des moments différents ; pour le
   billard, avec la visée active.
3. Relever dans le jeu : table des yaku et barème du koi-koi, règles de tirage de l'oicho-kabu.
