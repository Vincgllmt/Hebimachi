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
| Billard    | Partiellement, à prouver | Aide à la visée, sous condition de la vue de la table | 3        |

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

**À prouver d'abord.** Tout dépend de la vue. En vue de dessus, c'est de la géométrie ; avec une
caméra 3D derrière la queue, il faut redresser la table par ses 4 coins, comme le plateau de shogi,
et rien ne garantit que les coins soient visibles. Sans capture, on ne peut pas trancher.

**Détection.** Redressement de la table, puis billes par `HoughCircles` et couleur.

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

**Limite principale.** L'outil calcule un angle, mais le joueur vise à la manette, sans repère à
l'écran. Il faut soit une surimpression au-dessus du jeu (fenêtre transparente, ce qui suppose le
mode fenêtré sans bordure), soit traduire le conseil en repère visible dans le jeu, comme la ligne
de visée qu'il affiche peut-être. La physique du jeu (effet, frottement, rebonds) n'est pas connue
et devra être mesurée.

**Plan en deux temps.** D'abord un prototype qui lit une capture et trace la ligne sur l'image,
pour valider la détection. La surimpression seulement si ce prototype tient.

**Critères d'acceptation du prototype.** Billes localisées à moins d'un demi-rayon sur 20 captures ;
coups conseillés réussis sur au moins 7 essais sur 10 en jeu, à la main.

**Taille :** grande, risque élevé.

## Ce qu'il faut pour démarrer

1. Créer le dossier du jeu cible dans `data/games/<id>/`.
2. Pour chaque mini-jeu retenu, 5 à 10 captures en 1920×1080 à des moments différents ; pour le
   billard, avec la visée active.
3. Relever dans le jeu : table des yaku et barème du koi-koi, règles de tirage de l'oicho-kabu.
