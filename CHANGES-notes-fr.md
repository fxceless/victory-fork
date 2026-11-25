# Ce qui a changé, et ce qu'il reste à changer

Note tenue au fil du travail, en français. Elle dit ce qui a été fait (avec
les chiffres mesurés) puis, à la fin, ce qu'il faut encore changer, du plus
utile au moins utile. La version anglaise d'avant ce soir reste à côté, dans
`CHANGES-passage-de-vitesse.en.md`.

---

## 1. Fait : le passage de vitesse (commutateurs par défaut)

Le fork portait déjà les bons commutateurs, mais tous éteints : une exécution
normale prenait le chemin lent. Ils sont maintenant actifs par défaut, chacun
remis à l'ancien comportement par sa variable.

| Variable | Défaut maintenant | Ce qu'elle évite |
|---|---|---|
| `VICTORY_LATE_WAIT` | 3 | le titre et nous attendons chacun notre tour |
| `VICTORY_RELAY` | actif | une copie de relais en trop par envoi |
| `VICTORY_ASYNC_SUBMIT` | actif | l'envoi qui bloque le fil du titre |
| `VICTORY_QUEUE2` | 1 | la file de copie non découpée |
| `VICTORY_INDIRECT_WINDOW` (+ `_CHECK`) | actif | le bornage par enregistrement partout |
| `VICTORY_COPY_PLAIN` | actif | la copie page par page |
| `VICTORY_VIEW_KNOWN` | actif | les interrogations `VirtualQuery` ratées |
| `VICTORY_WIDE_THREADS` | 8 (avant : 6) | le nombre de threads qui bridait l'image |
| `VICTORY_FULL_CARD_PASS` | actif | la passe de textures quand la carte est pleine |
| `VICTORY_LEVELS_ON_DEMAND` | actif | les niveaux de texture que le titre ne garde pas |

## 2. Fait : le plafond d'images, de 30 à 120

Le titre se limite à 30 images/s en prenant l'intervalle le plus long qui
tombe juste sur des multiples de 16,67 ms. La couche répond maintenant comme
un écran à 120 Hz : l'intervalle le plus court gardé est 8,33 ms.
`VICTORY_MIN_INTERVAL_US` en règle un autre (16667 = 60, 33333 = 30 comme
avant, 0 = ce que le titre demande).

À savoir : cela enlève une attente, cela n'ajoute pas d'images quand c'est la
carte qui limite.

## 3. Fait : les niveaux à la demande (le gros gain)

Le titre crée ses textures avec tous leurs niveaux, mais n'en remplit qu'une
partie. La couche prenait la mémoire de la carte pour tous les niveaux :
**2 130 ms de préparation de textures par image** et 3 images/s sur le dossier
de test. Avec les niveaux à la demande, seuls les niveaux que le titre
remplit prennent de la mémoire.

| Mesure (même machine, même dossier) | avant | après |
|---|---|---|
| images/s | 3 | **27 à 34** (menus) |
| ms par image | 3 034 | 29 à 37 |
| préparation des textures | 2 130 ms | **5 à 12 ms** |
| mémoire carte non allouée | — | 3,1 Go en moins |

Son test sur la vraie carte passe (`cargo test -p victory-gpu --release tiles
-- --ignored` avec `VICTORY_TILES_TEST_CARD=1` : « all steps passed ON THE
CARD »).

## 4. Fait : la manette PlayStation

Le titre ne lit que XInput, et une manette DualSense ou DualShock 4 n'est pas
un périphérique XInput : elle n'était donc pas là du tout. Le premier
périphérique HID que Windows présente et qui se déclare « gamepad » est
maintenant ouvert et lu en entier ; chaque usage est pris par son numéro, donc
la disposition du rapport est celle de Windows et pas une table à nous. Ça
couvre DualSense, DualShock 4 et Switch Pro. `VICTORY_HID_PAD=0` désactive.

Non fait : les moteurs d'une manette HID ne sont pas encore écrits ; l'ancien
chemin XInput les fait pour une manette Xbox.

## 5. Fait : des sous-objets de pipeline en plus

Le flux de pipeline du titre nomme deux choses que la couche ne savait pas
lire (le nombre d'échantillons forcé du rasteriseur, et les formats de cible
un par un). Elles sont lues ; un flux qui les nomme n'est plus refusé.

## 6. Fusion de l'amont et méthode 0x218 (25 nov.)

- **L'amont `fxceless/victory-fork` est fusionné** (`3973179`, 16 fichiers :
  ce qui fait une image, les parties jointes, les copies de retour par notre
  propre vue, un compteur d'images dans le coin, les enregistrements indirects
  ramassés une fois par liste). Un seul conflit, sur `watch.rs` : notre
  `VICTORY_VIEW_KNOWN` allumé par défaut est gardé, et le mode `=alt` de
  l'amont (vingt secondes avec, vingt sans) est ajouté à côté. 176 tests
  passent. Mesure après fusion au dossier de test : **1,1 à 1,5 s par image**
  (contre 4 à 5 s avant), soit environ 0,6 à 0,9 images/s au chargement.
- **La méthode de création de pipeline au slot 0x218 est écrite.** C'est
  `CreatePipelineState` sur un flux de sous-objets, la même forme que la
  méthode 0x1a8 que la couche savait déjà lire : elle est maintenant lue par
  le même code. Le titre fait 13 appels au lancement et les 13 réussissent
  (`VICTORY_PIPELINE_STREAM=0` revient à l'ancien refus).
  **Ce que ça ne fait pas** : ces 13 flux commencent tous par un sous-objet
  de type console `0x1c`, dont on ne connaît pas la forme ; les shaders d'un
  état d'objet (lancer de rayons) n'y sont pas. Ce qui est rendu est donc un
  pipeline sans étage : le titre ne reçoit plus un échec, mais ce n'est pas
  un pipeline de lancer de rayons. Les 969 ms par image de DexNom viennent
  d'un travail qui reste à faire.

## 7. Repris de DexNom (25 nov.)

Trois correctifs de [DexNom/VIctory](https://github.com/DexNom/VIctory), portés
et adaptés à notre branche :

- **Une image qui change sans que le garde l'ait vue** (le film de l'intro :
  trois images qui changent une fois sur huit) était montrée jusqu'à huit
  images en retard, d'où les changements d'image brutaux au lancement. Elle
  est maintenant hachée à chaque vérification pendant 16 images après une
  première découverte (120 si c'était déjà le cas), et les images linéaires
  d'un seul niveau (les plans du film) sont toujours hachées.
  `crate::texture::Staging::unseen_write`.
- **Les copies de retour de la carte** (ce que les shaders écrivent dans la
  mémoire du titre) étaient une par morceau de 64 Kio. Celles qui se suivent
  dans la même copie sont jointes en une seule copie ; un compteur les compte
  (`fetches N pieces whole, M joined` dans la ligne de profil).
  `VICTORY_FETCH_JOIN=0` revient à l'ancien comportement.
- **La fenêtre d'envois indirects** était plafonnée à 64 enregistrements par
  passe : 54 passes par image pour 3 460 envois. Le plafond est passé à 256
  (réglable de 4 à 4096 par `VICTORY_INDIRECT_WINDOW_MOST`).

## 8. Essayé et abandonné (mesuré, pas supposé)

Deux pistes ont été écrites, compilées, mesurées, puis retirées parce
qu'elles n'aidaient pas :

- **Un budget de 8 ms par image sur la passe des tampons du tas**
  (`heap_buffers`, 260 à 970 ms par image). Quand la passe est coupée, les
  copies de l'image suivante sont plus grosses : le total ne bouge pas.
- **Lire une texture partiellement en mémoire niveau par niveau** (une
  texture dont seuls les niveaux du milieu sont arrivés). Le détilage refuse
  chaque niveau (`TooSmall`) et le total des textures passe de 310 à
  10 400 ms par image. La bonne façon est celle de DexNom (reprendre les
  textures éparses à l'envoi), pas de découper la lecture ici.

## 9. Ce qui reste à changer, du plus utile au moins utile

1. **Les zones à l'ombre sont noires** : le lancer de rayons (l'éclairage
   indirect, les ombres) ne tourne pas. C'est ce que le fork de DexNom a en
   plus, et ce n'est pas une question de pilote : c'est du D3D12 ordinaire,
   donc ça marchera aussi sur NVIDIA. Son propre chiffre, en ville avec le RT
   allumé, est **969 ms par image** : c'est de l'image avant d'être de la
   vitesse. Gros chantier.
2. **Les textures éparses** : une texture dont seules les pages du milieu sont
   en mémoire n'est pas envoyée. DexNom le fait ; ici, non. C'est ce qui donne
   des textures noires ou en basse résolution au chargement, et probablement
   une partie des micro-freezes aux changements brusques d'image.
3. **Le bornage par enregistrement** dans les listes du jeu reste l'essentiel
   du temps de la carte (102 ms sur 130). Les fenêtres en enlèvent une part ;
   le reste demande une autre façon d'émuler la lecture d'arguments de la
   console. Pas un commutateur.
4. **Les copies mémoire PC ↔ carte** restent le principal du temps de
   soumission. La sortie est `VICTORY_INPLACE` (les tampons du titre utilisés
   là où ils sont) ; seul son premier étage existe. `=2` est incompatible avec
   `VICTORY_LATE_WAIT`, donc l'essayer demande aussi `VICTORY_LATE_WAIT=0`.
5. **La passe sur les tampons du tas** (« heap buffers ») coûte 260 à
   970 ms par image sur le dossier de test. Un budget par image a été essayé
   et **n'aide pas** : quand la passe est coupée, les copies de l'image
   suivante sont plus grosses, et le total ne bouge pas. Le vrai travail est de
   ne pas copier du tout (point 4).
6. **Les micro-freezes au lancement** viennent de deux choses mesurées : la
   création des pipelines (`host pipeline refused`, et deux méthodes de
   création que le titre appelle et que la couche refuse) et les textures qui
   arrivent en retard. Le point 2 règle une partie des secondes ; les
   pipelines demandent d'écrire les deux méthodes manquantes.
7. **La base est en retard sur l'amont** : `fxceless/victory-fork` a un commit
   de plus (`3973179`, 16 fichiers) que cette branche n'a pas encore. À
   fusionner avant tout le reste, sinon les deux divergent.
