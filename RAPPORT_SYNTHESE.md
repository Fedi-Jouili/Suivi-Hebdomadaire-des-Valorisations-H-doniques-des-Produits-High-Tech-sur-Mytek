# Segmentation et Modélisation Hédonique des Prix — Marché High-Tech Mytek.tn

## Rapport de synthèse du stage

**Auteur :** Fedi Jouili · **Période couverte par les données :** 4 semaines de collecte (S1–S4, juin–juillet 2026)
**Dernière mise à jour de ce rapport :** 2026-08-02
**Dashboard public :** https://hedonique-mytek-dashboard.onrender.com

> **Changements majeurs depuis le 2026-07-29** (résumés ici, détaillés aux sections citées) :
> 1. **Correction scientifique du clustering N2** (§3.2) — le clustering marque × gamme mélangeait par erreur des
>    observations de semaines différentes ; corrigé pour n'opérer que sur une semaine de référence unique.
>    Conséquence directe et assumée : couverture N2 et stabilité bootstrap plus faibles qu'annoncé précédemment
>    (chiffres corrigés dans ce rapport), mais désormais scientifiquement valides.
> 2. **Harmonisation méthodologique notebooks ↔ production** (§3.2ter) — les notebooks ne redéfinissent plus leur
>    propre logique de classification de features/choix de k, ils délèguent à `src/models/hedonic_model.py`.
> 3. **Modélisation par cluster, N1 et N2 séparément** (nouveau §3.3ter) — un modèle unique par catégorie était
>    jugé trop général ; un OLS/Ridge/Random Forest dédié est désormais ajusté par cluster quand l'effectif le
>    permet ET qu'il bat démontrablement le modèle catégorie.
> 4. **Page Prédiction du dashboard** — le sélecteur de segmentation (N1/N2) affectait l'affichage mais jamais le
>    prix prédit ; corrigé (§5).
> 5. **Page « Modèles & clustering » enrichie** (§5) — formule propre par cluster, prix réel vs estimé par
>    semaine, produits du cluster par semaine.

---

## Comment lire ce document

Ce projet est documenté à plusieurs niveaux, chacun avec un rôle précis — ce rapport sert de **point d'entrée
unique** qui les relie entre eux, pas un résumé qui les remplacerait :

| Niveau | Où | Rôle |
|---|---|---|
| **Ce rapport** | `RAPPORT_SYNTHESE.md` | Vue d'ensemble : contexte, méthodologie, résultats clés, limites — à lire en premier. |
| **Notebooks académiques** | `notebooks/*.ipynb` | Démonstration complète, méthodologie détaillée, citations, code exécuté et ses sorties. |
| **Dashboard** | `src/dashboard/` | Exploration interactive des résultats, catégorie par catégorie, semaine par semaine. |
| **Rapports chiffrés** | `reports/*.csv` | Données agrégées sous-jacentes à toutes les analyses — jamais recalculées à la main. |
| **Code source** | `src/` | Implémentation, abondamment commentée sur le *pourquoi* de chaque décision. |
| **Tests** | `tests/` | Preuve exécutable que le code fait ce que ce rapport affirme (287 tests, cf. §8). |

---

## 1. Contexte et objectif

Mytek.tn est un revendeur tunisien de produits high-tech (PC de bureau, PC portables, smartphones, téléphones
portables « classiques », téléviseurs). Le stage vise à construire, sur ce marché, un pipeline complet
d'**analyse hédonique des prix** — au sens de Lancaster (1966) et Rosen (1974) : le prix d'un produit technologique
n'est pas un nombre isolé, mais la somme des prix implicites de ses caractéristiques (RAM, stockage, taille
d'écran, connectivité...). Trois questions structurent le travail :

1. **Quelle est la structure du marché ?** Peut-on regrouper les produits en segments cohérents (clusters), et
   selon quelle logique (technique pure, ou marque × positionnement prix) ?
2. **Quel est le prix implicite de chaque caractéristique ?** Un Go de RAM supplémentaire, un écran plus grand :
   combien cela vaut-il, toutes choses égales par ailleurs, et cette relation est-elle stable dans le temps ?
3. **Le marché change-t-il de PRIX ou de COMPOSITION ?** Une hausse du prix moyen affiché peut venir d'une vraie
   décision tarifaire, ou simplement du fait que le catalogue vendu une semaine donnée est plus haut de gamme —
   deux phénomènes économiquement très différents, que ce projet cherche à distinguer explicitement à chaque
   niveau d'analyse (catégorie entière, puis cluster par cluster).

## 2. Données

- **Source :** scraping hebdomadaire de Mytek.tn (`src/scraper/`), 5 catégories de produits.
- **Fenêtre disponible :** 4 semaines (S1 à S4) au moment de ce rapport — cf. §9 (Limites) pour ce que cette
  fenêtre encore courte permet et ne permet pas de conclure.
- **Volume par catégorie** (produits poolés sur les 4 semaines, après nettoyage) :

  | Catégorie | Produits (poolés) | Couverture clustering N1 | Couverture clustering N2 |
  |---|---:|---:|---:|
  | PC de bureau | 873 | 100 % | 94,4 % |
  | PC portables | 2 155 | 100 % | 92,8 % |
  | Smartphones | 1 012 | 100 % | 94,5 % |
  | Téléphones portables | 291 | 100 % | 73,6 % |
  | Téléviseurs | 448 | 100 % | 94,9 % |

  (N1 = clustering technique pur, structurellement exhaustif ; N2 = segmentation marque × gamme, cf. §3 —
  couverture mesurée, jamais supposée, `reports/couverture_clustering_hebdo.csv`. **Couverture N2 revue à la
  baisse le 2026-08-01** suite à la correction du clustering N2 décrite en §3.2 — chiffres antérieurs à cette
  date, proches de 100 %, calculés sur une méthodologie depuis reconnue incorrecte, cf. §3.2.)

## 3. Méthodologie

### 3.1 Prétraitement (`src/preprocessing/`)

Nettoyage, dédoublonnage (clé GTIN/URL), validation par bornes de plausibilité par catégorie
(`VALIDITY_BOUNDS`), récupération de valeurs manquantes depuis le texte libre (`specs_brutes`) quand le champ
structuré est absent ou aberrant, imputation en cascade (KNN puis repli médiane/mode) pour le résidu, et
sélection de caractéristiques par effet mesuré sur le prix (Kruskal-Wallis pour le catégoriel, |Spearman| pour
le numérique) plutôt qu'une liste fixée à la main. **Aucun filtrage silencieux** : toute ligne écartée, toute
valeur corrigée, est journalisée avec sa raison.

**Sélection de features stabilisée (2026-07-28)** : le test statistique de sélection était auparavant recalculé
indépendamment chaque semaine, sur un échantillon parfois inférieur à 100 lignes — bruité, produisant une
recommandation différente d'une semaine à l'autre pour la même variable (schéma non identique entre semaines,
constaté en pratique sur `cpu_brand`/`has_4g`). `python -m src.preprocessing.pipeline --all`
(`compute_stable_feature_selection`) calcule désormais cette sélection **une seule fois**, sur les 4 semaines
poolées : le schéma de colonnes est aujourd'hui rigoureusement identique sur les 4 semaines pour chaque
catégorie (vérifié directement sur les fichiers `data/processed/week_*/*_clean.csv`). L'ancien mode
semaine-par-semaine reste disponible (`build_processed_datasets` seul) pour un usage ponctuel.

**Recalibration des bornes de plausibilité (2026-07-28/29)** : `VALIDITY_BOUNDS` avait été calibrée une seule
fois, sur la semaine 1 seule (1181 produits, 2026-07-04) — jamais revue depuis malgré l'accumulation de 3
semaines supplémentaires. Re-vérification sur les 4 semaines poolées (4843 produits bruts) : la borne de prix
`telephones_portables` (0-120 TND) rejetait à tort un feature phone réel et vérifié produit par produit (NOKIA
2660 Flip, 249 TND) — corrigée à 300 TND. Les autres dépassements trouvés (RAM/stockage `telephones_portables`)
ont été vérifiés produit par produit et confirmés comme des artefacts de parsing (un NOKIA HMD 150 crédité de
825 Go de stockage SSD, un CLEVER F10 crédité de 256 Go de RAM — techniquement impossibles pour ces appareils
d'entrée de gamme) : bornes **confirmées**, pas élargies. Cf. `src/preprocessing/clean.py` pour le détail
produit par produit de chaque décision.

### 3.2 Deux lectures de la structure du marché (clustering)

- **N1 — technique pur** (`notebooks/Clustering_produits_technologiques.ipynb`) : K-Means sur les seules
  caractéristiques techniques (jamais le prix ni la marque), par catégorie. Exhaustif par construction.
- **N2 — marque × gamme** (`notebooks/Segmentation_Prix_Clustering_produits_technologiques.ipynb`) : chaque
  marque est d'abord divisée en gammes de prix (quantiles propres à la marque), puis un clustering technique est
  tenté au sein de chaque couple (marque, gamme) — une lecture commerciale directement actionnable.
- Les deux approches sont comparées explicitement dans `notebooks/Comparaison_Approches_Clustering.ipynb`
  (ARI, silhouette, test de Kruskal-Wallis par catégorie) : elles capturent des structures différentes, pas la
  même chose vue deux fois — conclusion qui motive `cluster_id`/`gamme_prix` à être **interdits comme
  régresseurs** du modèle de prix (garde-fou de circularité, `FORBIDDEN_REGRESSORS`, cf. `reports/audit_code.md`
  §3.1) : ces variables sont *dérivées* du prix, les utiliser pour prédire le prix serait circulaire.

**Correction scientifique critique (2026-07-31) — un cluster ne doit jamais mélanger des semaines.** Le
clustering N2 poolait auparavant les 4 semaines AVANT de clusteriser : un même produit vu 4 fois (une fois par
semaine) comptait comme 4 observations indépendantes dans l'ajustement K-Means d'une unité marque × gamme,
gonflant artificiellement son effectif au-delà du seuil minimal et faisant apparaître une structure qui n'existe
pas à l'échelle d'une seule semaine. Symptôme observé qui a motivé la correction : pour ASUS/Économique
(pc_bureau), `marque_gamme_estimations_hebdo.csv` affichait des clusters `c0`/`c1` identiques à chaque semaine —
scientifiquement incohérent, puisqu'un cluster K-Means n'a de sens que pour l'échantillon sur lequel il a été
ajusté.

Corrigé (`n2_reference_week`, `src/models/hedonic_model.py`) : `gamme_prix` et `cluster_id` sont désormais
calculés **une seule fois**, sur la semaine de référence (la plus ancienne semaine disponible), puis réappliqués
par produit (`url`) à toutes les semaines où il apparaît — un produit peut entrer/sortir d'un cluster d'une
semaine à l'autre (son effectif change), mais jamais deux semaines ne sont mélangées dans un même ajustement
K-Means. Conséquence directe, mesurée et assumée : la couverture N2 (§2) et la stabilité bootstrap (§3.2bis)
sont **plus faibles** qu'annoncé dans les versions précédentes de ce rapport — les anciens chiffres, plus
favorables, provenaient d'effectifs artificiellement gonflés par le mélange de semaines, pas d'un clustering
réellement plus robuste. `outputs/labels/produits_prix_cluster_semaine_<catégorie>.csv`
(`scripts/merge_segmentation_prix_semaine.py`) documente désormais, produit par produit et semaine par semaine,
le cluster attribué — une ligne par (produit, semaine), jamais un fichier qui laisserait croire à un clustering
recalculé chaque semaine.

### 3.2bis Rigueur du clustering K-Means : stabilité et choix de k (ajouté 2026-07-27)

Le clustering N2 (ci-dessus), utilisé ensuite par toutes les analyses de §3.5, n'avait jusqu'ici aucune mesure de
robustesse : le k retenu par la sélection interne (silhouette maximale sous contrainte d'effectif minimal par
cluster) n'était jamais comparé aux k voisins qu'il a écartés, ni testé pour sa stabilité à l'échantillonnage.
Deux diagnostics ajoutés (`src/models/weekly_report.py`, visibles dans le dashboard — page « Modèles &
clustering », onglet clustering) :

- **Stabilité bootstrap** (`cluster_stability_n2`, `reports/stabilite_clustering_n2.csv`) : pour chaque unité
  marque × gamme clusterisée, 100 réplications bootstrap (rééchantillonnage des produits, ré-ajustement d'un
  K-Means, comparaison par Adjusted Rand Index — Hubert & Arabie 1985 — aux labels originaux). **Chiffres revus
  le 2026-08-01** suite à la correction de §3.2 (clustering sur la seule semaine de référence, effectifs par
  unité mécaniquement plus petits qu'avec le pooling multi-semaines précédemment utilisé par erreur) : **21**
  unités clusterisées sur les 5 catégories (contre 60 avant correction), ARI moyen de **0,62** (médiane 0,60) —
  et **6 unités (28,6 %)** ont un ARI < 0,5 (MSI/Premium et MYTEK/Premium et Économique pour pc_bureau,
  DELL/Premium et LENOVO/Premium pour pc_portables, SAMSUNG/Premium pour smartphones). La stabilité mesurée est
  donc nettement moins favorable qu'annoncé précédemment — un coût direct et honnête de la correction
  méthodologique de §3.2 : un clustering ajusté sur un échantillon plus petit (une seule semaine, jamais quatre
  mélangées) est intrinsèquement plus sensible au bruit d'échantillonnage. Près d'un tiers des unités
  clusterisées doivent désormais être lues avec prudence, pas seulement les cas isolés rapportés auparavant.
- **Justification du choix de k** (`k_selection_justification`, `reports/justification_k_clustering.csv`) :
  expose la silhouette de TOUS les k testés (retenus ou non), avec la raison de rejet explicite quand applicable
  (effondrement K-Means sur points dupliqués, ou cluster sous l'effectif minimal). Rend auditable un choix qui ne
  l'était pas auparavant — seul le k final apparaissait dans les artefacts persistés.
- **Limite documentée, non corrigée** : le K-Means sous-jacent (N1 et N2) mélange, dans une seule distance
  euclidienne, des variables continues standardisées et des variables catégorielles one-hot — le poids relatif
  de chaque groupe dans la distance dépend en partie du nombre de modalités encodées (artefact du one-hot
  encoding, pas une pondération délibérée). Une distance dédiée aux données mixtes (K-Prototypes, Gower) serait
  plus rigoureuse ; non implémentée ici (cf. §10).

### 3.2ter Harmonisation méthodologique notebooks ↔ production (2026-08-01)

`src/models/hedonic_model.py` est la source de vérité du pipeline de production (`save_artifacts.py`,
`weekly_report.py`) pour la classification des features (continue vs catégorielle) et le choix de k. Les
notebooks académiques, eux, redéfinissaient historiquement leur propre copie de cette logique — jusqu'à 9 copies
collées dans un seul notebook (`Segmentation_Prix_Clustering_produits_technologiques.ipynb`) — avec des règles
naïves divergentes (ex. `is_numeric_dtype` seul, sans tenir compte du fait qu'une variable comme `cpu_serie` est
encodée numériquement mais représente des paliers non équidistants économiquement, donc traitée comme
catégorielle en production). Impact mesuré avant correction : composition exacte des clusters divergente entre
notebooks et production sur `pc_bureau`/`pc_portables`/`smartphones` (la gamme de prix résultante restait,
elle, identique à 100 %).

Corrigé : les notebooks importent désormais directement les fonctions de `hedonic_model.py`
(`_classify_features`, `_build_feature_matrix`, `_choose_k`, `_max_k_for`, `_min_cluster_size_for`) au lieu de
les redéfinir — élimine à la fois la divergence de règle de classification et une divergence annexe trouvée dans
le notebook N1 (`choose_k` y retournait le plus petit k testé au lieu du sentinel « aucune structure » utilisé
partout ailleurs, produisant un clustering forcé là où la production n'en trouvait aucun sur plusieurs unités).
Les 2 notebooks générateurs de labels et les 2 notebooks de comparaison ont été ré-exécutés après correction ;
`outputs/labels/produits_prix_cluster_semaine_<catégorie>.csv` (régénéré à partir de cette version harmonisée)
concorde désormais à 100 % — gamme ET composition exacte des clusters — avec `models/<catégorie>/
pooled_labeled.csv` (la version utilisée pour l'entraînement des modèles), vérification qui n'était pas possible
avant cette correction faute d'un référentiel commun.

### 3.3 Modélisation hédonique (`src/models/`)

Trois modèles, entraînés sur `log(prix)`, sur les mêmes données poolées (4 semaines, split train/test **groupé
par produit** pour interdire toute fuite d'un même produit vu à plusieurs semaines) :

| Modèle | Rôle | R² test (log) — plage observée |
|---|---|---|
| **Hedonic OLS** (statsmodels, erreurs-types **groupées par produit**, cf. §3.3bis) | Inférence : coefficients, p-values, prix implicites | 0,545 – 0,944 |
| **Ridge** (Rosen semi-log, GridSearchCV **groupé par produit**) | Alternative régularisée, mêmes coefficients réexprimés | 0,541 – 0,952 |
| **Random Forest** (GridSearchCV **groupé par produit**) | Non-linéaire, benchmark de plafond prédictif | 0,406 – 0,986 |

Écart notable : Random Forest domine largement sur `smartphones` (R² = 0,986 contre 0,839 pour OLS — la
relation prix/caractéristiques y est probablement non-linéaire, effets de seuil de marque/génération que le
modèle log-linéaire ne capture pas) mais reste en retrait sur `pc_bureau` (0,406 contre 0,545) — aucun modèle ne
domine partout, d'où la comparaison systématique plutôt qu'un choix a priori.

**Correctifs de rigueur (2026-07-28, audit méthodologique externe)** :
- Les erreurs-types de l'OLS sont désormais **groupées par produit** (`cov_type="cluster"`, `groups=url`) plutôt
  que seulement robustes à l'hétéroscédasticité (HC3) : un même produit est vu à plusieurs semaines dans les
  données poolées, ce ne sont pas des observations indépendantes — les traiter comme telles sous-estimait les
  erreurs-types et gonflait artificiellement la significativité des coefficients.
- La validation croisée de Ridge et Random Forest (sélection d'alpha / d'hyperparamètres) est désormais
  **groupée par produit** (`GroupKFold`) plutôt qu'un `KFold` ordinaire, pour la même raison — un même produit
  pouvait auparavant se retrouver à la fois dans le pli d'entraînement et le pli de validation d'une itération.
  Effet mesuré : la régularisation Ridge retenue est passée de α≈0,1 (faible) à α≈10-100 (nettement plus forte)
  pour 4 des 5 catégories, et la profondeur maximale des arbres Random Forest a diminué pour 3 catégories —
  cohérent avec l'hypothèse que la CV non groupée sélectionnait auparavant des modèles trop complexes.
- **Accord entre modèles** (`model_agreement.csv`, page « Modèles & clustering » du dashboard) : OLS et Ridge
  s'accordent sur le signe du coefficient pour 80 à 100 % des caractéristiques selon la catégorie ; en revanche,
  la corrélation de rang (Spearman) entre l'importance Random Forest et |coefficient| OLS est **faible ou
  négative** pour 3 des 5 catégories (pc_bureau, pc_portables, télévisions) — Random Forest et les modèles
  linéaires ne classent souvent PAS les mêmes variables comme les plus importantes. Une importance Random
  Forest ne doit donc jamais être citée comme confirmée par les modèles linéaires sans vérifier ce chiffre.

Diagnostics post-estimation (`run_diagnostics`) : VIF (colinéarité), test de White (hétéroscédasticité),
distance de Cook (observations influentes), QQ-plot des résidus — cf. `reports/hedonic_qq_*.png` et
`notebooks/*.ipynb` pour le détail par catégorie.

### 3.3bis Ce que ces coefficients établissent (et ce qu'ils n'établissent pas)

Les coefficients hédoniques ci-dessus sont couramment décrits comme le « prix implicite » d'une caractéristique
(Lancaster 1966 / Rosen 1974) — une formulation qui invite à une lecture causale (« un Go de RAM supplémentaire
COÛTE X % de plus ») que les données de ce projet ne permettent pas d'établir aussi directement :

- **Association conditionnelle, pas expérience contrôlée.** Les prix Mytek.tn sont des prix de catalogue
  observés, pas le résultat d'une expérience où les caractéristiques auraient été assignées aléatoirement à des
  prix. Un coefficient mesure une corrélation conditionnelle sur les variables incluses (marque, RAM,
  stockage...), pas un effet causal isolé de toute autre influence.
- **Confusion possible caractéristique ↔ segment.** Un fabricant choisit une configuration technique EN
  FONCTION du segment de prix visé (une configuration « haut de gamme » est conçue pour un prix haut de gamme),
  pas l'inverse — la causalité entre caractéristique et prix peut donc courir dans les deux sens, ou être
  conjointement déterminée par une décision produit en amont que le modèle ne peut pas observer.
- **« Marque » comme effet fixe suppose son exogénéité au prix.** hedonic_model.py traite la marque comme une
  caractéristique du produit, exogène à SON PROPRE prix (§NOTE DE CONCEPTION du module) — un choix standard en
  économétrie hédonique, mais qui reste une hypothèse, pas un fait vérifié : une marque peut être corrélée à des
  attributs non observés (qualité de fabrication perçue, garantie, réseau après-vente) qui influencent
  eux-mêmes le prix. Si c'est le cas, l'effet fixe de marque absorbe une partie de ce qui serait, en toute
  rigueur, un effet de caractéristique non mesurée — plutôt qu'un biais qui contaminerait les autres
  coefficients.

**Conséquence pratique** : lire les coefficients de ce rapport comme « la caractéristique X est associée à une
variation de Y % du prix, conditionnellement à la marque et aux autres caractéristiques observées » — jamais
comme « installer X ferait varier le prix de Y % ». Cette nuance ne change aucun chiffre déjà publié dans ce
rapport ; elle change uniquement la phrase qui doit accompagner leur citation.

### 3.3ter Modélisation par cluster, N1 et N2 séparément (2026-08-01)

Un seul modèle par catégorie (§3.3) traite la catégorie comme homogène — trop général pour une prédiction fine :
un PC de bureau économique et un PC de bureau premium n'ont pas nécessairement la même structure de prix
implicite. `fit_models_per_segment` (`src/models/hedonic_model.py`, généralisation du patron déjà utilisé par
`fit_strategy_b`) ajuste donc, **en plus** du modèle catégorie, un Hedonic OLS/Ridge/Random Forest dédié pour
**chaque cluster** de **chaque** segmentation (N1 technique et N2 marque × gamme, traités séparément) — mais
seulement quand deux conditions sont réunies :

1. **Effectif suffisant** (`n_lignes ≥ n_prédicteurs × ratio`), avec un ratio différencié par famille — OLS exige
   des degrés de liberté pour l'inférence (ratio = 10, la même règle empirique que `fit_strategy_b`), Ridge/Random
   Forest tolèrent un effectif plus petit (ratio = 5, la régularisation L2/l'agrégation d'arbres sont conçues pour
   ça). Sous le seuil pour une famille donnée → pas de modèle pour cette famille sur ce cluster, jamais un
   ajustement dégénéré.
2. **Bat démontrablement le modèle catégorie** sur le MÊME jeu de test hors-échantillon (R² log). Ce second
   critère n'est pas cosmétique : mesuré empiriquement sur ce projet, plusieurs clusters passant le seuil
   d'effectif produisaient un R² hors-échantillon fortement négatif (surapprentissage sur petit échantillon) —
   jamais retenus pour la prédiction malgré un ajustement techniquement réussi.

Chaque combinaison (segment, famille) est reportée dans une table d'audit (`models/<catégorie>/clusters_n1_
summary.csv` / `clusters_n2_summary.csv`, colonnes `ajuste`/`retenu_pour_prediction`/`raison_rejet`/`r2_test` vs
`r2_test_categorie`) — jamais un résultat silencieux, visible dans le dashboard (page « Modèles & clustering »,
onglet « Modèles par cluster »).

**Résultat mesuré, conforme à l'attente statistique** (les unités N2 ont des effectifs nettement plus petits que
les clusters N1, cf. §2/§3.2) :

| Segmentation | Clusters avec ≥ 1 modèle retenu | Combinaisons (cluster, famille) retenues |
|---|---:|---:|
| N1 (technique) | 10 / 19 (52,6 %) | 22 / 57 (38,6 %) |
| N2 (marque × gamme) | 32 / 118 (27,1 %) | 64 / 354 (18,1 %) |

Cas limite honnête : pour `televiseurs`, **aucun** des 17 clusters N2 ne dispose d'un modèle dédié retenu (0/17)
— le modèle catégorie reste utilisé pour toute prédiction sur cette combinaison catégorie/segmentation, jamais
remplacé par un modèle sous-alimenté. La page Prédiction du dashboard affiche systématiquement la source réelle
du prix retourné (« n2 », « n1 », ou « catégorie »), jamais ambigu sur le modèle qui a produit l'estimation.

### 3.4 Indice de prix temporel (effet fixe semaine)

`fit_strategy_c_pooled_time` (`src/models/hedonic_model.py`) poole les 4 semaines avec des pentes hédoniques
COMMUNES + une indicatrice « semaine » — ses coefficients donnent un indice de prix « toutes choses égales »
(Triplett 2004), net de la composition du catalogue. Sur la fenêtre actuelle, un seul mouvement est
statistiquement significatif (p < 0,05) : `smartphones` S2 (+1,97 %, p = 0,0495) — les autres catégories ne
montrent, pour l'instant, aucun mouvement de prix pur distinguable du bruit
(`reports/indice_prix_hedonique_hebdo.csv`, recalculé à chaque exécution). `telephones_portables` est exclue de
cet indice (coefficients hédoniques mesurés comme instables entre semaines sur cette catégorie précise, cf.
`POOLED_TIME_EXCLUDED_CATEGORIES`) — une exclusion documentée, pas un oubli.

### 3.5 Étude des transitions par cluster — prix réel vs valeur technique implicite

Contribution la plus récente du projet (`src/models/weekly_report.py`, `notebooks/
Etude_Transitions_Clusters_Marque_Gamme.ipynb`) : descend du niveau catégorie (§3.4) au niveau **cluster N2**
(marque × gamme × sous-cluster). Pour chaque cluster et chaque paire de semaines consécutives, une grille de
lecture 3×3 classe la transition selon la direction du prix RÉEL (moyenne géométrique observée) et du prix
ESTIMÉ (valeur technique implicite du mix vendu, via les 3 modèles de §3.3) :

- Les deux bougent ensemble → variation expliquée par un changement de caractéristiques (montée/baisse de
  gamme technique), pas une décision de prix.
- Un seul bouge → écart résiduel : le marché facture quelque chose que les caractéristiques n'expliquent pas.

**Upgrade statistique (2026-07-25) :** le seuil de matérialité initial (±3 %, une convention) est désormais
doublé d'un **test bootstrap** (1000 répliques, rééchantillonnage produit par produit) qui calcule un intervalle
de confiance à 95 % sur l'écart résiduel de chaque transition testable (effectif ≥ 2 des deux côtés). Résultat,
sur les données actuelles (**chiffres revus le 2026-08-01** suite à la correction du clustering N2, §3.2 — moins
d'unités clusterisées, donc moins de transitions testables) : sur **354** transitions (5 catégories, 3
périodes ; 803 avant correction), le seuil fixe en classait **14** comme « notables » (65 avant correction) — le
test bootstrap n'en confirme que **1** (26 % avant correction, désormais 1/14) avec un intervalle excluant 0. La
majorité des écarts détectés par un seuil naïf ne résistent toujours pas à un test tenant compte de la taille
d'échantillon — le résultat qualitatif tient, mais sur un échantillon nettement plus restreint et donc moins
concluant statistiquement qu'annoncé précédemment ; cf. §9 pour la discussion de ce compromis.

**Upgrade de rigueur (2026-07-26) — répondre explicitement à « prix, qualité, ou nombre de produits ? »** : la
grille 3×3 ci-dessus suppose implicitement que l'EFFECTIF du cluster est resté stable d'une semaine à l'autre.
Si des produits sont entrés/sortis du cluster entre les deux semaines, une lecture « prix et qualité bougent
ensemble » peut simplement refléter un catalogue différent (des produits moins chers sont partis, des plus
chers sont arrivés) — pas un changement de prix ni de qualité d'un produit donné. `cluster_transitions` teste
maintenant ce cas **en priorité**, avant toute lecture de la grille : si l'effectif a changé, la cause retenue
(colonne `cause_principale`) est directement `effectif`, et la grille prix/qualité n'est pas invoquée pour cette
transition ; sinon, `cause_principale` est déduite de la grille (`prix`, `qualite`, `prix_et_qualite`, ou
`aucune`).

Sur les données actuelles (**354 transitions**, chiffres revus le 2026-08-01 suite à la correction du clustering
N2 de §3.2 ; 803 avant correction) : **79,1 %** ne montrent aucun changement notable (`aucune`, 79,3 % avant
correction), **19,5 %** sont dominées par un changement de COMPOSITION du cluster (`effectif` — le nombre de
produits a changé, la lecture prix/qualité n'est pas isolable pour ces cas ; 19,3 % avant correction), et
seulement **1,4 %** combinent un vrai signal de prix et/ou de qualité (0,8 % qualité seule, 0,6 % prix seul, 0 %
les deux à la fois — proportions quasi identiques avant/après correction). Réponse directe à la question posée :
sur cette fenêtre de 4 semaines, le changement de CATALOGUE (quels produits sont vendus) domine très largement
sur un changement de PRIX ou de QUALITÉ d'un produit donné — la variation du panier moyen d'un cluster est, dans
la grande majorité des cas observés, due à autre chose qu'une décision tarifaire ou une montée en gamme. Notable
: la RÉPARTITION relative entre les 3 causes est restée quasiment inchangée par la correction du clustering N2
(seul le nombre absolu de transitions testables a diminué) — un indice de robustesse qualitative de cette
conclusion, indépendante du bug corrigé en §3.2.

## 4. Une découverte méthodologique : la vérification a trouvé de vrais bugs de pipeline

En sélectionnant systématiquement les 3 transitions à écart résiduel le plus extrême pour une étude de cas
détaillée (`notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb` §4), **2 sur 3 se sont révélées être des
artefacts de pipeline**, pas des signaux économiques :

1. **Confusion RAM/stockage** (`src/preprocessing/clean.py::fix_ram`) — sur certaines fiches produit (ex. Xiaomi
   Redmi Note 14), le champ générique `"Mémoire"` duplique parfois le stockage plutôt que la RAM ; la RAM réelle
   se trouvait fusionnée dans le champ `"Processeur"` par un artefact du site source. Corrigé : ordre de priorité
   des candidats revu, repli sur un motif `"RAM : NGo"` explicite, bornes de plausibilité resserrées
   (`smartphones` : (0, 192) Go → (1, 32) Go).
2. **Faux positif de connectivité cellulaire** (`src/preprocessing/encode.py::extract_connectivity_flags`) — la
   notation Wi-Fi double bande « 2.4G/5G » était lue comme une connectivité cellulaire 4G/5G, faisant
   apparaître des téléviseurs (sans aucune carte SIM) comme ayant du 4G. Corrigé : les notations de bande Wi-Fi
   sont retirées du texte avant la recherche de motifs cellulaires — vérifié sur les 4277 chaînes de
   connectivité du catalogue, exactement 1 chaîne distincte affectée, aucune régression ailleurs.

Les deux corrections ont été propagées de bout en bout (prétraitement → ré-entraînement des 3 modèles → rapports
hebdomadaires) et vérifiées disparues du classement des écarts résiduels
(`notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb` §7).

**Un troisième bug, plus sévère, trouvé par la même discipline de vérification (2026-07-31)** : ce n'est pas une
étude de cas produit par produit mais une relecture directe des CSV exportés (`reports/marque_gamme_estimations_
hebdo.csv`) qui a révélé le clustering N2 mélangeant des observations de semaines différentes (détail complet en
§3.2). Contrairement aux deux bugs ci-dessus, celui-ci ne faussait pas une valeur isolée mais la validité même de
l'unité d'analyse (« cluster ») utilisée par §3.3ter et §3.5 — un défaut structurel, pas ponctuel. Corrigé, avec
un coût honnête et mesuré : couverture N2 et stabilité bootstrap revues nettement à la baisse (§2, §3.2bis),
échantillon de transitions testables réduit de plus de moitié (§3.5).

**Conclusion méthodologique retenue comme la plus importante du projet à ce stade :** un signal détecté
automatiquement, même le plus extrême et même statistiquement significatif au sens bootstrap, ne doit jamais
être interprété économiquement sans redescendre au niveau produit — le bootstrap protège contre le bruit
d'échantillonnage, pas contre un biais systématique amont. Le troisième bug (weekly-mixing) montre que cette
discipline de vérification s'applique aussi à un niveau plus structurel : relire directement ce qu'un artefact
exporté affiche, pas seulement les métriques agrégées qui en découlent, reste nécessaire même pour du code déjà
testé.

## 5. Dashboard (`src/dashboard/`)

Application Dash multi-pages, strictement en lecture seule sur les artefacts persistés (jamais de `.fit()` dans
le dashboard) :

- **Accueil** — vue d'ensemble, fraîcheur des données.
- **Statistiques descriptives** — volumes, prix, marques, caractéristiques techniques, par catégorie.
- **Modèles & clustering** — coefficients hédoniques, importance des variables (RF), profils de clusters N1/N2,
  et depuis le 2026-07-27 les 2 diagnostics de rigueur de §3.2bis (ARI de stabilité bootstrap, silhouette du k
  retenu) directement dans la table des unités N2. **Onglet « Modèles par cluster » enrichi (2026-08-01ter,
  cf. §3.3ter)** : vue d'ensemble compacte (un clic → détail par accordéon, pour ne jamais tout afficher en
  même temps) donnant, pour chaque cluster retenu ou non, sa formule propre (équation log-linéaire lisible pour
  OLS/Ridge avec p-values, importances pour Random Forest), le prix réel (moyenne géométrique) vs estimé par
  semaine, et la liste des produits du cluster par semaine avec prix et caractéristiques — jamais seulement le
  résultat final, toujours le détail qui le justifie.
- **Prédiction** — produit hypothétique → prix estimé (avec intervalle pour OLS), segment assigné, produits
  réels comparables ; RAM/stockage en paliers réellement observés (pas un slider continu) ; ajustement optionnel
  par semaine via l'indice de §3.4. **Correctif (2026-08-01bis)** : le sélecteur « Type de segmentation »
  (N1/N2) ne modifiait auparavant que l'affichage du segment assigné, jamais le prix retourné — les deux
  segmentations affichaient donc silencieusement le même prix. Corrigé : le prix retourné respecte désormais
  strictement le niveau choisi (N2 → repli catégorie si aucun modèle N2 retenu, N1 → jamais de repli croisé vers
  N2), avec la source affichée explicitement (§3.3ter).
- **Évolution hebdomadaire** — couverture du clustering, prix par cluster, prix réel vs estimé par modèle,
  et l'étude de transitions de §3.5/§4 (grille de lecture, cas notables, confirmation bootstrap).
- **Téléchargements** (ajouté 2026-07-29) — notebooks en PDF (pré-générés hors-ligne, cf.
  `scripts/generate_notebook_pdfs.py`), estimations par cluster × semaine par catégorie (avec les colonnes
  d'erreur d'estimation par modèle), données produit par catégorie. **Correctif (2026-08-02)** : les liens de
  téléchargement naviguaient l'onglet du dashboard lui-même vers l'URL du fichier — si le navigateur restaure
  les onglets au démarrage, cela redéclenchait la boîte de dialogue de téléchargement à chaque ouverture.
  Corrigé (`target="_blank"` sur tous les liens de la page).

Lancement local : `python -m src.dashboard.app` → http://127.0.0.1:8050

**Déploiement public (2026-07-29)** : le dashboard est hébergé sur [Render](https://render.com)
(https://hedonique-mytek-dashboard.onrender.com), construit automatiquement depuis ce dépôt GitHub à chaque
push (`render.yaml`, build Docker via `Dockerfile`). `data/processed/` et `models/` sont désormais versionnés
(anciennement exclus du dépôt) : un déploiement basé sur git a besoin de ces artefacts pour avoir quoi que ce
soit à servir. `data/raw/` (scrape brut, volumineux, jamais lu par le dashboard) reste exclu. Serveur de
production : gunicorn (le serveur de développement Flask/Dash, mono-thread par défaut et dont le mode debug
expose un débogueur Python interactif dans le navigateur, n'est utilisé qu'en local). Niveau gratuit Render :
l'instance se met en veille après une période d'inactivité, avec un délai de réveil de 30 à 60 secondes sur la
première requête suivante.

## 6. Résultats clés (résumé)

- Les 5 catégories sont modélisées avec un pouvoir explicatif hors-échantillon substantiel (R² test entre 0,40
  et 0,99 selon catégorie/modèle) — cf. tableau §3.3 et `models/<catégorie>/metrics.json`. Ces R² catégorie sont
  inchangés par la correction du clustering N2 de §3.2 (le modèle catégorie ne dépend pas de `cluster_id`).
- La segmentation N2 (marque × gamme) couvre entre 73,6 % (`telephones_portables`) et 94,9 % du catalogue
  retenu selon la catégorie (§2) — couverture revue à la baisse le 2026-08-01 suite à la correction
  méthodologique de §3.2 (chiffres antérieurs, proches de 100 %, reposaient sur un clustering aujourd'hui
  reconnu incorrect).
- **Modélisation par cluster (§3.3ter, nouveau)** : un modèle dédié (OLS/Ridge/RF) est retenu — c'est-à-dire
  qu'il bat démontrablement le modèle catégorie sur son propre test — pour 10 des 19 clusters N1 (52,6 %) et
  32 des 118 clusters N2 (27,1 %) toutes catégories confondues ; ailleurs, le modèle catégorie reste utilisé,
  jamais silencieusement. Cas limite : aucun cluster N2 n'a de modèle retenu pour `televiseurs` (0/17).
- Sur la fenêtre actuelle, un seul mouvement de prix « toutes choses égales » est statistiquement significatif
  au niveau catégorie (`smartphones` S2, +1,97 %) — le marché ne montre pas, pour l'instant, de signal fort de
  re-tarification généralisée.
- Au niveau cluster, la grande majorité des transitions (79,1 %, cf. `reports/transitions_cluster_hebdo.csv`)
  ne montrent aucun écart entre prix réel et valeur technique implicite ; parmi les 14 écarts détectés par un
  seuil naïf, seul 1 résiste à un test statistique explicite (chiffres revus le 2026-08-01, échantillon réduit
  de 803 à 354 transitions par la correction du clustering N2 — cf. §3.5).
- **Prix, qualité, ou nombre de produits ?** (cf. §3.5) Sur les 354 transitions observées, **19,5 %** sont
  dominées par un changement du NOMBRE de produits du cluster (`cause_principale = effectif`), contre
  seulement **1,4 %** avec un vrai signal de prix et/ou qualité isolable — le changement de catalogue domine
  très largement sur la re-tarification ou la montée en gamme d'un produit donné, sur cette fenêtre de 4
  semaines. Cette répartition relative est restée quasi identique avant/après la correction du clustering N2.
- **Stabilité du clustering N2** (cf. §3.2bis) : ARI moyen de 0,62 (médiane 0,60) sur les 21 unités marque ×
  gamme clusterisées (chiffres revus le 2026-08-01) ; 6 unités (28,6 %) présentent une segmentation fragile
  (ARI < 0,5), à interpréter avec prudence.
- **Trois** bugs de pipeline réels ont été identifiés et corrigés grâce à cette méthodologie, avec effet
  mesurable sur les artefacts en aval (cf. §4) — une validation *a posteriori* de l'intérêt de descendre au
  niveau produit, y compris pour un défaut plus structurel (clustering mélangeant les semaines) que les deux
  premiers.

## 7. Reproduire ce projet de bout en bout

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 1. Pretraitement -- TOUTES les semaines sous data/raw/, selection de features
#    calculee une seule fois sur les donnees poolees (recommande, cf. §3.1)
python -m src.preprocessing.pipeline --all

# 2. Entrainement des modeles + clustering (tous les artefacts sous models/)
python -m src.models.save_artifacts

# 3. Rapports hebdomadaires (tous les CSV sous reports/, alimentent le dashboard)
python -m src.models.weekly_report

# 4. Dashboard (local)
python -m src.dashboard.app        # http://127.0.0.1:8050
# ... ou en conteneur, identique a la production (cf. §5) :
docker compose up --build          # http://localhost:8050

# 5. Notebooks (executes avec leurs sorties, mais reproductibles independamment) :
#    ordre de lecture recommande dans notebooks/README implicite par prefixe --
#    EDA -> Clustering -> Comparaison -> Segmentation -> Evolution_Temporelle -> Etude_Transitions
#    PDF (page Telechargements du dashboard) : python scripts/generate_notebook_pdfs.py

# 6. Tests
pytest tests/ -v
```

## 8. Qualité et reproductibilité du code

- **287 tests automatisés** (`pytest tests/`, 287 passants + 1 skipped), couvrant le prétraitement, les 3
  modèles, le clustering, les rapports hebdomadaires et les fonctions pures du dashboard — exécutés
  automatiquement par CI sur chaque push/PR depuis le 2026-07-28 (`.github/workflows/tests.yml`), plus
  manuellement avant. 65 tests ajoutés le 2026-07-25 : 8 régressions directes sur les bugs de §4, 40 sur
  `src/models/weekly_report.py`, 17 sur les fonctions pures de la page dashboard « Évolution hebdomadaire ». 19
  tests supplémentaires ajoutés le 2026-07-26/27 pour les upgrades de rigueur de §3.2bis/§3.5. 17 tests
  supplémentaires ajoutés le 2026-07-28 (audit méthodologique externe) : `tests/test_preprocessing/
  test_pipeline.py` (le point d'entrée réel de production, jusque-là jamais testé directement, y compris la
  sélection de features stable poolée) et `tests/test_models/test_save_artifacts.py`/`test_rf.py` (accord entre
  modèles, importance par permutation). **53 tests supplémentaires ajoutés le 2026-08-01/02** pour les
  correctifs et fonctionnalités de cette mise à jour : `tests/test_models/test_cluster_models.py` (19 tests —
  garde-fou d'effectif différencié par famille, table ajusté/retenu/écarté, schéma des artefacts par cluster,
  cf. §3.3ter) ; `tests/test_dashboard/test_prediction_utils.py` (11 tests — régression directe sur le
  correctif N1/N2 de §5, vérifie qu'une segmentation choisie ne retourne jamais silencieusement le prix de
  l'autre) ; `tests/test_dashboard/test_models_page.py` (14 tests — équations lisibles OLS/Ridge, statut
  retenu/écarté, vue d'ensemble par cluster) ; extensions de `test_data_loader.py` (6 tests) et
  `test_weekly_report.py` (3 tests) pour les nouveaux chargeurs et le rapport N1 hebdomadaire. Zones encore non
  couvertes par un test dédié : `src/scraper/spider.py`/`utils.py`/`scheduler.py` (pagination, retry,
  checkpoint), `bounds.py`/`impute.py`/`select_features.py` pris isolément (exercés indirectement via
  `test_pipeline.py`, pas unitairement).
- **Environnement figé** : `requirements.txt` (racine, généré par `pip freeze`) + `src/dashboard/requirements.txt`
  (sous-ensemble minimal du dashboard seul).
- **Convention du projet, tenue tout du long :** jamais de filtrage/repli silencieux — toute décision de nettoyage,
  seuil, ou exclusion est journalisée et documentée en commentaire à l'endroit où elle est prise (cf.
  `reports/audit_code.md` pour l'audit du garde-fou de circularité le plus critique).

## 9. Limites

- **Fenêtre de 4 semaines encore courte** — ni saisonnalité ni tendance de fond ne peuvent être établies à ce
  stade ; les notebooks (`Evolution_Temporelle_Marche_Mytek.ipynb` §4.2, `Etude_Transitions_Clusters_Marque_
  Gamme.ipynb` §1.5) sont conçus pour être ré-exécutés à mesure que de nouvelles semaines s'accumulent, sans
  modification de code (découverte dynamique des semaines, `discover_weeks`).
- **Biais de survie non corrigé** — le panel apparié (produits vus sur toutes les semaines) exclut par
  construction les produits retirés du catalogue.
- **Petits effectifs** — de nombreux clusters marque × gamme comptent moins de 10 produits ; le test bootstrap
  de §3.5 quantifie directement l'impact de cette contrainte plutôt que de le contourner.
- **Caractéristiques non observées** — garantie, bundle logiciel, SAV, disponibilité en stock ne sont pas dans
  le modèle ; un écart résiduel peut légitimement refléter l'une d'elles plutôt qu'une décision de prix.
- **Trois bugs corrigés, rien ne garantit l'exhaustivité** — la méthodologie de §3.5/§4 s'applique
  systématiquement aux cas les plus extrêmes à chaque exécution, mais n'audite pas l'intégralité du catalogue à
  chaque passage.
- **La correction du clustering N2 (§3.2) a un coût statistique direct, assumé mais réel** — un clustering
  scientifiquement valide (jamais deux semaines mélangées) réduit mécaniquement l'effectif disponible par unité
  marque × gamme par rapport à l'ancienne méthode (incorrecte). Conséquence mesurée : couverture N2 en baisse
  (§2), stabilité bootstrap plus faible (ARI moyen 0,62 contre 0,82 annoncé précédemment, §3.2bis), échantillon
  de transitions testables réduit de plus de moitié (803 → 354, §3.5). Le compromis est délibéré (validité
  scientifique avant tout) mais réduit la puissance statistique de plusieurs analyses en aval — à garder en tête
  en particulier pour toute conclusion tirée d'une unité marque × gamme à ARI < 0,5 (6 unités sur 21, §3.2bis).
- **Modélisation par cluster (§3.3ter) : couverture partielle, surtout côté N2** — seuls 27,1 % des clusters N2
  (contre 52,6 % des clusters N1) disposent d'un modèle dédié retenu ; le reste continue d'utiliser le modèle
  catégorie. Ce n'est pas un défaut d'implémentation mais une conséquence directe et honnête des petits
  effectifs par unité marque × gamme (cf. point précédent) — le garde-fou empêche délibérément un ajustement
  sous-alimenté d'être utilisé pour la prédiction, au prix d'une couverture plus faible plutôt que d'un résultat
  trompeur.
- **Distance mixte du K-Means (N1/N2), documentée mais non corrigée** (cf. §3.2bis) — la distance euclidienne
  mélange variables continues standardisées et catégorielles one-hot ; le poids relatif de chaque groupe dépend
  en partie du nombre de modalités encodées, un artefact du one-hot encoding plutôt qu'une pondération choisie.
- **Les diagnostics de stabilité/justification de k sont purement descriptifs** (§3.2bis) — aucun mécanisme
  n'écarte ou ne réajuste automatiquement une unité à ARI faible (< 0,5) ; l'interprétation reste manuelle,
  cluster par cluster.
- **Les coefficients hédoniques sont des associations conditionnelles, pas des effets causaux établis** (cf.
  §3.3bis pour la discussion complète) — confusion possible caractéristique/segment de prix, exogénéité de la
  marque supposée plutôt que vérifiée.
- **Méthode de collecte non auditée sur le plan légal/ToS** — le scraper (`src/scraper/`) utilise un fetcher en
  mode furtif (`stealthy_headers`, User-Agent navigateur, cf. `src/scraper/utils.py`) pour accéder à Mytek.tn ;
  aucune revue des conditions d'utilisation du site ni de politique de rétention des données n'est documentée
  dans ce dépôt. Question à trancher avant toute citation externe de ce projet, pas quelque chose que ce rapport
  peut affirmer être réglé.

## 10. Pistes d'amélioration envisageables

Par ordre de priorité suggéré (détail dans les notebooks concernés, §« Conclusion et perspectives ») :

1. **Accumuler davantage de semaines** — le levier scientifique le plus direct ; permettrait un vrai test de
   tendance/saisonnalité plutôt que des comparaisons point à point.
2. **Suivi d'identité produit inter-semaines** (`url`/`gtin`) au sein d'un cluster — distinguerait explicitement,
   transition par transition, un changement de composition d'une re-tarification du même produit (actuellement
   seul `composition_stable`, basé sur l'effectif, en donne un indice indirect).
3. **Modèle à effets fixes cluster × semaine** — alternative paramétrique au bootstrap actuel (non-paramétrique),
   partageant l'information entre clusters voisins plutôt que de traiter chaque transition isolément.
4. **Étendre la grille de lecture de §3.5 au clustering N1** en parallèle, comme contre-épreuve indépendante de
   la segmentation marque × gamme — **partiellement fait le 2026-08-01** : la brique de base (prix réel vs
   estimé par les 3 modèles, par cluster N1 et par semaine, `reports/n1_cluster_estimations_hebdo.csv`) existe
   désormais et est visible dans le dashboard (§5), mais la grille de classification complète (quadrants
   prix/qualité, `cause_principale`, confirmation bootstrap) de `cluster_transitions` reste, elle, spécifique à
   N2 — reste à généraliser.
5. **Audit systématique des chaînes `specs_brutes`** pour d'autres artefacts du même type que ceux de §4 (ex.
   d'autres notations ambiguës non couvertes par les motifs de retrait actuels).
6. **K-Prototypes ou distance de Gower pour le clustering N1/N2** — corrigerait la limite documentée en §3.2bis
   (poids implicite du one-hot encoding dans la distance euclidienne mixte), au prix d'une dépendance
   supplémentaire (K-Prototypes n'est pas dans scikit-learn).
7. **Seuil d'alerte automatique sur l'ARI de stabilité** — les 6 unités N2 à ARI < 0,5 (§3.2bis) sont
   actuellement identifiées mais pas signalées automatiquement à l'utilisateur du dashboard au-delà de la
   coloration de la table ; un bandeau d'alerte dédié serait plus visible.
8. **Renforcer la couverture des modèles par cluster N2 à mesure que les semaines s'accumulent** (§3.3ter) —
   27,1 % des clusters N2 seulement disposent aujourd'hui d'un modèle dédié retenu (0/17 pour `televiseurs`) ;
   le garde-fou d'effectif est correct mais conservateur sur une fenêtre de 4 semaines encore courte (point 1) —
   à réévaluer périodiquement, jamais en assouplissant le seuil lui-même.
