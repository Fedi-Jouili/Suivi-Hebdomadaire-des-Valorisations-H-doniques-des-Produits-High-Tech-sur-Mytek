# Segmentation et Modélisation Hédonique des Prix — Marché High-Tech Mytek.tn

## Rapport de synthèse du stage

**Auteur :** Fedi Jouili · **Période couverte par les données :** 4 semaines de collecte (S1–S4, juin–juillet 2026)
**Dernière mise à jour de ce rapport :** 2026-07-29
**Dashboard public :** https://hedonique-mytek-dashboard.onrender.com

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
| **Tests** | `tests/` | Preuve exécutable que le code fait ce que ce rapport affirme (233 tests, cf. §8). |

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
  | PC de bureau | 873 | 100 % | 100 % |
  | PC portables | 2 155 | 100 % | 100 % |
  | Smartphones | 1 012 | 100 % | 100 % |
  | Téléphones portables | 291 | 100 % | 95,8 % |
  | Téléviseurs | 448 | 100 % | 100 % |

  (N1 = clustering technique pur, structurellement exhaustif ; N2 = segmentation marque × gamme, cf. §3 —
  couverture mesurée, jamais supposée, `reports/couverture_clustering_hebdo.csv`.)

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

### 3.2bis Rigueur du clustering K-Means : stabilité et choix de k (ajouté 2026-07-27)

Le clustering N2 (ci-dessus), utilisé ensuite par toutes les analyses de §3.5, n'avait jusqu'ici aucune mesure de
robustesse : le k retenu par la sélection interne (silhouette maximale sous contrainte d'effectif minimal par
cluster) n'était jamais comparé aux k voisins qu'il a écartés, ni testé pour sa stabilité à l'échantillonnage.
Deux diagnostics ajoutés (`src/models/weekly_report.py`, visibles dans le dashboard — page « Modèles &
clustering », onglet clustering) :

- **Stabilité bootstrap** (`cluster_stability_n2`, `reports/stabilite_clustering_n2.csv`) : pour chaque unité
  marque × gamme clusterisée (60 sur les 5 catégories), 100 réplications bootstrap (rééchantillonnage des
  produits, ré-ajustement d'un K-Means, comparaison par Adjusted Rand Index — Hubert & Arabie 1985 — aux labels
  originaux). Résultat : ARI moyen de **0,82** (médiane 0,88) sur les 60 unités clusterisées ; seules **4 unités
  (6,7 %)** ont un ARI < 0,5 (segmentation fragile — ex. SMARTEC/téléphones portables, MYTEK Premium/PC de
  bureau). La grande majorité du clustering N2 est donc robuste au bruit d'échantillonnage, mais ces cas précis
  doivent être lus avec prudence.
- **Justification du choix de k** (`k_selection_justification`, `reports/justification_k_clustering.csv`) :
  expose la silhouette de TOUS les k testés (retenus ou non), avec la raison de rejet explicite quand applicable
  (effondrement K-Means sur points dupliqués, ou cluster sous l'effectif minimal). Rend auditable un choix qui ne
  l'était pas auparavant — seul le k final apparaissait dans les artefacts persistés.
- **Limite documentée, non corrigée** : le K-Means sous-jacent (N1 et N2) mélange, dans une seule distance
  euclidienne, des variables continues standardisées et des variables catégorielles one-hot — le poids relatif
  de chaque groupe dans la distance dépend en partie du nombre de modalités encodées (artefact du one-hot
  encoding, pas une pondération délibérée). Une distance dédiée aux données mixtes (K-Prototypes, Gower) serait
  plus rigoureuse ; non implémentée ici (cf. §10).

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
sur les données actuelles : sur 803 transitions (5 catégories, 3 périodes), le seuil fixe en classait 65 comme
« notables » — le test bootstrap n'en confirme que **17** (26 %) avec un intervalle excluant 0. La majorité des
écarts détectés par un seuil naïf ne résistent pas à un test tenant compte de la taille d'échantillon — un
résultat qui, en lui-même, justifie l'upgrade.

**Upgrade de rigueur (2026-07-26) — répondre explicitement à « prix, qualité, ou nombre de produits ? »** : la
grille 3×3 ci-dessus suppose implicitement que l'EFFECTIF du cluster est resté stable d'une semaine à l'autre.
Si des produits sont entrés/sortis du cluster entre les deux semaines, une lecture « prix et qualité bougent
ensemble » peut simplement refléter un catalogue différent (des produits moins chers sont partis, des plus
chers sont arrivés) — pas un changement de prix ni de qualité d'un produit donné. `cluster_transitions` teste
maintenant ce cas **en priorité**, avant toute lecture de la grille : si l'effectif a changé, la cause retenue
(colonne `cause_principale`) est directement `effectif`, et la grille prix/qualité n'est pas invoquée pour cette
transition ; sinon, `cause_principale` est déduite de la grille (`prix`, `qualite`, `prix_et_qualite`, ou
`aucune`).

Sur les données actuelles (803 transitions) : **79,3 %** ne montrent aucun changement notable (`aucune`),
**19,3 %** sont dominées par un changement de COMPOSITION du cluster (`effectif` — le nombre de produits a
changé, la lecture prix/qualité n'est pas isolable pour ces cas), et seulement **1,4 %** combinent un vrai
signal de prix et/ou de qualité (1,0 % qualité seule, 0,4 % prix seul, 0 % les deux à la fois). Réponse directe
à la question posée : sur cette fenêtre de 4 semaines, le changement de CATALOGUE (quels produits sont vendus)
domine très largement sur un changement de PRIX ou de QUALITÉ d'un produit donné — la variation du panier moyen
d'un cluster est, dans la grande majorité des cas observés, due à autre chose qu'une décision tarifaire ou une
montée en gamme.

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
(`notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb` §7). **Conclusion méthodologique retenue comme la
plus importante du projet à ce stade :** un signal détecté automatiquement, même le plus extrême et même
statistiquement significatif au sens bootstrap, ne doit jamais être interprété économiquement sans redescendre
au niveau produit — le bootstrap protège contre le bruit d'échantillonnage, pas contre un biais systématique
amont.

## 5. Dashboard (`src/dashboard/`)

Application Dash multi-pages, strictement en lecture seule sur les artefacts persistés (jamais de `.fit()` dans
le dashboard) :

- **Accueil** — vue d'ensemble, fraîcheur des données.
- **Statistiques descriptives** — volumes, prix, marques, caractéristiques techniques, par catégorie.
- **Modèles & clustering** — coefficients hédoniques, importance des variables (RF), profils de clusters N1/N2,
  et depuis le 2026-07-27 les 2 diagnostics de rigueur de §3.2bis (ARI de stabilité bootstrap, silhouette du k
  retenu) directement dans la table des unités N2.
- **Prédiction** — produit hypothétique → prix estimé (avec intervalle pour OLS), segment assigné, produits
  réels comparables ; RAM/stockage en paliers réellement observés (pas un slider continu) ; ajustement optionnel
  par semaine via l'indice de §3.4.
- **Évolution hebdomadaire** — couverture du clustering, prix par cluster, prix réel vs estimé par modèle,
  et l'étude de transitions de §3.5/§4 (grille de lecture, cas notables, confirmation bootstrap).
- **Téléchargements** (ajouté 2026-07-29) — notebooks en PDF (pré-générés hors-ligne, cf.
  `scripts/generate_notebook_pdfs.py`), estimations par cluster × semaine par catégorie (avec les colonnes
  d'erreur d'estimation par modèle), données produit par catégorie.

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
  et 0,99 selon catégorie/modèle) — cf. tableau §3.3 et `models/<catégorie>/metrics.json`.
- La segmentation N2 (marque × gamme) couvre 100 % du catalogue retenu sur 4 des 5 catégories, 95,8 % sur
  `telephones_portables` (écarts documentés, jamais silencieux).
- Sur la fenêtre actuelle, un seul mouvement de prix « toutes choses égales » est statistiquement significatif
  au niveau catégorie (`smartphones` S2, +1,97 %) — le marché ne montre pas, pour l'instant, de signal fort de
  re-tarification généralisée.
- Au niveau cluster, la grande majorité des transitions (≈ 91 %, cf. `reports/transitions_cluster_hebdo.csv`)
  ne montrent aucun écart entre prix réel et valeur technique implicite ; parmi les écarts détectés par un
  seuil naïf, seule une minorité (26 %) résiste à un test statistique explicite.
- **Prix, qualité, ou nombre de produits ?** (cf. §3.5) Sur les 803 transitions observées, **19,3 %** sont
  dominées par un changement du NOMBRE de produits du cluster (`cause_principale = effectif`), contre
  seulement **1,4 %** avec un vrai signal de prix et/ou qualité isolable — le changement de catalogue domine
  très largement sur la re-tarification ou la montée en gamme d'un produit donné, sur cette fenêtre de 4
  semaines.
- **Stabilité du clustering N2** (cf. §3.2bis) : ARI moyen de 0,82 (médiane 0,88) sur les 60 unités marque ×
  gamme clusterisées ; 4 unités (6,7 %) présentent une segmentation fragile (ARI < 0,5), à interpréter avec
  prudence.
- Deux bugs de pipeline réels ont été identifiés et corrigés grâce à cette méthodologie, avec effet mesurable
  sur les artefacts en aval (cf. §4) — une validation *a posteriori* de l'intérêt de descendre au niveau produit.

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

- **234 tests automatisés** (`pytest tests/`, tous passants), couvrant le prétraitement, les 3 modèles, le
  clustering, les rapports hebdomadaires et les fonctions pures du dashboard — exécutés automatiquement par CI
  sur chaque push/PR depuis le 2026-07-28 (`.github/workflows/tests.yml`), plus manuellement avant. 65 tests
  ajoutés le 2026-07-25 : 8 régressions directes sur les bugs de §4, 40 sur `src/models/weekly_report.py`, 17 sur
  les fonctions pures de la page dashboard « Évolution hebdomadaire ». 19 tests supplémentaires ajoutés le
  2026-07-26/27 pour les upgrades de rigueur de §3.2bis/§3.5. 17 tests supplémentaires ajoutés le 2026-07-28
  (audit méthodologique externe) : `tests/test_preprocessing/test_pipeline.py` (le point d'entrée réel de
  production, jusque-là jamais testé directement, y compris la sélection de features stable poolée) et
  `tests/test_models/test_save_artifacts.py`/`test_rf.py` (accord entre modèles, importance par permutation).
  Zones encore non couvertes par un test dédié : `src/scraper/spider.py`/`utils.py`/`scheduler.py` (pagination,
  retry, checkpoint), `bounds.py`/`impute.py`/`select_features.py` pris isolément (exercés indirectement via
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
- **Deux bugs corrigés, rien ne garantit l'exhaustivité** — la méthodologie de §3.5/§4 s'applique
  systématiquement aux cas les plus extrêmes à chaque exécution, mais n'audite pas l'intégralité du catalogue à
  chaque passage.
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
   la segmentation marque × gamme.
5. **Audit systématique des chaînes `specs_brutes`** pour d'autres artefacts du même type que ceux de §4 (ex.
   d'autres notations ambiguës non couvertes par les motifs de retrait actuels).
6. **K-Prototypes ou distance de Gower pour le clustering N1/N2** — corrigerait la limite documentée en §3.2bis
   (poids implicite du one-hot encoding dans la distance euclidienne mixte), au prix d'une dépendance
   supplémentaire (K-Prototypes n'est pas dans scikit-learn).
7. **Seuil d'alerte automatique sur l'ARI de stabilité** — les 4 unités N2 à ARI < 0,5 (§3.2bis) sont
   actuellement identifiées mais pas signalées automatiquement à l'utilisateur du dashboard au-delà de la
   coloration de la table ; un bandeau d'alerte dédié serait plus visible.
