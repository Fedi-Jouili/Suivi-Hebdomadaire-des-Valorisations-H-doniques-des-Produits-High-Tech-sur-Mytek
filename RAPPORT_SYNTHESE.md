# Segmentation et Modélisation Hédonique des Prix — Marché High-Tech Mytek.tn

## Rapport de synthèse du stage

**Auteur :** Fedi Jouili · **Période couverte par les données :** 4 semaines de collecte (S1–S4, juin–juillet 2026)
**Dernière mise à jour de ce rapport :** 2026-07-25

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
| **Tests** | `tests/` | Preuve exécutable que le code fait ce que ce rapport affirme (197 tests, cf. §8). |

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

### 3.3 Modélisation hédonique (`src/models/`)

Trois modèles, entraînés sur `log(prix)`, sur les mêmes données poolées (4 semaines, split train/test **groupé
par produit** pour interdire toute fuite d'un même produit vu à plusieurs semaines) :

| Modèle | Rôle | R² test (log) — plage observée |
|---|---|---|
| **Hedonic OLS** (statsmodels, erreurs-types HC3) | Inférence : coefficients, p-values, prix implicites | 0,545 – 0,944 |
| **Ridge** (Rosen semi-log, GridSearchCV) | Alternative régularisée, mêmes coefficients réexprimés | 0,545 – 0,945 |
| **Random Forest** | Non-linéaire, benchmark de plafond prédictif | 0,404 – 0,985 |

Écart notable : Random Forest domine largement sur `smartphones` (R² = 0,985 contre 0,839 pour OLS/Ridge — la
relation prix/caractéristiques y est probablement non-linéaire, effets de seuil de marque/génération que le
modèle log-linéaire ne capture pas) mais reste en retrait sur `pc_bureau` (0,404 contre 0,545) — aucun modèle ne
domine partout, d'où la comparaison systématique plutôt qu'un choix a priori.

Diagnostics post-estimation (`run_diagnostics`) : VIF (colinéarité), test de White (hétéroscédasticité),
distance de Cook (observations influentes), QQ-plot des résidus — cf. `reports/hedonic_qq_*.png` et
`notebooks/*.ipynb` pour le détail par catégorie.

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
- **Modèles & clustering** — coefficients hédoniques, importance des variables (RF), profils de clusters N1/N2.
- **Prédiction** — produit hypothétique → prix estimé (avec intervalle pour OLS), segment assigné, produits
  réels comparables ; RAM/stockage en paliers réellement observés (pas un slider continu) ; ajustement optionnel
  par semaine via l'indice de §3.4.
- **Évolution hebdomadaire** — couverture du clustering, prix par cluster, prix réel vs estimé par modèle,
  et l'étude de transitions de §3.5/§4 (grille de lecture, cas notables, confirmation bootstrap).

Lancement : `python -m src.dashboard.app` → http://127.0.0.1:8050

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
- Deux bugs de pipeline réels ont été identifiés et corrigés grâce à cette méthodologie, avec effet mesurable
  sur les artefacts en aval (cf. §4) — une validation *a posteriori* de l'intérêt de descendre au niveau produit.

## 7. Reproduire ce projet de bout en bout

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 1. Pretraitement (par semaine deja scrapee sous data/raw/week_N/)
python -m src.preprocessing.pipeline --raw-dir data/raw/week_1 --out-dir data/processed/week_1
# ... repeter pour chaque semaine disponible

# 2. Entrainement des modeles + clustering (tous les artefacts sous models/)
python -m src.models.save_artifacts

# 3. Rapports hebdomadaires (tous les CSV sous reports/, alimentent le dashboard)
python -m src.models.weekly_report

# 4. Dashboard
python -m src.dashboard.app        # http://127.0.0.1:8050

# 5. Notebooks (executes avec leurs sorties, mais reproductibles independamment) :
#    ordre de lecture recommande dans notebooks/README implicite par prefixe --
#    EDA -> Clustering -> Comparaison -> Segmentation -> Evolution_Temporelle -> Etude_Transitions

# 6. Tests
pytest tests/ -v
```

## 8. Qualité et reproductibilité du code

- **197 tests automatisés** (`pytest tests/`, tous passants), couvrant le prétraitement, les 3 modèles, le
  clustering, les rapports hebdomadaires et les fonctions pures du dashboard. 65 tests ajoutés le 2026-07-25 :
  8 régressions directes sur les bugs de §4 (garantissent qu'ils ne reviennent pas silencieusement), 40 sur
  `src/models/weekly_report.py` (couverture, écarts résiduels, test bootstrap), 17 sur les fonctions pures de la
  page dashboard « Évolution hebdomadaire ».
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
