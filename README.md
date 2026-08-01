# Suivi Hebdomadaire des Valorisations Hédoniques des Produits High-Tech sur Mytek

Projet réalisé dans le cadre d'un stage à l'**Institut National de la Statistique (INS), Tunisie**. Il collecte hebdomadairement les prix des produits high-tech sur [Mytek.tn](https://www.mytek.tn) et décompose leur prix par la méthode des **prix hédoniques** (Lancaster 1966 / Rosen 1974), afin d'isoler la contribution de chaque caractéristique technique au prix observé.

**Dashboard public :** https://hedonique-mytek-dashboard.onrender.com (niveau gratuit Render — la première visite après une période d'inactivité peut prendre 30-60s).

## Catégories suivies

- PC de bureau (`pc_bureau`)
- PC portables (`pc_portables`)
- Smartphones (`smartphones`)
- Téléphones portables (`telephones_portables`)
- Téléviseurs (`televiseurs`)

Chaque catégorie est analysée séparément : les échelles de prix et les caractéristiques pertinentes diffèrent trop d'une catégorie à l'autre pour une analyse groupée.

## Structure du projet

```
src/
  scraper/          # Collecte hebdomadaire des annonces Mytek.tn
  preprocessing/     # Nettoyage, imputation, encodage, sélection de features
  models/           # Régression hédonique (OLS/Ridge), Random Forest, rapports hebdomadaires
  dashboard/        # Dashboard Dash (Plotly + Dash Mantine Components + dash-ag-grid)
  utils/            # Configuration et logging

notebooks/          # Analyses exploratoires, clustering, segmentation, comparaisons, transitions
scripts/            # Utilitaires hors-ligne (ex. generate_notebook_pdfs.py)
data/
  raw/              # Données brutes scrapées, par semaine (non versionné -- volumineux, pas relu par le dashboard)
  processed/        # Données nettoyées par catégorie/semaine (VERSIONNÉ -- nécessaire au déploiement public)
outputs/
  labels/           # Export de clustering figé de la semaine 1 (voir note ci-dessous -- ne pas confondre avec
                     # le clustering courant, recalculé à la volée par src/models/hedonic_model.py)
models/             # Modèles entraînés + métriques (VERSIONNÉ, cf. ci-dessus)
reports/            # Rapports hebdomadaires générés, diagnostics, PDF des notebooks (VERSIONNÉ)
tests/              # Tests unitaires et d'intégration (mirroir partiel de src/, cf. §Tests)

Dockerfile, docker-compose.yml, render.yaml, requirements-deploy.txt   # Déploiement du dashboard (cf. §Déploiement)
```

> `outputs/labels/*.csv` provient des notebooks de clustering (`Clustering_produits_technologiques.ipynb` pour
> N1, `Segmentation_Prix_Clustering_produits_technologiques.ipynb` pour N2), calculé sur le seul instantané
> semaine 1 — `produits_prix_cluster_semaine_<catégorie>.csv` étend ensuite ces clusters à toutes les semaines
> disponibles, PAR PRODUIT (un cluster ne doit jamais mélanger des observations de semaines différentes,
> décision utilisateur du 2026-07-31, cf. `hedonic_model.n2_reference_week`), pas en le recalculant. Depuis le
> correctif d'harmonisation du 2026-08-01, ces notebooks délèguent leur règle de classification continue/
> catégorielle à `src/models/hedonic_model.py::_classify_features` (source unique) — leurs clusters concordent
> désormais à 100% (gamme ET composition exacte) avec `models/<catégorie>/pooled_labeled.csv`, le clustering
> utilisé par le reste du projet (dashboard, rapports), lui-même ancré sur la même semaine de référence.

## Pipeline

1. **Scraping** (`src/scraper`) — collecte hebdomadaire des fiches produits.
2. **Preprocessing** (`src/preprocessing`) — nettoyage, bornage des valeurs aberrantes, imputation, encodage, sélection de features pertinentes pour `log(prix_tnd)` (tests de Spearman/Kruskal-Wallis, ou justification théorique). `python -m src.preprocessing.pipeline --all` sélectionne les features UNE SEULE FOIS sur toutes les semaines poolées (recommandé, évite la dérive de schéma d'une semaine à l'autre) ; le mode une-semaine-à-la-fois reste disponible pour un usage ponctuel.
3. **Modélisation** (`src/models`) — régression hédonique (OLS + Ridge) pour la décomposition du prix, Random Forest pour l'importance des features, un modèle **par catégorie entière** puis, en plus, un modèle **par cluster** (N1 technique et N2 marque × gamme séparément) quand l'effectif le permet ET que le résultat bat démontrablement le modèle catégorie sur son propre test hors-échantillon (`save_artifacts.fit_models_per_segment` — jamais un modèle de cluster simplement estimable, cf. sa docstring), rapports hebdomadaires agrégés (`weekly_report.py`).
4. **Dashboard** (`src/dashboard`) — visualisation interactive (statistiques descriptives, modèles & clustering, évolution hebdomadaire, prédiction, téléchargements) sur les artefacts générés aux étapes 2-3, strictement en lecture seule. La page Prédiction utilise automatiquement le modèle le plus spécifique retenu pour le produit hypothétique saisi (N2 > N1 > catégorie).
5. **Notebooks** (`notebooks/`) — EDA, segmentation prix/marque, clustering, comparaison de modèles, évolution temporelle du marché, étude des transitions par cluster.

## Dashboard

```bash
python -m src.dashboard.app        # -> http://127.0.0.1:8050
```

Nécessite `data/processed/` et `models/` (déjà versionnés dans ce dépôt — un clone frais peut lancer le
dashboard immédiatement, sans re-scraper ni ré-entraîner). Pour régénérer ces artefacts depuis zéro : voir le
Pipeline ci-dessus.

## Déploiement

Le dashboard est déployé publiquement sur [Render](https://render.com) (`render.yaml`, build Docker via
`Dockerfile`) — voir le lien en tête de ce document. Pour un lancement local en conteneur (identique à ce qui
tourne en production) :

```bash
docker compose up --build   # -> http://localhost:8050
```

`requirements-deploy.txt` est un sous-ensemble allégé de `requirements.txt` (sans le scraper ni les notebooks,
inutiles à l'exécution du dashboard). Penser à régénérer `reports/notebooks_pdf/` (`python
scripts/generate_notebook_pdfs.py`, hors-ligne) avant un rebuild si un notebook a changé.

## Conventions

- Code et notebooks documentés en **français**, avec justification méthodologique (référence à la littérature académique).
- Aucun filtrage silencieux : toute ligne ou colonne exclue est reportée avec son nombre et sa raison.
- Analyse par catégorie (jamais groupée), et segmentation par marque avant clustering.

## Tests

```bash
pytest
```

**CI** : chaque push/PR exécute automatiquement `pytest` sur GitHub Actions (`.github/workflows/tests.yml`,
Ubuntu, Python 3.12) — badge d'état visible sur la page GitHub du dépôt.

Organisés en sous-dossiers miroir de `src/` (`tests/test_preprocessing`, `tests/test_models`,
`tests/test_scraper`, `tests/test_dashboard`), mais **pas encore une couverture fichier-pour-fichier** :
`src/scraper/spider.py`, `utils.py` et `scheduler.py` (pagination, retry, checkpoint/reprise) n'ont pas de test
dédié, seul `parser.py` en a un. `python -m src.preprocessing.pipeline` (le point d'entrée réel de production)
est couvert par `tests/test_preprocessing/test_pipeline.py` ; `bounds.py`/`impute.py`/`select_features.py`
n'ont pas encore de tests unitaires dédiés.

## Licence

Voir [LICENSE](LICENSE).
