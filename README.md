# Suivi Hebdomadaire des Valorisations Hédoniques des Produits High-Tech sur Mytek

Projet réalisé dans le cadre d'un stage à l'**Institut National de la Statistique (INS), Tunisie**. Il collecte hebdomadairement les prix des produits high-tech sur [Mytek.tn](https://www.mytek.tn) et décompose leur prix par la méthode des **prix hédoniques** (Lancaster 1966 / Rosen 1974), afin d'isoler la contribution de chaque caractéristique technique au prix observé.

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
  scraper/         # Collecte hebdomadaire des annonces Mytek.tn
  preprocessing/    # Nettoyage, imputation, encodage, sélection de features
  models/          # Régression hédonique (Ridge) et Random Forest (feature importance)
  utils/           # Configuration et logging

notebooks/         # Analyses exploratoires, clustering et segmentation
data/
  raw/             # Données brutes scrapées, par semaine (non versionné)
  processed/       # Données nettoyées et encodées, par catégorie (non versionné)
outputs/
  labels/          # Résultats de clustering/segmentation par catégorie
reports/           # Graphiques et rapports générés
tests/             # Tests unitaires et d'intégration (miroir de src/)
```

## Pipeline

1. **Scraping** (`src/scraper`) — collecte hebdomadaire des fiches produits.
2. **Preprocessing** (`src/preprocessing`) — nettoyage, bornage des valeurs aberrantes, imputation, encodage, sélection de features pertinentes pour `log(prix_tnd)` (tests de Spearman/Kruskal-Wallis, ou justification théorique).
3. **Modélisation** (`src/models`) — régression hédonique (Ridge) pour la décomposition du prix, Random Forest pour l'importance des features.
4. **Notebooks** (`notebooks/`) — EDA, segmentation prix/marque, clustering, comparaison de modèles, évolution temporelle du marché.

## Conventions

- Code et notebooks documentés en **français**, avec justification méthodologique (référence à la littérature académique).
- Aucun filtrage silencieux : toute ligne ou colonne exclue est reportée avec son nombre et sa raison.
- Analyse par catégorie (jamais groupée), et segmentation par marque avant clustering.

## Tests

```bash
pytest
```

Les tests suivent la même arborescence que `src/` (`tests/test_preprocessing`, `tests/test_models`, `tests/test_scraper`).
