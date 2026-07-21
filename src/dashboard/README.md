# Dashboard hédonique Mytek.tn

Dashboard Dash (Dash Mantine Components + dash-ag-grid + Plotly) pour visualiser et servir l'analyse hédonique
du projet — décomposition du prix des produits high-tech collectés sur Mytek.tn (stage à l'Institut National de
la Statistique, INS Tunisie).

## Prérequis

1. **Environnement** : le `.venv` du projet, avec les dépendances du dashboard installées :

   ```bash
   pip install -r src/dashboard/requirements.txt
   ```

   (pandas/scikit-learn/statsmodels/joblib sont déjà requis par `src/preprocessing` et `src/models` — non
   répétés dans ce fichier.)

2. **Données traitées** : `data/processed/week_*/<catégorie>_clean.csv` doivent exister (pipeline de
   prétraitement déjà exécuté, cf. `README.md` racine du projet). Sans elles, seule la page « Statistiques
   descriptives » sera partiellement fonctionnelle (message explicite affiché sinon, jamais une page blanche).

3. **Artefacts de modèles** : les pages « Modèles & clustering » et « Prédiction » nécessitent les artefacts
   entraînés par `src/models/save_artifacts.py` (Ridge/Random Forest/Hedonic OLS + clustering technique N1 +
   données étiquetées, un jeu par catégorie). Le dashboard est **strictement lecture seule** : il ne réentraîne
   jamais rien lui-même. À exécuter une fois avant le premier lancement, puis à chaque fois qu'une nouvelle
   semaine de collecte est disponible :

   ```bash
   python -m src.models.save_artifacts
   # ou pour une seule catégorie :
   python -m src.models.save_artifacts --category pc_bureau
   ```

   Écrit sous `models/<catégorie>/` (répertoire ignoré par git, régénérable à volonté).

## Lancement

```bash
python -m src.dashboard.app
```

Puis ouvrir <http://127.0.0.1:8050>. Le serveur démarre en `debug=True` par défaut (rechargement automatique,
barre d'outils de développement Dash visible en bas à droite) — pratique en développement, à désactiver pour une
démo/un déploiement (`app.run(debug=False)` dans `src/dashboard/app.py`, ou variable d'environnement selon le
mode de déploiement choisi).

## Pages

| Page | Route | Contenu |
|---|---|---|
| Accueil | `/` | Présentation, chiffres-clés cumulés, accès rapide |
| Statistiques descriptives | `/descriptif` | Volumes, prix, marques, specs techniques, valeurs manquantes, évolution sur les 4 dernières semaines — par catégorie |
| Modèles & clustering | `/modeles` | Méthodologie, métriques hors-échantillon (R²/RMSE, log ET TND), coefficients hédoniques avec incertitude, importances Random Forest, profils de clusters N1/N2 |
| Prédiction | `/prediction` | Composer un produit hypothétique, choisir un modèle + un type de segmentation, obtenir un prix prédit et des produits réels comparables |
| À propos | `/a-propos` | Profil du créateur (placeholders `TODO` à compléter) |

## Décisions méthodologiques notables

- **Jamais de circularité** : les 3 modèles de prédiction (OLS/Ridge/RF) utilisent `marque` comme effet fixe,
  jamais `gamme_prix`/`cluster_id` (dérivés du prix) — cf. le correctif du 2026-07-21 dans
  `src/models/hedonic_model.py` et `reports/audit_code.md` §3.1.
- **Segmentation N2 en 2 temps** (page Prédiction) : `gamme_prix` ne peut pas être calculée pour un produit
  hypothétique (elle dépend du prix, qu'on cherche justement à prédire) — le dashboard prédit d'abord un prix
  provisoire, en déduit une gamme à partir des bornes déjà observées pour la marque choisie, puis assigne le
  segment du produit réel le plus proche dans cette gamme. Cf. `src/dashboard/prediction_utils.py`.
- **Formation des modèles** : sur les 4 semaines de collecte poolées (`train.csv`+`test.csv` régénérés avec un
  split **groupé par produit**, `GroupShuffleSplit` sur `url` — un même produit vu à plusieurs semaines ne se
  retrouve jamais à la fois en train et en test). Cf. `src/models/save_artifacts.py`.
- **Retro-transformation** : les métriques RMSE/MAE en TND sont obtenues par simple `exp()` des prédictions log,
  sans correction de biais de retransformation (Duan/Miller) — explicité dans l'UI (page Modèles & clustering).

## Tests

```bash
pytest tests/test_dashboard/
```

## Structure

```
src/dashboard/
  app.py                  # point d'entrée (AppShell, MantineProvider, theme)
  theme.py                 # theme DMC + template Plotly (source unique)
  format_utils.py          # formatage numérique français
  data_loader.py            # accès caché aux données/artefacts (lecture seule)
  prediction_utils.py       # logique de la page Prédiction
  components/               # KPI card, en-tête de section, bandeau de provenance, sélecteur de catégorie, nav
  pages/                    # accueil, descriptif, modèles, prédiction, à propos
  assets/                    # CSS custom (chiffres tabulaires, thème AG Grid)
  requirements.txt
  README.md                 # ce fichier
```
