# -*- coding: utf-8 -*-
"""
=============================================================================
Script : src/models/demo_hedonic_pooled_time.py
=============================================================================
ROLE :
    Demo de bout en bout de fit_strategy_c_pooled_time (src/models/
    hedonic_model.py) sur des donnees REELLES : pour chaque categorie
    poolable (toutes sauf telephones_portables, cf.
    POOLED_TIME_EXCLUDED_CATEGORIES), compare le modele POOLE (pentes
    communes + effet fixe semaine) a 3 modeles SEPARES (un par semaine,
    fit_strategy_a) -- pour verifier EMPIRIQUEMENT que le pooling est
    justifie (adj R²/AIC comparables ou meilleurs pour le modele poole),
    pas suppose sur la seule base de la stabilite des coefficients deja
    observee dans Evolution_Temporelle_Marche_Mytek.ipynb.

UTILISATION :
    python -m src.models.demo_hedonic_pooled_time
=============================================================================
"""
from pathlib import Path

import pandas as pd

from src.models.hedonic_model import (
    POOLED_TIME_EXCLUDED_CATEGORIES,
    compute_price_tiers,
    compute_cluster_labels,
    fit_strategy_a,
    fit_strategy_c_pooled_time,
    _classify_features,
)
from src.preprocessing.split import discover_weeks

CATEGORY_ORDER = ["pc_bureau", "pc_portables", "smartphones", "telephones_portables", "televiseurs"]
DATA_PROCESSED_ROOT = Path(__file__).resolve().parents[2] / "data" / "processed"


def run_demo():
    for category in CATEGORY_ORDER:
        print(f"{'=' * 70}\n{category}\n{'=' * 70}")

        if category in POOLED_TIME_EXCLUDED_CATEGORIES:
            print("Exclue du pooling temporel (coefficients hédoniques instables entre semaines, "
                  "cf. Evolution_Temporelle_Marche_Mytek.ipynb) -- non testée ici.\n")
            continue

        # -- Modeles separes, un par semaine (Strategie A) -----------------------
        # Semaines decouvertes dynamiquement (jamais figees a (1, 2, 3), cf.
        # discover_weeks / fit_strategy_c_pooled_time) -- une semaine 4+ ajoutee
        # au depot est incluse ici sans modification de code.
        per_week_rows = []
        for w in discover_weeks(DATA_PROCESSED_ROOT):
            path = DATA_PROCESSED_ROOT / f"week_{w}" / f"{category}_clean.csv"
            if not path.exists():
                continue
            df_w = pd.read_csv(path)
            df_w, _ = compute_price_tiers(df_w)
            cont, cat = _classify_features(df_w)
            df_w, _ = compute_cluster_labels(df_w, category, continuous_features=cont, categorical_features=cat)
            model_w, X_w, y_w = fit_strategy_a(df_w, cont, cat)
            per_week_rows.append({
                "semaine": w, "n": int(model_w.nobs), "n_predicteurs": X_w.shape[1] + 1,
                "adj_r2": model_w.rsquared_adj, "aic": model_w.aic, "bic": model_w.bic,
            })
        per_week_df = pd.DataFrame(per_week_rows)
        print(f"\nModèles séparés (Stratégie A, {len(per_week_rows)} semaine(s)) :")
        print(per_week_df.to_string(index=False))

        # -- Modele poole avec effet fixe semaine (Strategie C) -----------------
        model_c, X_c, y_c, df_pooled = fit_strategy_c_pooled_time(category)
        print(f"\nModèle poolé (Stratégie C, effet fixe semaine, SE groupées par produit) :")
        print(f"n={int(model_c.nobs)}, {X_c.shape[1] + 1} paramètres, "
              f"adj_R²={model_c.rsquared_adj:.3f}, AIC={model_c.aic:.1f}, BIC={model_c.bic:.1f}")

        coefs_c = model_c.get_coefficients()
        semaine_rows = coefs_c[coefs_c["feature"].str.startswith("semaine_")]
        if not semaine_rows.empty:
            print("\nIndice de prix hédonique (effet des dummies semaine, référence = S1) :")
            print(semaine_rows[["feature", "pct_effect", "p_value"]].to_string(index=False))
        else:
            print("\n(Une seule semaine disponible pour cette catégorie -- pas de dummy semaine à afficher.)")

        # -- Verdict empirique ---------------------------------------------------
        mean_adj_r2_separate = per_week_df["adj_r2"].mean()
        print(f"\nadj_R² moyen des {len(per_week_rows)} modèles séparés = {mean_adj_r2_separate:.3f} vs. "
              f"adj_R² du modèle poolé = {model_c.rsquared_adj:.3f}")
        if model_c.rsquared_adj >= mean_adj_r2_separate - 0.03:
            print("-> Pooling statistiquement défendable : le modèle poolé n'est pas nettement moins bon "
                  "que la moyenne des modèles séparés, pour un modèle plus parcimonieux et un indice de "
                  "prix temporel directement interprétable.")
        else:
            print("-> Pooling discutable ici : le modèle poolé perd sensiblement en ajustement par rapport "
                  "aux modèles séparés -- les pentes ne sont peut-être pas aussi communes qu'espéré.")
        print()


if __name__ == "__main__":
    run_demo()
