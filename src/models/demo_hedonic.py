# -*- coding: utf-8 -*-
"""
=============================================================================
Script : src/models/demo_hedonic.py
=============================================================================
ROLE :
    Demo de bout en bout de src/models/hedonic_model.py sur des donnees
    REELLES (categorie pc_bureau, data/processed/week_1/) : chargement,
    gammes de prix, clusters techniques recalcules, strategie A (pentes
    communes), strategie B (pentes libres par marque dominante),
    comparaison formelle (test de Chow), monotonie des gammes,
    diagnostics (VIF, test de White, distance de Cook, QQ-plot).

    Separe du __main__ de hedonic_model.py (qui ne fait qu'un sanity
    check avec des donnees jouet, meme convention que ridge_model.py /
    rf_model.py) : ceci est le livrable "demo qui tourne de bout en bout
    sur une vraie categorie" demande explicitement.

UTILISATION :
    python -m src.models.demo_hedonic
    python -m src.models.demo_hedonic --category pc_portables
=============================================================================
"""

import argparse
from pathlib import Path

from src.models.hedonic_model import (
    MIN_BRAND_COUNT,
    load_category_data,
    compute_price_tiers,
    compute_cluster_labels,
    fit_strategy_a,
    fit_strategy_b,
    compare_strategies,
    check_tier_monotonicity,
    run_diagnostics,
    _classify_features,
)

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def run_demo(category: str = "pc_bureau") -> None:
    print(f"{'=' * 70}\nDémo hédonique -- {category}\n{'=' * 70}\n")

    # 1. Chargement ----------------------------------------------------------
    df_raw = load_category_data(category)
    print(f"{len(df_raw)} produits chargés (data/processed/week_1/{category}_clean.csv)")

    # 2. Gammes de prix (marque x prix uniquement) ----------------------------
    df_tiers, brand_plan = compute_price_tiers(df_raw)
    print(f"\n{len(brand_plan)} marques retenues (>= {MIN_BRAND_COUNT} produits) :")
    print(brand_plan.to_string(index=False))

    # 3. Caractéristiques + clusters techniques (recalculés à la volée) -------
    continuous_features, categorical_features = _classify_features(df_tiers)
    print(f"\nCaractéristiques continues   : {continuous_features}")
    print(f"Caractéristiques catégorielles : {categorical_features}")

    df_clustered, unit_summary = compute_cluster_labels(
        df_tiers, category, continuous_features=continuous_features, categorical_features=categorical_features,
    )
    print(f"\nUnités marque × gamme ({len(unit_summary)}) :")
    print(unit_summary.to_string(index=False))
    print(f"\n  -> {(unit_summary['outcome'] == 'clustered').sum()} clusterisées, "
          f"{(unit_summary['outcome'] == 'no_structure').sum()} sans structure retenue, "
          f"{(unit_summary['outcome'] == 'too_small').sum()} trop petites (jamais tentées).")

    # 4. Stratégie A -----------------------------------------------------------
    print(f"\n{'-' * 70}\nStratégie A -- un modèle pour {category}, pentes communes, "
          f"cluster_id en effets fixes\n{'-' * 70}")
    model_a, X_a, y_a = fit_strategy_a(df_clustered, continuous_features, categorical_features)
    print(f"n={int(model_a.nobs)}, {X_a.shape[1] + 1} paramètres, adj_R²={model_a.rsquared_adj:.3f}, "
          f"AIC={model_a.aic:.1f}, BIC={model_a.bic:.1f}")
    print("\nCoefficients (10 plus grands effets en valeur absolue) :")
    coefs_a = model_a.get_coefficients()
    print(coefs_a.head(10).to_string(index=False))

    # 5. Stratégie B -------------------------------------------------------------
    print(f"\n{'-' * 70}\nStratégie B -- régressions séparées par marque dominante "
          f"(pentes libres)\n{'-' * 70}")
    results_b, skipped_b = fit_strategy_b(df_clustered, continuous_features, categorical_features, top_n_brands=3)
    for brand, res in results_b.items():
        print(f"  [{brand}] n={res['n']}, adj_R²={res['model'].rsquared_adj:.3f}")
    if not skipped_b.empty:
        print("\nMarques écartées (effectif insuffisant pour le nombre de prédicteurs) :")
        print(skipped_b.to_string(index=False))

    # 6. Comparaison formelle A vs B (test de Chow) -------------------------------
    print(f"\n{'-' * 70}\nComparaison A vs B -- test de Chow (interactions marque × "
          f"caractéristique)\n{'-' * 70}")
    comparison, chow, restricted, unrestricted = compare_strategies(
        df_clustered, continuous_features, categorical_features, top_n_brands=3,
    )
    print(comparison.to_string(index=False))
    print(f"\nTest de Chow : F={chow['F']:.3f}, p={chow['p_value']:.4g} (df_diff={chow['df_diff']})")
    print(f"-> {chow['conclusion']}")

    # 7. Monotonie des gammes ------------------------------------------------------
    print(f"\n{'-' * 70}\nMonotonie des gammes -- prix prédit croissant Économique -> "
          f"Milieu -> Premium ?\n{'-' * 70}")
    monotonicity = check_tier_monotonicity(df_clustered, model_a)
    if monotonicity.empty:
        print("Aucune marque avec >= 2 gammes distinctes -- rien à vérifier.")
    else:
        print(monotonicity.to_string(index=False))
        n_flagged = int((~monotonicity["monotone"]).sum())
        if n_flagged:
            print(f"\n{n_flagged} marque(s) où 'Premium' n'est PAS hédoniquement plus chère une "
                  f"fois les caractéristiques contrôlées (majoration de marque pure).")
        else:
            print("\nToutes les marques vérifiées sont monotones.")

    # 8. Diagnostics (modèle A) --------------------------------------------------
    print(f"\n{'-' * 70}\nDiagnostics -- modèle A\n{'-' * 70}")
    diag = run_diagnostics(model_a)
    print("\nVIF (colinéarité, > 10 = préoccupant) :")
    print(diag["vif"].to_string(index=False))
    print(f"\nTest de White : LM={diag['white_test']['LM_stat']:.2f}, p={diag['white_test']['LM_p_value']:.4g}")
    print(f"-> {diag['white_test']['conclusion']}")
    print(f"\nPoints influents (distance de Cook > {diag['cooks_threshold']:.4f}) : "
          f"{diag['n_influential_points']} / {int(model_a.nobs)}")
    if diag["n_singular_leverage_points"]:
        print(f"  dont {diag['n_singular_leverage_points']} à levier singulier (Cook's D non défini, "
              f"levier=1 -- un point qui EST son propre paramètre, ex. seule observation d'un cluster_id).")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    qq_path = REPORTS_DIR / f"hedonic_qq_{category}.png"
    diag["qq_fig"].savefig(qq_path, dpi=110, bbox_inches="tight")
    print(f"\nQQ-plot des résidus sauvegardé : {qq_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Démo du modèle hédonique OLS sur une catégorie réelle.")
    parser.add_argument("--category", type=str, default="pc_bureau",
                         choices=["pc_bureau", "pc_portables", "smartphones", "telephones_portables", "televiseurs"])
    args = parser.parse_args()
    run_demo(args.category)
