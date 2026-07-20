# -*- coding: utf-8 -*-
"""
=============================================================================
Package : src/models
=============================================================================
ROLE :
    Modeles hedoniques du projet Mytek.tn -- log(prix) ~ caracteristiques.

      - ridge_model.py         : RidgeModel, regression log-lineaire
                                 (Rosen, 1974) avec interactions
                                 polynomiales et selection d'alpha par
                                 validation croisee.
      - rf_model.py            : RandomForestModel, alternative
                                 non-lineaire avec selection
                                 d'hyperparametres par validation croisee.
      - feature_importance.py  : compare_importances, croise coefficients
                                 Ridge et importances Random Forest.
      - hedonic_model.py       : HedonicOLS (statsmodels, inference --
                                 coefficients, erreurs-types robustes,
                                 p-values, tests F), avec pipeline complet
                                 gammes de prix -> clusters techniques ->
                                 regression (pentes communes ou libres par
                                 marque) -> comparaison formelle (test de
                                 Chow) -> diagnostics.

    Ce fichier re-exporte les classes/fonctions principales, par ex. :

        from src.models import RidgeModel, RandomForestModel, compare_importances
        from src.models import HedonicOLS, fit_strategy_a, compare_strategies
=============================================================================
"""

from .ridge_model import RidgeModel
from .rf_model import RandomForestModel
from .feature_importance import compare_importances
from .hedonic_model import (
    HedonicOLS,
    CircularityError,
    load_category_data,
    compute_price_tiers,
    compute_cluster_labels,
    build_design_matrix,
    fit_strategy_a,
    fit_strategy_b,
    compare_strategies,
    check_tier_monotonicity,
    run_diagnostics,
)

__all__ = [
    "RidgeModel",
    "RandomForestModel",
    "compare_importances",
    "HedonicOLS",
    "CircularityError",
    "load_category_data",
    "compute_price_tiers",
    "compute_cluster_labels",
    "build_design_matrix",
    "fit_strategy_a",
    "fit_strategy_b",
    "compare_strategies",
    "check_tier_monotonicity",
    "run_diagnostics",
]
