# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_models/test_save_artifacts.py
=============================================================================
ROLE :
    Tests des fonctions PURES de src/models/save_artifacts.py (aucun acces
    disque, aucun entrainement reel) -- jusqu'ici ce module n'avait aucun
    test dedie, seulement un exercice indirect via test_weekly_report.py
    (requires_artifacts). compute_model_agreement en particulier est une
    fonction nouvelle (2026-07-28, audit methodologique reviewer 1) :
    rien ne verifiait jusqu'ici que Random Forest confirme ou contredit
    OLS/Ridge sur les MEMES variables.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from src.models.save_artifacts import compute_model_agreement


def _coefs(feature_coef: dict, extra_cols: dict | None = None) -> pd.DataFrame:
    rows = [{"feature": "const", "coefficient": 6.9}]
    rows += [{"feature": f, "coefficient": c} for f, c in feature_coef.items()]
    df = pd.DataFrame(rows)
    if extra_cols:
        for col, val in extra_cols.items():
            df[col] = val
    return df


def _importances(feature_importance: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature": f, "importance": v} for f, v in feature_importance.items()]
    )


class TestComputeModelAgreement:
    def test_signes_identiques_toujours_accord(self):
        """OLS et Ridge coefficient de meme signe -- accord=True pour toutes
        les features."""
        ols = _coefs({"ram_go": 0.05, "stockage_go": 0.02}, extra_cols={"std_err": 0.01, "p_value": 0.01, "pct_effect": 5.0})
        ridge = _coefs({"ram_go": 0.04, "stockage_go": 0.03})
        rf = _importances({"ram_go": 0.6, "stockage_go": 0.4})

        comparison, resume = compute_model_agreement(ols, ridge, rf)

        assert comparison["ols_ridge_signes_accordent"].all()
        assert resume["pct_signes_ols_ridge_accordent"] == 100.0
        assert resume["n_features_comparees"] == 2

    def test_signes_opposes_detecte_le_desaccord(self):
        """Un signe oppose entre OLS et Ridge doit etre signale, pas noye
        dans la moyenne."""
        ols = _coefs({"ram_go": 0.05, "stockage_go": -0.02})
        ridge = _coefs({"ram_go": 0.04, "stockage_go": 0.03})  # signe oppose sur stockage_go
        rf = _importances({"ram_go": 0.6, "stockage_go": 0.4})

        comparison, resume = compute_model_agreement(ols, ridge, rf)

        row_stockage = comparison[comparison["feature"] == "stockage_go"].iloc[0]
        assert row_stockage["ols_ridge_signes_accordent"] == False  # noqa: E712
        assert resume["pct_signes_ols_ridge_accordent"] == 50.0

    def test_const_exclue_de_la_comparaison(self):
        """'const' (intercept OLS) n'a pas d'equivalent Ridge/RF -- ne doit
        jamais apparaitre dans la comparaison."""
        ols = _coefs({"ram_go": 0.05})
        ridge = _coefs({"ram_go": 0.04})
        rf = _importances({"ram_go": 1.0})

        comparison, _resume = compute_model_agreement(ols, ridge, rf)

        assert "const" not in set(comparison["feature"])

    def test_correlation_rf_ols_forte_et_positive_quand_classements_identiques(self):
        """RF et OLS d'accord sur le classement des variables -> Spearman
        rho proche de 1."""
        ols = _coefs({"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.05})
        ridge = _coefs({"a": 0.4, "b": 0.35, "c": 0.12, "d": 0.06})
        rf = _importances({"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05})

        _comparison, resume = compute_model_agreement(ols, ridge, rf)

        assert resume["rf_ols_spearman_rho"] == pytest.approx(1.0, abs=1e-6)

    def test_feature_absente_dun_modele_est_ecartee_proprement(self):
        """Une feature presente dans OLS/Ridge mais absente de RF (ne
        devrait normalement pas arriver -- meme X_train partout -- mais ne
        doit jamais planter) est simplement exclue de la comparaison."""
        ols = _coefs({"ram_go": 0.05, "stockage_go": 0.02})
        ridge = _coefs({"ram_go": 0.04, "stockage_go": 0.03})
        rf = _importances({"ram_go": 1.0})  # stockage_go absente

        comparison, resume = compute_model_agreement(ols, ridge, rf)

        assert set(comparison["feature"]) == {"ram_go"}
        assert resume["n_features_comparees"] == 1

    def test_aucune_feature_commune_ne_plante_pas(self):
        ols = _coefs({"ram_go": 0.05})
        ridge = _coefs({"autre_colonne": 0.04})
        rf = _importances({"encore_une_autre": 1.0})

        comparison, resume = compute_model_agreement(ols, ridge, rf)

        assert comparison.empty
        assert resume["n_features_comparees"] == 0
        assert resume["pct_signes_ols_ridge_accordent"] is None
        assert resume["rf_ols_spearman_rho"] is None
