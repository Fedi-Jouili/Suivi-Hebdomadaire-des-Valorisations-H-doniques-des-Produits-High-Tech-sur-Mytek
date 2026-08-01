# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_dashboard/test_models_page.py
=============================================================================
ROLE :
    Tests des fonctions PURES de src/dashboard/pages/models.py (page
    "Modèles & clustering", onglet "Modèles par cluster") -- transformation
    de DataFrames deja charges (equations lisibles, statut retenu/ecarte,
    vue d'ensemble par cluster), jamais d'acces disque ici, meme convention
    que test_evolution.py. Donnees synthetiques mais realistes (memes noms
    de colonnes que les CSV de models/<categorie>/, cf.
    src/models/save_artifacts.py::fit_models_per_segment/
    persist_segment_models).

    Ajoute le 2026-08-01ter -- redesign de l'onglet "Modèles par cluster"
    (formule propre par cluster pour OLS/Ridge/RF, prix reel vs estime par
    semaine, produits du cluster par semaine, navigation par boutons/
    accordeon pour optimiser l'espace, cf. demande utilisateur).
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

import src.dashboard.app  # noqa: F401 -- instancie l'app Dash avant d'importer une page
from src.dashboard.pages.models import (
    _cluster_overview_rows,
    _format_linear_equation,
    _format_ols_table,
    _ols_equation_text,
    _rf_importance_text,
    _ridge_equation_text,
    _segment_status_info,
)


class TestFormatLinearEquation:
    def test_avec_intercept_signes_corrects(self):
        eq = _format_linear_equation("log(prix)", 5.0, [("ram_go", 0.02), ("has_wifi", -0.3)])
        assert eq == "log(prix) = 5.000 + 0.020·ram_go − 0.300·has_wifi"

    def test_sans_intercept_premier_terme_garde_son_signe(self):
        """Ridge n'a pas d'ordonnee a l'origine -- le premier terme ne doit
        jamais recevoir un '+' de tete artificiel, et un premier terme
        negatif doit rester explicitement negatif."""
        eq_pos = _format_linear_equation("log(prix)", None, [("ram_go", 0.02)])
        assert eq_pos == "log(prix) = 0.020·ram_go"
        eq_neg = _format_linear_equation("log(prix)", None, [("has_wifi", -0.3)])
        assert eq_neg == "log(prix) = − 0.300·has_wifi"

    def test_aucun_terme_ni_intercept(self):
        assert _format_linear_equation("log(prix)", None, []) == "log(prix) = —"


class TestOlsEquationText:
    def test_intercept_extrait_de_la_ligne_const(self):
        coefs = pd.DataFrame({
            "feature": ["const", "ram_go", "stockage_go"],
            "coefficient": [5.6, 0.02, 0.0003],
            "std_err": [0.08, 0.003, 0.0001],
            "p_value": [0.0, 0.0, 0.08],
            "pct_effect": [np.nan, 2.0, 0.03],
        })
        equation, n_hidden = _ols_equation_text(coefs, max_terms=8)
        assert equation.startswith("log(prix) = 5.600")
        assert "const" not in equation
        assert n_hidden == 0

    def test_troncature_signale_par_n_hidden(self):
        coefs = pd.DataFrame({
            "feature": ["const"] + [f"var_{i}" for i in range(10)],
            "coefficient": [1.0] + [float(i + 1) for i in range(10)],
            "std_err": [0.1] * 11,
            "p_value": [0.01] * 11,
            "pct_effect": [np.nan] + [float(i) for i in range(10)],
        })
        equation, n_hidden = _ols_equation_text(coefs, max_terms=8)
        assert n_hidden == 2
        # les termes montres doivent etre les 8 plus grands en |coefficient|
        assert "var_9" in equation and "var_0" not in equation


class TestRidgeEquationText:
    def test_pas_de_const_dans_les_donnees_pas_dintercept_affiche(self):
        coefs = pd.DataFrame({"feature": ["ram_go", "has_wifi"], "coefficient": [0.02, -0.3]})
        equation, n_hidden = _ridge_equation_text(coefs, max_terms=8)
        assert n_hidden == 0
        assert equation.startswith("log(prix)")
        assert "5." not in equation.split("=")[1][:6]  # pas d'intercept invente


class TestRfImportanceText:
    def test_pas_de_formule_fermee_top_n(self):
        imp = pd.DataFrame({
            "feature": ["stockage_go", "ram_go", "cpu_serie_5.0"],
            "importance": [0.5, 0.3, 0.2],
        })
        text = _rf_importance_text(imp, max_terms=2)
        assert "Pas de formule fermée" in text
        assert "stockage_go" in text and "ram_go" in text
        assert "cpu_serie_5.0" not in text  # troncature a max_terms=2


class TestFormatOlsTable:
    def test_p_value_et_pct_effect_formates_lisiblement(self):
        coefs = pd.DataFrame({
            "feature": ["const", "has_wifi"],
            "coefficient": [5.639636, 1.881389],
            "std_err": [0.0876, 0.1337],
            "p_value": [0.0, 6.068826e-45],
            "pct_effect": [np.nan, 556.261673],
        })
        rows = _format_ols_table(coefs)
        assert rows[0]["p_value"] == "<0.001"
        assert rows[0]["pct_effect"] == "—"
        assert rows[1]["p_value"] == "<0.001"
        assert rows[1]["pct_effect"] == "556.26"


class TestSegmentStatusInfo:
    _SUMMARY = pd.DataFrame({
        "segment": ["0", "0", "0", "1", "1", "1"],
        "famille": ["hedonic_ols", "ridge", "random_forest"] * 2,
        "ajuste": [True, True, True, False, False, False],
        "retenu_pour_prediction": [False, True, True, False, False, False],
        "raison_rejet": [None, None, None, "n_lignes=30 < n_min_requis=60 (ratio=10)"] * 1 + [None, None],
        "r2_test": [-60.9, 0.586, -0.126, np.nan, np.nan, np.nan],
        "r2_test_categorie": [-1.4, -0.66, -0.16, np.nan, np.nan, np.nan],
    })

    def test_retenu_quand_bat_la_categorie(self):
        info = _segment_status_info(self._SUMMARY, "0", "ridge")
        assert info["statut"] == "Retenu"
        assert info["r2_test"] == pytest.approx(0.586)
        assert info["r2_test_categorie"] == pytest.approx(-0.66)

    def test_ajuste_mais_pas_meilleur(self):
        info = _segment_status_info(self._SUMMARY, "0", "hedonic_ols")
        assert info["statut"] == "Ajusté mais pas meilleur"

    def test_ecarte_sous_le_seuil_effectif(self):
        info = _segment_status_info(self._SUMMARY, "1", "hedonic_ols")
        assert info["statut"] == "Écarté"
        assert info["raison"] is not None

    def test_segment_absent_retourne_n_d(self):
        info = _segment_status_info(self._SUMMARY, "segment_inexistant", "ridge")
        assert info["statut"] == "n/d"
        assert info["r2_test"] is None


class TestClusterOverviewRows:
    def test_une_ligne_par_segment_triee_par_n_decroissant(self):
        summary = pd.DataFrame({
            "segment": ["0"] * 3 + ["1"] * 3,
            "famille": ["hedonic_ols", "ridge", "random_forest"] * 2,
            "n_lignes": [50] * 3 + [200] * 3,
            "n_produits_distincts": [12] * 3 + [40] * 3,
            "ajuste": [True] * 6,
            "retenu_pour_prediction": [True, False, False, True, True, False],
            "raison_rejet": [None] * 6,
            "r2_test": [0.1] * 6,
            "r2_test_categorie": [0.05] * 6,
        })
        rows = _cluster_overview_rows("pc_bureau", "clusters_n2", summary)
        assert len(rows) == 2
        assert rows[0]["n_lignes"] == 200  # segment "1" en premier (plus gros effectif)
        assert rows[0]["segment"] == "1"
        assert rows[0]["statut_hedonic_ols"] == "Retenu"
        assert rows[1]["statut_ridge"] == "Ajusté mais pas meilleur"

    def test_summary_vide_retourne_liste_vide(self):
        assert _cluster_overview_rows("pc_bureau", "clusters_n1", pd.DataFrame()) == []
