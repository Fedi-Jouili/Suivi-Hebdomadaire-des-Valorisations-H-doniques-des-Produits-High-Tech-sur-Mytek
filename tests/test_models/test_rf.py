# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_models/test_rf.py
=============================================================================
ROLE :
    Tests de RandomForestModel (src/models/rf_model.py).
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
from sklearn.exceptions import NotFittedError

from src.models.rf_model import RandomForestModel


# Grille minuscule -- accelere le test (le defaut du projet est plus large,
# adapte a des vraies donnees, pas necessaire ici pour verifier le cablage).
GRILLE_RAPIDE = {"n_estimators": [10], "max_depth": [5], "min_samples_split": [2]}


class TestRandomForestModelFitPredict:
    """fit() et predict() s'executent sans erreur sur de petites donnees."""

    def test_fit_sans_erreur(self, dummy_X, dummy_y):
        model = RandomForestModel(param_grid=GRILLE_RAPIDE)
        model.fit(dummy_X, dummy_y)
        assert model.best_estimator_ is not None

    def test_predict_forme_correcte(self, dummy_X, dummy_y):
        model = RandomForestModel(param_grid=GRILLE_RAPIDE).fit(dummy_X, dummy_y)
        predictions = model.predict(dummy_X)
        assert len(predictions) == len(dummy_X)

    def test_predict_avant_fit_leve_notfittederror(self, dummy_X):
        model = RandomForestModel()
        with pytest.raises(NotFittedError):
            model.predict(dummy_X)


class TestRandomForestModelGetImportances:
    """DataFrame ['feature', 'importance'], somme des importances ~= 1.0
    (propriete de RandomForestRegressor.feature_importances_, normalisees
    par construction)."""

    def test_colonnes_et_somme_egale_un(self, dummy_X, dummy_y):
        model = RandomForestModel(param_grid=GRILLE_RAPIDE).fit(dummy_X, dummy_y)
        importances = model.get_importances(feature_names=list(dummy_X.columns))

        assert list(importances.columns) == ["feature", "importance"]
        assert importances["importance"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_trie_par_importance_decroissante(self, dummy_X, dummy_y):
        model = RandomForestModel(param_grid=GRILLE_RAPIDE).fit(dummy_X, dummy_y)
        importances = model.get_importances(feature_names=list(dummy_X.columns))
        valeurs = importances["importance"].tolist()
        assert valeurs == sorted(valeurs, reverse=True)
