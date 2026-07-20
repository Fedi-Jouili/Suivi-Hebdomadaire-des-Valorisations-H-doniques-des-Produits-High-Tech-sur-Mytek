# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_models/test_ridge.py
=============================================================================
ROLE :
    Tests de RidgeModel (src/models/ridge_model.py).

    NOTE sur get_coefficients() : le DataFrame retourne contient UNIQUEMENT
    les caracteristiques etendues par PolynomialFeatures (via
    poly.get_feature_names_out()) -- PAS de ligne pour l'intercept
    (ridge.intercept_ est une valeur scalaire separee, jamais ajoutee comme
    ligne du DataFrame). Pour interaction_only=True, degree=2 et N
    variables d'origine, le nombre de lignes attendu est
    N + C(N, 2) = N + N*(N-1)/2 (N termes simples + toutes les paires).
=============================================================================
"""

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.ridge_model import RidgeModel


class TestRidgeModelFit:
    """fit() s'execute sans erreur et peuple best_estimator_."""

    def test_fit_sans_erreur(self, dummy_X, dummy_y):
        model = RidgeModel(degree=2, alphas=[0.1, 1.0, 10.0])
        model.fit(dummy_X, dummy_y)
        assert model.best_estimator_ is not None

    def test_get_best_alpha_parmi_la_grille(self, dummy_X, dummy_y):
        alphas = [0.1, 1.0, 10.0]
        model = RidgeModel(degree=2, alphas=alphas).fit(dummy_X, dummy_y)
        assert model.get_best_alpha() in alphas


class TestRidgeModelPredict:
    """predict() retourne un tableau de la meme longueur que X."""

    def test_forme_des_predictions(self, dummy_X, dummy_y):
        model = RidgeModel().fit(dummy_X, dummy_y)
        predictions = model.predict(dummy_X)
        assert len(predictions) == len(dummy_X)

    def test_predict_avant_fit_leve_notfittederror(self, dummy_X):
        model = RidgeModel()
        with pytest.raises(NotFittedError):
            model.predict(dummy_X)


class TestRidgeModelGetCoefficients:
    """DataFrame ['feature', 'coefficient'], longueur = N + C(N,2) pour
    interaction_only=True (pas de ligne intercept -- cf. docstring)."""

    def test_colonnes_attendues(self, dummy_X, dummy_y):
        model = RidgeModel(degree=2).fit(dummy_X, dummy_y)
        coefs = model.get_coefficients(feature_names=list(dummy_X.columns))
        assert list(coefs.columns) == ["feature", "coefficient"]

    def test_longueur_n_plus_combinaisons_paires(self, dummy_X, dummy_y):
        model = RidgeModel(degree=2).fit(dummy_X, dummy_y)
        coefs = model.get_coefficients(feature_names=list(dummy_X.columns))

        n = len(dummy_X.columns)
        longueur_attendue = n + comb(n, 2)  # termes simples + interactions de degre 2
        assert len(coefs) == longueur_attendue

    def test_coefficients_echelle_originale_plausibles(self, dummy_X, dummy_y):
        """Les coefficients re-mis a l'echelle d'origine doivent rester
        dans un ordre de grandeur raisonnable (pas les valeurs brutes
        standardisees, potentiellement >> 1)."""
        model = RidgeModel(degree=2).fit(dummy_X, dummy_y)
        coefs = model.get_coefficients(feature_names=list(dummy_X.columns))
        assert coefs["coefficient"].abs().max() < 10  # generes par une relation log-lineaire modeste
