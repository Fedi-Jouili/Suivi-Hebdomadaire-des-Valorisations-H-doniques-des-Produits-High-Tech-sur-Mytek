# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_preprocessing/test_transform.py
=============================================================================
ROLE :
    Tests de src/preprocessing/transform.py : apply_log_transform,
    create_polynomial_features, prepare_final_features.

    Pour prepare_final_features (qui charge un KMeans/StandardScaler via
    joblib), on AJUSTE et SERIALISE un vrai petit modele dans tmp_path
    (fixture pytest standard) plutot que de mocker joblib.load -- plus
    realiste qu'un faux objet qui pourrait masquer un vrai bug de
    .predict()/.transform().
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.preprocessing.transform import (
    apply_log_transform,
    add_cluster_labels_online,
    create_polynomial_features,
    prepare_final_features,
)


class TestApplyLogTransform:
    """log_prix = ln(prix_tnd), sans NaN quand tous les prix sont positifs."""

    def test_log_prix_correctement_calcule(self):
        df = pd.DataFrame({"prix_tnd": [100.0, 1000.0, 10000.0]})
        result = apply_log_transform(df)
        assert np.allclose(result["log_prix"], np.log(df["prix_tnd"]))

    def test_aucun_nan_si_prix_tous_positifs(self):
        df = pd.DataFrame({"prix_tnd": [100.0, 250.0, 999.0]})
        result = apply_log_transform(df)
        assert result["log_prix"].isna().sum() == 0


class TestCreatePolynomialFeatures:
    """interaction_only=True : uniquement des termes croises, jamais de
    terme au carre (ram_go^2)."""

    def test_termes_interaction_presents(self):
        X = pd.DataFrame({"ram_go": [8.0, 16.0, 32.0], "cpu_serie": [3.0, 5.0, 7.0]})
        X_poly, poly = create_polynomial_features(X, degree=2, interaction_only=True)
        assert "ram_go cpu_serie" in X_poly.columns

    def test_aucun_terme_au_carre(self):
        X = pd.DataFrame({"ram_go": [8.0, 16.0, 32.0], "cpu_serie": [3.0, 5.0, 7.0]})
        X_poly, poly = create_polynomial_features(X, degree=2, interaction_only=True)
        # sklearn nomme les puissances avec un accent circonflexe ("ram_go^2") ;
        # interaction_only=True doit garantir qu'aucune colonne n'en contient.
        assert not any("^" in col for col in X_poly.columns)

    def test_include_bias_false_pas_de_colonne_constante(self):
        X = pd.DataFrame({"ram_go": [8.0, 16.0, 32.0]})
        X_poly, poly = create_polynomial_features(X)
        assert "1" not in X_poly.columns


class TestAddClusterLabelsOnline:
    """Applique un KMeans/StandardScaler DEJA AJUSTES (jamais de re-fit)."""

    @pytest.fixture
    def modele_clustering_sauvegarde(self, tmp_path, dummy_X):
        """Ajuste un vrai petit KMeans sur dummy_X et le serialise dans
        tmp_path -- imite l'export attendu du notebook de clustering."""
        scaler = StandardScaler().fit(dummy_X)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaler.transform(dummy_X))

        model_path = tmp_path / "kmeans.pkl"
        scaler_path = tmp_path / "scaler.pkl"
        joblib.dump(kmeans, model_path)
        joblib.dump(scaler, scaler_path)
        return model_path, scaler_path

    def test_fichiers_absents_leve_filenotfounderror_explicite(self, dummy_X, tmp_path):
        with pytest.raises(FileNotFoundError, match="notebook"):
            add_cluster_labels_online(
                dummy_X,
                model_path=tmp_path / "absent_kmeans.pkl",
                scaler_path=tmp_path / "absent_scaler.pkl",
            )

    def test_colonne_cluster_ajoutee_en_categoriel(self, dummy_X, modele_clustering_sauvegarde):
        model_path, scaler_path = modele_clustering_sauvegarde
        result = add_cluster_labels_online(dummy_X, model_path=model_path, scaler_path=scaler_path)

        assert "cluster" in result.columns
        assert isinstance(result["cluster"].dtype, pd.CategoricalDtype)
        assert result["cluster"].notna().all()


class TestPrepareFinalFeatures:
    """Orchestrateur : log-prix + cluster + interactions, cluster/dummies
    exclus de l'expansion polynomiale."""

    @pytest.fixture
    def modele_clustering_sauvegarde(self, tmp_path, dummy_X):
        scaler = StandardScaler().fit(dummy_X)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaler.transform(dummy_X))
        model_path = tmp_path / "kmeans.pkl"
        scaler_path = tmp_path / "scaler.pkl"
        joblib.dump(kmeans, model_path)
        joblib.dump(scaler, scaler_path)
        return model_path, scaler_path

    def test_cluster_present_categoriel_et_non_polynomial(self, dummy_X, dummy_y, modele_clustering_sauvegarde):
        model_path, scaler_path = modele_clustering_sauvegarde
        df = dummy_X.copy()
        df["prix_tnd"] = np.exp(dummy_y)  # prepare_final_features refait le log lui-meme

        X, y, poly = prepare_final_features(
            df, cluster_model_path=model_path, cluster_scaler_path=scaler_path,
            cluster_features=list(dummy_X.columns), add_polynomials=True,
        )

        assert "cluster" in X.columns
        assert isinstance(X["cluster"].dtype, pd.CategoricalDtype)
        # Aucune colonne generee par PolynomialFeatures ne doit impliquer "cluster"
        assert not any("cluster" in col for col in X.columns if col != "cluster")
        assert "prix_tnd" not in X.columns
        assert len(y) == len(df)
