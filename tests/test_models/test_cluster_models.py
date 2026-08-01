# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_models/test_cluster_models.py
=============================================================================
ROLE :
    Tests de la modelisation PAR CLUSTER (N1/N2 separement, decision
    utilisateur du 2026-08-01) : src.models.save_artifacts.
    fit_models_per_segment/persist_segment_models/_sanitize_segment_name/
    _drop_constant_categorical, et le n_splits GroupKFold adaptatif de
    ridge_model.py/rf_model.py.

    Priorite au garde-fou en 2 temps (le point le plus critique de ce
    correctif, cf. sa docstring) : (1) effectif suffisant pour ESTIMER le
    modele (_min_rows_required, differencie par famille), (2) le modele
    ajuste doit REELLEMENT battre le modele categorie sur le meme test
    hors-echantillon (retenu_pour_prediction) -- mesure empirique sur ce
    projet (pc_bureau) qu'un modele "estimable" peut neanmoins avoir un R²
    hors-echantillon fortement negatif (surapprentissage sur petit
    echantillon), donc PIRE que le repli categorie.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.models.ridge_model import RidgeModel
from src.models.rf_model import RandomForestModel
from src.models.save_artifacts import (
    DEFAULT_MIN_ROWS_RATIOS,
    _drop_constant_categorical,
    _sanitize_segment_name,
    fit_models_per_segment,
    persist_segment_models,
)


# ─────────────────────────────────────────────────────────────────────────────
# DONNEES SYNTHETIQUES -- 3 segments choisis pour exercer les 3 issues
# possibles du garde-fou d'effectif : trop petit (skip), assez grand pour
# Ridge/RF mais pas OLS (ratio different par famille), assez grand pour
# les 3 familles.
# ─────────────────────────────────────────────────────────────────────────────

def _make_segment_rows(segment_value, n_products, marque="MARQUEA", seed=0):
    rng = np.random.default_rng(seed)
    ram_go = rng.choice([8.0, 16.0, 32.0], size=n_products)
    stockage_go = rng.choice([256.0, 512.0, 1024.0], size=n_products)
    log_prix = 6.5 + 0.02 * ram_go + 0.0006 * stockage_go + rng.normal(0, 0.02, size=n_products)
    rows = []
    for i in range(n_products):
        rows.append({
            "url": f"https://example.test/{segment_value}-{i}", "nom": f"produit {segment_value}-{i}",
            "marque": marque, "prix_tnd": float(np.exp(log_prix[i])),
            "ram_go": ram_go[i], "stockage_go": stockage_go[i],
            "cluster_id": segment_value,
        })
    return rows


@pytest.fixture
def segments_df() -> pd.DataFrame:
    """seg_tiny (n=6, < tous les seuils) ; seg_medium (n=24, ~19-20 lignes
    apres le split 80/20 ci-dessous -- franchit Ridge/RF ratio=5 mais pas
    OLS ratio=10 -- 3 predicteurs (ram_go/stockage_go + constante) =>
    min_requis OLS=30, Ridge/RF=15) ; seg_big (n=90, franchit les 3 seuils
    meme apres le split). Une ligne = un produit (url unique), pas de
    pooling multi-semaines ici -- non pertinent pour ce garde-fou."""
    rows = (
        _make_segment_rows("seg_tiny", 6, seed=1)
        + _make_segment_rows("seg_medium", 24, seed=2)
        + _make_segment_rows("seg_big", 90, seed=3)
    )
    return pd.DataFrame(rows)


@pytest.fixture
def segments_train_test(segments_df):
    """Split 80/20 deterministe (pas GroupShuffleSplit -- pas besoin de
    proteger une fuite inter-semaines ici, une seule ligne par url)."""
    df = segments_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    cut = int(len(df) * 0.8)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


class _StubModel:
    """Modele categorie factice -- .predict() delegue a une fonction
    fournie, pour tester retenu_pour_prediction SANS devoir ajuster un vrai
    modele categorie complet (hors-sujet pour ce test, deja couvert par
    test_save_artifacts.py/l'integration reelle)."""

    def __init__(self, predict_fn):
        self._predict_fn = predict_fn

    def predict(self, X):
        return self._predict_fn(X)


class TestSanitizeSegmentName:
    def test_deux_points_remplaces(self):
        assert ":" not in _sanitize_segment_name("MSI::Premium::c0")

    def test_espaces_remplaces(self):
        assert " " not in _sanitize_segment_name("ASUS::Milieu de gamme::c0")

    def test_entier_convertible_en_chaine(self):
        """cluster_direct (N1) est un entier -- doit rester utilisable comme
        nom de dossier sans lever d'exception."""
        assert _sanitize_segment_name(3) == "3"


class TestDropConstantCategorical:
    def test_colonne_constante_retiree(self):
        df = pd.DataFrame({"marque": ["MSI"] * 5, "os_platform": ["Windows", "Windows", "FreeDos", "Windows", "FreeDos"]})
        kept = _drop_constant_categorical(df, ["marque", "os_platform"])
        assert kept == ["os_platform"]

    def test_aucune_colonne_constante_rien_retire(self):
        df = pd.DataFrame({"marque": ["MSI", "ASUS", "MSI"], "os_platform": ["Windows", "FreeDos", "Windows"]})
        kept = _drop_constant_categorical(df, ["marque", "os_platform"])
        assert kept == ["marque", "os_platform"]


class TestFitModelsPerSegment:
    def test_segment_trop_petit_ecarte_pour_les_3_familles(self, segments_train_test):
        df_train, df_test = segments_train_test
        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        tiny_rows = summary[summary["segment"] == "seg_tiny"]
        assert not tiny_rows.empty
        assert not tiny_rows["ajuste"].any()
        assert "seg_tiny" not in fitted

    def test_seuil_differencie_par_famille(self, segments_train_test):
        """seg_medium (n=40) doit franchir le ratio Ridge/RF (5) mais pas
        necessairement celui d'OLS (10, plus strict) -- verifie que les
        ratios par defaut sont bien DIFFERENTS et appliques independamment,
        pas un seuil unique deguise en 3."""
        assert DEFAULT_MIN_ROWS_RATIOS["hedonic_ols"] > DEFAULT_MIN_ROWS_RATIOS["ridge"]
        assert DEFAULT_MIN_ROWS_RATIOS["hedonic_ols"] > DEFAULT_MIN_ROWS_RATIOS["random_forest"]

        df_train, df_test = segments_train_test
        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        medium = summary[summary["segment"] == "seg_medium"]
        ols_row = medium[medium["famille"] == "hedonic_ols"].iloc[0]
        ridge_row = medium[medium["famille"] == "ridge"].iloc[0]
        # Meme n_predicteurs (memes features), ratio OLS strictement plus
        # eleve => n_min_requis strictement plus eleve, TOUJOURS (jamais
        # tributaire du hasard exact du split 80/20 ci-dessus).
        assert ols_row["n_predicteurs"] == ridge_row["n_predicteurs"]
        assert ols_row["n_min_requis"] > ridge_row["n_min_requis"]
        # Et, avec l'effectif choisi pour ce segment, OLS est effectivement
        # ecarte alors que Ridge/RF sont ajustables (illustre la difference,
        # pas seulement les seuils en abstrait).
        assert not ols_row["ajuste"]
        assert bool(ridge_row["ajuste"])

    def test_segment_assez_grand_est_ajuste(self, segments_train_test):
        df_train, df_test = segments_train_test
        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        big_rows = summary[summary["segment"] == "seg_big"]
        assert big_rows["ajuste"].any()
        assert "seg_big" in fitted

    def test_valeurs_nan_du_segment_ignorees(self, segments_train_test):
        """Une ligne sans segment (produit non couvert, cf. n2_reference_week)
        ne doit jamais faire planter ni apparaitre comme un segment nomme
        'nan'."""
        df_train, df_test = segments_train_test
        df_train = df_train.copy()
        df_train.loc[df_train.index[0], "cluster_id"] = np.nan
        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        assert "nan" not in set(summary["segment"].astype(str))

    def test_marque_constante_retiree_du_segment_mais_pas_categorie(self, segments_train_test):
        """Cf. _drop_constant_categorical : `marque` est constante DANS
        chaque segment de ce jeu de test (tous MARQUEA) -- ne doit jamais
        apparaitre dans categorical_features du segment ajuste, meme si
        elle est passee en entree."""
        df_train, df_test = segments_train_test
        fitted, _summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], ["marque"],
        )
        assert "seg_big" in fitted
        assert "marque" not in fitted["seg_big"]["categorical_features"]

    def test_retenu_pour_prediction_true_quand_segment_bat_categorie(self, segments_train_test):
        """Modele categorie factice tres mauvais (constante loin du vrai
        prix) -- le modele de segment (ajuste sur des donnees log-lineaires
        propres) doit le battre et etre retenu."""
        df_train, df_test = segments_train_test
        bad_log_price = float(np.log(df_train["prix_tnd"]).mean()) + 5.0  # tres loin du vrai prix
        category_models = {
            "ridge": _StubModel(lambda X: np.full(len(X), bad_log_price)),
            "random_forest": _StubModel(lambda X: np.full(len(X), bad_log_price)),
        }

        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
            category_models=category_models, category_design_columns=["ram_go", "stockage_go"],
        )
        big_ridge = summary[(summary["segment"] == "seg_big") & (summary["famille"] == "ridge")]
        assert not big_ridge.empty
        if bool(big_ridge.iloc[0]["ajuste"]):
            assert bool(big_ridge.iloc[0]["comparaison_possible"])
            assert bool(big_ridge.iloc[0]["retenu_pour_prediction"])

    def test_retenu_pour_prediction_false_quand_categorie_meilleure(self, segments_train_test):
        """Modele categorie factice QUASI PARFAIT (reproduit directement la
        formule log-lineaire SANS le bruit qui a servi a generer les
        donnees, cf. _make_segment_rows) -- aucun Ridge ajuste sur un
        echantillon bruite ne peut le battre, jamais retenu. NB : predire
        la simple MOYENNE de y_test donnerait R²=0 par definition (pas 1),
        d'ou la reconstruction explicite de la formule ici plutot qu'une
        constante."""
        df_train, df_test = segments_train_test
        seg_big_test = df_test[df_test["cluster_id"] == "seg_big"]
        if seg_big_test.empty:
            pytest.skip("aucune ligne de test pour seg_big avec ce seed -- rien a comparer")

        def _true_formula(X):
            return 6.5 + 0.02 * X["ram_go"].to_numpy() + 0.0006 * X["stockage_go"].to_numpy()

        category_models = {"ridge": _StubModel(_true_formula), "random_forest": _StubModel(_true_formula)}

        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
            category_models=category_models, category_design_columns=["ram_go", "stockage_go"],
        )
        big_ridge = summary[(summary["segment"] == "seg_big") & (summary["famille"] == "ridge")]
        assert not big_ridge.empty
        if bool(big_ridge.iloc[0]["ajuste"]) and bool(big_ridge.iloc[0]["comparaison_possible"]):
            assert not bool(big_ridge.iloc[0]["retenu_pour_prediction"])

    def test_sans_test_comparaison_impossible_mais_retenu_par_defaut(self):
        """Un segment sans AUCUNE ligne de test (rien a comparer) ne doit
        jamais etre rejete faute de donnees -- comparaison_possible=False,
        retenu_pour_prediction=True par defaut (cf. docstring)."""
        rows = _make_segment_rows("seg_solo", 60, seed=7)
        df_train = pd.DataFrame(rows)
        df_test = df_train.iloc[0:0]  # aucune ligne de test

        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        ridge_row = summary[(summary["segment"] == "seg_solo") & (summary["famille"] == "ridge")]
        assert not ridge_row.empty
        if bool(ridge_row.iloc[0]["ajuste"]):
            assert not bool(ridge_row.iloc[0]["comparaison_possible"])
            assert bool(ridge_row.iloc[0]["retenu_pour_prediction"])


class TestPersistSegmentModels:
    def test_ecrit_les_artefacts_attendus(self, segments_train_test, tmp_path):
        df_train, df_test = segments_train_test
        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        persist_segment_models(fitted, summary, tmp_path, "clusters_n2")

        assert (tmp_path / "clusters_n2_summary.csv").exists()
        seg_dir = tmp_path / "clusters_n2" / "seg_big"
        assert seg_dir.exists()
        assert (seg_dir / "feature_schema.json").exists()
        # Au moins un fichier de modele parmi les 3 familles potentiellement ajustees.
        assert any((seg_dir / f).exists() for f in ("ols.joblib", "ridge.joblib", "rf.joblib"))

    def test_segment_ecarte_nobtient_aucun_dossier(self, segments_train_test, tmp_path):
        df_train, df_test = segments_train_test
        fitted, summary = fit_models_per_segment(
            df_train, df_test, "cluster_id", ["ram_go", "stockage_go"], [],
        )
        persist_segment_models(fitted, summary, tmp_path, "clusters_n2")
        assert not (tmp_path / "clusters_n2" / "seg_tiny").exists()


class TestGroupKFoldAdaptatif:
    """RidgeModel.fit/RandomForestModel.fit -- n_splits = min(5, n_groupes),
    ValueError explicite si < 2 groupes (correctif du 2026-08-01, cf. leurs
    docstrings). Avant ce correctif, n_splits=5 fixe plantait des qu'un
    segment avait moins de 5 produits distincts."""

    @staticmethod
    def _toy_xy(n):
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"ram_go": rng.choice([8.0, 16.0, 32.0], size=n)})
        y = pd.Series(6.5 + 0.02 * X["ram_go"] + rng.normal(0, 0.01, size=n), name="log_prix")
        return X, y

    def test_ridge_fonctionne_avec_moins_de_5_groupes(self):
        X, y = self._toy_xy(12)
        groups = pd.Series([f"p{i % 3}" for i in range(12)])  # seulement 3 groupes distincts
        model = RidgeModel(degree=1).fit(X, y, groups=groups)
        assert model.best_estimator_ is not None

    def test_ridge_leve_value_error_sous_2_groupes(self):
        X, y = self._toy_xy(8)
        groups = pd.Series(["p0"] * 8)  # un seul groupe distinct
        with pytest.raises(ValueError):
            RidgeModel(degree=1).fit(X, y, groups=groups)

    def test_rf_fonctionne_avec_moins_de_5_groupes(self):
        X, y = self._toy_xy(12)
        groups = pd.Series([f"p{i % 3}" for i in range(12)])
        model = RandomForestModel().fit(X, y, groups=groups)
        assert model.best_estimator_ is not None

    def test_rf_leve_value_error_sous_2_groupes(self):
        X, y = self._toy_xy(8)
        groups = pd.Series(["p0"] * 8)
        with pytest.raises(ValueError):
            RandomForestModel().fit(X, y, groups=groups)
