# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_dashboard/test_data_loader.py
=============================================================================
ROLE :
    Tests de src/dashboard/data_loader.py -- module strictement lecture
    seule (pas de fit()). Exerce les vrais artefacts sur disque (data/
    processed/, models/), comme tests/test_integration.py le fait deja pour
    la chaine de modelisation -- pas de mock, ce module N'EST QUE de la
    lecture de fichiers deja produits par le pipeline/save_artifacts.

    Suppose que `python -m src.models.save_artifacts` a deja ete execute au
    moins une fois (cf. src/dashboard/README.md) ; les tests qui en
    dependent sont explicitement SKIPPED (jamais un echec trompeur) si
    aucun artefact n'est trouve pour aucune categorie.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.dashboard.data_loader import (
    ArtifactsMissingError,
    artifacts_available,
    available_weeks,
    get_feature_ranges,
    last_n_weeks,
    load_clean_category_recent,
    load_clean_category_week,
    load_cluster_models_summary,
    load_cluster_products,
    load_cluster_segment_detail,
    load_coefficients,
    load_metrics,
    load_model_artifact,
    load_pooled_category_full,
    load_pooled_labeled,
    raw_vs_clean_counts,
)
from src.utils.config import CATEGORY_ORDER

_ANY_ARTIFACTS = any(artifacts_available(c) for c in CATEGORY_ORDER)
_ANY_WEEKS = len(available_weeks()) > 0

requires_processed_data = pytest.mark.skipif(
    not _ANY_WEEKS, reason="Aucune semaine sous data/processed/ -- executer le pipeline de pretraitement d'abord."
)
requires_artifacts = pytest.mark.skipif(
    not _ANY_ARTIFACTS,
    reason="Aucun artefact sous models/ -- executer `python -m src.models.save_artifacts` d'abord.",
)


class TestWeeks:
    @requires_processed_data
    def test_available_weeks_non_vide_et_triee(self):
        weeks = available_weeks()
        assert len(weeks) > 0
        assert list(weeks) == sorted(weeks)

    @requires_processed_data
    def test_last_n_weeks_respecte_n(self):
        weeks = last_n_weeks(4)
        assert len(weeks) <= 4
        assert set(weeks) <= set(available_weeks())


class TestCleanCategoryData:
    @requires_processed_data
    @pytest.mark.parametrize("category", CATEGORY_ORDER)
    def test_load_clean_category_week_colonnes_attendues(self, category):
        """<categorie>_clean.csv doit toujours contenir url/nom/prix_tnd/
        marque -- schema minimal garanti par select_features.ALWAYS_KEEP,
        quelle que soit la categorie ou la semaine."""
        weeks = available_weeks()
        if not weeks:
            pytest.skip("aucune semaine disponible")
        df = load_clean_category_week(category, weeks[-1])
        if df.empty:
            pytest.skip(f"aucune donnee pour {category} semaine {weeks[-1]}")
        for col in ("url", "nom", "prix_tnd", "marque"):
            assert col in df.columns

    @requires_processed_data
    def test_load_clean_category_week_absente_retourne_df_vide(self):
        """Une semaine inexistante ne doit JAMAIS lever d'exception --
        DataFrame vide, cf. docstring du module."""
        df = load_clean_category_week("pc_bureau", 9999)
        assert df.empty

    @requires_processed_data
    def test_load_clean_category_recent_pas_de_nan_residuel(self):
        """La reconciliation de schema (_reconcile_pooled_schema) doit
        eliminer tout NaN introduit par une colonne absente de certaines
        semaines (ex. has_4g pour televiseurs S1-S3) -- jamais de NaN
        residuel apres pooling, meme garantie que <categorie>_clean.csv
        pris individuellement."""
        for category in CATEGORY_ORDER:
            df = load_clean_category_recent(category, 4)
            if df.empty:
                continue
            assert df.isna().sum().sum() == 0, f"NaN residuel apres pooling pour {category}"

    @requires_processed_data
    def test_raw_vs_clean_counts_coherent(self):
        weeks = available_weeks()
        if not weeks:
            pytest.skip("aucune semaine disponible")
        counts = raw_vs_clean_counts("pc_bureau", weeks[-1])
        assert counts["n_clean"] <= counts["n_raw"] or counts["n_raw"] == 0
        assert counts["n_excluded"] == max(counts["n_raw"] - counts["n_clean"], 0)

    @requires_processed_data
    def test_raw_vs_clean_counts_reference_week(self):
        counts = raw_vs_clean_counts("pc_bureau", 1)
        assert counts == {"n_raw": 221, "n_clean": 217, "n_excluded": 4}


class TestArtifacts:
    @requires_artifacts
    @pytest.mark.parametrize("category", CATEGORY_ORDER)
    def test_metrics_schema_par_categorie(self, category):
        if not artifacts_available(category):
            pytest.skip(f"pas d'artefacts pour {category}")
        metrics = load_metrics(category)
        for key in ("hedonic_ols", "ridge", "random_forest", "continuous_features",
                    "categorical_features", "design_matrix_columns"):
            assert key in metrics
        for model_key in ("hedonic_ols", "ridge", "random_forest"):
            for metric_key in ("r2_log", "rmse_log", "rmse_tnd", "retransformation_note"):
                assert metric_key in metrics[model_key]

    @requires_artifacts
    def test_marque_toujours_effet_fixe_jamais_circulaire(self):
        """Regression test direct du correctif de circularite (reports/
        audit_code.md §3.1) : aucun artefact sauvegarde ne doit contenir
        gamme_prix/cluster_id parmi ses regresseurs."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            metrics = load_metrics(category)
            forbidden = {"gamme_prix", "cluster_id", "prix_tnd", "log_prix_tnd"}
            assert not (forbidden & set(metrics["design_matrix_columns"]))
            assert "marque" in metrics["categorical_features"]

    @requires_artifacts
    def test_load_coefficients_colonnes(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            coefs = load_coefficients(category)
            assert {"feature", "coefficient", "std_err", "p_value", "pct_effect"} <= set(coefs.columns)
            break
        else:
            pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_load_model_artifact_leve_si_absent(self):
        with pytest.raises(ArtifactsMissingError):
            load_model_artifact("pc_bureau", "modele_qui_nexiste_pas")

    @requires_artifacts
    def test_pooled_labeled_a_gamme_et_clusters(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = load_pooled_labeled(category)
            for col in ("gamme_prix", "cluster_id", "cluster_direct", "marque", "prix_tnd"):
                assert col in df.columns
            break
        else:
            pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_get_feature_ranges_coherent(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            ranges = get_feature_ranges(category)
            assert "continuous" in ranges and "categorical" in ranges and "marque" in ranges
            for col, r in ranges["continuous"].items():
                assert r["min"] <= r["median"] <= r["max"]
            assert len(ranges["marque"]) > 0
            break
        else:
            pytest.skip("aucun artefact disponible")


class TestPooledCategoryFull:
    @requires_processed_data
    def test_load_pooled_category_full_pas_vide(self):
        for category in CATEGORY_ORDER:
            df = load_pooled_category_full(category)
            if not df.empty:
                assert "semaine" in df.columns
                return
        pytest.skip("aucune categorie avec donnees")


class TestClusterSegmentDetail:
    """load_cluster_segment_detail / load_cluster_products -- ajoutes le
    2026-08-01ter pour la page Modeles enrichie (formule par cluster +
    produits du cluster par semaine, cf. demande utilisateur)."""

    @staticmethod
    def _a_segment(subdir: str):
        """Un (categorie, segment) reel AVEC AU MOINS UNE famille ajustee
        (persist_segment_models n'ecrit un dossier/feature_schema.json que
        pour ces segments-la, cf. save_artifacts.py -- un segment ou les 3
        familles sont ecartees (effectif insuffisant partout) n'existe
        simplement pas sous disque, jamais une erreur mais pas un candidat
        valide pour ce test)."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            summary = load_cluster_models_summary(category, subdir)
            if summary.empty:
                continue
            ajustes = summary[summary["ajuste"] == True]  # noqa: E712
            if ajustes.empty:
                continue
            return category, str(ajustes["segment"].iloc[0])
        return None, None

    @requires_artifacts
    @pytest.mark.parametrize("subdir", ["clusters_n1", "clusters_n2"])
    def test_segment_detail_schema(self, subdir):
        category, segment = self._a_segment(subdir)
        if category is None:
            pytest.skip(f"aucun cluster {subdir} disponible")
        detail = load_cluster_segment_detail(category, subdir, segment)
        assert "schema" in detail and "coefficients" in detail
        for key in ("continuous_features", "categorical_features", "design_matrix_columns",
                    "n_train", "n_test", "familles_ajustees", "retenu_pour_prediction"):
            assert key in detail["schema"]
        for famille, coefs in detail["coefficients"].items():
            assert famille in ("hedonic_ols", "ridge", "random_forest")
            assert not coefs.empty

    @requires_artifacts
    def test_segment_detail_vide_si_segment_absent(self):
        category = next((c for c in CATEGORY_ORDER if artifacts_available(c)), None)
        if category is None:
            pytest.skip("aucun artefact disponible")
        assert load_cluster_segment_detail(category, "clusters_n1", "segment_qui_nexiste_pas") == {}

    @requires_artifacts
    @pytest.mark.parametrize("subdir", ["clusters_n1", "clusters_n2"])
    def test_cluster_products_non_vide_et_coherent(self, subdir):
        """Chaque produit retourne doit reellement appartenir au cluster
        demande -- jamais un melange avec d'autres clusters (cf. bug de
        pooling multi-semaines deja corrige pour N2, meme classe de risque)."""
        category, segment = self._a_segment(subdir)
        if category is None:
            pytest.skip(f"aucun cluster {subdir} disponible")
        products = load_cluster_products(category, subdir, segment)
        assert not products.empty
        key_col = "cluster_direct" if subdir == "clusters_n1" else "cluster_id"
        assert (products[key_col].astype(str) == segment).all()
        for col in ("nom", "prix_tnd", "marque", "semaine"):
            assert col in products.columns

    @requires_artifacts
    def test_cluster_products_vide_si_colonne_absente(self):
        """subdir invalide (ni clusters_n1 ni clusters_n2) -- jamais une
        KeyError, DataFrame vide comme toute autre absence de donnees."""
        category = next((c for c in CATEGORY_ORDER if artifacts_available(c)), None)
        if category is None:
            pytest.skip("aucun artefact disponible")
        assert load_cluster_products(category, "subdir_invalide", "x").empty
