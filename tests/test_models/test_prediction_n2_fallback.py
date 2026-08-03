# -*- coding: utf-8 -*-
"""
=============================================================================
Module : tests/test_models/test_prediction_n2_fallback.py
=============================================================================
ROLE :
    Tests unitaires du fallback hierarchique N2 pour la prediction de prix
    (decision utilisateur du 2026-08-03) :
      1. Modele dedie du cluster N2 exact
      2. Modele dedie marque x gamme (n'importe quel cluster retenu
         "MARQUE::GAMME::*")
      3. Modele dedie marque (n'importe quel cluster retenu "MARQUE::*")
      4. Aucun modele N2 disponible -> repli categorie

    Utilise UNIQUEMENT des stubs/mocks -- pas de modele reel, pas de disque,
    pas de donnees reelles. Teste _resolve_n2_model_with_fallback en
    isolation (resolution pure du niveau hierarchique) et
    predict_price_cluster_aware via un patch minimal de ses dependances
    externes (load_metrics, load_model_artifact, load_retained_cluster_models,
    assign_n1_cluster, assign_n2_segment, _week_adjustment_pct).
=============================================================================
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.dashboard.prediction_utils import (
    _resolve_n2_model_with_fallback,
    predict_price_cluster_aware,
)


# ─────────────────────────────────────────────────────────────────────────────
# STUBS
# ─────────────────────────────────────────────────────────────────────────────

def _stub_model_info(segment_name: str) -> dict:
    """Info dict factice pour un segment (meme schema que
    load_retained_cluster_models retourne)."""
    return {
        "model": _StubModel(segment_name),
        "continuous_features": ["ram_go", "stockage_go"],
        "categorical_features": [],
        "design_columns": ["ram_go", "stockage_go"],
    }


class _StubModel:
    """Modele factice -- .predict() retourne une constante identifiable par
    le nom du segment (permet de verifier QUEL modele a ete appele)."""

    def __init__(self, segment_name: str):
        self.segment_name = segment_name
        # Valeur log(prix) = 7.0 + hash du segment (pour distinguer les modeles)
        self._log_price = 7.0 + (hash(segment_name) % 100) / 1000

    def predict(self, X):
        return pd.Series([self._log_price] * len(X))


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DE _resolve_n2_model_with_fallback (resolution pure, sans I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveN2ModelWithFallback:

    def test_niveau_1_cluster_exact(self):
        """Cas 1 : le cluster N2 exact a un modele retenu pour la famille
        demandee -- on le prend directement, aucun fallback."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        info, source = _resolve_n2_model_with_fallback(
            "SAMSUNG::Premium::c0", n2_models, "ridge",
        )
        assert info is not None
        assert source == "n2"
        assert info["model"].segment_name == "SAMSUNG::Premium::c0"

    def test_niveau_2_marque_gamme(self):
        """Cas 2 : le cluster exact (c1) n'a pas de modele, mais un AUTRE
        cluster du meme couple marque x gamme (c0) est retenu."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        info, source = _resolve_n2_model_with_fallback(
            "SAMSUNG::Premium::c1", n2_models, "ridge",
        )
        assert info is not None
        assert source == "n2_marque_gamme"
        assert info["model"].segment_name == "SAMSUNG::Premium::c0"

    def test_niveau_3_marque_seule(self):
        """Cas 3 : aucun cluster du couple marque x gamme n'est retenu, mais
        un cluster d'une AUTRE gamme de la meme marque l'est."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        info, source = _resolve_n2_model_with_fallback(
            "SAMSUNG::Économique::c0", n2_models, "ridge",
        )
        assert info is not None
        assert source == "n2_marque"
        assert info["model"].segment_name == "SAMSUNG::Premium::c0"

    def test_niveau_4_aucun_modele(self):
        """Cas 4 : aucun cluster de la marque n'a de modele retenu -- repli
        categorie (info=None, source=None)."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        info, source = _resolve_n2_model_with_fallback(
            "OPPO::Premium::c0", n2_models, "ridge",
        )
        assert info is None
        assert source is None

    def test_cluster_id_none(self):
        """cluster_id=None (pas de segment N2 assigne) -> repli direct."""
        n2_models = {"SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")}}
        info, source = _resolve_n2_model_with_fallback(None, n2_models, "ridge")
        assert info is None
        assert source is None

    def test_n2_models_vides(self):
        """Aucun modele N2 retenu du tout -> repli direct."""
        info, source = _resolve_n2_model_with_fallback(
            "SAMSUNG::Premium::c0", {}, "ridge",
        )
        assert info is None
        assert source is None

    def test_famille_absente_du_cluster_exact(self):
        """Le cluster exact a un modele retenu, mais PAS pour la famille
        demandee (ex: ridge retenu, hedonic_ols non). Fallback vers un autre
        cluster qui aurait la bonne famille."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
            "SAMSUNG::Premium::c1": {"hedonic_ols": _stub_model_info("SAMSUNG::Premium::c1")},
        }
        info, source = _resolve_n2_model_with_fallback(
            "SAMSUNG::Premium::c0", n2_models, "hedonic_ols",
        )
        assert info is not None
        assert source == "n2_marque_gamme"
        assert info["model"].segment_name == "SAMSUNG::Premium::c1"

    def test_format_cluster_id_inattendu(self):
        """Un cluster_id sans assez de '::' -> pas de niveau intermediaire
        deductible, repli propre (None, None)."""
        n2_models = {"SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")}}
        info, source = _resolve_n2_model_with_fallback(
            "format_bizarre", n2_models, "ridge",
        )
        assert info is None
        assert source is None

    def test_determinisme_marque_gamme(self):
        """Quand plusieurs clusters du meme marque x gamme sont retenus, le
        choix est DETERMINISTE (tri lexicographique de sorted()) -- jamais
        un resultat aleatoire d'un run a l'autre."""
        n2_models = {
            "SAMSUNG::Premium::c2": {"ridge": _stub_model_info("SAMSUNG::Premium::c2")},
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        info_1, source_1 = _resolve_n2_model_with_fallback(
            "SAMSUNG::Premium::c3", n2_models, "ridge",
        )
        info_2, source_2 = _resolve_n2_model_with_fallback(
            "SAMSUNG::Premium::c3", n2_models, "ridge",
        )
        assert info_1["model"].segment_name == info_2["model"].segment_name
        # Le premier par tri lexicographique est c0
        assert info_1["model"].segment_name == "SAMSUNG::Premium::c0"
        assert source_1 == "n2_marque_gamme"


# ─────────────────────────────────────────────────────────────────────────────
# TESTS INTEGRES DE predict_price_cluster_aware (avec mocks des I/O)
# ─────────────────────────────────────────────────────────────────────────────

_MODULE = "src.dashboard.prediction_utils"

# Metriques factices (meme schema que metrics.json reel)
_FAKE_METRICS = {
    "design_matrix_columns": ["ram_go", "stockage_go"],
    "continuous_features": ["ram_go", "stockage_go"],
    "categorical_features": [],
}

_FAKE_VALUES = {"ram_go": 8.0, "stockage_go": 256.0, "marque": "SAMSUNG"}


def _make_stub_model(log_price: float = 7.0):
    """Cree un stub model compatible predict(), avec une valeur identifiable."""
    m = MagicMock()
    m.predict.return_value = pd.Series([log_price])
    return m


def _patch_predict_deps(n2_models_dict: dict):
    """Context manager qui patche TOUTES les dependances externes de
    predict_price_cluster_aware pour un test isole."""
    category_model = _make_stub_model(log_price=7.0)

    return (
        patch(f"{_MODULE}.load_metrics", return_value=_FAKE_METRICS),
        patch(f"{_MODULE}.load_model_artifact", return_value=category_model),
        patch(f"{_MODULE}.load_retained_cluster_models",
              side_effect=lambda cat, subdir: n2_models_dict if subdir == "clusters_n2" else {}),
        patch(f"{_MODULE}.assign_n1_cluster", return_value=0),
        patch(f"{_MODULE}.assign_n2_segment",
              return_value={"cluster_id": "SAMSUNG::Premium::c1", "gamme_estimee": "Premium"}),
        patch(f"{_MODULE}._week_adjustment_pct", return_value=(None, [])),
    )


class TestPredictPriceClusterAwareFallback:
    """Tests de bout en bout (mockés) vérifiant que price_source reflète
    correctement le niveau de fallback utilisé."""

    def test_n2_exact_retourne_price_source_n2(self):
        """Un modele N2 exact retenu -> price_source='n2'."""
        n2_models = {
            "SAMSUNG::Premium::c1": {"ridge": _stub_model_info("SAMSUNG::Premium::c1")},
        }
        patches = _patch_predict_deps(n2_models)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = predict_price_cluster_aware("smartphones", "ridge", _FAKE_VALUES, segmentation="n2")
        assert result["price_source"] == "n2"

    def test_marque_gamme_fallback_retourne_price_source_n2_marque_gamme(self):
        """Cluster exact absent, mais un autre cluster du meme marque x gamme
        retenu -> price_source='n2_marque_gamme'."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        patches = _patch_predict_deps(n2_models)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = predict_price_cluster_aware("smartphones", "ridge", _FAKE_VALUES, segmentation="n2")
        assert result["price_source"] == "n2_marque_gamme"

    def test_marque_fallback_retourne_price_source_n2_marque(self):
        """Aucun cluster du couple marque x gamme, mais un cluster d'une
        autre gamme de la meme marque retenu -> price_source='n2_marque'."""
        n2_models = {
            "SAMSUNG::Économique::c0": {"ridge": _stub_model_info("SAMSUNG::Économique::c0")},
        }
        patches = _patch_predict_deps(n2_models)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = predict_price_cluster_aware("smartphones", "ridge", _FAKE_VALUES, segmentation="n2")
        assert result["price_source"] == "n2_marque"

    def test_aucun_modele_n2_retourne_price_source_categorie(self):
        """Aucun modele N2 de la marque -> price_source='categorie'."""
        n2_models = {
            "XIAOMI::Premium::c0": {"ridge": _stub_model_info("XIAOMI::Premium::c0")},
        }
        patches = _patch_predict_deps(n2_models)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = predict_price_cluster_aware("smartphones", "ridge", _FAKE_VALUES, segmentation="n2")
        assert result["price_source"] == "categorie"

    def test_note_contient_indication_du_niveau(self):
        """La note retournee doit mentionner le mecanisme de fallback pour
        que l'utilisateur du dashboard sache d'ou vient le prix."""
        n2_models = {
            "SAMSUNG::Économique::c0": {"ridge": _stub_model_info("SAMSUNG::Économique::c0")},
        }
        patches = _patch_predict_deps(n2_models)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = predict_price_cluster_aware("smartphones", "ridge", _FAKE_VALUES, segmentation="n2")
        assert "même marque" in result["note"]

    def test_segmentation_n1_non_affectee(self):
        """Le fallback N2 ne doit JAMAIS intervenir quand segmentation='n1'."""
        n2_models = {
            "SAMSUNG::Premium::c0": {"ridge": _stub_model_info("SAMSUNG::Premium::c0")},
        }
        patches = _patch_predict_deps(n2_models)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            result = predict_price_cluster_aware("smartphones", "ridge", _FAKE_VALUES, segmentation="n1")
        # N1 pas retenu (mocked as empty) -> categorie
        assert result["price_source"] == "categorie"
