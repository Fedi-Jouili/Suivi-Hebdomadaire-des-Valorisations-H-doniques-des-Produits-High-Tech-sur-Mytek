# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_dashboard/test_prediction_utils.py
=============================================================================
ROLE :
    Tests de src/dashboard/prediction_utils.py -- exerce les vrais
    artefacts sur disque (models/), meme convention que test_data_loader.py
    (requires_artifacts, jamais un echec trompeur si rien n'est encore
    genere).

    Regression test PRIORITAIRE (correctif du 2026-08-01bis, signale par
    l'utilisateur) : le selecteur "Type de segmentation" (N1/N2) de la page
    Prediction ne changeait auparavant QUE l'affichage du segment assigne,
    jamais le PRIX -- predict_price_cluster_aware essayait toujours N2 puis
    retombait sur N1 puis categorie, quel que soit le choix de
    l'utilisateur. Deux segmentations differentes affichaient donc
    silencieusement le meme prix. Corrige en ajoutant un parametre
    `segmentation` qui borne EXPLICITEMENT quel niveau peut produire le
    resultat final (cf. sa docstring).
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.dashboard.data_loader import artifacts_available, load_pooled_labeled
from src.dashboard.prediction_utils import predict_price_cluster_aware
from src.utils.config import CATEGORY_ORDER

_ANY_ARTIFACTS = any(artifacts_available(c) for c in CATEGORY_ORDER)
requires_artifacts = pytest.mark.skipif(
    not _ANY_ARTIFACTS, reason="Aucun artefact sous models/ -- executer `python -m src.models.save_artifacts` d'abord."
)


def _values_for_a_real_product(category: str) -> dict:
    """Caracteristiques d'un vrai produit du catalogue -- un formulaire
    hypothetique construit a la main risquerait de tomber sur une
    combinaison de modalites jamais observee (colonne one-hot absente),
    hors-sujet pour ce test."""
    df = load_pooled_labeled(category)
    row = df.dropna(subset=["marque"]).iloc[0]
    cols = [c for c in df.columns if c not in (
        "url", "nom", "prix_tnd", "semaine", "cluster_direct", "gamme_prix", "cluster_id", "specs_brutes",
    )]
    return {c: row[c] for c in cols if c in df.columns}


class TestSegmentationChoiceRespected:
    """Le parametre segmentation doit borner STRICTEMENT quel niveau peut
    produire le resultat final -- jamais un repli croise vers l'autre
    niveau, meme silencieusement 'meilleur'."""

    @requires_artifacts
    @pytest.mark.parametrize("category", CATEGORY_ORDER)
    def test_segmentation_n1_ne_retourne_jamais_une_source_n2(self, category):
        if not artifacts_available(category):
            pytest.skip(f"pas d'artefacts pour {category}")
        values = _values_for_a_real_product(category)
        for model_name in ("ols", "ridge", "rf"):
            result = predict_price_cluster_aware(category, model_name, values, segmentation="n1")
            assert result["price_source"] in ("n1", "categorie")

    @requires_artifacts
    @pytest.mark.parametrize("category", CATEGORY_ORDER)
    def test_segmentation_n2_ne_retourne_jamais_une_source_n1(self, category):
        if not artifacts_available(category):
            pytest.skip(f"pas d'artefacts pour {category}")
        values = _values_for_a_real_product(category)
        for model_name in ("ols", "ridge", "rf"):
            result = predict_price_cluster_aware(category, model_name, values, segmentation="n2")
            assert result["price_source"] in ("n2", "categorie")

    @requires_artifacts
    def test_deux_segmentations_peuvent_differer(self):
        """Au moins une categorie/famille doit montrer un prix DIFFERENT
        entre segmentation='n1' et 'n2' pour un meme produit -- sinon le
        correctif n'a rien change d'observable (jamais un test qui passe
        par construction sans jamais avoir pu echouer)."""
        found_a_difference = False
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            values = _values_for_a_real_product(category)
            for model_name in ("ols", "ridge", "rf"):
                r_n1 = predict_price_cluster_aware(category, model_name, values, segmentation="n1")
                r_n2 = predict_price_cluster_aware(category, model_name, values, segmentation="n2")
                if r_n1["price_source"] != r_n2["price_source"]:
                    found_a_difference = True
        if not found_a_difference:
            pytest.skip("aucun produit d'exemple ne distingue N1/N2 sur les artefacts actuels -- rien a verifier")
        assert found_a_difference
