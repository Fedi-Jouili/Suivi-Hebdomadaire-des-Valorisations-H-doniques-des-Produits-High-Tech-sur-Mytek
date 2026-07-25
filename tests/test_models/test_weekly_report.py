# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_models/test_weekly_report.py
=============================================================================
ROLE :
    Tests de src/models/weekly_report.py. Meme convention que
    tests/test_dashboard/test_data_loader.py : les fonctions qui
    consomment les artefacts persistes (models/<categorie>/) sont
    exercees sur les VRAIS artefacts sur disque, jamais mockees -- ce
    module ne fait que relire/agreger ce que save_artifacts.py a deja
    produit (joblib.load/.predict(), pas de fit()). Les tests concernes
    sont SKIPPED (jamais un echec trompeur) si aucun artefact n'est
    trouve pour aucune categorie, cf. requires_artifacts ci-dessous.

    Les fonctions purement numeriques (_geo_mean, _pct_change,
    _direction) sont testees independamment, sans dependance a
    models/ -- ce sont elles qui portent la logique la plus facilement
    cassable silencieusement (division par zero, signe, seuil).
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.dashboard.data_loader import artifacts_available
from src.models.hedonic_model import POOLED_TIME_EXCLUDED_CATEGORIES
from src.models.weekly_report import (
    MATERIALITY_THRESHOLD_PCT,
    MIN_RELIABLE_N,
    QUADRANT_LABELS,
    _bootstrap_ecart_residuel,
    _direction,
    _geo_mean,
    _marque_gamme_product_level,
    _pct_change,
    catalog_composition_by_week,
    cluster_geometric_means,
    cluster_transitions,
    coverage_by_week,
    hedonic_price_index,
    marque_gamme_model_estimates,
    weekly_model_estimates,
)
from src.utils.config import CATEGORY_ORDER

_ANY_ARTIFACTS = any(artifacts_available(c) for c in CATEGORY_ORDER)
requires_artifacts = pytest.mark.skipif(
    not _ANY_ARTIFACTS,
    reason="Aucun artefact sous models/ -- executer `python -m src.models.save_artifacts` d'abord.",
)


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions numeriques pures -- aucune dependance a models/
# ─────────────────────────────────────────────────────────────────────────────

class TestGeoMean:
    def test_geo_mean_deux_valeurs(self):
        # sqrt(4*9) = 6
        assert _geo_mean([4.0, 9.0]) == pytest.approx(6.0)

    def test_geo_mean_valeur_unique(self):
        assert _geo_mean([42.0]) == pytest.approx(42.0)

    def test_geo_mean_ecarte_valeurs_non_positives(self):
        # 0 et negatif ecartes -- la moyenne geometrique de log(x<=0) n'existe pas
        assert _geo_mean([0.0, -5.0, 10.0, 10.0]) == pytest.approx(10.0)

    def test_geo_mean_aucune_valeur_positive_retourne_nan(self):
        assert np.isnan(_geo_mean([0.0, -1.0]))

    def test_geo_mean_coherente_avec_moyenne_arithmetique_sur_valeurs_egales(self):
        assert _geo_mean([7.0, 7.0, 7.0]) == pytest.approx(7.0)

    def test_geo_mean_toujours_inferieure_ou_egale_a_arithmetique(self):
        """Inegalite AM-GM (inégalité arithmético-géométrique) : jamais violée,
        quelle que soit la distribution -- garde-fou contre une erreur de
        formule (ex. exp(sum(log)) au lieu de exp(mean(log)))."""
        valeurs = [100.0, 200.0, 5000.0, 150.0]
        assert _geo_mean(valeurs) <= np.mean(valeurs) + 1e-9


class TestPctChangeDirection:
    def test_pct_change_hausse(self):
        assert _pct_change(100.0, 110.0) == pytest.approx(10.0)

    def test_pct_change_baisse(self):
        assert _pct_change(100.0, 90.0) == pytest.approx(-10.0)

    def test_pct_change_valeur_initiale_nulle_retourne_nan(self):
        assert np.isnan(_pct_change(0.0, 50.0))

    def test_pct_change_valeur_manquante_retourne_nan(self):
        assert np.isnan(_pct_change(None, 50.0))
        assert np.isnan(_pct_change(50.0, None))

    def test_direction_seuil_materialite(self):
        # Seuil par defaut MATERIALITY_THRESHOLD_PCT = 3.0
        assert _direction(3.1) == "hausse"
        assert _direction(-3.1) == "baisse"
        assert _direction(2.9) == "stable"
        assert _direction(-2.9) == "stable"
        assert _direction(0.0) == "stable"

    def test_direction_valeur_manquante(self):
        assert _direction(float("nan")) == "indéterminé"


class TestQuadrantLabels:
    def test_neuf_combinaisons_couvertes(self):
        """La grille 3x3 (direction prix reel x direction prix estime)
        doit couvrir EXACTEMENT les 9 combinaisons -- un cas non couvert
        retomberait sur le repli \"Cas non classé\" de cluster_transitions,
        jamais silencieux mais jamais souhaitable non plus."""
        directions = {"stable", "hausse", "baisse"}
        attendu = {(a, b) for a in directions for b in directions}
        assert set(QUADRANT_LABELS.keys()) == attendu

    def test_libelles_tous_non_vides_et_distincts(self):
        labels = list(QUADRANT_LABELS.values())
        assert all(isinstance(lbl, str) and lbl for lbl in labels)
        assert len(set(labels)) == len(labels)  # jamais deux cases avec le meme texte


class TestBootstrapEcartResiduel:
    """_bootstrap_ecart_residuel remplace le seuil de materialite FIXE par
    une inference statistique (upgrade demande par l'utilisateur,
    2026-07-25) -- ces tests verifient qu'elle se comporte comme un test
    d'hypothese correct sur des cas synthetiques ou la reponse attendue
    est connue avec certitude (contrairement aux artefacts reels, ou le
    "vrai" signal n'est jamais connu a l'avance)."""

    def test_aucun_ecart_reel_intervalle_contient_zero(self):
        """Prix reels et estimes tires INDEPENDAMMENT de la MEME
        distribution, des deux cotes (aucun ecart structurel entre reel et
        estime, ni entre les deux semaines) -- l'IC doit contenir 0,
        jamais signaler un ecart residuel significatif la ou il n'y en a
        structurellement aucun. Quatre tirages INDEPENDANTS (pas le meme
        array reutilise, cf. test_vecteur_parfaitement_constant... pour le
        cas degenere ou reel et estime coincident exactement) -- sinon
        delta_reel == delta_est sur chaque replique par construction, ce
        qui degenere artificiellement le p-value a 0."""
        rng_data = np.random.default_rng(7)
        prix_reel_t = 1000.0 + rng_data.normal(0, 20, size=30)
        prix_reel_t1 = 1000.0 + rng_data.normal(0, 20, size=30)
        prix_est_t = 1000.0 + rng_data.normal(0, 20, size=30)
        prix_est_t1 = 1000.0 + rng_data.normal(0, 20, size=30)
        rng = np.random.default_rng(0)
        ic_bas, ic_haut, p_value = _bootstrap_ecart_residuel(
            prix_reel_t, prix_reel_t1, prix_est_t, prix_est_t1, n_boot=2000, rng=rng,
        )
        assert ic_bas <= 0.0 <= ic_haut
        assert p_value > 0.05

    def test_vecteur_parfaitement_constant_intervalle_ponctuel_a_zero(self):
        """Cas degenere (zero variance des deux cotes, ex. UN SEUL prix
        distinct dans tout le cluster) : l'IC s'effondre sur un point
        (0.0, 0.0) -- ni ic_bas ni ic_haut ne depassent 0, donc
        `ecart_residuel_significatif` (bas > 0 OU haut < 0, cf.
        cluster_transitions) reste correctement False, meme si le p-value
        brut de la methode "proportion de repliques de part et d'autre de
        0" degenere a 0 dans ce cas limite precis (bruit d'echantillonnage
        nul, pas un vrai signal) -- ce test verrouille cette distinction."""
        rng = np.random.default_rng(0)
        prix = np.full(20, 1000.0)
        ic_bas, ic_haut, _ = _bootstrap_ecart_residuel(prix, prix, prix, prix, n_boot=500, rng=rng)
        assert ic_bas == pytest.approx(0.0)
        assert ic_haut == pytest.approx(0.0)
        significatif = ic_bas > 0 or ic_haut < 0  # meme formule que cluster_transitions
        assert significatif is False

    def test_ecart_large_et_consistant_detecte_comme_significatif(self):
        """Prix reel qui double, prix estime inchange, sur un effectif
        assez grand pour que le bruit d'echantillonnage ne puisse pas
        expliquer un ecart de cette ampleur -- l'IC doit EXCLURE 0."""
        rng = np.random.default_rng(1)
        prix_reel_t = np.full(30, 1000.0)
        prix_reel_t1 = np.full(30, 2000.0)  # +100%
        prix_est_t = np.full(30, 1000.0)
        prix_est_t1 = np.full(30, 1000.0)  # inchange
        ic_bas, ic_haut, p_value = _bootstrap_ecart_residuel(
            prix_reel_t, prix_reel_t1, prix_est_t, prix_est_t1, n_boot=2000, rng=rng,
        )
        assert ic_bas > 0  # l'IC entier est au-dessus de 0
        assert p_value < 0.01

    def test_n_boot_zero_retourne_none(self):
        rng = np.random.default_rng(0)
        result = _bootstrap_ecart_residuel(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]),
            n_boot=0, rng=rng,
        )
        assert result == (None, None, None)

    def test_reproductible_avec_meme_graine(self):
        prix = np.array([100.0, 150.0, 200.0])
        r1 = _bootstrap_ecart_residuel(prix, prix * 1.1, prix, prix, n_boot=500, rng=np.random.default_rng(42))
        r2 = _bootstrap_ecart_residuel(prix, prix * 1.1, prix, prix, n_boot=500, rng=np.random.default_rng(42))
        assert r1 == r2


# ─────────────────────────────────────────────────────────────────────────────
# Fonctions sur artefacts reels (models/<categorie>/) -- skip si absents
# ─────────────────────────────────────────────────────────────────────────────

class TestCoverageByWeek:
    @requires_artifacts
    @pytest.mark.parametrize("category", CATEGORY_ORDER)
    def test_colonnes_et_plages(self, category):
        if not artifacts_available(category):
            pytest.skip(f"pas d'artefacts pour {category}")
        df = coverage_by_week(category)
        for col in ("categorie", "semaine", "n_produits_poole", "n_retenus_marque_suffisante",
                    "n1_couverts", "n1_pct_couverture", "n2_couverts", "n2_pct_couverture"):
            assert col in df.columns
        assert list(df["semaine"]) == sorted(df["semaine"])
        assert (df["n1_pct_couverture"].between(0, 100)).all()
        assert (df["n2_pct_couverture"].between(0, 100)).all()
        assert (df["n_retenus_marque_suffisante"] <= df["n_produits_poole"]).all()

    @requires_artifacts
    def test_n1_structurellement_complet(self):
        """N1 (clustering technique) est construit sur la totalite du
        catalogue retenu, sans filtrage de marque -- sa couverture doit
        etre structurellement de 100% (cf. docstring du module),
        contrairement a N2 qui peut etre partielle."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = coverage_by_week(category)
            assert (df["n1_pct_couverture"] >= 99.9).all(), f"{category} : N1 incomplet"


class TestClusterGeometricMeans:
    @requires_artifacts
    def test_approche_seulement_n1_ou_n2(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_geometric_means(category)
            assert set(df["approche"].unique()) <= {"N1_technique", "N2_marque_gamme"}
            assert (df["prix_geometrique_tnd"] > 0).all()
            assert (df["n_produits"] > 0).all()
            return
        pytest.skip("aucun artefact disponible")


class TestWeeklyModelEstimates:
    @requires_artifacts
    def test_trois_modeles_prix_positifs(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = weekly_model_estimates(category)
            assert set(df["modele"].unique()) == {"hedonic_ols", "ridge", "random_forest"}
            assert (df["prix_reel_geometrique_tnd"] > 0).all()
            assert (df["prix_estime_geometrique_tnd"] > 0).all()
            return
        pytest.skip("aucun artefact disponible")


class TestMarqueGammeModelEstimates:
    @requires_artifacts
    def test_schema_exact_demande(self):
        """Verrouille le schema EXACT demande (decision utilisateur du
        2026-07-25) : categorie, marque, gamme, cluster, semaine,
        moyenne_geometrique, moyenne_estimee_ridge, moyenne_estimee_hedonic,
        moyenne_estimee_rf (+ n_produits, ajout documente)."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = marque_gamme_model_estimates(category)
            assert list(df.columns) == [
                "categorie", "marque", "gamme", "cluster", "semaine",
                "moyenne_geometrique", "moyenne_estimee_ridge", "moyenne_estimee_hedonic",
                "moyenne_estimee_rf", "n_produits",
            ]
            assert (df["categorie"] == category).all()
            assert (df["moyenne_geometrique"] > 0).all()
            return
        pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_restreint_aux_lignes_n2_couvertes(self):
        """Chaque ligne provient d'un produit avec cluster_id non NaN
        (cf. docstring) -- jamais une gamme/cluster \"None\" residuel."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = marque_gamme_model_estimates(category)
            assert df["gamme"].notna().all()
            assert df["cluster"].notna().all()
            return
        pytest.skip("aucun artefact disponible")


class TestClusterTransitions:
    @requires_artifacts
    def test_composition_stable_coherente_avec_effectifs(self):
        """composition_stable doit etre l'exacte egalite n_produits_t ==
        n_produits_t1 -- invariant garanti par construction, jamais
        approximatif."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category)
            if df.empty:
                continue
            assert (df["composition_stable"] == (df["n_produits_t"] == df["n_produits_t1"])).all()
            return
        pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_fiabilite_limitee_coherente_avec_seuil(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category)
            if df.empty:
                continue
            attendu = (df["n_produits_t"] < MIN_RELIABLE_N) | (df["n_produits_t1"] < MIN_RELIABLE_N)
            assert (df["fiabilite_limitee"] == attendu).all()
            return
        pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_classification_toujours_dans_la_grille_connue(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category)
            if df.empty:
                continue
            labels_connus = set(QUADRANT_LABELS.values())
            inconnus = set(df["classification"].unique()) - labels_connus
            assert not inconnus, f"{category} : classification(s) hors grille : {inconnus}"
            return
        pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_semaines_toujours_consecutives_et_croissantes(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category)
            if df.empty:
                continue
            assert (df["semaine_t1"] > df["semaine_t"]).all()
            return
        pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_bootstrap_desactivable_pour_tests_rapides(self):
        """n_boot=0 doit desactiver entierement le bootstrap (toutes les
        colonnes associees a None/False) -- utilise par les tests ci-dessus
        qui n'ont pas besoin de l'inference statistique, pour rester
        rapides."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category, n_boot=0)
            if df.empty:
                continue
            assert (~df["bootstrap_possible"]).all()
            assert df["ecart_residuel_ic_bas"].isna().all()
            assert not df["ecart_residuel_significatif"].any()
            return
        pytest.skip("aucun artefact disponible")

    @requires_artifacts
    def test_bootstrap_impossible_sous_effectif_minimal(self):
        """bootstrap_possible doit etre False des que n_produits_t ou
        n_produits_t1 < 2 (reechantillonnage d'une seule valeur degenere,
        cf. docstring de _bootstrap_ecart_residuel) -- jamais un IC
        trompeur de largeur nulle."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category, n_boot=200)
            sous_effectif = df[(df["n_produits_t"] < 2) | (df["n_produits_t1"] < 2)]
            if sous_effectif.empty:
                continue
            assert not sous_effectif["bootstrap_possible"].any()
            return
        pytest.skip("aucune transition sous-effectif trouvee dans les artefacts disponibles")

    @requires_artifacts
    def test_intervalle_de_confiance_coherent_et_contient_le_point_estime(self):
        """Verification de coherence interne : quand le bootstrap est
        possible, ic_bas <= ic_haut toujours, et le point estime
        (ecart_residuel_pct, calcule independamment sur les moyennes
        completes) doit tomber A L'INTERIEUR de l'IC bootstrap -- sinon la
        methode de reechantillonnage serait biaisee par rapport a
        l'estimateur ponctuel qu'elle est censee encadrer."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = cluster_transitions(category, n_boot=500)
            possible = df[df["bootstrap_possible"]]
            if possible.empty:
                continue
            assert (possible["ecart_residuel_ic_bas"] <= possible["ecart_residuel_ic_haut"]).all()
            # petite marge (0.5 pt) pour l'arrondi/le bruit bootstrap sur les cas limites
            marge = 0.5
            assert (
                (possible["ecart_residuel_pct"] >= possible["ecart_residuel_ic_bas"] - marge)
                & (possible["ecart_residuel_pct"] <= possible["ecart_residuel_ic_haut"] + marge)
            ).all()
            return
        pytest.skip("aucun artefact disponible")


class TestMarqueGammeProductLevel:
    @requires_artifacts
    def test_base_commune_agregation_coherente(self):
        """_marque_gamme_product_level (une ligne par produit) et
        marque_gamme_model_estimates (agregee) doivent rester coherentes :
        agreger product_level "a la main" doit redonner les memes moyennes
        geometriques que la fonction dediee -- verrouille le refactor du
        2026-07-25 qui a extrait cette base commune."""
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            product_level = _marque_gamme_product_level(category)
            estimates = marque_gamme_model_estimates(category, product_level=product_level)

            premiere_ligne = estimates.iloc[0]
            sous_ensemble = product_level[
                (product_level["marque"] == premiere_ligne["marque"])
                & (product_level["gamme"] == premiere_ligne["gamme"])
                & (product_level["cluster"] == premiere_ligne["cluster"])
                & (product_level["semaine"] == premiere_ligne["semaine"])
            ]
            assert len(sous_ensemble) == premiere_ligne["n_produits"]
            assert _geo_mean(sous_ensemble["prix_reel"]) == pytest.approx(premiere_ligne["moyenne_geometrique"], abs=0.01)
            return
        pytest.skip("aucun artefact disponible")


class TestHedonicPriceIndex:
    @requires_artifacts
    def test_categories_exclues_retournent_none(self):
        for category in POOLED_TIME_EXCLUDED_CATEGORIES:
            if not artifacts_available(category):
                continue
            assert hedonic_price_index(category) is None

    @requires_artifacts
    def test_semaine_reference_a_indice_zero(self):
        for category in CATEGORY_ORDER:
            if category in POOLED_TIME_EXCLUDED_CATEGORIES or not artifacts_available(category):
                continue
            df = hedonic_price_index(category)
            ref = df[df["semaine_reference"]]
            assert len(ref) == 1
            assert ref.iloc[0]["indice_prix_ajuste_qualite_pct"] == 0.0
            return
        pytest.skip("aucune categorie non exclue avec artefacts disponible")


class TestCatalogCompositionByWeek:
    @requires_artifacts
    def test_prix_geometrique_positif_et_n_coherent(self):
        for category in CATEGORY_ORDER:
            if not artifacts_available(category):
                continue
            df = catalog_composition_by_week(category)
            assert (df["prix_geometrique_tnd"] > 0).all()
            assert (df["n_produits"] > 0).all()
            return
        pytest.skip("aucun artefact disponible")
