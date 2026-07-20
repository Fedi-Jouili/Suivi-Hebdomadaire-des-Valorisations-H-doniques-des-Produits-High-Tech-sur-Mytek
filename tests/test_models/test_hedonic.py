# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_models/test_hedonic.py
=============================================================================
ROLE :
    Tests de src/models/hedonic_model.py. Couvre en priorite le garde-fou
    de circularite (le point le plus critique du module -- une regression
    qui "expliquerait" le prix par une variable derivee du prix serait
    silencieusement fausse sans lui), puis la coherence de bout en bout du
    pipeline gammes -> clusters -> regression sur un petit jeu de donnees
    synthetique mais realiste (memes noms de colonnes que
    data/processed/<categorie>_clean.csv).
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.models.hedonic_model import (
    CircularityError,
    HedonicOLS,
    _check_no_circularity,
    build_design_matrix,
    compute_cluster_labels,
    compute_price_tiers,
    fit_strategy_a,
    fit_strategy_b,
    compare_strategies,
)


@pytest.fixture
def hedonic_df() -> pd.DataFrame:
    """
    45 produits synthetiques sur 3 marques choisies pour exercer les 3
    paliers de compute_price_tiers : BRANDA (n=20 -> 3 gammes), BRANDB
    (n=12 -> 2 gammes), BRANDC (n=6 -> 1 gamme unique). Prix log-lineaire
    en ram_go/stockage_go + bruit, pour une regression non degenerée.
    """
    rng = np.random.default_rng(42)
    rows = []
    specs = [("BRANDA", 20), ("BRANDB", 12), ("BRANDC", 6)]
    for brand, n in specs:
        ram_go = rng.choice([8.0, 16.0, 32.0], size=n)
        stockage_go = rng.choice([256.0, 512.0, 1024.0], size=n)
        cpu_serie = rng.choice([3.0, 5.0, 7.0], size=n)
        os_platform = rng.choice(["Windows 11 Famille", "Windows 11 Pro"], size=n)
        log_prix = (
            6.5 + 0.02 * ram_go + 0.0006 * stockage_go + 0.05 * cpu_serie
            + rng.normal(0, 0.03, size=n)
        )
        for i in range(n):
            rows.append({
                "nom": f"{brand} produit {i}", "url": f"https://example.test/{brand}-{i}",
                "marque": brand, "prix_tnd": float(np.exp(log_prix[i])),
                "ram_go": ram_go[i], "stockage_go": stockage_go[i],
                "cpu_serie": cpu_serie[i], "os_platform": os_platform[i],
            })
    return pd.DataFrame(rows)


@pytest.fixture
def hedonic_df_tiered(hedonic_df):
    df, _ = compute_price_tiers(hedonic_df, min_brand_count=5)
    return df


@pytest.fixture
def hedonic_df_clustered(hedonic_df_tiered):
    df, _ = compute_cluster_labels(
        hedonic_df_tiered, category="test",
        continuous_features=["ram_go", "stockage_go"],
        categorical_features=["cpu_serie", "os_platform"],
    )
    return df


class TestCircularityGuard:
    """Le garde-fou est le point le plus critique du module : prix_tnd et
    gamme_prix ne doivent JAMAIS pouvoir atteindre la matrice de design."""

    def test_prix_tnd_leve_circularity_error(self):
        with pytest.raises(CircularityError):
            _check_no_circularity(["ram_go", "prix_tnd"])

    def test_gamme_prix_leve_circularity_error(self):
        with pytest.raises(CircularityError):
            _check_no_circularity(["ram_go", "gamme_prix"])

    def test_cluster_id_est_autorise(self):
        _check_no_circularity(["ram_go", "cluster_id"])  # ne doit pas lever

    def test_build_design_matrix_leve_si_gamme_prix_demandee(self, hedonic_df_clustered):
        with pytest.raises(CircularityError):
            build_design_matrix(hedonic_df_clustered, ["ram_go", "gamme_prix"], ["cluster_id"])

    def test_hedonic_ols_fit_leve_si_prix_tnd_dans_X(self, hedonic_df_clustered):
        X = hedonic_df_clustered[["ram_go", "prix_tnd"]].astype(float)
        y = np.log(hedonic_df_clustered["prix_tnd"])
        with pytest.raises(CircularityError):
            HedonicOLS().fit(X, y)


class TestComputePriceTiers:
    """n>=15 -> 3 gammes, 10<=n<15 -> 2, n<10 -> 1 (adaptatif, §2.3bis du
    notebook de segmentation -- meme logique reprise ici)."""

    def test_marque_negligeable_ecartee(self, hedonic_df):
        df_out, plan = compute_price_tiers(hedonic_df, min_brand_count=10)
        assert "BRANDC" not in df_out["marque"].unique()  # n=6 < 10
        assert "BRANDC" not in plan["marque"].values

    def test_n_gammes_adaptatif(self, hedonic_df):
        _, plan = compute_price_tiers(hedonic_df, min_brand_count=5)
        plan_by_brand = plan.set_index("marque")["n_gammes_planifie"]
        assert plan_by_brand["BRANDA"] == 3   # n=20
        assert plan_by_brand["BRANDB"] == 2   # n=12
        assert plan_by_brand["BRANDC"] == 1   # n=6

    def test_gamme_prix_dans_le_vocabulaire_attendu(self, hedonic_df_tiered):
        assert set(hedonic_df_tiered["gamme_prix"].unique()) <= {
            "Économique", "Milieu de gamme", "Premium", "Gamme unique"
        }


class TestComputeClusterLabels:
    """cluster_id est toujours defini et globalement unique (imbrique
    marque x gamme x sous-cluster) -- jamais de collision entre marques."""

    def test_requiert_gamme_prix(self, hedonic_df):
        with pytest.raises(ValueError):
            compute_cluster_labels(hedonic_df, category="test")

    def test_cluster_id_jamais_manquant(self, hedonic_df_clustered):
        assert hedonic_df_clustered["cluster_id"].isna().sum() == 0

    def test_cluster_id_encode_la_marque(self, hedonic_df_clustered):
        """Un cluster_id ne doit jamais etre partage entre deux marques
        differentes (sinon 2 clusters sans rapport seraient confondus
        dans la meme indicatrice)."""
        for cid, sub in hedonic_df_clustered.groupby("cluster_id"):
            assert sub["marque"].nunique() == 1

    def test_unit_summary_outcome_vocabulaire(self, hedonic_df_tiered):
        _, unit_summary = compute_cluster_labels(
            hedonic_df_tiered, category="test",
            continuous_features=["ram_go", "stockage_go"], categorical_features=["cpu_serie", "os_platform"],
        )
        assert set(unit_summary["outcome"].unique()) <= {"clustered", "no_structure", "too_small"}


class TestHedonicOLS:
    def test_fit_avant_predict_requis(self, hedonic_df_clustered):
        X, y = build_design_matrix(hedonic_df_clustered, ["ram_go", "stockage_go"], ["cluster_id"])
        with pytest.raises(NotFittedError):
            HedonicOLS().predict(X)

    def test_predict_forme_attendue(self, hedonic_df_clustered):
        X, y = build_design_matrix(hedonic_df_clustered, ["ram_go", "stockage_go"], ["cluster_id"])
        model = HedonicOLS().fit(X, y, continuous_cols=["ram_go", "stockage_go"])
        assert len(model.predict(X)) == len(X)

    def test_get_coefficients_colonnes(self, hedonic_df_clustered):
        X, y = build_design_matrix(hedonic_df_clustered, ["ram_go", "stockage_go"], ["cluster_id"])
        model = HedonicOLS().fit(X, y, continuous_cols=["ram_go", "stockage_go"])
        coefs = model.get_coefficients()
        assert list(coefs.columns) == ["feature", "coefficient", "std_err", "p_value", "pct_effect"]

    def test_pct_effect_formule_exacte_pour_categoriel(self, hedonic_df_clustered):
        """Pour une variable categorielle, pct_effect doit etre
        EXACTEMENT (exp(beta)-1)*100, jamais l'approximation beta*100."""
        X, y = build_design_matrix(hedonic_df_clustered, ["ram_go", "stockage_go"], ["cluster_id"])
        model = HedonicOLS().fit(X, y, continuous_cols=["ram_go", "stockage_go"])
        coefs = model.get_coefficients().set_index("feature")
        cluster_rows = [f for f in coefs.index if f.startswith("cluster_id_")]
        assert cluster_rows, "aucune colonne cluster_id dans la matrice de design"
        row = coefs.loc[cluster_rows[0]]
        expected = (np.exp(row["coefficient"]) - 1) * 100
        assert row["pct_effect"] == pytest.approx(expected)

    def test_pct_effect_approx_lineaire_pour_continu(self, hedonic_df_clustered):
        X, y = build_design_matrix(hedonic_df_clustered, ["ram_go", "stockage_go"], ["cluster_id"])
        model = HedonicOLS().fit(X, y, continuous_cols=["ram_go", "stockage_go"])
        coefs = model.get_coefficients().set_index("feature")
        assert coefs.loc["ram_go", "pct_effect"] == pytest.approx(coefs.loc["ram_go", "coefficient"] * 100)


class TestStrategiesAB:
    def test_strategie_a_ajuste_sans_erreur(self, hedonic_df_clustered):
        model, X, y = fit_strategy_a(hedonic_df_clustered, ["ram_go", "stockage_go"], ["cpu_serie", "os_platform"])
        assert model.nobs == len(hedonic_df_clustered)

    def test_strategie_b_ecarte_les_segments_trop_petits(self, hedonic_df_clustered):
        """BRANDC (n=6, la plus petite marque) doit toujours etre ecartee
        pour un nombre de predicteurs qui exige >= 10 lignes/predicteur --
        sur un jeu de donnees aussi petit, meme BRANDA (n=20) peut l'etre
        aussi une fois cluster_id/cpu_serie/os_platform pris en compte,
        ce qui est le comportement ATTENDU du garde-fou, pas une erreur."""
        results, skipped = fit_strategy_b(
            hedonic_df_clustered, ["ram_go", "stockage_go"], ["cpu_serie", "os_platform"],
            top_n_brands=3, min_rows_ratio=10,
        )
        assert not skipped.empty
        assert "BRANDC" in set(skipped["marque"])
        for _, row in skipped.iterrows():
            assert row["n"] < row["n_min_requis"]

    def test_compare_strategies_modeles_embottes(self, hedonic_df_clustered):
        """Le modele non-restreint (B) doit avoir STRICTEMENT plus de
        parametres que le restreint (A) -- condition necessaire pour
        qu'un test F embotte (Chow) ait un sens."""
        comparison, chow, restricted, unrestricted = compare_strategies(
            hedonic_df_clustered, ["ram_go", "stockage_go"], ["cpu_serie", "os_platform"], top_n_brands=2,
        )
        n_params_restricted = comparison.loc[comparison["modele"].str.startswith("A"), "n_predicteurs"].iloc[0]
        n_params_unrestricted = comparison.loc[comparison["modele"].str.startswith("B"), "n_predicteurs"].iloc[0]
        assert n_params_unrestricted > n_params_restricted
        assert np.isfinite(chow["F"])
        assert 0.0 <= chow["p_value"] <= 1.0
