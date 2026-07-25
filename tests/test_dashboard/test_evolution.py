# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_dashboard/test_evolution.py
=============================================================================
ROLE :
    Tests des fonctions PURES de src/dashboard/pages/evolution.py (page
    "Évolution hebdomadaire") -- transformation/agregation de DataFrames
    deja charges, jamais d'acces disque ici (contrairement a
    test_weekly_report.py qui exerce les artefacts reels). Donnees
    synthetiques mais realistes (memes noms de colonnes que les CSV de
    reports/, cf. src/models/weekly_report.py).

    `import src.dashboard.app` AVANT d'importer le module de page est
    necessaire : dash.register_page() (appele a l'import de n'importe
    quelle page) leve PageError si aucune app Dash(use_pages=True) n'a
    encore ete instanciee -- importer app.py une fois suffit pour tout le
    fichier de test (le module reste en cache sys.modules).
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

import src.dashboard.app  # noqa: F401 -- instancie l'app Dash avant d'importer une page (cf. docstring)
from src.dashboard.pages.evolution import (
    _pivot_cluster_table,
    _severity,
    _severity_shares,
    _spec_change_sentence,
)
from src.models.weekly_report import QUADRANT_LABELS


class TestSeverity:
    def test_stable_stable_est_neutre(self):
        assert _severity(QUADRANT_LABELS[("stable", "stable")]) == "neutre"

    @pytest.mark.parametrize("direction", ["hausse", "baisse"])
    def test_memes_directions_est_explique(self, direction):
        assert _severity(QUADRANT_LABELS[(direction, direction)]) == "explique"

    @pytest.mark.parametrize("dirs", [("stable", "hausse"), ("stable", "baisse"),
                                       ("hausse", "stable"), ("baisse", "stable")])
    def test_une_seule_direction_bouge_est_ecart_partiel(self, dirs):
        assert _severity(QUADRANT_LABELS[dirs]) == "ecart_partiel"

    @pytest.mark.parametrize("dirs", [("hausse", "baisse"), ("baisse", "hausse")])
    def test_directions_opposees_est_ecart_maximal(self, dirs):
        assert _severity(QUADRANT_LABELS[dirs]) == "ecart_maximal"

    def test_classification_inconnue_repli_neutre(self):
        """Jamais de KeyError sur un libelle non reconnu -- repli sur
        \"neutre\", jamais un crash de page (cf. _LABEL_TO_DIRS.get)."""
        assert _severity("un libellé qui n'existe pas dans QUADRANT_LABELS") == "neutre"

    def test_toutes_les_9_cases_de_la_grille_couvertes_sans_kfaute(self):
        for label in QUADRANT_LABELS.values():
            assert _severity(label) in {"neutre", "explique", "ecart_partiel", "ecart_maximal"}


class TestSeverityShares:
    def test_repartition_somme_a_cent(self):
        transitions = pd.DataFrame({
            "classification": [
                QUADRANT_LABELS[("stable", "stable")],
                QUADRANT_LABELS[("stable", "stable")],
                QUADRANT_LABELS[("hausse", "hausse")],
                QUADRANT_LABELS[("hausse", "baisse")],
            ],
        })
        shares = _severity_shares(transitions)
        assert shares["neutre"] == pytest.approx(50.0)
        assert shares["explique"] == pytest.approx(25.0)
        assert shares["ecart_maximal"] == pytest.approx(25.0)
        assert shares["ecart_partiel"] == pytest.approx(0.0)
        assert sum(shares.values()) == pytest.approx(100.0)

    def test_dataframe_vide_ne_leve_pas(self):
        shares = _severity_shares(pd.DataFrame({"classification": []}))
        assert all(v == 0.0 for v in shares.values())


class TestPivotClusterTable:
    def test_n1_pivot_colonnes_et_variation(self):
        cluster_means = pd.DataFrame([
            {"approche": "N1_technique", "cluster": "c0", "semaine": 1, "n_produits": 5, "prix_geometrique_tnd": 1000.0},
            {"approche": "N1_technique", "cluster": "c0", "semaine": 2, "n_produits": 6, "prix_geometrique_tnd": 1100.0},
            {"approche": "N1_technique", "cluster": "c1", "semaine": 1, "n_produits": 3, "prix_geometrique_tnd": 500.0},
            {"approche": "N1_technique", "cluster": "c1", "semaine": 2, "n_produits": 3, "prix_geometrique_tnd": 500.0},
            # bruit d'une autre approche -- ne doit jamais se mélanger au pivot N1
            {"approche": "N2_marque_gamme", "cluster": "c9", "semaine": 1, "n_produits": 9, "prix_geometrique_tnd": 42.0},
        ])
        pivot, weeks = _pivot_cluster_table(cluster_means, "N1_technique")

        assert weeks == [1, 2]
        assert set(pivot["cluster"]) == {"c0", "c1"}
        assert {"S1", "S2", "n_produits", "variation_pct"} <= set(pivot.columns)

        row_c0 = pivot[pivot["cluster"] == "c0"].iloc[0]
        assert row_c0["S1"] == pytest.approx(1000.0)
        assert row_c0["S2"] == pytest.approx(1100.0)
        assert row_c0["variation_pct"] == pytest.approx(10.0)  # (1100-1000)/1000*100
        assert row_c0["n_produits"] == 6  # effectif de la DERNIERE semaine, pas la premiere

        row_c1 = pivot[pivot["cluster"] == "c1"].iloc[0]
        assert row_c1["variation_pct"] == pytest.approx(0.0)  # prix inchangé

    def test_n2_pivot_inclut_marque_et_gamme(self):
        cluster_means = pd.DataFrame([
            {"approche": "N2_marque_gamme", "cluster": "ASUS::Premium::c0", "marque": "ASUS", "gamme_prix": "Premium",
             "semaine": 1, "n_produits": 2, "prix_geometrique_tnd": 2000.0},
            {"approche": "N2_marque_gamme", "cluster": "ASUS::Premium::c0", "marque": "ASUS", "gamme_prix": "Premium",
             "semaine": 2, "n_produits": 2, "prix_geometrique_tnd": 1800.0},
        ])
        pivot, weeks = _pivot_cluster_table(cluster_means, "N2_marque_gamme")
        assert {"marque", "gamme_prix"} <= set(pivot.columns)
        assert pivot.iloc[0]["variation_pct"] == pytest.approx(-10.0)


class TestSpecChangeSentence:
    def test_dict_vide_retourne_chaine_vide(self):
        assert _spec_change_sentence({}) == ""

    def test_dict_non_vide_mentionne_direction(self):
        phrase = _spec_change_sentence({"ram_go": 12.5, "stockage_go": -2.0})
        assert "hausse" in phrase
        assert "baisse" in phrase
        assert "RAM" in phrase or "ram" in phrase.lower()
