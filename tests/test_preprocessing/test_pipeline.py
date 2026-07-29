# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_preprocessing/test_pipeline.py
=============================================================================
ROLE :
    Tests de src/preprocessing/pipeline.py::build_processed_datasets -- le
    VRAI point d'entree de production (`python -m src.preprocessing.pipeline`),
    jusqu'ici jamais exerce par la suite de tests : test_integration.py
    reproduit la meme LOGIQUE (clean_products -> encode.py -> ...) mais en
    appelant transform.py::prepare_final_features a la place, sans jamais
    passer par pipeline.py ni ecrire les CSV finaux. Un bug dans
    build_processed_datasets/process_category (ex. mauvais argument a
    select_features_for_category, mauvaise colonne droppee avant l'ecriture)
    pouvait donc passer inapercu de `pytest` en entier -- cf. audit
    methodologique, reviewer 4 (Critical).
=============================================================================
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from src.preprocessing.pipeline import build_all_weeks, build_processed_datasets, compute_stable_feature_selection

MARQUES_PORTABLES = [
    ("ASUS", "Intel Core i5-1235U"), ("HP", "Intel Core i7-1255U"),
    ("DELL", "Intel Core i3-1115G4"), ("LENOVO", "AMD Ryzen 5 5500U"),
    ("MSI", "Intel Core i7-1360P"), ("ACER", "Intel Core i5-1240P"),
    ("GIGABYTE", "AMD Ryzen 7 5800H"), ("TOSHIBA", "Intel Core i3-1215U"),
    ("APPLE", "Intel Core i9-9880H"), ("SAMSUNG", "Intel Core i5-1135G7"),
    ("HUAWEI", "Intel Core i5-1155G7"), ("XIAOMI", "Intel Core i7-1165G7"),
    ("MEDION", "AMD Ryzen 3 5300U"), ("VAIO", "Intel Core i7-10510U"),
]
MARQUES_BUREAU = [
    ("HP", "Intel Core i5-12400"), ("DELL", "Intel Core i3-10105"),
    ("LENOVO", "Intel Core i7-12700"), ("ASUS", "AMD Ryzen 5 5600G"),
    ("ACER", "Intel Core i5-11400"), ("MSI", "Intel Core i7-11700"),
    ("APPLE", "Apple M2"), ("GIGABYTE", "AMD Ryzen 7 5700G"),
    ("HUAWEI", "Intel Core i5-10400"), ("SAMSUNG", "Intel Core i3-9100"),
]


def _produits_pc_portables() -> list:
    """14 produits valides + 1 prix aberrant (neutralise par les bornes de
    clean.py, puis la ligne est ecartee faute de prix fiable -- meme
    convention que test_integration.py::_generer_produits_bruts)."""
    produits = []
    for i, (marque, processeur) in enumerate(MARQUES_PORTABLES):
        prix = -50.0 if i == 0 else 1200.0 + i * 300.0
        produits.append({
            "nom": f"PC Portable {marque} Modele{i} {processeur}",
            "marque": marque,
            "prix_tnd": prix,
            "processeur": processeur,
            "ram_go": float(8 + 4 * (i % 4)),
            "stockage_go": float(256 * (1 + i % 4)),
            "type_stockage": "SSD",
            "taille_ecran": 14.0 + (i % 3),
            "os": "Windows 11 Famille" if i % 2 == 0 else "FreeDos",
            "connectivite": "Wi-Fi; Bluetooth",
            "url": f"https://exemple.test/pc-portable-{marque.lower()}-{i}.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:00:00",
            "specs_brutes": {"Marque": marque},
        })
    return produits


def _produits_pc_bureau() -> list:
    """10 produits, tous valides -- categorie plus petite pour verifier que
    build_processed_datasets traite correctement PLUSIEURS categories dans
    le meme repertoire (pas seulement une, contrairement a test_integration.py)."""
    produits = []
    for i, (marque, processeur) in enumerate(MARQUES_BUREAU):
        produits.append({
            "nom": f"PC Bureau {marque} Modele{i} {processeur}",
            "marque": marque,
            "prix_tnd": 900.0 + i * 250.0,
            "processeur": processeur,
            "ram_go": float(8 + 4 * (i % 4)),
            "stockage_go": float(512 * (1 + i % 3)),
            "type_stockage": "SSD",
            "taille_ecran": None,  # ecran non pertinent pour pc_bureau (VALIDITY_BOUNDS["pc_bureau"]["screen"]=None)
            "os": "Windows 11 Famille" if i % 2 == 0 else "Windows 11 Pro",
            "connectivite": "Wi-Fi; Bluetooth" if i % 3 else None,
            "url": f"https://exemple.test/pc-bureau-{marque.lower()}-{i}.html",
            "categorie": "pc_bureau",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:00:00",
            "specs_brutes": {"Marque": marque},
        })
    return produits


def _ecrire_raw_json(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "produits_pc_portables_test.json", "w", encoding="utf-8") as fh:
        json.dump(_produits_pc_portables(), fh, ensure_ascii=False)
    with open(raw_dir / "produits_pc_bureau_test.json", "w", encoding="utf-8") as fh:
        json.dump(_produits_pc_bureau(), fh, ensure_ascii=False)


class TestBuildProcessedDatasets:
    """build_processed_datasets(raw_dir, out_dir) -- le point d'entree reel
    de `python -m src.preprocessing.pipeline`."""

    def test_ecrit_les_bons_fichiers_pour_chaque_categorie(self, tmp_path):
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _ecrire_raw_json(raw_dir)

        summary = build_processed_datasets(raw_dir, out_dir)

        assert set(summary.keys()) == {"pc_portables", "pc_bureau"}
        # 14 produits pc_portables generes, le prix aberrant (-50.0) est
        # neutralise (None) par les bornes de clean.py puis la ligne
        # ecartee faute de prix fiable (build_processed_datasets) -- 13
        # restants.
        assert summary["pc_portables"]["clean_rows"] == 13
        assert summary["pc_bureau"]["clean_rows"] == 10

        for cat in ("pc_portables", "pc_bureau"):
            assert (out_dir / f"{cat}_clean.csv").exists()
            assert (out_dir / f"{cat}_encoded.csv").exists()

    def test_aucune_valeur_manquante_dans_les_csv_ecrits(self, tmp_path):
        """Invariant central du pipeline (cf. son propre garde-fou interne,
        pipeline.py::process_category) -- verifie ici sur le FICHIER
        REELLEMENT ECRIT, pas seulement sur un DataFrame en memoire avant
        l'ecriture (l'invariant pourrait tenir en memoire et se corrompre a
        l'ecriture/relecture CSV -- ex. un NaN mal serialise)."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _ecrire_raw_json(raw_dir)
        build_processed_datasets(raw_dir, out_dir)

        for cat in ("pc_portables", "pc_bureau"):
            df_clean = pd.read_csv(out_dir / f"{cat}_clean.csv")
            assert df_clean.isna().sum().sum() == 0, f"NaN residuel dans {cat}_clean.csv"
            df_encoded = pd.read_csv(out_dir / f"{cat}_encoded.csv")
            assert df_encoded.isna().sum().sum() == 0, f"NaN residuel dans {cat}_encoded.csv"

    def test_encoded_csv_ne_contient_ni_nom_ni_url(self, tmp_path):
        """process_category retire explicitement nom/url avant encode_for_ridge
        (pipeline.py) -- des identifiants de tracabilite, jamais des features."""
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "processed"
        _ecrire_raw_json(raw_dir)
        build_processed_datasets(raw_dir, out_dir)

        df_encoded = pd.read_csv(out_dir / "pc_portables_encoded.csv")
        assert "nom" not in df_encoded.columns
        assert "url" not in df_encoded.columns

    def test_idempotent_deux_executions_identiques(self, tmp_path):
        """Meme entree -> memes fichiers de sortie, executee deux fois --
        aucun etat cache/aleatoire non-determinisme ne doit se glisser
        (imputation KNN, selection de features, encodage)."""
        raw_dir = tmp_path / "raw"
        out_dir_1 = tmp_path / "processed_1"
        out_dir_2 = tmp_path / "processed_2"
        _ecrire_raw_json(raw_dir)

        build_processed_datasets(raw_dir, out_dir_1)
        build_processed_datasets(raw_dir, out_dir_2)

        for cat in ("pc_portables", "pc_bureau"):
            df1 = pd.read_csv(out_dir_1 / f"{cat}_clean.csv")
            df2 = pd.read_csv(out_dir_2 / f"{cat}_clean.csv")
            pd.testing.assert_frame_equal(df1, df2)

    def test_repertoire_sans_json_leve_erreur_explicite(self, tmp_path):
        """Aucun fichier .json trouve -- une erreur explicite (RuntimeError),
        jamais un pipeline qui continue silencieusement sur un DataFrame vide."""
        raw_dir = tmp_path / "raw_vide"
        raw_dir.mkdir()
        out_dir = tmp_path / "processed"
        with pytest.raises(RuntimeError):
            build_processed_datasets(raw_dir, out_dir)


class TestSelectionStablePooleeSurToutesLesSemaines:
    """compute_stable_feature_selection()/build_all_weeks() -- correctif du
    2026-07-28 (audit methodologique, reviewer 1, Major) : select_features_
    for_category rejouee independamment chaque semaine produisait une
    recommandation bruitee (schema qui derive d'une semaine a l'autre pour
    la MEME variable, cf. l'ancienne _reconcile_pooled_schema de
    save_artifacts.py). Ces tests verifient que la nouvelle selection,
    calculee une seule fois sur les semaines poolees, est bien la MEME
    utilisee pour ecrire CHAQUE semaine -- plus de derive possible."""

    def _ecrire_deux_semaines(self, raw_root: Path) -> None:
        _ecrire_raw_json(raw_root / "week_1")
        _ecrire_raw_json(raw_root / "week_2")

    def test_selection_stable_retourne_une_liste_et_un_rapport_par_categorie(self, tmp_path):
        raw_root = tmp_path / "raw"
        self._ecrire_deux_semaines(raw_root)

        selection = compute_stable_feature_selection(raw_root=raw_root)

        assert set(selection.keys()) == {"pc_portables", "pc_bureau"}
        for cat, (kept_columns, report) in selection.items():
            assert isinstance(kept_columns, list) and len(kept_columns) > 0
            assert {"url", "nom", "prix_tnd"} <= set(kept_columns)  # ALWAYS_KEEP, cf. select_features.py
            assert not report.empty

    def test_build_all_weeks_ecrit_le_meme_schema_pour_chaque_semaine(self, tmp_path):
        """La derive de schema entre semaines (colonne presente une semaine,
        absente une autre pour la MEME categorie) est precisement ce que la
        selection stable doit eliminer."""
        raw_root = tmp_path / "raw"
        out_root = tmp_path / "processed"
        self._ecrire_deux_semaines(raw_root)

        summary = build_all_weeks(raw_root=raw_root, out_root=out_root)
        assert set(summary.keys()) == {1, 2}

        for cat in ("pc_portables", "pc_bureau"):
            cols_s1 = list(pd.read_csv(out_root / "week_1" / f"{cat}_clean.csv").columns)
            cols_s2 = list(pd.read_csv(out_root / "week_2" / f"{cat}_clean.csv").columns)
            assert cols_s1 == cols_s2, f"schema divergent entre semaines pour {cat} : {cols_s1} vs {cols_s2}"

    def test_build_all_weeks_utilise_bien_la_selection_stable_pas_une_selection_locale(self, tmp_path):
        """Verifie le lien entre les deux fonctions : les colonnes ecrites
        par build_all_weeks doivent etre EXACTEMENT celles retournees par
        compute_stable_feature_selection (pas une redecouverte locale par
        semaine qui pourrait par coincidence tomber d'accord)."""
        raw_root = tmp_path / "raw"
        out_root = tmp_path / "processed"
        self._ecrire_deux_semaines(raw_root)

        selection = compute_stable_feature_selection(raw_root=raw_root)
        build_all_weeks(raw_root=raw_root, out_root=out_root)

        for cat, (kept_columns, _report) in selection.items():
            cols_written = list(pd.read_csv(out_root / "week_1" / f"{cat}_clean.csv").columns)
            assert set(cols_written) == set(kept_columns)
