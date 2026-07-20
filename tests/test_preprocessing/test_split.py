# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_preprocessing/test_split.py
=============================================================================
ROLE :
    Tests de src/preprocessing/split.py : construit un faux
    data/processed/week_<N>/<categorie>_clean.csv (fixture tmp_path,
    schema minimal mais realiste -- colonne "url" unique, quelques
    caracteristiques) puis verifie :

      (a) le split respecte ~80/20 ;
      (b) aucun chevauchement train/test (par url) ;
      (c) full_data == train UNION test (aucune ligne perdue) ;
      (d) la reproductibilite : deux executions produisent des fichiers
          strictement identiques (meme RANDOM_STATE).

    Donnees synthetiques plutot que les vraies data/processed/ : isole le
    test du contenu reel (qui evolue au fil des semaines de collecte),
    coherent avec test_transform.py qui utilise deja tmp_path pour les
    memes raisons.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import pytest

from src.preprocessing.split import (
    build_all_weeks,
    build_week_splits,
    check_schema_consistency,
    load_full_data_for_week,
    split_train_test,
)

CATEGORIES = ["cat_a", "cat_b"]
RANDOM_STATE = 42
TEST_SIZE = 0.20


def _make_category_df(category: str, n: int, start_id: int) -> pd.DataFrame:
    """n produits synthetiques pour une categorie, avec une colonne url
    unique (cle produit reelle du projet) et deux caracteristiques."""
    return pd.DataFrame({
        "url": [f"https://example.test/{category}-{i}.html" for i in range(start_id, start_id + n)],
        "nom": [f"{category} produit {i}" for i in range(start_id, start_id + n)],
        "prix_tnd": [100.0 + i for i in range(n)],
        "ram_go": [4.0 * (i % 4 + 1) for i in range(n)],
    })


@pytest.fixture
def fake_processed_dir(tmp_path):
    """
    Construit data/processed/week_1/ et week_2/ dans tmp_path, chacune
    avec 2 categories d'effectifs DIFFERENTS (cat_a=50, cat_b=12) --
    cat_b assez petite pour que la stratification par categorie soit
    reellement exercee (sans elle, une petite categorie pourrait tomber
    entierement d'un cote lors d'un split non stratifie).
    """
    processed_dir = tmp_path / "data" / "processed"
    for week in (1, 2):
        week_dir = processed_dir / f"week_{week}"
        week_dir.mkdir(parents=True)
        _make_category_df("cat_a", 50, start_id=week * 1000).to_csv(week_dir / "cat_a_clean.csv", index=False)
        _make_category_df("cat_b", 12, start_id=week * 1000).to_csv(week_dir / "cat_b_clean.csv", index=False)
    return processed_dir


class TestSplitRatio:
    def test_ratio_respected_overall(self, fake_processed_dir):
        full_df = load_full_data_for_week(1, fake_processed_dir, CATEGORIES)
        train_df, test_df = split_train_test(full_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        actual_ratio = len(test_df) / len(full_df)
        assert abs(actual_ratio - TEST_SIZE) < 0.05, f"Ratio test observé {actual_ratio:.2%}, attendu ~20%"

    def test_ratio_respected_per_category(self, fake_processed_dir):
        """La stratification doit garder ~80/20 y compris pour cat_b (12 produits seulement) --
        c'est precisement le cas qu'un split NON stratifie risquerait de mal traiter."""
        full_df = load_full_data_for_week(1, fake_processed_dir, CATEGORIES)
        train_df, test_df = split_train_test(full_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        for cat in CATEGORIES:
            n_cat = (full_df["categorie"] == cat).sum()
            n_test_cat = (test_df["categorie"] == cat).sum()
            ratio = n_test_cat / n_cat
            assert 0 < n_test_cat < n_cat, f"{cat} : la catégorie est tombée entièrement d'un côté (n_test={n_test_cat})"
            assert abs(ratio - TEST_SIZE) < 0.15, f"{cat} : ratio test {ratio:.2%} trop loin de 20%"


class TestNoOverlap:
    def test_no_train_test_overlap(self, fake_processed_dir):
        full_df = load_full_data_for_week(1, fake_processed_dir, CATEGORIES)
        train_df, test_df = split_train_test(full_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        overlap = set(train_df["url"]) & set(test_df["url"])
        assert overlap == set(), f"Chevauchement train/test détecté : {overlap}"


class TestFullEqualsTrainUnionTest:
    def test_full_equals_union(self, fake_processed_dir):
        full_df = load_full_data_for_week(1, fake_processed_dir, CATEGORIES)
        train_df, test_df = split_train_test(full_df, test_size=TEST_SIZE, random_state=RANDOM_STATE)

        assert len(train_df) + len(test_df) == len(full_df)
        union_urls = set(train_df["url"]) | set(test_df["url"])
        assert union_urls == set(full_df["url"])


class TestReproducibility:
    def test_two_runs_produce_identical_files(self, fake_processed_dir):
        build_week_splits(1, fake_processed_dir, CATEGORIES, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        week_dir = fake_processed_dir / "week_1"
        first_run = {
            name: pd.read_csv(week_dir / name) for name in ("full_data.csv", "train.csv", "test.csv")
        }

        build_week_splits(1, fake_processed_dir, CATEGORIES, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        second_run = {
            name: pd.read_csv(week_dir / name) for name in ("full_data.csv", "train.csv", "test.csv")
        }

        for name in first_run:
            pd.testing.assert_frame_equal(first_run[name], second_run[name]), (
                f"{name} diffère entre deux exécutions malgré RANDOM_STATE fixe"
            )

    def test_reproducible_across_full_pipeline(self, fake_processed_dir):
        """Meme verification que ci-dessus mais via build_all_weeks() (le
        point d'entree reellement utilise par `python -m src.preprocessing.split`),
        sur les 2 semaines de la fixture."""
        build_all_weeks(fake_processed_dir, CATEGORIES, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        snapshot_1 = {
            (week, name): pd.read_csv(fake_processed_dir / f"week_{week}" / name)
            for week in (1, 2) for name in ("full_data.csv", "train.csv", "test.csv")
        }

        build_all_weeks(fake_processed_dir, CATEGORIES, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        snapshot_2 = {
            (week, name): pd.read_csv(fake_processed_dir / f"week_{week}" / name)
            for week in (1, 2) for name in ("full_data.csv", "train.csv", "test.csv")
        }

        for key in snapshot_1:
            pd.testing.assert_frame_equal(snapshot_1[key], snapshot_2[key])


class TestSchemaConsistency:
    def test_identical_schema_detected_as_consistent(self, fake_processed_dir):
        full_by_week = {
            1: load_full_data_for_week(1, fake_processed_dir, CATEGORIES),
            2: load_full_data_for_week(2, fake_processed_dir, CATEGORIES),
        }
        assert check_schema_consistency(full_by_week) is True

    def test_divergent_schema_detected_and_flagged(self, fake_processed_dir):
        """Ajoute une colonne propre a la semaine 2 -- check_schema_consistency
        doit le detecter (return False) sans lever d'exception (jamais
        silencieux, mais jamais bloquant non plus)."""
        full_1 = load_full_data_for_week(1, fake_processed_dir, CATEGORIES)
        full_2 = load_full_data_for_week(2, fake_processed_dir, CATEGORIES)
        full_2["colonne_nouvelle_s2"] = 1

        result = check_schema_consistency({1: full_1, 2: full_2})
        assert result is False
