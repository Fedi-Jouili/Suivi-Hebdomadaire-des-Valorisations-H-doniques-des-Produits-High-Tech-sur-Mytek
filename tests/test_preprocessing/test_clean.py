# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_preprocessing/test_clean.py
=============================================================================
ROLE :
    Tests de clean_products() (src/preprocessing/clean.py).

    Comportement REEL teste ici (different de "clean_data() supprime les
    lignes prix<=0" -- cette fonction n'existe pas ; voir le recapitulatif
    envoye avant ce fichier) :
      - une ligne dupliquee (meme URL) est fusionnee en une seule (garde la
        plus recente par date_collecte) ;
      - un prix hors bornes (ex: negatif) est mis a None, PAS supprime --
        la suppression des lignes sans prix fiable est une etape ULTERIEURE
        (src/preprocessing/pipeline.py::build_processed_datasets) ;
      - une RAM absente du champ structure mais presente dans specs_brutes
        est recuperee (fix_ram), pas laissee a None.

    L'imputation par la MEDIANE (explicitement demandee) ne vit PAS dans
    clean.py -- c'est la responsabilite de src/preprocessing/impute.py,
    testee dans TestImputationMediane ci-dessous pour couvrir fidelement
    ce qui etait demande, meme si le module differe.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.clean import clean_products
from src.preprocessing.impute import impute_numeric_cascade


class TestClProduitsDedoublonnage:
    """Deduplication par (GTIN ou URL) + semaine (dedup_key)."""

    def test_deux_lignes_meme_url_meme_semaine_fusionnees(self):
        produit_v1 = {
            "nom": "PC Test", "marque": "ASUS", "prix_tnd": 1000.0,
            "processeur": "Intel Core i5", "ram_go": 8.0, "stockage_go": 256.0,
            "type_stockage": "SSD", "taille_ecran": 15.6, "os": "Windows 11",
            "connectivite": "Wi-Fi", "url": "https://exemple.test/produit.html",
            "categorie": "pc_portables", "semaine": 1,
            "date_collecte": "2026-07-01 10:00:00", "specs_brutes": {},
        }
        produit_v2 = {**produit_v1, "prix_tnd": 1050.0, "date_collecte": "2026-07-01 12:00:00"}

        df = clean_products([produit_v1, produit_v2])

        assert len(df) == 1  # les deux lignes ont fusionne en une seule
        assert df.iloc[0]["prix_tnd"] == 1050.0  # la plus recente (date_collecte) est conservee


class TestClProduitsBornes:
    """Une valeur hors bornes est neutralisee (None), pas supprimee -- la
    ligne elle-meme reste presente dans le DataFrame retourne."""

    def test_prix_negatif_devient_none_ligne_conservee(self, sample_raw_df):
        df = clean_products(sample_raw_df.to_dict("records"))

        ligne_dell = df[df["marque"] == "DELL"]
        assert len(ligne_dell) == 1  # la ligne existe toujours
        assert pd.isna(ligne_dell.iloc[0]["prix_tnd"])  # mais son prix aberrant a ete neutralise


class TestClProduitsRecuperationRam:
    """fix_ram recupere la RAM depuis specs_brutes quand le champ
    structure est absent -- l'equivalent visee de 'imputer avec la
    mediane', mais par recuperation de la vraie valeur plutot que par une
    estimation statistique."""

    def test_ram_absente_recuperee_depuis_specs_brutes(self, sample_raw_df):
        df = clean_products(sample_raw_df.to_dict("records"))

        ligne_hp = df[df["marque"] == "HP"]
        assert len(ligne_hp) == 1
        assert ligne_hp.iloc[0]["ram_go"] == 16.0  # recupere depuis specs_brutes["Mémoire"] = "16 Go"


class TestClProduitsRamMemoireStockageAmbigu:
    """
    Regression directe sur le bug trouve et corrige le 2026-07-25 (cf.
    notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb §4.1) : sur
    certaines fiches Mytek.tn, le champ "Memoire" duplique numeriquement
    "Stockage" (128Go == 128 Go) plutot que de designer la RAM -- fix_ram
    doit ecarter "Memoire" dans ce cas precis et chercher un motif
    "RAM : NGo" explicite ailleurs dans specs_brutes (ici fusionne dans le
    champ "Processeur", artefact de scraping reellement observe sur une
    fiche Xiaomi Redmi Note 14).
    """

    _SPECS_XIAOMI_BUGUE = {
        "Marque": "XIAOMI",
        "Processeur": "Processeur Hélio G99-Ultra, jusqu'à 2,5 GHz- RAM : 8Go",
        "Mémoire": "128Go",
        "Stockage": "128 Go",
    }

    def _produit_smartphone(self, ram_go, specs_brutes):
        return {
            "nom": "Smartphone Xiaomi Redmi Note 14 4G  8Go 128Go - Lime Green",
            "marque": "XIAOMI", "prix_tnd": 749.0, "processeur": "Hélio G99-Ultra",
            "ram_go": ram_go, "stockage_go": 128.0, "type_stockage": None,
            "taille_ecran": 6.67, "os": "Xiaomi HyperOS", "connectivite": "4G; Wi-Fi",
            "url": "https://www.mytek.tn/redmi-note-14-8-128-lime.html",
            "categorie": "smartphones", "semaine": 3, "date_collecte": "2026-07-15 03:43:16",
            "specs_brutes": specs_brutes,
        }

    def test_memoire_dupliquant_stockage_ecartee_repli_sur_ram_fusionnee(self):
        """raw ram_go=128 (bug scraper, hors de la borne smartphones
        (1.0, 32.0) resserree le 2026-07-25) -> rejete par clean_products,
        fix_ram doit retrouver 8.0 via le motif "RAM : 8Go" fusionne dans
        "Processeur", PAS 128.0 via le doublon "Memoire"/"Stockage"."""
        produit = self._produit_smartphone(128.0, self._SPECS_XIAOMI_BUGUE)
        df = clean_products([produit])
        assert df.iloc[0]["ram_go"] == 8.0

    def test_memoire_distincte_de_stockage_reste_utilisee_normalement(self):
        """Cas majoritaire (990/1020 smartphones du catalogue reel) : quand
        "Memoire" NE duplique PAS "Stockage", elle designe bien la RAM et
        doit continuer a etre utilisee -- le garde-fou ne doit pas
        sur-corriger le cas normal."""
        specs = {"Marque": "SAMSUNG", "Mémoire": "2 Go", "Stockage": "16 Go"}
        # ram_go volontairement hors bornes pour forcer la re-derivation
        # (0.05 -- imite le second bug corrige le meme jour, cf.
        # TestClProduitsRamBornesResserrees).
        produit = self._produit_smartphone(0.05, specs)
        df = clean_products([produit])
        assert df.iloc[0]["ram_go"] == 2.0

    def test_aucun_signal_ram_exploitable_retombe_sur_none(self):
        """Ni "RAM"/"Memoire RAM" specifique, ni motif "RAM : NGo" fusionne
        ailleurs, et "Memoire" dedoublonne avec "Stockage" -- fix_ram ne
        doit JAMAIS deviner (retourner 128.0 par defaut serait pire que
        l'absence de valeur) : None, a charge de l'imputation statistique
        en aval (impute.py) de completer proprement."""
        specs = {"Marque": "XIAOMI", "Mémoire": "128Go", "Stockage": "128 Go"}
        produit = self._produit_smartphone(128.0, specs)
        df = clean_products([produit])
        assert pd.isna(df.iloc[0]["ram_go"])


class TestClProduitsRamBornesResserrees:
    """Regression sur le second volet du meme correctif (2026-07-25) : les
    bornes basses de plausibilite RAM (auparavant 0.0 pour toutes les
    categories) laissaient passer des valeurs quasi nulles (bug de
    confusion cache/RAM historique, cf. docstring de fix_ram) sans jamais
    declencher de re-derivation."""

    def test_smartphone_ram_quasi_nulle_rejetee_et_rederivee(self):
        specs = {"Marque": "SAMSUNG", "RAM": "8Go", "Stockage": "128 Go"}
        produit = {
            "nom": "Smartphone Samsung Galaxy Test 8Go 128Go",
            "marque": "SAMSUNG", "prix_tnd": 899.0, "processeur": "Exynos",
            "ram_go": 0.0234375,  # artefact reel observe (cache/RAM confondus)
            "stockage_go": 128.0, "type_stockage": None, "taille_ecran": 6.5,
            "os": "Android", "connectivite": "4G; Wi-Fi",
            "url": "https://www.mytek.tn/galaxy-test.html",
            "categorie": "smartphones", "semaine": 1, "date_collecte": "2026-07-01 10:00:00",
            "specs_brutes": specs,
        }
        df = clean_products([produit])
        assert df.iloc[0]["ram_go"] == 8.0  # re-derive depuis specs_brutes["RAM"], plus 0.0234

    def test_telephone_portable_ram_quasi_nulle_reste_intacte(self):
        """A la difference des smartphones/pc_portables, telephones_portables
        n'a PAS eu sa borne basse relevee (decision documentee dans
        VALIDITY_BOUNDS) : certains feature phones basiques ont une RAM
        reellement de l'ordre de quelques dizaines de Mo (ex. "32 Mo"
        explicite en specs_brutes) -- ce n'est pas le bug cache/RAM, une
        borne basse relevee ecarterait ici une valeur correcte."""
        specs = {"Marque": "NOKIA", "RAM": "32 Mo"}
        produit = {
            "nom": "Téléphone Portable Nokia Test",
            "marque": "NOKIA", "prix_tnd": 45.0, "processeur": None,
            "ram_go": 32 / 1024,  # ~0.03125 -- valeur EXACTE et legitime, pas un artefact
            "stockage_go": 32.0, "type_stockage": None, "taille_ecran": 2.0,
            "os": None, "connectivite": None,
            "url": "https://www.mytek.tn/nokia-test.html",
            "categorie": "telephones_portables", "semaine": 1, "date_collecte": "2026-07-01 10:00:00",
            "specs_brutes": specs,
        }
        df = clean_products([produit])
        assert df.iloc[0]["ram_go"] == pytest.approx(32 / 1024)


class TestImputationMediane:
    """
    L'imputation par la mediane demandee dans la tache initiale vit dans
    src/preprocessing/impute.py (impute_numeric_cascade), appelee par
    categorie dans le pipeline final -- pas dans clean.py. Teste ici pour
    couvrir fidelement le comportement demande.
    """

    def test_ram_manquante_imputee_par_mediane_quand_non_recuperable(self):
        # 4 valeurs connues (mediane = 12.0) + 1 manquante, sans specs_brutes
        # exploitable -- rien a "recuperer", seule l'imputation statistique
        # peut combler ce trou.
        df = pd.DataFrame({
            "categorie": ["pc_portables"] * 5,
            "ram_go": [8.0, 8.0, 16.0, 16.0, np.nan],
            "stockage_go": [256.0, 512.0, 256.0, 512.0, 256.0],
        })

        result = impute_numeric_cascade(
            df, target_columns=["ram_go"], neighbor_columns=["ram_go", "stockage_go"],
            category_col="categorie",
        )

        assert result["ram_go"].isna().sum() == 0  # plus aucune valeur manquante
        assert result["ram_go"].iloc[4] in (8.0, 16.0, 12.0)  # valeur plausible (KNN ou repli mediane)
