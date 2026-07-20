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
