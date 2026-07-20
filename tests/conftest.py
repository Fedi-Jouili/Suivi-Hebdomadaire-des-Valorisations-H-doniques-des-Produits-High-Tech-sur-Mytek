# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/conftest.py
=============================================================================
ROLE :
    Fixtures partagees par toute la suite de tests (src/tests/ et ses
    sous-dossiers test_scraper/, test_preprocessing/, test_models/).

    sys.path.insert ci-dessous : place la RACINE du projet (parent de src/)
    sur sys.path une seule fois, ici, pour que n'importe quel fichier de
    test sous src/tests/ puisse faire `from src.preprocessing.clean import
    ...` quelle que soit la maniere dont pytest est invoque (`pytest` nu,
    `python -m pytest`, depuis un autre repertoire...) -- centralise ici
    plutot que repete dans chaque fichier de test.
=============================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # tests/conftest.py -> tests -> racine
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """
    5 produits mimant des donnees brutes scrapees (schema reel du projet,
    cf. src/scraper/parser.py::parse_product) -- pas 5 produits generiques
    mais 5 CAS DE TEST distincts, chacun choisi pour exercer un
    comportement precis de clean_products() :

      1. ASUS   : cas nominal, toutes les valeurs presentes et valides.
      2. HP     : ram_go ABSENT du champ structure, mais recuperable
                  depuis specs_brutes["Mémoire"] (teste fix_ram).
      3. DELL   : prix_tnd NEGATIF (donnee aberrante), doit etre mis a
                  None par la validation par bornes (teste flag_out_of_range
                  + le fait que clean_products() NULL une valeur hors
                  bornes plutot que de la garder telle quelle).
      4. LENOVO : connectivite ABSENTE (None), doit degrader proprement
                  (pas de crash) en aval dans extract_connectivity_flags.
      5. MSI    : cas nominal, marque/os differents des 4 premiers pour
                  varier les modalites categorielles (utile aux tests
                  d'encodage one-hot).

    NOTE : clean_products() attend une LISTE DE DICTS (pas un DataFrame
    directement) -- cf. test_clean.py qui fait
    sample_raw_df.to_dict("records") avant l'appel, pour rester fidele au
    contrat reel de la fonction tout en gardant une fixture lisible sous
    forme de DataFrame.
    """
    rows = [
        {
            "nom": "PC Portable ASUS VivoBook 15 i5 8Go 512Go SSD",
            "marque": "ASUS",
            "prix_tnd": 1500.0,
            "processeur": "Intel Core i5-1235U",
            "ram_go": 8.0,
            "stockage_go": 512.0,
            "type_stockage": "SSD",
            "taille_ecran": 15.6,
            "os": "Windows 11 Famille",
            "connectivite": "Wi-Fi; Bluetooth",
            "url": "https://www.mytek.tn/pc-portable-asus-vivobook-15.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:00:00",
            "specs_brutes": {"Marque": "ASUS", "Référence Processeur": "Intel® Core™ i5-1235U"},
        },
        {
            "nom": "PC Portable HP 15 i7 16Go 512Go SSD",
            "marque": "HP",
            "prix_tnd": 1800.0,
            "processeur": "Intel Core i7-1255U",
            "ram_go": None,  # absent du champ structure -- recuperable via specs_brutes
            "stockage_go": 512.0,
            "type_stockage": "SSD",
            "taille_ecran": 15.6,
            "os": "Windows 11 Pro",
            "connectivite": "Wi-Fi",
            "url": "https://www.mytek.tn/pc-portable-hp-15.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:05:00",
            "specs_brutes": {"Mémoire": "16 Go", "Référence Processeur": "Intel® Core™ i7-1255U"},
        },
        {
            "nom": "PC Portable DELL Vostro i3 8Go 256Go SSD",
            "marque": "DELL",
            "prix_tnd": -50.0,  # aberrant -- doit etre neutralise par les bornes
            "processeur": "Intel Core i3-1115G4",
            "ram_go": 8.0,
            "stockage_go": 256.0,
            "type_stockage": "SSD",
            "taille_ecran": 15.6,
            "os": "FreeDos",
            "connectivite": "Wi-Fi; Bluetooth",
            "url": "https://www.mytek.tn/pc-portable-dell-vostro.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:10:00",
            "specs_brutes": {"Marque": "DELL"},
        },
        {
            "nom": "PC Portable LENOVO IdeaPad Ryzen 5 16Go 512Go SSD",
            "marque": "LENOVO",
            "prix_tnd": 2200.0,
            "processeur": "AMD Ryzen 5 5500U",
            "ram_go": 16.0,
            "stockage_go": 512.0,
            "type_stockage": "SSD",
            "taille_ecran": 15.6,
            "os": "Windows 11 Famille",
            "connectivite": None,  # absente -- ne doit pas faire planter le traitement aval
            "url": "https://www.mytek.tn/pc-portable-lenovo-ideapad.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:15:00",
            "specs_brutes": {"Marque": "LENOVO", "Référence Processeur": "AMD Ryzen™ 5 5500U"},
        },
        {
            "nom": "PC Portable MSI Modern 14 i7 32Go 1To SSD",
            "marque": "MSI",
            "prix_tnd": 3500.0,
            "processeur": "Intel Core i7-1360P",
            "ram_go": 32.0,
            "stockage_go": 1024.0,
            "type_stockage": "SSD",
            "taille_ecran": 14.0,
            "os": "Windows 11 Pro",
            "connectivite": "Wi-Fi; Bluetooth",
            "url": "https://www.mytek.tn/pc-portable-msi-modern-14.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:20:00",
            "specs_brutes": {"Marque": "MSI", "Référence Processeur": "Intel® Core™ i7-1360P"},
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def sample_encoded_df(sample_raw_df) -> pd.DataFrame:
    """
    sample_raw_df apres nettoyage (clean_products) et extraction des
    caracteristiques structurees (extract_cpu_features, extract_os_platform,
    extract_connectivity_flags) -- l'equivalent du "encode_categorical()"
    demande, mais avec les VRAIS noms de fonctions du projet. Le one-hot
    final (encode_for_ridge) n'est PAS applique ici : chaque test qui en a
    besoin l'appelle explicitement, pour ne pas figer un choix d'encodage
    (Ridge vs RF ont des logiques differentes) dans une fixture partagee.
    """
    from src.preprocessing.clean import clean_products
    from src.preprocessing.encode import extract_cpu_features, extract_os_platform, extract_connectivity_flags

    df = clean_products(sample_raw_df.to_dict("records"))
    df = extract_cpu_features(df)
    df = extract_os_platform(df)
    df = extract_connectivity_flags(df)
    return df


@pytest.fixture
def dummy_X() -> pd.DataFrame:
    """Matrice de caracteristiques numeriques, pret a l'emploi pour RidgeModel/RandomForestModel."""
    rng = np.random.default_rng(42)
    n = 40
    return pd.DataFrame({
        "ram_go": rng.choice([4.0, 8.0, 16.0, 32.0], size=n),
        "stockage_go": rng.choice([128.0, 256.0, 512.0, 1024.0], size=n),
        "cpu_serie": rng.choice([3.0, 5.0, 7.0, 9.0], size=n),
        "taille_ecran": rng.uniform(10.0, 17.0, size=n),
    })


@pytest.fixture
def dummy_y(dummy_X) -> pd.Series:
    """log(prix) correspondant a dummy_X -- relation log-lineaire + bruit, pour des tests deterministes."""
    rng = np.random.default_rng(43)
    log_prix = (
        6.0
        + 0.03 * dummy_X["ram_go"]
        + 0.0008 * dummy_X["stockage_go"]
        + 0.1 * dummy_X["cpu_serie"]
        + 0.05 * dummy_X["taille_ecran"]
        + rng.normal(0, 0.05, size=len(dummy_X))
    )
    return pd.Series(log_prix, name="log_prix")
