# -*- coding: utf-8 -*-
"""
=============================================================================
Module : tests/test_encode.py
=============================================================================
ROLE :
    Tests unitaires pour src/preprocessing/encode.py.

    Couvre en priorite :
      - parse_cpu sur des cas REELS observes dans le scrape pc_bureau
        S1 (Intel "Core iX | Core X" double nommage, Celeron/Jasper
        Lake sans tier, AMD Ryzen avec generation extraite de la
        reference constructeur)
      - la regression du bug AMD : \D{0,15} ne pouvait pas sauter le
        chiffre du tier (5) avant le modele (3400G) -- corrige en
        .{0,20}? ; ce test verrouille le comportement correct.
      - extract_connectivity_flags : flags independants, jamais de
        combinatoire ; regression sur le bug 'wifi' vs 'Wi-Fi' (tiret
        optionnel desormais matche, l'ancien pattern ne matchait jamais
        l'orthographe reellement dominante dans les donnees)
      - extract_os_platform : simplification categorie-consciente d'un
        champ 'os' a tres forte cardinalite, y compris le cas de valeurs
        manifestement fausses (specs de luminosite/frequence capturees
        a la place d'un OS sur des televiseurs)
      - encode_for_ridge : one-hot drop_first, NaN traites comme
        categorie explicite, cpu_serie reste numerique ordinal
      - encode_for_rf : codes entiers stables, mapping reutilisable
        via apply_rf_encoders pour coder nouvelles semaines

UTILISATION :
    pytest tests/test_encode.py -v
=============================================================================
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.encode import (
    parse_cpu,
    extract_cpu_features,
    extract_os_platform,
    extract_connectivity_flags,
    encode_for_ridge,
    encode_for_rf,
    apply_rf_encoders,
    NOMINAL_COLUMNS,
)


# ─────────────────────────────────────────────────────────────────────────────
# parse_cpu : cas reels du scrape pc_bureau S1
# ─────────────────────────────────────────────────────────────────────────────

class TestParseCpu:

    def test_intel_dual_naming_core_i3_pipe_core3(self):
        """'Intel Core i3 | Core 3' -- les deux conventions de nommage
        Intel cohabitent dans la meme chaine, le tier doit etre detecte
        une seule fois (3), pas en double ou en conflit."""
        result = parse_cpu(
            "Intel Core i3 | Core 3",
            {"Référence Processeur": "Intel® Core™ i3-1000NG4"},
        )
        assert result["cpu_brand"] == "Intel"
        assert result["cpu_serie"] == 3
        assert result["cpu_gen"] == 10

    def test_intel_celeron_has_no_serie_tier(self):
        """Celeron n'a pas de tier 3/5/7/9 -- cpu_serie doit etre None,
        ce qui est correct (pas une absence de donnee a corriger)."""
        result = parse_cpu("Intel Celeron", {"Référence Processeur": "Intel® Celeron® G5900"})
        assert result["cpu_brand"] == "Intel"
        assert result["cpu_serie"] is None

    def test_intel_jasper_lake_atom_class_no_tier(self):
        result = parse_cpu("Intel Jasper Lake N5095", {"Référence Processeur": "Intel® Jasper Lake N5095"})
        assert result["cpu_brand"] == "Intel"
        assert result["cpu_serie"] is None
        assert result["cpu_gen"] is None

    def test_amd_ryzen_generation_extracted_correctly(self):
        """Regression directe sur le bug trouve et corrige : la
        reference 'AMD Ryzen™ 5 3400G' doit donner generation=3, pas
        None. Le bug original venait de \\D{0,15} qui ne pouvait pas
        sauter le chiffre du tier (5) avant le modele (3400)."""
        result = parse_cpu("AMD Ryzen 5", {"Référence Processeur": "AMD Ryzen™ 5 3400G"})
        assert result["cpu_brand"] == "AMD"
        assert result["cpu_serie"] == 5
        assert result["cpu_gen"] == 3

    @pytest.mark.parametrize("reference,expected_gen", [
        ("AMD Ryzen™ 3 3200G", 3),
        ("AMD Ryzen™ 5 3400G", 3),
        ("AMD Ryzen™ 5 3600", 3),
        ("AMD Ryzen™ 5 5500GT", 5),
        ("AMD Ryzen™ 5 5600GT", 5),
    ])
    def test_amd_generation_all_real_references(self, reference, expected_gen):
        """Verrouille le comportement sur l'ensemble des references AMD
        reellement observees dans le scrape (pas seulement un exemple)."""
        result = parse_cpu("AMD Ryzen", {"Référence Processeur": reference})
        assert result["cpu_gen"] == expected_gen

    def test_intel_generation_two_digit(self):
        result = parse_cpu("Intel Core i5", {"Référence Processeur": "Intel® Core™ i5-13420H"})
        assert result["cpu_gen"] == 13

    def test_apple_brand_detected_no_serie(self):
        result = parse_cpu("Apple M2", {})
        assert result["cpu_brand"] == "Apple"
        assert result["cpu_serie"] is None  # Apple n'a pas de tier 3/5/7/9

    def test_unknown_brand_falls_back_to_autre(self):
        result = parse_cpu("Unisoc Tiger T612", {})
        assert result["cpu_brand"] == "Autre"

    @pytest.mark.parametrize("processeur,expected_brand", [
        ("Snapdragon 8 Gen 3", "Qualcomm"),
        ("Qualcomm Snapdragon 695", "Qualcomm"),
        ("MediaTek Dimensity 7050", "MediaTek"),
        ("Helio G99", "MediaTek"),
        ("Samsung Exynos 2400", "Samsung Exynos"),
        ("Google Tensor G4", "Google Tensor"),
        ("HiSilicon Kirin 990", "HiSilicon Kirin"),
    ])
    def test_smartphone_chipset_brands_detected(self, processeur, expected_brand):
        """Regression : ces marques etaient toutes classees 'Autre' avant
        l'extension du detecteur -- confirme sur le scrape smartphones S1
        que cela rendait cpu_brand constant (donc inutile) pour toute la
        categorie (203/254 produits concernes)."""
        result = parse_cpu(processeur, {})
        assert result["cpu_brand"] == expected_brand

    def test_none_processeur_returns_all_none(self):
        result = parse_cpu(None, {})
        assert result == {"cpu_brand": None, "cpu_serie": None, "cpu_gen": None}

    def test_missing_specs_brutes_does_not_crash(self):
        """specs_brutes=None (au lieu de {}) ne doit pas planter --
        cpu_gen sera simplement None faute de reference constructeur."""
        result = parse_cpu("AMD Ryzen 5", None)
        assert result["cpu_brand"] == "AMD"
        assert result["cpu_serie"] == 5
        assert result["cpu_gen"] is None


class TestExtractCpuFeatures:

    def test_dataframe_integration(self):
        df = pd.DataFrame([
            {"processeur": "AMD Ryzen 5", "specs_brutes": {"Référence Processeur": "AMD Ryzen™ 5 3400G"}},
            {"processeur": "Intel Celeron", "specs_brutes": {"Référence Processeur": "Intel® Celeron® G5900"}},
        ])
        result = extract_cpu_features(df)
        assert list(result["cpu_brand"]) == ["AMD", "Intel"]
        # Note : pd.DataFrame.apply(result_type="expand") sur une colonne
        # mixte int/None la convertit en float64 (5 -> 5.0, None -> NaN).
        # C'est un comportement pandas attendu, pas un defaut de
        # extract_cpu_features -- d'ou la comparaison via isna()/== plutot
        # qu'une egalite de liste stricte avec None.
        assert result["cpu_serie"].iloc[0] == 5
        assert pd.isna(result["cpu_serie"].iloc[1])
        assert result.loc[0, "cpu_gen"] == 3

    def test_missing_specs_column_does_not_crash(self):
        """Si specs_brutes a deja ete supprime du DataFrame (ex: apres
        clean_products), l'extraction doit degrader proprement plutot
        que planter -- cpu_gen sera majoritairement None."""
        df = pd.DataFrame([{"processeur": "Intel Core i5"}])
        result = extract_cpu_features(df)
        assert result.loc[0, "cpu_brand"] == "Intel"

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["processeur", "specs_brutes"])
        result = extract_cpu_features(df)
        assert len(result) == 0
        assert "cpu_brand" in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# extract_connectivity_flags
# ─────────────────────────────────────────────────────────────────────────────

class TestConnectivityFlags:

    def test_single_value_4g(self):
        """Cas reel observe : connectivite='4G' (valeur unique, pas une
        liste) sur le scrape pc_bureau S1."""
        df = pd.DataFrame([{"connectivite": "4G"}])
        result = extract_connectivity_flags(df)
        assert result.loc[0, "has_4g"] == 1
        assert result.loc[0, "has_5g"] == 0
        assert result.loc[0, "has_wifi"] == 0

    def test_multivalue_string(self):
        df = pd.DataFrame([{"connectivite": "WiFi, Bluetooth, 4G, 5G"}])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_wifi"] == 1
        assert row["has_bluetooth"] == 1
        assert row["has_4g"] == 1
        assert row["has_5g"] == 1
        assert row["has_nfc"] == 0

    def test_wifi6_does_not_doublecount_as_plain_wifi(self):
        """'WiFi 6' doit activer has_wifi6 ; le pattern has_wifi exclut
        explicitement 'wifi 6' via negative lookahead pour eviter un
        double comptage trompeur (produit qui semblerait avoir a la
        fois du wifi generique ET du wifi6 a partir d'une seule
        mention)."""
        df = pd.DataFrame([{"connectivite": "WiFi 6, Bluetooth 5.0"}])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_wifi6"] == 1
        assert row["has_wifi"] == 0

    def test_missing_value_all_zero(self):
        df = pd.DataFrame([{"connectivite": None}])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_4g"] == 0
        assert row["has_wifi"] == 0

    def test_missing_column_warns_but_does_not_crash(self):
        df = pd.DataFrame([{"nom": "produit sans colonne connectivite"}])
        result = extract_connectivity_flags(df)
        assert result.loc[0, "has_4g"] == 0

    def test_hyphenated_wifi_is_detected(self):
        """Regression directe sur le bug trouve et corrige : 'Wi-Fi'
        (avec tiret) est l'orthographe DOMINANTE dans les donnees reelles
        (parser.py l'ecrit systematiquement ainsi), mais l'ancien pattern
        litteral 'wifi' ne la matchait jamais. Confirme sur le scrape
        reel : has_wifi etait a 0/541 pour pc_portables alors que
        541/541 ont bien 'Wi-Fi' dans connectivite."""
        df = pd.DataFrame([{"connectivite": "Interface Réseau: RJ45; Connectivité: Wi-Fi 5, Bluetooth 5.2"}])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_wifi"] == 1
        assert row["has_bluetooth"] == 1
        assert row["has_ethernet"] == 1

    def test_hyphenated_wifi6_does_not_doublecount(self):
        """Meme garde-fou que test_wifi6_does_not_doublecount_as_plain_wifi,
        mais avec l'orthographe a tiret reellement observee."""
        df = pd.DataFrame([{"connectivite": "Wi-Fi 6, Bluetooth 5.3"}])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_wifi6"] == 1
        assert row["has_wifi"] == 0

    def test_ethernet_flag_detects_rj45(self):
        df = pd.DataFrame([
            {"connectivite": "Interface Réseau: RJ45; Connectivité: Wi-Fi 5"},
            {"connectivite": "Wi-Fi seulement"},
        ])
        result = extract_connectivity_flags(df)
        assert result.loc[0, "has_ethernet"] == 1
        assert result.loc[1, "has_ethernet"] == 0

    def test_wifi_dual_band_notation_not_read_as_cellular(self):
        """Regression directe sur le bug trouve et corrige le 2026-07-25
        (cf. notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb §4.2) :
        'Wi-Fi 2.4G/5G' decrit une connectivite Wi-Fi double bande (2,4 GHz
        et 5 GHz), jamais une connectivite cellulaire -- mais l'ancien motif
        \\b4g\\b / \\b5g\\b matchait quand meme "4G" dans "2.4G" et "5G" dans
        "/5G". Chaine reelle observee sur le scrape televiseurs (aucun
        televiseur de ce catalogue n'a de carte SIM) : has_4g/has_5g
        doivent rester a 0, has_wifi doit rester a 1 (seule la lecture
        cellulaire est corrigee, pas la detection Wi-Fi elle-meme)."""
        df = pd.DataFrame([{"connectivite": "Interface Réseau: RJ45; Connectivité: Wi-Fi 2.4G/5G, Bluetooth 5.4"}])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_4g"] == 0
        assert row["has_5g"] == 0
        assert row["has_wifi"] == 1
        assert row["has_bluetooth"] == 1
        assert row["has_ethernet"] == 1

    def test_genuine_cellular_mentions_still_detected_alongside_wifi_band(self):
        """Meme garde-fou que ci-dessus, mais dans l'autre sens : une vraie
        mention cellulaire ('Reseaux Mobiles: 4G') combinee a une notation
        de bande Wi-Fi dans la MEME chaine ne doit pas etre masquee par le
        retrait du motif de bande -- has_4g doit rester a 1."""
        df = pd.DataFrame([{
            "connectivite": "Réseaux Mobiles: 4G; Connectivité: Wi-Fi (2,4 GHz | 5 GHz), 4G et Bluetooth 5.4",
        }])
        result = extract_connectivity_flags(df)
        row = result.iloc[0]
        assert row["has_4g"] == 1
        assert row["has_wifi"] == 1

    def test_real_smartphone_connectivity_strings_unaffected(self):
        """Echantillon de chaines connectivite REELLEMENT observees sur le
        scrape smartphones/telephones_portables (jamais de notation de
        bande Wi-Fi 'X.XG/Y.YG') -- verifie que le retrait ajoute pour le
        cas televiseurs ne change RIEN pour ces cas, les plus frequents du
        catalogue."""
        cas = [
            ("4G; Wi-Fi", {"has_4g": 1, "has_5g": 0}),
            ("5G", {"has_4g": 0, "has_5g": 1}),
            ("Réseaux Mobiles: 5G; Connectivité: 5G, Wifi 7 et Bluetooth 5.4; 4G; Wi-Fi", {"has_4g": 1, "has_5g": 1}),
            ("Réseaux Mobiles: 4G; Connectivité: 4G, NFC et Bluetooth", {"has_4g": 1, "has_5g": 0}),
        ]
        df = pd.DataFrame([{"connectivite": c} for c, _ in cas])
        result = extract_connectivity_flags(df)
        for i, (_, attendu) in enumerate(cas):
            for flag, valeur in attendu.items():
                assert result.loc[i, flag] == valeur, f"ligne {i} ({cas[i][0]!r}) : {flag} attendu={valeur}"


# ─────────────────────────────────────────────────────────────────────────────
# extract_os_platform
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractOsPlatform:

    def test_smartphone_android_oem_skins_collapse_to_android(self):
        """ColorOS/HyperOS/XOS/MagicOS/EMUI... sont toutes des surcouches
        Android -- ne justifient pas une plateforme a part (cas reels du
        scrape smartphones S1)."""
        df = pd.DataFrame([
            {"categorie": "smartphones", "os": "Android 15, HyperOS 2"},
            {"categorie": "smartphones", "os": "ColorOS 15.0"},
            {"categorie": "smartphones", "os": "Android 16, XOS 16"},
            {"categorie": "smartphones", "os": "EMUI 13"},
        ])
        result = extract_os_platform(df)
        assert list(result["os_platform"]) == ["Android"] * 4

    def test_smartphone_ios_and_harmonyos_detected(self):
        df = pd.DataFrame([
            {"categorie": "smartphones", "os": "iOS 17"},
            {"categorie": "smartphones", "os": "HarmonyOS 4.2"},
        ])
        result = extract_os_platform(df)
        assert list(result["os_platform"]) == ["iOS", "HarmonyOS"]

    def test_smartphone_genuinely_wrong_value_falls_back_to_autre(self):
        """os='FreeDos' sur un smartphone est une erreur de collecte
        reelle (constatee sur le scrape) -- pas de plateforme mobile
        connue ne matche, elle tombe donc dans 'Autre' plutot que d'etre
        propagee comme si elle etait valide."""
        df = pd.DataFrame([{"categorie": "smartphones", "os": "FreeDos"}])
        result = extract_os_platform(df)
        assert result.loc[0, "os_platform"] == "Autre"

    def test_tv_google_tv_wins_over_generic_android(self):
        """'Google TV (Android 11)' contient a la fois 'google tv' et
        'android' -- doit rester Google TV (marketing plus specifique),
        pas retomber sur le libelle generique Android TV."""
        df = pd.DataFrame([{"categorie": "televiseurs", "os": "Google TV (Android 11)"}])
        result = extract_os_platform(df)
        assert result.loc[0, "os_platform"] == "Google TV"

    def test_tv_bare_android_maps_to_android_tv(self):
        df = pd.DataFrame([{"categorie": "televiseurs", "os": "Android"}])
        result = extract_os_platform(df)
        assert result.loc[0, "os_platform"] == "Android TV"

    def test_tv_garbage_value_falls_back_to_autre(self):
        """Cas reel observe sur le scrape televiseurs S1 : le parser
        capture parfois une spec de luminosite/frequence a la place d'un
        OS. Aucun mot-cle de plateforme connue ne matche -- 'Autre'
        plutot que de conserver ce texte parasite tel quel."""
        df = pd.DataFrame([{
            "categorie": "televiseurs",
            "os": '200 nits -Taux de rafraîchissement : 60 Hz  -Transmission Numérique Terrestre "TNT"',
        }])
        result = extract_os_platform(df)
        assert result.loc[0, "os_platform"] == "Autre"

    def test_pc_os_passed_through_normalized(self):
        """Pour les categories PC, 'os' est deja propre -- os_platform en
        est une copie normalisee (None -> 'Inconnu'), pas un mapping."""
        df = pd.DataFrame([
            {"categorie": "pc_bureau", "os": "FreeDos"},
            {"categorie": "pc_bureau", "os": None},
        ])
        result = extract_os_platform(df)
        assert result.loc[0, "os_platform"] == "FreeDos"
        assert result.loc[1, "os_platform"] == "Inconnu"

    def test_missing_columns_degrade_to_inconnu(self):
        df = pd.DataFrame([{"nom": "produit sans categorie ni os"}])
        result = extract_os_platform(df)
        assert result.loc[0, "os_platform"] == "Inconnu"


# ─────────────────────────────────────────────────────────────────────────────
# encode_for_ridge
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeForRidge:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([
            {"marque": "ASUS", "categorie": "pc_bureau", "type_stockage": "SSD",
             "os_platform": "Windows 11", "cpu_brand": "Intel", "cpu_serie": 5,
             "has_4g": 0, "has_wifi": 1, "has_wifi6": 0, "has_bluetooth": 1,
             "has_5g": 0, "has_nfc": 0},
            {"marque": "DELL", "categorie": "pc_bureau", "type_stockage": "HDD",
             "os_platform": None, "cpu_brand": "AMD", "cpu_serie": 3,
             "has_4g": 1, "has_wifi": 0, "has_wifi6": 0, "has_bluetooth": 0,
             "has_5g": 0, "has_nfc": 0},
        ])

    def test_drop_first_avoids_dummy_trap(self, sample_df):
        """Avec drop_first=True et 2 marques distinctes, une seule
        colonne marque_* doit survivre (pas les deux) -- evite la
        colinearite parfaite entre dummies qui fausserait
        l'interpretation des coefficients Ridge comme prix implicites."""
        result = encode_for_ridge(sample_df)
        marque_cols = [c for c in result.columns if c.startswith("marque_")]
        assert len(marque_cols) == 1  # 2 categories -> 1 dummy apres drop_first

    def test_missing_os_becomes_explicit_category_not_silently_dropped(self, sample_df):
        """Un OS manquant (None) doit devenir la categorie explicite
        'Manquant', pas disparaitre silencieusement de get_dummies (ce
        qui rendrait un produit sans OS renseigne indistinguable de la
        categorie de reference droppee par drop_first)."""
        result = encode_for_ridge(sample_df)
        os_cols = [c for c in result.columns if c.startswith("os_platform_")]
        # 2 valeurs distinctes ('Windows 11', 'Manquant') -> 1 dummy apres drop_first
        assert len(os_cols) == 1

    def test_cpu_serie_stays_numeric_not_onehot(self, sample_df):
        """cpu_serie est ORDINAL -- doit rester une colonne numerique
        unique, jamais eclatee en dummies (contrairement aux variables
        nominales)."""
        result = encode_for_ridge(sample_df)
        assert "cpu_serie" in result.columns
        assert pd.api.types.is_numeric_dtype(result["cpu_serie"])
        cpu_serie_cols = [c for c in result.columns if c.startswith("cpu_serie_")]
        assert len(cpu_serie_cols) == 0

    def test_binary_flags_passed_through_unchanged(self, sample_df):
        result = encode_for_ridge(sample_df)
        assert list(result["has_4g"]) == [0, 1]

    def test_missing_nominal_column_does_not_crash(self):
        """Si une colonne nominale attendue (ex: 'os_platform') est
        totalement absente du DataFrame, l'encodage doit continuer sans
        planter."""
        df = pd.DataFrame([{"marque": "ASUS", "categorie": "pc_bureau"}])
        result = encode_for_ridge(df)
        assert any(c.startswith("marque_") for c in result.columns) or True  # ne plante pas


# ─────────────────────────────────────────────────────────────────────────────
# encode_for_rf / apply_rf_encoders
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeForRf:

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame([
            {"marque": "ASUS", "categorie": "pc_bureau", "type_stockage": "SSD",
             "os_platform": "Windows 11", "cpu_brand": "Intel", "cpu_serie": 5},
            {"marque": "DELL", "categorie": "pc_bureau", "type_stockage": "HDD",
             "os_platform": "FreeDos", "cpu_brand": "AMD", "cpu_serie": 3},
        ])

    def test_produces_integer_codes_not_onehot(self, sample_df):
        result, encoders = encode_for_rf(sample_df)
        assert "marque_code" in result.columns
        assert pd.api.types.is_integer_dtype(result["marque_code"])
        # Pas de colonnes one-hot du type marque_ASUS
        assert not any(c.startswith("marque_") and c != "marque_code" for c in result.columns)

    def test_encoder_mapping_returned_and_consistent(self, sample_df):
        result, encoders = encode_for_rf(sample_df)
        assert "marque" in encoders
        assert encoders["marque"]["ASUS"] == result.loc[0, "marque_code"]
        assert encoders["marque"]["DELL"] == result.loc[1, "marque_code"]

    def test_cpu_serie_remains_numeric(self, sample_df):
        result, _ = encode_for_rf(sample_df)
        assert list(result["cpu_serie"]) == [5, 3]

    def test_apply_rf_encoders_reuses_mapping_on_new_data(self, sample_df):
        """Le mapping genere sur S1 doit etre reutilisable tel quel
        pour encoder S2 -- les codes doivent rester identiques pour les
        memes categories, condition necessaire pour qu'un modele RF
        deja entraine reste valide sur de nouvelles semaines."""
        _, encoders = encode_for_rf(sample_df)
        new_week_df = pd.DataFrame([
            {"marque": "ASUS", "categorie": "pc_bureau", "type_stockage": "SSD",
             "os_platform": "Windows 11", "cpu_brand": "Intel"},
        ])
        result = apply_rf_encoders(new_week_df, encoders)
        assert result.loc[0, "marque_code"] == encoders["marque"]["ASUS"]

    def test_apply_rf_encoders_unseen_category_gets_minus_one(self, sample_df):
        """Une marque jamais vue lors du fit original (ex: nouvelle
        marque apparue en S3) doit recevoir le code -1, pas planter ni
        etre silencieusement assignee a une categorie existante."""
        _, encoders = encode_for_rf(sample_df)
        new_week_df = pd.DataFrame([
            {"marque": "MARQUE_INCONNUE_S3", "categorie": "pc_bureau",
             "type_stockage": "SSD", "os_platform": "Windows 11", "cpu_brand": "Intel"},
        ])
        result = apply_rf_encoders(new_week_df, encoders)
        assert result.loc[0, "marque_code"] == -1