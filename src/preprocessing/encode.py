# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/encode.py
=============================================================================
ROLE :
    Encodage des variables catégorielles pour les modèles hédoniques.

    Deux modes d'encodage :
      - encode_for_ridge : one-hot (get_dummies, drop_first=True) pour
        éviter le dummy trap et permettre l'interprétation des coefficients
        Ridge comme prix implicites (théorie de Rosen 1974).
      - encode_for_rf    : label encoding entier avec mapping réutilisable
        sur S2/S3 (codes stables, catégories inconnues → -1).

    Fonctions auxiliaires :
      - parse_cpu / extract_cpu_features : marque (Intel/AMD/Apple/Qualcomm/
        MediaTek/Samsung Exynos/Google Tensor/HiSilicon Kirin), série (tier)
        et génération du processeur, extraits du champ 'processeur' et de
        la référence constructeur dans specs_brutes.
      - extract_os_platform : simplifie le champ 'os' (texte libre à très
        haute cardinalité pour smartphones/téléviseurs) en une plateforme
        à faible cardinalité (Android/iOS/HarmonyOS ; Google TV/Android TV/
        Tizen/webOS/VIDAA), adaptée par catégorie.
      - extract_connectivity_flags : flags binaires indépendants (has_4g,
        has_5g, has_wifi, has_wifi6, has_bluetooth, has_nfc, has_ethernet)
        depuis le champ 'connectivite'.
=============================================================================
"""

import json
import re
import logging
import warnings

import pandas as pd

logger = logging.getLogger("preprocessing.encode")

# Colonnes nominales traitées par encode_for_ridge et encode_for_rf.
# cpu_serie est ORDINAL (entier 3/5/7/9) → reste numérique, jamais éclaté.
# "os" -> "os_platform" (texte brut trop haute cardinalité, cf.
# extract_os_platform) ; "type_stockage" retiré (confirmé quasi-constant
# ou 100% manquant sur toutes les catégories réelles -- aucun pouvoir
# discriminant, cf. select_features.py).
NOMINAL_COLUMNS = ["marque", "categorie", "os_platform", "cpu_brand"]

_CONNECTIVITY_FLAGS = [
    "has_4g", "has_5g", "has_wifi", "has_wifi6", "has_bluetooth", "has_nfc",
    "has_ethernet",
]


# ─────────────────────────────────────────────────────────────────────────────
# PARSE CPU
# ─────────────────────────────────────────────────────────────────────────────

def parse_cpu(processeur, specs_brutes):
    """
    Extrait cpu_brand, cpu_serie (tier ordinal 3/5/7/9) et cpu_gen depuis
    le champ 'processeur' et la référence constructeur dans specs_brutes.

    Conventions :
      - Intel Core iX → brand=Intel, serie=X, gen depuis "iX-NNNNN" dans ref
      - AMD Ryzen X   → brand=AMD,   serie=X, gen=premier chiffre du modèle
      - Celeron / Jasper Lake / Apple → serie=None, gen=None
      - Qualcomm (Snapdragon) / MediaTek (Dimensity/Helio) / Samsung Exynos /
        Google Tensor / HiSilicon Kirin → brand detecte (serie/gen restent
        None : les conventions de nommage n'ont pas d'equivalent au tier
        ordinal Intel/AMD). Ajoute car sur le scrape smartphones S1, ces
        marques representaient a elles seules 203/254 produits, tous
        auparavant classes "Autre" faute de detection -- rendant cpu_brand
        constant (donc inutile) pour toute la categorie smartphones.
      - Marque inconnue → brand='Autre'

    Returns: dict {"cpu_brand": str|None, "cpu_serie": int|None, "cpu_gen": int|None}
    """
    result = {"cpu_brand": None, "cpu_serie": None, "cpu_gen": None}

    if processeur is None or (isinstance(processeur, float) and pd.isna(processeur)):
        return result

    specs = specs_brutes
    if isinstance(specs, str):
        # specs_brutes survit au round-trip CSV sous forme de chaine JSON
        # (clean.py le serialise via json.dumps) -- on le redeserialise ici.
        try:
            specs = json.loads(specs)
        except (json.JSONDecodeError, TypeError):
            specs = {}
    if not isinstance(specs, dict):
        specs = {}  # couvre NaN (float), None, ou toute autre valeur inattendue
    ref = specs.get("Référence Processeur", "") or ""

    # ── Marque ───────────────────────────────────────────────────────────────
    proc_lower = processeur.lower()
    if "intel" in proc_lower:
        result["cpu_brand"] = "Intel"
    elif "amd" in proc_lower:
        result["cpu_brand"] = "AMD"
    elif "apple" in proc_lower:
        result["cpu_brand"] = "Apple"
    elif "snapdragon" in proc_lower or "qualcomm" in proc_lower:
        result["cpu_brand"] = "Qualcomm"
    elif "dimensity" in proc_lower or "mediatek" in proc_lower or "helio" in proc_lower:
        result["cpu_brand"] = "MediaTek"
    elif "exynos" in proc_lower:
        result["cpu_brand"] = "Samsung Exynos"
    elif "tensor" in proc_lower:
        result["cpu_brand"] = "Google Tensor"
    elif "kirin" in proc_lower:
        result["cpu_brand"] = "HiSilicon Kirin"
    else:
        result["cpu_brand"] = "Autre"

    # ── Série (tier) ─────────────────────────────────────────────────────────
    if result["cpu_brand"] == "Intel":
        # Deux conventions coexistent : "Core i5" et "Core 5" (Arrow Lake)
        m = re.search(r'\bi([3579])\b', processeur, re.IGNORECASE)
        if not m:
            m = re.search(r'Core\s+([3579])\b', processeur, re.IGNORECASE)
        if m:
            result["cpu_serie"] = int(m.group(1))
    elif result["cpu_brand"] == "AMD":
        m = re.search(r'Ryzen\s+([3579])\b', processeur, re.IGNORECASE)
        if m:
            result["cpu_serie"] = int(m.group(1))

    # ── Génération ───────────────────────────────────────────────────────────
    if result["cpu_brand"] == "Intel" and result["cpu_serie"] is not None:
        # Uniquement pour les Core (pas Celeron/Atom qui n'ont pas de tier)
        # Format ref : "Intel® Core™ i5-13420H" → modèle "13420" → gen 13
        # Bug corrigé : ancienne regex \D{0,15} ne sautait pas le tier AMD
        m = re.search(r'i\d-(\d+)', ref, re.IGNORECASE)
        if m:
            n = int(m.group(1))
            # Gen 10+ : modèle 5 chiffres (10xxx–13xxx) ou 4 chiffres (1000NG4)
            if n >= 10000:
                result["cpu_gen"] = n // 1000   # 13420 → 13
            elif n >= 1000:
                result["cpu_gen"] = n // 100    # 1000  → 10
    elif result["cpu_brand"] == "AMD":
        # Format ref : "AMD Ryzen™ 5 3400G" → modèle "3400" → gen 3
        # .{0,20}? saute le symbole ™ et l'espace avant le tier (5)
        m = re.search(r'Ryzen.{0,20}?\d+\s+(\d+)', ref, re.IGNORECASE)
        if m:
            result["cpu_gen"] = int(m.group(1)[0])

    return result


def extract_cpu_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique parse_cpu ligne à ligne et ajoute cpu_brand, cpu_serie, cpu_gen.
    Si specs_brutes est absent du DataFrame, cpu_gen sera None pour toutes
    les lignes (dégradation propre, pas d'exception).
    """
    df = df.copy()

    if df.empty:
        for col in ("cpu_brand", "cpu_serie", "cpu_gen"):
            df[col] = pd.Series(dtype=object)
        return df

    def _parse_row(row):
        specs = row["specs_brutes"] if "specs_brutes" in row.index else None
        processeur = row["processeur"] if "processeur" in row.index else None
        return pd.Series(parse_cpu(processeur, specs))

    cpu_df = df.apply(_parse_row, axis=1, result_type="expand")
    return pd.concat([df, cpu_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# OS / PLATEFORME
# ─────────────────────────────────────────────────────────────────────────────
# 'os' brut est une variable a tres forte cardinalite pour smartphones et
# televiseurs (ex: "Android 15, jusqu'à 3 mises à jour Android majeures,
# XOS 15.1") -- inexploitable telle quelle comme categorielle (quasi-
# unique par ligne), et parfois franchement fausse pour les televiseurs
# (le parser capte parfois une spec de luminosite/frequence a la place,
# ex: "200 nits -Taux de rafraîchissement : 60 Hz..."). os_platform la
# remplace par un petit nombre de familles reconnues ; tout ce qui ne
# matche aucun mot-cle connu tombe dans "Autre" plutot que de conserver
# du texte parasite.

_OS_KEYWORDS_PHONE = [
    ("iOS", ("ios", "iphone")),
    ("HarmonyOS", ("harmonyos", "harmony os")),
    # Les surcouches constructeur (ColorOS, HyperOS, XOS, OriginOS,
    # FuntouchOS, MagicOS, MIUI, OxygenOS...) sont toutes baties sur
    # Android -- elles ne justifient pas une plateforme a part.
    ("Android", ("android", "coloros", "hyperos", "xos", "originos",
                 "funtouchos", "magicos", "miui", "oxygenos", "realme ui",
                 "emui")),
]

_OS_KEYWORDS_TV = [
    # "Google TV" verifie AVANT "android" : "Google TV (Android 11)" doit
    # rester Google TV, pas retomber sur le libelle generique Android TV.
    ("Google TV", ("google tv",)),
    ("Android TV", ("android",)),
    ("Tizen", ("tizen",)),
    ("webOS", ("webos",)),
    ("VIDAA", ("vidaa",)),
]


def _classify_os(text, keyword_table, default="Autre"):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return "Inconnu"
    norm = str(text).lower()
    for label, keywords in keyword_table:
        if any(kw in norm for kw in keywords):
            return label
    return default


def extract_os_platform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplifie 'os' en 'os_platform', une plateforme a faible cardinalite,
    avec une logique adaptee par famille de categorie :
      - smartphones  : Android / iOS / HarmonyOS / Autre / Inconnu.
      - televiseurs   : Google TV / Android TV / Tizen / webOS / VIDAA /
                        Autre / Inconnu (absorbe aussi les valeurs
                        manifestement fausses, ex. specs de luminosite).
      - autres categories (pc_bureau, pc_portables, telephones_portables) :
        'os' y est deja une categorielle propre (FreeDos/Windows 11 Pro/
        Famille) -- copie normalisee (None -> "Inconnu"), pas de mapping.

    Si 'categorie' ou 'os' est absent du DataFrame, os_platform="Inconnu"
    partout (degradation propre, pas d'exception).
    """
    df = df.copy()

    if "os" not in df.columns or "categorie" not in df.columns:
        df["os_platform"] = "Inconnu"
        return df

    def _row_platform(row):
        cat = row.get("categorie")
        os_val = row.get("os")
        if cat == "smartphones":
            return _classify_os(os_val, _OS_KEYWORDS_PHONE)
        if cat == "televiseurs":
            return _classify_os(os_val, _OS_KEYWORDS_TV)
        if os_val is None or (isinstance(os_val, float) and pd.isna(os_val)):
            return "Inconnu"
        return str(os_val).strip()

    df["os_platform"] = df.apply(_row_platform, axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# CONNECTIVITÉ
# ─────────────────────────────────────────────────────────────────────────────

def extract_connectivity_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des flags binaires indépendants depuis 'connectivite'.
    has_wifi exclut explicitement 'wifi 6' via negative lookahead pour
    éviter le double comptage : un produit avec "WiFi 6" n'a pas aussi
    du WiFi générique.
    Si la colonne est absente ou None, tous les flags sont 0.

    "wi-?fi" (tiret optionnel) et non "wifi" : bug corrigé -- la quasi-
    totalité des specs Mytek.tn (parser + specs_brutes) écrivent
    "Wi-Fi" avec un tiret, jamais "wifi" sans tiret. L'ancien pattern
    littéral "wifi" ne matchait donc JAMAIS cette orthographe : confirmé
    sur les données réelles, has_wifi était à 0/541 pour pc_portables
    (541/541 ont pourtant bien "Wi-Fi" dans connectivite), et sous-compté
    ailleurs (34/112 au lieu de 96/112 pour les téléviseurs).
    """
    df = df.copy()

    if "connectivite" not in df.columns:
        warnings.warn(
            "Colonne 'connectivite' absente du DataFrame — flags mis à 0.",
            stacklevel=2,
        )
        for flag in _CONNECTIVITY_FLAGS:
            df[flag] = 0
        return df

    def _flags(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return {f: 0 for f in _CONNECTIVITY_FLAGS}
        s = str(val)
        # has_4g/has_5g : cherches sur `s` SANS les mentions de bande Wi-Fi
        # (ex. "Wi-Fi 2.4G/5G") -- sans ce retrait, "2.4G" et "/5G" declenchent
        # un faux positif cellulaire (confirme sur le scrape reel : la chaine
        # "Interface Reseau: RJ45; Connectivite: Wi-Fi 2.4G/5G, Bluetooth 5.4",
        # exclusive aux televiseurs -- aucun televiseur de ce catalogue n'a de
        # connectivite cellulaire reelle -- faisait basculer has_4g/has_5g a 1
        # pour des TV sans aucune carte SIM). Les vraies mentions cellulaires
        # ("Reseaux Mobiles: 4G", "4G; Wi-Fi"...) ne suivent jamais ce format
        # "Wi-Fi X(.X)G/Y(.Y)G" et restent donc intactes. has_wifi/has_wifi6
        # restent calcules sur `s` original (la bande Wi-Fi elle-meme doit
        # rester detectee).
        s_no_wifi_band = re.sub(
            r'wi-?fi\s*[:\-]?\s*\d(\.\d)?\s*g(hz)?\s*/\s*\d(\.\d)?\s*g(hz)?', '', s, flags=re.IGNORECASE,
        )
        return {
            "has_4g":        int(bool(re.search(r'\b4g\b',         s_no_wifi_band, re.IGNORECASE))),
            "has_5g":        int(bool(re.search(r'\b5g\b',         s_no_wifi_band, re.IGNORECASE))),
            "has_wifi6":     int(bool(re.search(r'wi-?fi\s*6',     s, re.IGNORECASE))),
            "has_wifi":      int(bool(re.search(r'wi-?fi(?!\s*6)', s, re.IGNORECASE))),
            "has_bluetooth": int(bool(re.search(r'bluetooth',      s, re.IGNORECASE))),
            "has_nfc":       int(bool(re.search(r'\bnfc\b',        s, re.IGNORECASE))),
            "has_ethernet":  int(bool(re.search(r'ethernet|rj-?45', s, re.IGNORECASE))),
        }

    flags_df = df["connectivite"].apply(_flags).apply(pd.Series)
    return pd.concat([df, flags_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE POUR RIDGE (one-hot, drop_first)
# ─────────────────────────────────────────────────────────────────────────────

def encode_for_ridge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode NOMINAL_COLUMNS en one-hot avec drop_first=True.

    Garanties :
      - NaN dans une colonne nominale → catégorie explicite 'Manquant'
        (pas de disparition silencieuse qui rendrait la valeur manquante
        indistinguable de la catégorie de référence droppée).
      - cpu_serie reste numérique (ordinal, jamais éclaté en dummies).
      - Colonnes nominales absentes ignorées sans exception.
    """
    df = df.copy()

    cols_present = [c for c in NOMINAL_COLUMNS if c in df.columns]
    for col in cols_present:
        df[col] = df[col].fillna("Manquant").astype(str)

    if cols_present:
        df = pd.get_dummies(df, columns=cols_present, drop_first=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE POUR RANDOM FOREST (label encoding entier)
# ─────────────────────────────────────────────────────────────────────────────

def encode_for_rf(df: pd.DataFrame) -> tuple:
    """
    Encode NOMINAL_COLUMNS en codes entiers stables pour Random Forest.

    Retourne (DataFrame encodé, dict encodeurs) où encodeurs =
    {col: {categorie → code}} — à conserver pour appliquer le même
    mapping sur S2/S3 via apply_rf_encoders.
    """
    df = df.copy()
    encoders = {}

    for col in NOMINAL_COLUMNS:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Manquant").astype(str)
        categories = sorted(df[col].unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        encoders[col] = mapping
        df[f"{col}_code"] = df[col].map(mapping).astype(int)
        df = df.drop(columns=[col])

    return df, encoders


def apply_rf_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Applique les encodeurs générés par encode_for_rf à un nouveau DataFrame.
    Catégories jamais vues lors du fit → code -1 (pas d'exception, pas de
    contamination silencieuse d'une catégorie existante).
    """
    df = df.copy()

    for col, mapping in encoders.items():
        if col not in df.columns:
            df[f"{col}_code"] = -1
            continue
        df[col] = df[col].fillna("Manquant").astype(str)
        df[f"{col}_code"] = df[col].map(mapping).fillna(-1).astype(int)
        df = df.drop(columns=[col])

    return df


__all__ = [
    "NOMINAL_COLUMNS",
    "parse_cpu",
    "extract_cpu_features",
    "extract_os_platform",
    "extract_connectivity_flags",
    "encode_for_ridge",
    "encode_for_rf",
    "apply_rf_encoders",
]