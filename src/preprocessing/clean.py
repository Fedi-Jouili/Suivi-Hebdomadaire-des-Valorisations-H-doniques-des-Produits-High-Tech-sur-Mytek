# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/clean.py
=============================================================================
ROLE :
    Pipeline de data cleaning du projet de modele hedonique Mytek.tn.
    Lit tous les fichiers JSON bruts dans data/raw/ (recursivement, toutes
    semaines et categories confondues), applique les regles de nettoyage,
    et ecrit un CSV unique consolide dans data/processed/.

REGLES DE NETTOYAGE APPLIQUEES :
    1. Deduplication par (GTIN ou URL) + SEMAINE. La semaine fait partie
       de la cle car le meme produit est *legitimement* observe sur
       plusieurs semaines (S1, S2, S3) -- c'est la structure meme du
       panel temporel du projet. Seuls les vrais doublons (meme produit
       scrape deux fois la MEME semaine, ex: scheduler relance par
       erreur) sont elimines. Le nom n'est jamais utilise comme cle.
    2. Marque : toujours prise depuis specs_brutes["Marque"] en priorite
       (source fiable, remplie par Magento), avec repli sur le champ
       'marque' deja calcule par le parser seulement si l'attribut est
       absent des specs -- puis NORMALISEE a l'echelle du DataFrame
       (fusion des doublons de casse type "EVERTEK"/"Evertek" et des
       variantes de gamme d'un meme fabricant type "POWERED-BY-MSI-*").
    3. RAM/stockage/ecran : recalcules depuis specs_brutes PUIS le nom du
       produit quand le champ structure est absent OU rejete (hors
       bornes -- cf. regle 5) -- une valeur deja presente n'est plus une
       source aveuglement prioritaire si elle est aberrante.
    4. Harmonisation du stockage : comparaison entre la capacite annoncee
       dans le nom du produit et celle de specs_brutes ; flag en cas
       d'ecart (utile pour les modeles a variantes multiples).
    5. Validation par bornes PAR CATEGORIE : prix, RAM, stockage et
       taille d'ecran hors plage plausible sont a la fois FLAGUES
       (colonne *_suspect=True, calculee sur la valeur finale -- indique
       si le probleme PERSISTE apres tentative de recuperation) et
       CORRIGES : la valeur aberrante est rejetee et une re-derivation
       est tentee depuis les autres sources disponibles (regle 3) ; le
       prix (variable cible, non derivable d'une autre source) est mis a
       None plutot que corrige -- la ligne est ecartee en aval faute de
       cible fiable. Les bornes sont calibrees empiriquement par
       categorie (voir bounds.py) plutot que choisies a la main, car
       desktops/laptops/smartphones/feature phones ont des distributions
       tres differentes (prix, RAM, ecran).
    6. Deduplication finale : en cas de doublon (GTIN/URL, semaine)
       strict, on garde la ligne la plus recente (date_collecte la plus
       tardive).

    Ce que clean_products() NE FAIT PAS (voir impute.py/select_features.py/
    pipeline.py) : combler les valeurs encore manquantes apres l'etape 5
    (imputation par similarite) et choisir les colonnes finales par
    categorie pour la modelisation/le clustering -- ces etapes sont
    volontairement separees pour que clean_products() reste une fonction
    pure, reutilisable telle quelle par d'autres usages futurs.

CALIBRATION DES BORNES :
    Les bornes ci-dessous (VALIDITY_BOUNDS) doivent etre regenerees a
    partir des donnees reelles via bounds.compute_all_bounds() quand le
    volume de donnees augmente significativement (nouvelles semaines,
    nouvelles categories). Ne jamais adopter aveuglement la sortie brute
    de compute_all_bounds() : elle doit d'abord etre comparee aux valeurs
    MIN/MAX reellement observees (loggees par compute_bounds_from_data)
    -- un quantile 1-99% + marge peut proposer une borne plus etroite que
    des valeurs deja confirmees legitimes (constate ici : l'auto-
    calibration proposait 12.3 pouces comme minimum d'ecran pour les PC
    portables, alors que des netbooks reels a 10.5 pouces existent deja
    dans les donnees -- l'aurait fait rejeter a tort). Ne jamais resserrer
    une borne en dessous du MIN/MAX observe.

    Derniere calibration : 2026-07-04, sur 1181 produits dedupliques
    (5 categories, semaine 1) via bounds.compute_all_bounds(), puis
    ajustee a la main pour ne jamais etre plus stricte que les valeurs
    observees (cf. commentaires par categorie ci-dessous).

UTILISATION :
    python -m src.preprocessing.clean
    python -m src.preprocessing.clean --raw-dir data/raw --out data/processed/produits_clean.csv
=============================================================================
"""

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger("cleaning")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# ─────────────────────────────────────────────────────────────────────────────
# BORNES DE VALIDITE PAR CATEGORIE
# ─────────────────────────────────────────────────────────────────────────────
# PROVISOIRE -- a regenerer avec bounds.compute_all_bounds() une fois le
# scrape complet disponible (500+ produits/categorie). Format :
#   {categorie: {"price": (lo, hi), "ram": (lo, hi),
#                "storage": (lo, hi), "screen": (lo, hi) | None}}
# screen=None signifie "variable non pertinente pour cette categorie"
# (ex: desktop sans ecran integre) -> jamais flaguee.

VALIDITY_BOUNDS = {
    "pc_bureau": {
        "price": (0.0, 12574.0),   # resserre (etait 30000) -- max observe 10659 sur 217 produits
        "ram": (0.0, 128.0),       # auto-calibration (0,44) trop etroite (max observe 64) -- marge de securite au-dessus
        "storage": (0.0, 4096.0),  # resserre (etait 16384) -- large marge au-dessus du max observe (2048)
        "screen": None,
    },
    "pc_portables": {
        "price": (0.0, 16500.0),   # resserre (etait 30000) -- couvre le max observe (16129) avec marge
        "ram": (0.0, 128.0),       # inchange -- deja bien calibre (max observe 64)
        "storage": (0.0, 4096.0),  # resserre (etait 8192) -- couvre le max observe (2048) avec marge
        "screen": (9.0, 19.0),     # inchange -- deja bien calibre (observe [10.5,17.3])
    },
    "smartphones": {
        "price": (0.0, 10500.0),   # resserre (etait 20000) -- couvre le max observe (6999) avec marge
        "ram": (0.0, 192.0),       # auto-calibration (0,23) trop etroite (max observe 128 !) -- marge au-dessus
        "storage": (0.0, 4096.0),  # releve (etait 2048) -- couvre le max observe (2048) avec marge
        "screen": (5.0, 7.5),      # resserre (etait 3,8) -- bien calibre sur 254 produits (observe [5.5,7.0])
    },
    "telephones_portables": {
        "price": (0.0, 120.0),     # fortement resserre (etait 1000) -- calibre sur 47 produits (max observe 85)
        "ram": (0.0, 24.0),        # releve (etait 2.0) -- des feature phones "connectes" ont jusqu'a 16 Go reels
        "storage": (0.0, 64.0),    # echantillon auto degenere (n=5, toutes valeurs=32) -- non fiable pour resserrer,
                                   # marge raisonnable gardee plutot que le point unique [32,32] propose
        "screen": (0.8, 3.0),      # resserre (etait 1,5) -- bien calibre sur 70 produits (observe [1.4,2.4]) ;
                                   # capture toujours le bug AMI C14 (14.0 pouces, tres au-dessus de 3.0)
    },
    "televiseurs": {
        "price": (0.0, 15000.0),   # legerement resserre (etait 20000) -- couvre le max observe (10799) avec marge
        "ram": None,       # non pertinent pour les TVs (~16/112 Smart TV seulement le renseignent, cf. pipeline.py)
        "storage": None,   # non pertinent pour les TVs (~6/112 Smart TV seulement, cf. pipeline.py)
        "screen": (20.0, 130.0),   # releve (etait 20,100) -- anticipe les diagonales premium 100"+ deja sur le marche
    },
}

# Bornes par defaut si la categorie n'est pas reconnue dans le dict
# ci-dessus (ne devrait pas arriver en pratique, filet de securite).
DEFAULT_BOUNDS = {
    "price": (0.0, 30000.0),
    "ram": (0.0, 256.0),
    "storage": (0.0, 16384.0),
    "screen": (1.0, 19.0),
}


def get_bounds(category: str) -> dict:
    """Retourne le dict de bornes pour une categorie, ou DEFAULT_BOUNDS
    si la categorie est inconnue (avec avertissement logge une seule
    fois par categorie inconnue rencontree)."""
    bounds = VALIDITY_BOUNDS.get(category)
    if bounds is None:
        logger.warning(
            f"Categorie '{category}' absente de VALIDITY_BOUNDS, "
            f"DEFAULT_BOUNDS utilise (bornes larges, peu discriminantes)."
        )
        return DEFAULT_BOUNDS
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES TEXTE (insensibles aux accents, pour matcher specs_brutes)
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text):
    """Supprime les accents et met en minuscule pour comparaison robuste."""
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', str(text))
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _find_spec(specs, *candidate_keys, exclude=None):
    """
    Cherche une valeur dans specs_brutes en testant plusieurs cles
    candidates, de facon insensible aux accents/casse.

    Strategie en deux passes pour eviter les faux positifs du type
    "Memoire" matchant "Memoire Cache" au lieu de "Memoire" (RAM) :
      1. Egalite exacte (normalisee) entre la cle candidate et la cle
         specs_brutes -- priorite absolue, zero ambiguite.
      2. A defaut, correspondance partielle (substring), en ignorant
         les cles contenant un terme de la liste `exclude` (ex: "cache",
         "type", "reference" qui sont des attributs voisins mais distincts).

    Args:
        specs: dict specs_brutes
        candidate_keys: cles a chercher, par ordre de priorite
        exclude: liste de termes (str) -- toute cle specs_brutes les
                 contenant est ignoree lors du matching partiel.

    Returns: la valeur (str) ou None.
    """
    if not specs:
        return None
    exclude = exclude or []
    norm_exclude = [_normalize(e) for e in exclude]

    # Passe 1 : egalite exacte normalisee (priorite absolue)
    for candidate in candidate_keys:
        norm_candidate = _normalize(candidate)
        for spec_key, spec_val in specs.items():
            if _normalize(spec_key) == norm_candidate:
                return spec_val

    # Passe 2 : correspondance partielle, en excluant les cles voisines
    for candidate in candidate_keys:
        norm_candidate = _normalize(candidate)
        for spec_key, spec_val in specs.items():
            norm_spec_key = _normalize(spec_key)
            if any(ex in norm_spec_key for ex in norm_exclude):
                continue
            if norm_candidate in norm_spec_key:
                return spec_val
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES FICHIERS JSON BRUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_files(raw_dir: Path) -> list:
    """
    Charge tous les fichiers .json trouves recursivement sous raw_dir.
    Chaque fichier est une liste de dicts produits (format du scheduler).

    Returns: liste plate de tous les produits, toutes semaines/categories
             confondues.
    """
    json_files = sorted(raw_dir.rglob("*.json"))
    if not json_files:
        logger.warning(f"Aucun fichier .json trouve sous {raw_dir}")
        return []

    all_products = []
    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                products = json.load(fh)
            if not isinstance(products, list):
                logger.warning(f"  {filepath.name} ignore (pas une liste JSON).")
                continue
            logger.info(f"  {filepath.name} : {len(products)} produit(s)")
            all_products.extend(products)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"  Erreur lecture {filepath.name} : {exc}")

    logger.info(f"Total brut charge : {len(all_products)} produit(s) "
                f"depuis {len(json_files)} fichier(s).")
    return all_products


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION DE LA CLE DE DEDUPLICATION (GTIN)
# ─────────────────────────────────────────────────────────────────────────────

def extract_gtin(product: dict) -> str | None:
    """
    Extrait le GTIN/reference produit depuis specs_brutes, via _find_spec
    (insensible accents/casse) plutot qu'un acces direct ["gtin"], pour
    ne pas dependre d'une casse exacte de la cle source. C'est la cle de
    deduplication PRIORITAIRE : deux produits au nom quasi identique mais
    a GTIN different (ex: variante FreeDOS vs Windows pre-installe) sont
    des SKU legitimement distincts.
    """
    specs = product.get("specs_brutes") or {}
    gtin = _find_spec(specs, "gtin", "GTIN", "EAN", "Code-barres", "Reference")
    if gtin:
        return str(gtin).strip()
    return None


def dedup_key(product: dict) -> str:
    """
    Cle de deduplication : (GTIN ou URL) + SEMAINE.

    La semaine est incluse car le meme produit est *legitimement*
    present sur plusieurs semaines -- c'est la structure meme du panel
    temporel du projet (S1, S2, S3). Sans la semaine dans la cle, le
    dedup collapserait les observations multi-semaines d'un meme
    produit en une seule ligne (la plus recente), detruisant les
    donnees necessaires a l'analyse temporelle.

    Le NOM n'est jamais utilise comme cle (deux SKU peuvent partager
    un nom tres proche tout en etant des produits distincts).
    """
    gtin = extract_gtin(product)
    week = product.get("semaine")
    if gtin:
        base = f"gtin:{gtin}"
    elif product.get("url"):
        base = f"url:{product.get('url')}"
    else:
        # Dernier recours improbable : pas de gtin ni d'url -> pas de
        # dedup possible au-dela de l'identite de l'objet en memoire.
        base = f"row:{id(product)}"
    return f"{base}|semaine:{week}"


# ─────────────────────────────────────────────────────────────────────────────
# RECALCUL / CORRECTION DES CHAMPS DEPUIS specs_brutes
# ─────────────────────────────────────────────────────────────────────────────

def fix_brand(product: dict) -> str | None:
    """
    La marque doit TOUJOURS provenir de specs_brutes['Marque'] en
    priorite (source fiable generee par Magento), avec repli sur le
    champ 'marque' deja calcule par le parser (heuristique sur le nom)
    seulement si l'attribut est absent des specs.
    """
    specs = product.get("specs_brutes") or {}
    brand_from_specs = _find_spec(specs, "Marque", "Brand", "Fabricant")
    if brand_from_specs:
        return str(brand_from_specs).strip()
    return product.get("marque")


# Meme fabricant, decline en plusieurs libelles marketing dans les specs
# Magento -- fusionnes pour eviter des dummies quasi-identiques dans le
# modele hedonique (constate sur pc_bureau S1 : POWERED-BY-MSI-ADVANCED/
# -ESSENTIAL/-ULTIMATE sont tous des PC assembles par/pour MSI, pas des
# fabricants distincts).
_BRAND_ALIAS_MAP = {
    "powered-by-msi-advanced": "MSI",
    "powered-by-msi-essential": "MSI",
    "powered-by-msi-ultimate": "MSI",
}


def normalize_brand_column(series: pd.Series) -> pd.Series:
    """
    Normalise une colonne de marques a l'echelle du DataFrame (pas ligne
    par ligne, car la casse canonique se decide par frequence globale) :

      1. Applique _BRAND_ALIAS_MAP (meme fabricant, plusieurs libelles).
      2. Regroupe les valeurs identiques a la casse pres (ex: "EVERTEK"
         (n=8) vs "Evertek" (n=1), constate sur telephones_portables) et
         retient la graphie la PLUS FREQUENTE comme forme canonique --
         pas de liste a maintenir a la main, s'adapte a tout nouveau cas.

    Aucune marque n'est supprimee/filtree : l'audit du scrape actuel n'a
    trouve aucune marque manifestement inventee parmi les valeurs
    observees, seulement des variantes de graphie/gamme du meme
    fabricant -- d'ou une normalisation plutot qu'un filtrage.
    """
    if series.empty:
        return series

    aliased = series.map(
        lambda v: _BRAND_ALIAS_MAP.get(_normalize(v), v) if v is not None else v
    )

    canonical_by_key: dict[str, str] = {}
    for value in aliased.value_counts().index:  # deja trie par frequence desc.
        canonical_by_key.setdefault(_normalize(value), value)

    return aliased.map(lambda v: canonical_by_key.get(_normalize(v), v) if v is not None else v)


def fix_ram(product: dict, seed: float | None = "__unset__") -> float | None:
    """
    Recalcule la RAM (Go) depuis specs_brutes/nom si le champ de depart
    est absent OU a ete rejete en amont (hors bornes -- cf. clean_products,
    qui appelle cette fonction avec seed=None quand la valeur du parser est
    connue pour etre aberrante, au lieu de lui faire confiance aveuglement).

    Args:
        seed: valeur de depart a faire confiance si non None. Par defaut
              (sentinelle "__unset__"), on lit product["ram_go"] --
              permet d'appeler fix_ram(product) comme avant (retro-
              compatible) tout en permettant a clean_products de forcer
              une re-derivation via seed=None.

    exclude=["cache"] est indispensable ici : sans lui, le candidat
    generique "Memoire" matcherait par substring la cle "Memoire Cache"
    (cache processeur) au lieu de "Memoire" (RAM) quand cette derniere
    n'est pas trouvee par les candidats plus specifiques -- confusion
    reellement observee sur le scrape (RAM aberrante de l'ordre de
    quelques Mo au lieu de plusieurs Go, corrigee a la source dans
    scraper/parser.py, mais toujours presente dans les JSON deja scrapes).
    """
    if seed == "__unset__":
        seed = product.get("ram_go")
    if seed is not None:
        return seed

    specs = product.get("specs_brutes") or {}
    val = _find_spec(specs, "Memoire RAM", "Memoire vive", "Memoire", "RAM", exclude=["cache"])
    if val:
        match = re.search(r'(\d+)\s*(Go|GB|Mo|MB)', str(val), re.IGNORECASE)
        if match:
            num = float(match.group(1))
            unit = match.group(2).upper()
            if unit in ("MO", "MB"):
                num = num / 1024
            return num

    name = product.get("nom") or ""
    match = re.search(r'(\d+)\s*(Go|GB)\s*(RAM|DDR)', name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


def fix_screen_size(product: dict, category: str = None, seed: float | None = "__unset__") -> float | None:
    """
    Recalcule la taille d'ecran (pouces) depuis specs_brutes/nom si le
    champ de depart est absent OU a ete rejete en amont (hors bornes).
    Voir fix_ram pour la logique de `seed`.

    Garde-fous :
      1. Si la categorie n'a pas d'ecran pertinent (ex: pc_bureau,
         screen=None dans VALIDITY_BOUNDS), on ne tente meme pas
         l'extraction -- evite de remonter une valeur parasite (ex:
         un nombre capte ailleurs dans le texte des specs).
      2. Si la valeur textuelle contient explicitement "Sans Ecran"
         (cas reel observe : mini PC BMAX), on ne tente pas de parser
         un nombre dedans -- evite un faux positif comme un "3" capte
         par erreur dans un texte qui ne decrit pas une taille d'ecran.
      3. Toute valeur extraite du NOM (bien moins fiable que specs_brutes
         -- un numero de modele comme "AMI C14" peut etre confondu avec
         une diagonale de 14 pouces) est immediatement revalidee contre
         les bornes de la categorie avant d'etre acceptee ; si elle est
         hors bornes, elle est rejetee plutot que retournee telle quelle.
    """
    bounds = get_bounds(category) if category is not None else None
    screen_bounds = bounds.get("screen") if bounds else None
    if bounds is not None and screen_bounds is None:
        # Categorie sans ecran pertinent : verite structurelle qui prime
        # sur n'importe quelle valeur deja presente (seed) -- sinon une
        # valeur aberrante comme les 1260 "pouces" constates sur des
        # PC de bureau (confusion avec une dimension de boitier en mm)
        # survivrait simplement parce qu'elle est non-None.
        return None

    if seed == "__unset__":
        seed = product.get("taille_ecran")
    if seed is not None:
        return seed

    specs = product.get("specs_brutes") or {}
    val = _find_spec(specs, "Taille de l'ecran", "Taille ecran", "Diagonale")
    if val and "sans ecran" in _normalize(val):
        return None
    if val:
        match = re.search(r'([\d.,]+)', str(val))
        if match:
            try:
                return float(match.group(1).replace(',', '.'))
            except ValueError:
                pass

    # Repli sur le nom du produit -- source moins fiable, d'ou la
    # revalidation immediate contre les bornes (garde-fou 3 ci-dessus).
    name = product.get("nom") or ""
    match = re.search(r'([\d.,]+)\s*["”pouces\']', name, re.IGNORECASE)
    if match:
        try:
            candidate = float(match.group(1).replace(',', '.'))
        except ValueError:
            return None
        if screen_bounds is not None and not (screen_bounds[0] <= candidate <= screen_bounds[1]):
            return None
        return candidate
    return None


def parse_storage_from_text(text: str):
    """Extrait (capacite_go, type) depuis un texte libre."""
    if not text:
        return None, None
    text = str(text)
    match = re.search(r'(\d+)\s*To\b', text, re.IGNORECASE)
    if match:
        capacity = float(match.group(1)) * 1024
    else:
        match = re.search(r'(\d+)\s*Go\b', text, re.IGNORECASE)
        if match:
            capacity = float(match.group(1))
        else:
            match = re.search(r'(\d+)\s*GB\b', text, re.IGNORECASE)
            if match:
                capacity = float(match.group(1))
            else:
                return None, None

    text_upper = text.upper()
    if "SSD" in text_upper:
        storage_type = "SSD"
    elif "HDD" in text_upper:
        storage_type = "HDD"
    elif "EMMC" in text_upper:
        storage_type = "eMMC"
    else:
        storage_type = None
    return capacity, storage_type


def fix_os(product: dict) -> str | None:
    """
    Recalcule l'OS depuis specs_brutes si le champ structure est
    absent. Ajoute apres constat reel : sur le scrape pc_bureau S1,
    `os` etait null pour 100% des produits alors que
    specs_brutes["Système d'exploitation"] contenait des valeurs
    exploitables (ex: "Windows 11") -- meme pattern que fix_brand/
    fix_ram/fix_screen_size, juste applique a une colonne qui ne
    beneficiait pas encore de ce repli.
    """
    if product.get("os") is not None:
        return product["os"]
    specs = product.get("specs_brutes") or {}
    val = _find_spec(specs, "Systeme d'exploitation", "OS", "Operating System")
    if val:
        return str(val).strip()
    return None


def fix_hdr(product: dict) -> bool | None:
    """
    Recalcule le support HDR des televiseurs depuis specs_brutes -- sans
    faire confiance au champ deja present, contrairement aux autres
    fix_*. Le parser (scraper/parser.py::_parse_hdr) avait un bug
    corrige a la source : il ne cherchait "HDR" que dans les VALEURS
    de specs_brutes, jamais dans les CLES, alors que la cle est presque
    toujours "Compatible HDR" avec une valeur "Oui"/"Non" qui ne
    mentionne jamais "HDR" elle-meme. Constat reel sur le scrape deja
    collecte : 88/112 televiseurs etaient a tort hdr=False. Cette
    fonction reproduit la logique corrigee pour les donnees deja
    scrapees (specs_brutes suffit, pas besoin de re-scraper).
    """
    if product.get("categorie") != "televiseurs":
        return product.get("hdr")

    specs = product.get("specs_brutes") or {}
    for key in ("Compatible HDR", "HDR"):
        for spec_key, spec_val in specs.items():
            if _normalize(key) in _normalize(spec_key):
                v = str(spec_val).upper()
                if "OUI" in v or "YES" in v:
                    return True
                if "NON" in v or "NO" in v:
                    return False

    name = product.get("nom") or ""
    all_text = " ".join(f"{k} {v}" for k, v in specs.items()) + " " + name
    return bool(re.search(r'\bHDR\b', all_text, re.IGNORECASE))


def fix_refresh_rate(product: dict) -> float | None:
    """
    Recalcule le taux de rafraichissement des televiseurs depuis
    specs_brutes si absent -- repli supplementaire par rapport au
    parser d'origine : sur 2 fiches, "Taux de rafraichissement : 60 Hz"
    se retrouvait fusionne dans la valeur d'une cle sans rapport
    (ex. "Luminosite (cd/m2)"), un artefact de mise en page cote source.
    Voir scraper/parser.py::_parse_refresh_rate pour la meme logique
    appliquee aux futurs scrapes.
    """
    if product.get("taux_rafraichissement") is not None:
        return product["taux_rafraichissement"]
    if product.get("categorie") != "televiseurs":
        return None

    specs = product.get("specs_brutes") or {}
    for spec_val in specs.values():
        normalized = _normalize(str(spec_val))
        if "rafraichissement" in normalized:
            match = re.search(r'rafraichissement[^\d]*(\d+)\s*hz', normalized)
            if match:
                return float(match.group(1))
    return None


def fix_storage(product: dict, storage_bounds: tuple | None = None):
    """
    Recalcule (stockage_go, type_stockage) depuis specs_brutes si absent,
    et detecte les incoherences entre la capacite FINALEMENT RETENUE
    (final_capacity, celle ecrite dans la ligne) et les sources
    disponibles (nom du produit, specs_brutes).

    Version corrigee : la version initiale comparait uniquement
    capacity_name vs capacity_specs, independamment de la valeur
    reellement utilisee (qui pouvait venir d'un champ deja rempli en
    amont par le parser, capacity_field). Une ligne pouvait donc avoir
    stockage_incoherent_nom_vs_specs=False alors meme que la valeur
    utilisee ne correspondait ni au nom ni aux specs. Desormais le flag
    compare ce qui est ECRIT dans la ligne a chacune des deux sources.

    Args:
        storage_bounds: (lo, hi) de la categorie, ou None si non
            pertinent. Si fourni et que la valeur DEJA PRESENTE dans le
            produit est hors bornes, elle est rejetee avant meme de
            servir de "valeur finale" -- meme logique que fix_ram/
            fix_screen_size : un champ deja rempli par le parser n'est
            plus une source aveuglement prioritaire s'il est aberrant.

    Returns: (stockage_go, type_stockage, stockage_incoherent: bool)
    """
    name = product.get("nom") or ""
    specs = product.get("specs_brutes") or {}

    capacity_field = product.get("stockage_go")
    type_field = product.get("type_stockage")
    if storage_bounds is not None and flag_out_of_range(capacity_field, *storage_bounds):
        capacity_field = None

    spec_val = _find_spec(specs, "Disque Dur", "Stockage", "Capacite", "SSD", "HDD")
    capacity_specs, type_specs = parse_storage_from_text(spec_val)

    capacity_name, type_name = parse_storage_from_text(name)

    # Valeur finale : priorite au champ deja rempli, sinon specs, sinon nom
    final_capacity = capacity_field if capacity_field is not None else (
        capacity_specs if capacity_specs is not None else capacity_name
    )
    final_type = type_field if type_field is not None else (
        type_specs if type_specs is not None else type_name
    )

    # Incoherence : la valeur RETENUE (final_capacity) ne correspond pas
    # a une source disponible qui la contredit (nom ou specs).
    incoherent = final_capacity is not None and (
        (capacity_name is not None and capacity_name != final_capacity)
        or (capacity_specs is not None and capacity_specs != final_capacity)
    )

    return final_capacity, final_type, incoherent


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION PAR BORNES (FLAGS, JAMAIS DE SUPPRESSION AUTOMATIQUE)
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_DATA_CORRECTIONS = {
    # "Smartphone Xiaomi Redmi Note 14 4G 8Go 128Go - Lime Green" : ram_go=128.0
    # dans le JSON scrape (specs_brutes n'a pas de cle "RAM" dediee pour
    # CETTE page precise -- contrairement a ses 7 variantes de couleur
    # identiques -- si bien que fix_ram() retombe sur specs_brutes["Memoire"]
    # = "128Go", qui decrit en realite le STOCKAGE ici). Verifie via 7
    # variantes de couleur du meme modele (Ocean blue, Midnight Black,
    # Purple, Sand Gold...), TOUTES a ram_go=8.0 -- une seule page mytek.tn
    # presente cette anomalie de mise en page, ce n'est pas un cas
    # systemique (fix_ram calcule correctement "Memoire"->RAM sur 935/1016
    # produits testes) : une correction generale du candidat "Memoire"
    # casserait ces 935 cas pour corriger 1 seul. Correction ponctuelle et
    # documentee, plutot qu'une heuristique d'extraction plus large et plus
    # risquee pour une anomalie confirmee isolee.
    "https://www.mytek.tn/smartphone-xiaomi-redmi-note-14-8go-128go-lime-vert.html": {"ram_go": 8.0},
}


def apply_known_corrections(product: dict) -> dict:
    """Applique KNOWN_DATA_CORRECTIONS (cf. sa docstring) sur une copie du
    produit AVANT toute autre regle de nettoyage -- une correction
    ponctuelle, verifiee manuellement, prime sur la valeur scrapee."""
    correction = KNOWN_DATA_CORRECTIONS.get(product.get("url"))
    if not correction:
        return product
    corrected = dict(product)
    corrected.update(correction)
    return corrected


def flag_out_of_range(value, lo, hi) -> bool:
    """True si la valeur est hors plage plausible (ou None -> False, on
    ne flague pas l'absence de donnee, seulement les valeurs aberrantes).
    Si lo/hi est None (variable non pertinente pour la categorie, ex:
    ecran sur un desktop), retourne toujours False."""
    if value is None or lo is None or hi is None:
        return False
    try:
        return not (lo <= float(value) <= hi)
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def clean_products(raw_products: list) -> pd.DataFrame:
    """
    Applique l'ensemble des regles de nettoyage a la liste de produits
    bruts et retourne un DataFrame pret a l'export CSV.
    """
    rows = []

    for product in raw_products:
        product = apply_known_corrections(product)
        category = product.get("categorie")
        bounds = get_bounds(category)

        price_lo, price_hi = bounds["price"]
        ram_lo, ram_hi = bounds["ram"] if bounds["ram"] is not None else (None, None)
        storage_lo, storage_hi = bounds["storage"] if bounds["storage"] is not None else (None, None)
        screen_bounds = bounds["screen"]  # peut etre None

        brand = fix_brand(product)

        # RAM/ecran/stockage : si la valeur DEJA PRESENTE dans le produit
        # est hors bornes, on la rejette (seed=None) plutot que de lui
        # faire confiance aveuglement -- force fix_* a retenter une
        # derivation depuis specs_brutes/nom. C'est ce qui corrige
        # retroactivement le bug cache/RAM (0.0039 Go au lieu de 8 Go)
        # deja corrige a la source dans scraper/parser.py mais toujours
        # present dans les JSON deja scrapes.
        raw_ram = product.get("ram_go")
        ram_seed = None if flag_out_of_range(raw_ram, ram_lo, ram_hi) else raw_ram
        ram = fix_ram(product, seed=ram_seed)

        raw_screen = product.get("taille_ecran")
        screen_lo_hi = screen_bounds if screen_bounds is not None else (None, None)
        screen_seed = None if flag_out_of_range(raw_screen, *screen_lo_hi) else raw_screen
        screen = fix_screen_size(product, category=category, seed=screen_seed)

        storage_bounds = (storage_lo, storage_hi) if storage_lo is not None else None
        storage_go, storage_type, storage_incoherent = fix_storage(product, storage_bounds=storage_bounds)

        os_value = fix_os(product)

        # Prix : c'est la variable CIBLE, on ne peut pas la "recalculer"
        # depuis une autre source comme la RAM/l'ecran -- une valeur hors
        # bornes est mise a None (la ligne sera ecartee faute de cible
        # fiable, cf. select_features/pipeline) plutot que conservee.
        # prix_suspect doit etre calcule AVANT le rejet : une fois price
        # mis a None, flag_out_of_range(None, ...) est toujours False, ce
        # qui rendrait le flag muet sur les valeurs qu'on vient de rejeter.
        price = product.get("prix_tnd")
        price_suspect = flag_out_of_range(price, price_lo, price_hi)
        if price_suspect:
            price = None

        row = {
            "gtin": extract_gtin(product),
            "dedup_key": dedup_key(product),
            "url": product.get("url"),
            "nom": product.get("nom"),
            "marque": brand,
            "prix_tnd": price,
            "prix_suspect": price_suspect,
            "processeur": product.get("processeur"),
            "ram_go": ram,
            "ram_suspecte": flag_out_of_range(ram, ram_lo, ram_hi),
            "stockage_go": storage_go,
            "stockage_suspect": flag_out_of_range(storage_go, storage_lo, storage_hi),
            "type_stockage": storage_type,
            "stockage_incoherent_nom_vs_specs": storage_incoherent,
            "taille_ecran": screen,
            "ecran_suspect": (
                flag_out_of_range(screen, *screen_bounds)
                if screen_bounds is not None else False
            ),
            "os": os_value,
            "connectivite": product.get("connectivite"),
            "categorie": category,
            "semaine": product.get("semaine"),
            "date_collecte": product.get("date_collecte"),
            "specs_brutes": (
                json.dumps(product["specs_brutes"], ensure_ascii=False)
                if product.get("specs_brutes") else None
            ),
            # Champs specifiques aux televiseurs (absents/None pour les
            # autres categories -- le parser ne les ajoute qu'a la ligne
            # 610 de parser.py pour category == "televiseurs").
            "resolution_affichage": product.get("resolution_affichage"),
            "technologie_dalle": product.get("technologie_dalle"),
            "hdr": fix_hdr(product),
            "smart_tv": product.get("smart_tv"),
            "taux_rafraichissement": fix_refresh_rate(product),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"DataFrame initial : {len(df)} ligne(s).")

    if df.empty:
        # Aucun produit (ex: scrape vide pour une categorie/semaine, ou
        # echec reseau en amont) -- retourne un DataFrame vide plutot
        # que de planter sur df["date_collecte"] (colonne inexistante
        # quand rows est vide).
        logger.warning("Aucun produit a nettoyer -- DataFrame vide retourne.")
        return df

    # ── Normalisation des marques ────────────────────────────────────────────
    # A l'echelle du DataFrame (pas ligne par ligne) : la casse canonique
    # se decide par frequence globale (cf. normalize_brand_column).
    before_unique = df["marque"].nunique()
    df["marque"] = normalize_brand_column(df["marque"])
    after_unique = df["marque"].nunique()
    if after_unique < before_unique:
        logger.info(
            f"Normalisation des marques : {before_unique} -> {after_unique} "
            f"valeurs distinctes ({before_unique - after_unique} doublon(s) "
            f"de casse/gamme fusionne(s))."
        )

    # ── Deduplication ──────────────────────────────────────────────────────
    # Cle = (GTIN ou URL) + semaine. On garde la ligne la plus recente
    # (date_collecte) en cas de doublon exact sur dedup_key -- c'est-a-dire
    # le meme produit re-scrape PLUSIEURS FOIS LA MEME SEMAINE (ex:
    # scheduler relance par erreur), jamais a travers des semaines
    # differentes (qui sont des observations legitimement distinctes du
    # panel temporel).
    df["date_collecte"] = pd.to_datetime(df["date_collecte"], errors="coerce")
    before = len(df)
    df = (
        df.sort_values("date_collecte")
        .drop_duplicates(subset="dedup_key", keep="last")
        .reset_index(drop=True)
    )
    removed = before - len(df)
    logger.info(
        f"Deduplication par (GTIN/URL, semaine) : {removed} doublon(s) "
        f"supprime(s) ({len(df)} produit(s) unique(s) restant(s))."
    )

    # ── Tri final pour lisibilite ───────────────────────────────────────────
    df = df.sort_values(["categorie", "semaine", "marque", "prix_tnd"], na_position="last")
    df = df.reset_index(drop=True)

    return df


def write_summary(df: pd.DataFrame) -> None:
    """Affiche un resume des flags de qualite pour revue manuelle."""
    logger.info("─" * 60)
    logger.info("RESUME QUALITE")
    logger.info(f"  Total produits uniques (toutes semaines) : {len(df)}")
    logger.info(f"  Prix suspects               : {df['prix_suspect'].sum()}")
    logger.info(f"  RAM suspecte                : {df['ram_suspecte'].sum()}")
    logger.info(f"  Stockage suspect            : {df['stockage_suspect'].sum()}")
    logger.info(f"  Ecran suspect                : {df['ecran_suspect'].sum()}")
    logger.info(
        f"  Stockage incoherent nom/specs : "
        f"{df['stockage_incoherent_nom_vs_specs'].sum()}"
    )
    logger.info("  Repartition par categorie x semaine :")
    if "semaine" in df.columns:
        for (cat, week), count in df.groupby(["categorie", "semaine"]).size().items():
            logger.info(f"    {cat} / semaine {week}: {count}")
    logger.info("  Valeurs manquantes par colonne :")
    for col in ["marque", "prix_tnd", "ram_go", "stockage_go", "taille_ecran", "os", "connectivite"]:
        n_null = df[col].isna().sum()
        pct = 100 * n_null / len(df) if len(df) else 0
        logger.info(f"    {col}: {n_null}/{len(df)} ({pct:.0f}%)")
    logger.info("─" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTREE CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Nettoyage des donnees brutes Mytek.tn -> CSV consolide."
    )
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw",
        help="Repertoire contenant les fichiers JSON bruts (defaut: data/raw)",
    )
    parser.add_argument(
        "--out", type=str, default="data/processed/produits_clean.csv",
        help="Chemin du fichier CSV de sortie "
             "(defaut: data/processed/produits_clean.csv)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)

    if not raw_dir.exists():
        logger.error(f"Le repertoire {raw_dir} n'existe pas.")
        sys.exit(1)

    logger.info(f"Chargement des fichiers JSON depuis {raw_dir} ...")
    raw_products = load_raw_files(raw_dir)
    if not raw_products:
        logger.error("Aucun produit charge, arret.")
        sys.exit(1)

    logger.info("Nettoyage en cours ...")
    df = clean_products(raw_products)

    write_summary(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV ecrit : {out_path} ({len(df)} lignes)")


if __name__ == "__main__":
    main()