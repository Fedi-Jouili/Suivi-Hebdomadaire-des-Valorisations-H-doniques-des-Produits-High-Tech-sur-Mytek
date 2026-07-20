# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/scraper/__init__.py
=============================================================================
RÔLE :
    Point d'entrée du package scraper. Ce fichier centralise :
    - Les imports des classes principales (MytekSpider, ProductParser,
      ScrapingScheduler) pour un accès simplifié depuis l'extérieur.
    - Les constantes globales partagées par tous les modules du scraper
      (URLs des catégories, domaine de base, délais, configuration du
      navigateur Brave).
    - La configuration du système de logging pour tracer l'avancement
      et les erreurs durant le scraping.

UTILISATION :
    from src.scraper import MytekSpider, ProductParser, ScrapingScheduler
    from src.scraper import CATEGORIES, BASE_URL
=============================================================================
"""
import os
import logging
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES GLOBALES
# ─────────────────────────────────────────────────────────────────────────────

# Domaine de base du site cible
BASE_URL = "https://www.mytek.tn"

# Catégories de produits à scraper avec leurs URLs et noms de fichiers
# Chaque entrée : clé = identifiant interne, valeur = dict(url, label, filename)
CATEGORIES = {
    "pc_portables": {
        "url": f"{BASE_URL}/informatique/ordinateurs-portables/pc-portable.html",
        "label": "PC Portables",
    },
    "pc_bureau": {
        "url": f"{BASE_URL}/informatique/ordinateur-de-bureau/pc-de-bureau.html",
        "label": "PC de Bureau",
    },
    "smartphones": {
        "url": f"{BASE_URL}/smartphone.html",
        "label": "Smartphones",
    },
    "telephones_portables": {
        "url": f"{BASE_URL}/telephonie-tunisie/smartphone-mobile-tunisie/telephone-portable.html",
        "label": "Téléphones Portables",
    },
    "televiseurs": {
        "url": f"{BASE_URL}/image-son/televiseurs.html",
        "label": "Téléviseurs",
    },
}

# Délai entre les requêtes (en secondes) — fourchette min/max
# Respecte le serveur et réduit le risque de blocage
DELAY_MIN = 2.0
DELAY_MAX = 5.0

# Chemin vers l'exécutable Brave (utilisé par StealthyFetcher si nécessaire)
# Adapter ce chemin si Brave est installé ailleurs sur votre machine
BRAVE_PATH = os.environ.get(
    "BRAVE_PATH",
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe"
)

# Répertoire racine pour les données brutes
DATA_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
KNOWN_BRANDS = [
    "HP", "Dell", "Lenovo", "ASUS", "Acer", "MSI", "Apple",
    "Samsung", "Huawei", "Xiaomi", "OPPO", "Realme", "Infinix",
    "Nokia", "Itel", "Tecno", "iPhone", "Toshiba", "LG", "Sony",
    "Alcatel", "TCL", "Honor", "Vivo", "ZTE", "Condor", "Evertek",
    # Marques TV
    "Philips", "Hisense", "Panasonic", "Sharp", "Grundig",
    "Thomson", "Haier", "Skyworth", "Changhong",
]

INTER_CAT_DELAY_MIN = 8.0
INTER_CAT_DELAY_MAX = 15.0

# Nombre maximum de tentatives par requête en cas d'erreur réseau
MAX_RETRIES = 3

# User-Agent réaliste pour les requêtes HTTP
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DU LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level=logging.INFO):
    """
    Configure le système de logging pour le package scraper.
    
    - Affiche les messages dans la console (stdout)
    - Format : [HEURE] NIVEAU | NOM_MODULE — message
    - Niveau par défaut : INFO (peut être changé en DEBUG pour plus de détails)
    """
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    # Logger racine du package scraper
    logger = logging.getLogger("scraper")
    logger.setLevel(level)
    
    # Éviter les doublons si setup_logging est appelé plusieurs fois
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


# Initialiser le logging au chargement du package
if __name__ == "__main__":
    setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS DES CLASSES PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "CATEGORIES",
    "BASE_URL",
    "DATA_RAW_DIR",
    "BRAVE_PATH",
    "DELAY_MIN",
    "DELAY_MAX",
    "MAX_RETRIES",
    "USER_AGENT",
    "KNOWN_BRANDS",
]
