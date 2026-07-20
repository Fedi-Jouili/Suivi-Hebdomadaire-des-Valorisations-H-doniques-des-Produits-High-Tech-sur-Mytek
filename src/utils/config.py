# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/utils/config.py
=============================================================================
ROLE :
    Source UNIQUE des constantes partagees par le projet : chemins de
    donnees (pathlib, relatifs a la racine du depot -- jamais de chemin
    absolu type C:\\Users\\...), liste/libelles des 5 categories, seed de
    reproductibilite, et parametres de split train/test.

    Avant ce module, RANDOM_STATE=42, CATEGORY_ORDER et la palette par
    categorie etaient redefinis independamment dans plusieurs notebooks et
    dans src/models/hedonic_model.py -- ce module centralise ces valeurs
    pour src/preprocessing/split.py (et pour toute future consolidation des
    autres modules).
=============================================================================
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS (relatifs a la racine du depot, jamais absolus)
# ─────────────────────────────────────────────────────────────────────────────

# Ce fichier vit dans <racine>/src/utils/config.py -> .parents[2] = <racine>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_COMPARISONS_DIR = DATA_DIR / "comparisons"

# ─────────────────────────────────────────────────────────────────────────────
# CATEGORIES (ordre et libelles stables, reutilises par tous les notebooks)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_ORDER = ["pc_bureau", "pc_portables", "smartphones", "telephones_portables", "televiseurs"]

CATEGORY_LABELS = {
    "pc_bureau": "PC de Bureau",
    "pc_portables": "PC Portables",
    "smartphones": "Smartphones",
    "telephones_portables": "Téléphones Portables",
    "televiseurs": "Téléviseurs",
}

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCTIBILITE ET SPLIT TRAIN/TEST
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_STATE = 42  # seed unique du projet -- KMeans, train_test_split, etc.
TEST_SIZE = 0.20   # part reservee au test lors du split train/test par semaine
