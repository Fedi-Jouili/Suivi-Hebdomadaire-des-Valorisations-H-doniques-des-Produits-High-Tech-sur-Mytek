# -*- coding: utf-8 -*-
"""
=============================================================================
Package : src/utils
=============================================================================
ROLE :
    Utilitaires transverses partages par src/preprocessing, src/models et
    src/scraper -- config.py (chemins, categories, seed) et logger.py
    (logging homogene), pour eviter que chaque module redefinisse ses
    propres constantes/handlers (cf. RANDOM_STATE, CATEGORY_ORDER, format
    de log auparavant repetes independamment dans plusieurs fichiers).
=============================================================================
"""

from .config import (
    PROJECT_ROOT,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    DATA_COMPARISONS_DIR,
    CATEGORY_ORDER,
    CATEGORY_LABELS,
    RANDOM_STATE,
    TEST_SIZE,
)
from .logger import get_logger

__all__ = [
    "PROJECT_ROOT",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "DATA_COMPARISONS_DIR",
    "CATEGORY_ORDER",
    "CATEGORY_LABELS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "get_logger",
]
