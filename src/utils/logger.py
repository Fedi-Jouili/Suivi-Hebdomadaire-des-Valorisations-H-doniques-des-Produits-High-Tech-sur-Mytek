# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/utils/logger.py
=============================================================================
ROLE :
    Fabrique un logger configure de maniere homogene (meme format que
    src/preprocessing/clean.py, pipeline.py et src/scraper/scheduler.py :
    "[%(asctime)s] %(levelname)-8s | %(message)s"), pour que les nouveaux
    modules n'aient plus a repeter leur propre logging.basicConfig et
    n'utilisent jamais de print() bruts pour tracer ce qu'ils font.

UTILISATION :
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("...")
=============================================================================
"""

import logging

_LOG_FORMAT = "[%(asctime)s] %(levelname)-8s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Retourne un logger nomme `name`, avec un unique StreamHandler formate.

    Idempotent : un appel repete avec le meme `name` (ex: import multiple
    du module appelant) ne duplique jamais les handlers -- sans cette
    garde, chaque re-import afficherait les messages en plusieurs
    exemplaires.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
