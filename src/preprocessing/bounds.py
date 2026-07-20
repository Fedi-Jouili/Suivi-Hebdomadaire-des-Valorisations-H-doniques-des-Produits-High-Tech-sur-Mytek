# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/bounds.py
=============================================================================
ROLE :
    Calibration des bornes de plausibilite (prix, RAM, stockage, ecran)
    utilisees par clean.py pour flaguer les valeurs suspectes.

    Separe volontairement de clean.py : clean.py APPLIQUE des bornes
    deja figees (constantes), ce module sert a les DERIVER a partir des
    donnees observees plutot que de les choisir a la main. Les deux
    etapes sont decouplees pour que clean_products() ne depende jamais
    d'un DataFrame de calibration pour fonctionner.

WORKFLOW TYPIQUE :
    1. Charger un premier passage de donnees brutes/peu nettoyees
       (toutes categories, idealement plusieurs semaines).
    2. Appeler compute_bounds_from_data() par categorie et par variable
       (prix, ram, stockage, ecran).
    3. Inspecter manuellement les bornes proposees (sanity check metier
       -- ex: un mini PC a 79 TND est-il plausible ? oui).
    4. Figer les bornes retenues en constantes dans clean.py
       (VALIDITY_BOUNDS), avec un commentaire indiquant la date/semaine
       source de la calibration.
    5. Recalibrer ponctuellement quand de nouvelles semaines/categories
       arrivent, sans jamais modifier clean.py "a l'aveugle".

UTILISATION :
    from src.preprocessing.bounds import compute_bounds_from_data
    bounds = compute_bounds_from_data(df, column="prix_tnd")
=============================================================================
"""

import logging

import pandas as pd

logger = logging.getLogger("preprocessing.bounds")

MIN_OBSERVATIONS = 5


def compute_bounds_from_data(
    df: pd.DataFrame,
    column: str,
    by: str = "categorie",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    pad_factor: float = 0.5,
    min_observations: int = MIN_OBSERVATIONS,
) -> dict:
    """
    Calcule des bornes de plausibilite par groupe (categorie) a partir
    des quantiles observes dans les donnees, plutot que des constantes
    choisies a la main.
    """
    if column not in df.columns:
        raise ValueError(f"Colonne '{column}' absente du DataFrame.")
    if by not in df.columns:
        raise ValueError(f"Colonne de groupement '{by}' absente du DataFrame.")

    bounds = {}
    for group_value, group_df in df.groupby(by):
        values = pd.to_numeric(group_df[column], errors="coerce").dropna()

        if len(values) < min_observations:
            logger.warning(
                f"  '{group_value}' / {column} : seulement {len(values)} "
                f"observation(s) (< {min_observations}) -- ignore, "
                f"a calibrer manuellement."
            )
            continue

        lo = values.quantile(lower_q)
        hi = values.quantile(upper_q)
        span = hi - lo

        padded_lo = max(0.0, lo - pad_factor * span)
        padded_hi = hi + pad_factor * span

        bounds[group_value] = (round(float(padded_lo), 2), round(float(padded_hi), 2))

        logger.info(
            f"  '{group_value}' / {column} : n={len(values)}, "
            f"observe=[{values.min():.1f}, {values.max():.1f}], "
            f"quantiles[{lower_q}-{upper_q}]=[{lo:.1f}, {hi:.1f}], "
            f"bornes proposees=[{padded_lo:.1f}, {padded_hi:.1f}]"
        )

    return bounds


def compute_all_bounds(
    df: pd.DataFrame,
    columns: list = None,
    by: str = "categorie",
    **kwargs,
) -> dict:
    """
    Calibre les bornes pour plusieurs colonnes en une seule passe,
    pratique pour generer d'un coup la structure VALIDITY_BOUNDS
    attendue par clean.py.
    """
    if columns is None:
        columns = ["prix_tnd", "ram_go", "stockage_go", "taille_ecran"]

    result = {}
    for column in columns:
        if column not in df.columns:
            logger.warning(f"Colonne '{column}' absente, ignoree.")
            continue
        logger.info(f"Calibration de '{column}' par '{by}' :")
        result[column] = compute_bounds_from_data(df, column, by=by, **kwargs)

    return result


__all__ = ["compute_bounds_from_data", "compute_all_bounds", "MIN_OBSERVATIONS"]