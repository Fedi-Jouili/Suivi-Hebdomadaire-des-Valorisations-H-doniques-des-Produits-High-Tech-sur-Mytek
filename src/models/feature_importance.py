# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/models/feature_importance.py
=============================================================================
ROLE :
    Compare les coefficients Ridge (semi-elasticites) et les importances
    Random Forest (reduction d'impurete) sur un meme referentiel de
    variables, pour croiser un modele parametrique interpretable et un
    modele non-parametrique plus flexible.

NOTE IMPORTANTE :
    RidgeModel travaille sur des caracteristiques POLYNOMIALEMENT
    ETENDUES (RAM, stockage, mais aussi "RAM x stockage", etc. -- cf.
    ridge_model.py), alors que RandomForestModel travaille sur les
    caracteristiques D'ORIGINE uniquement (pas d'expansion polynomiale
    dans ce module). L'ensemble des variables de Ridge est donc un
    SUR-ENSEMBLE de celui de Random Forest : la fusion (outer join)
    laissera naturellement `rf_importance` a NaN pour les termes
    d'interaction propres a Ridge (ex: "ram_go stockage_go"), puisque
    Random Forest n'a jamais vu cette variable en tant que colonne
    explicite -- ce n'est pas une erreur, c'est attendu.
=============================================================================
"""

import pandas as pd


def compare_importances(ridge_model, rf_model, feature_names: list) -> pd.DataFrame:
    """
    Fusionne les coefficients Ridge et les importances Random Forest dans
    un DataFrame unique, pour comparaison directe variable par variable.

    Args:
        ridge_model: instance de RidgeModel deja ajustee (fit() appele).
        rf_model: instance de RandomForestModel deja ajustee (fit() appele).
        feature_names: noms des colonnes ORIGINALES de X (avant
            expansion polynomiale), dans le meme ordre que celui utilise
            lors du fit() des deux modeles.

    Returns: DataFrame ["feature", "ridge_coef", "rf_importance"], trie
        par |ridge_coef| decroissant. Les termes d'interaction propres a
        Ridge (absents de Random Forest) ont rf_importance = NaN --
        cf. note ci-dessus.
    """
    ridge_df = ridge_model.get_coefficients(feature_names).rename(columns={"coefficient": "ridge_coef"})
    rf_df = rf_model.get_importances(feature_names).rename(columns={"importance": "rf_importance"})

    merged = pd.merge(ridge_df, rf_df, on="feature", how="outer")
    return merged.sort_values("ridge_coef", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
