# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/impute.py
=============================================================================
ROLE :
    Cascade d'imputation des valeurs manquantes, TOUJOURS operee PAR
    CATEGORIE (les "produits similaires" d'un telephone sont d'autres
    telephones, jamais un PC de bureau) :

      1. Recuperation par regle -- deja faite en amont, avant d'atteindre
         ce module (clean.py : fix_ram/fix_screen_size/fix_storage/fix_os
         tentent specs_brutes puis le nom du produit).
      2. Imputation par similarite (KNN) sur les valeurs encore
         manquantes -- se base sur les AUTRES caracteristiques du produit
         plutot qu'une simple mediane plate (cf. impute_numeric_cascade
         pour le detail numerique, impute_categorical_by_neighbors pour
         le detail categoriel).
      3. Repli sur mediane/mode de la categorie quand une colonne n'a pas
         assez de valeurs connues pour qu'un KNN soit pertinent (seuil
         documente : MIN_NON_NULL_FOR_KNN), toujours logge -- jamais
         silencieux.

    Le prix (variable CIBLE) n'est jamais impute ici : une ligne sans
    prix fiable est ecartee en amont (cf. pipeline.py), pas devinee.

UTILISATION :
    from src.preprocessing.impute import impute_numeric_cascade, impute_categorical_by_neighbors
=============================================================================
"""

import logging

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("preprocessing.impute")

MIN_NON_NULL_FOR_KNN = 10
DEFAULT_N_NEIGHBORS = 5


def _neighbor_feature_matrix(df: pd.DataFrame, idx, neighbor_columns: list) -> pd.DataFrame:
    """
    Matrice numerique (memes lignes que idx) pour la recherche de
    voisins : colonnes coercees en numerique, NaN remplaces par la
    mediane de la colonne -- NearestNeighbors ne tolere aucun NaN.
    """
    cols = [c for c in neighbor_columns if c in df.columns]
    sub = df.loc[idx, cols].apply(pd.to_numeric, errors="coerce")
    return sub.fillna(sub.median())


def impute_numeric_cascade(
    df: pd.DataFrame,
    target_columns: list,
    neighbor_columns: list,
    category_col: str = "categorie",
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_non_null: int = MIN_NON_NULL_FOR_KNN,
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de `target_columns` (numeriques), par
    categorie, via KNNImputer (sklearn) sur `target_columns +
    neighbor_columns` -- typiquement les autres variables numeriques du
    produit. `neighbor_columns` NE DOIT JAMAIS inclure prix_tnd : imputer
    une caracteristique a partir du prix qui servira plus tard a PREDIRE
    ce meme prix est une fuite de donnees legere cote regression (sans
    consequence pour du clustering, mais on exclut par defaut pour que
    le meme jeu de donnees serve aux deux usages sans mise en garde
    particuliere -- cf. plan de la tache).

    Repli sur la mediane de la categorie si une colonne cible a moins de
    `min_non_null` valeurs connues dans cette categorie -- un KNN n'a
    pas plus de sens qu'une mediane avec si peu de signal.

    Returns: DataFrame avec les colonnes de target_columns completees.
    """
    df = df.copy()

    for cat, group in df.groupby(category_col):
        idx = group.index

        cols_for_knn = []
        for col in target_columns:
            if col not in df.columns:
                continue
            series = df.loc[idx, col]
            n_non_null = series.notna().sum()
            n_missing = series.isna().sum()
            if n_missing == 0 or n_non_null == 0:
                continue

            if n_non_null < min_non_null:
                median = series.median()
                df.loc[idx, col] = series.fillna(median)
                logger.info(
                    f"  [{cat}] '{col}' : repli mediane ({median:.2f}) pour "
                    f"{n_missing} ligne(s) -- seulement {n_non_null} valeur(s) "
                    f"connue(s) (< {min_non_null}), KNN non pertinent."
                )
                continue

            cols_for_knn.append((col, n_missing, n_non_null))

        if not cols_for_knn:
            continue

        feature_cols = list(dict.fromkeys(
            [c for c, _, _ in cols_for_knn] + [c for c in neighbor_columns if c in df.columns]
        ))
        sub = df.loc[idx, feature_cols].apply(pd.to_numeric, errors="coerce")

        min_non_null_in_group = min(n for _, _, n in cols_for_knn)
        k = min(n_neighbors, max(1, min_non_null_in_group - 1))
        imputer = KNNImputer(n_neighbors=k)
        imputed = pd.DataFrame(imputer.fit_transform(sub), index=idx, columns=feature_cols)

        for col, n_missing, _ in cols_for_knn:
            df.loc[idx, col] = imputed[col]
            logger.info(
                f"  [{cat}] '{col}' : {n_missing} valeur(s) imputee(s) par "
                f"similarite (KNN, k={k})."
            )

    return df


def impute_categorical_by_neighbors(
    df: pd.DataFrame,
    target_columns: list,
    neighbor_columns: list,
    category_col: str = "categorie",
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
) -> pd.DataFrame:
    """
    Impute les valeurs manquantes de colonnes CATEGORIELLES par le mode
    des k plus proches voisins (memes `neighbor_columns` numeriques que
    impute_numeric_cascade, distance euclidienne standard) -- sklearn
    n'a pas d'imputeur "mode des voisins" pret a l'emploi pour du
    categoriel, d'ou cette implementation dediee.

    Repli sur le mode global de la categorie si aucune valeur connue
    n'existe dans le voisinage (cas limite, ex: colonne quasi-integralement
    manquante pour cette categorie).
    """
    df = df.copy()

    for cat, group in df.groupby(category_col):
        idx = group.index
        feature_matrix = _neighbor_feature_matrix(df, idx, neighbor_columns)
        if feature_matrix.shape[1] == 0:
            continue

        for col in target_columns:
            if col not in df.columns:
                continue
            missing = df.loc[idx, col].isna()
            n_missing = int(missing.sum())
            if n_missing == 0:
                continue

            known_idx = idx[~missing.to_numpy()]
            if len(known_idx) == 0:
                continue  # rien de connu dans toute la categorie -- laisse tel quel

            fallback_mode = df.loc[idx, col].mode(dropna=True)
            fallback_mode = fallback_mode.iloc[0] if not fallback_mode.empty else None

            k = min(n_neighbors, len(known_idx))
            nn = NearestNeighbors(n_neighbors=k).fit(feature_matrix.loc[known_idx])

            missing_idx = idx[missing.to_numpy()]
            _, neighbor_positions = nn.kneighbors(feature_matrix.loc[missing_idx])
            known_idx_array = known_idx.to_numpy()

            for row_id, positions in zip(missing_idx, neighbor_positions):
                neighbor_labels = df.loc[known_idx_array[positions], col]
                mode = neighbor_labels.mode(dropna=True)
                df.loc[row_id, col] = mode.iloc[0] if not mode.empty else fallback_mode

            logger.info(
                f"  [{cat}] '{col}' : {n_missing} valeur(s) imputee(s) par mode "
                f"des {k} plus proche(s) voisin(s)."
            )

    return df


__all__ = [
    "impute_numeric_cascade",
    "impute_categorical_by_neighbors",
    "MIN_NON_NULL_FOR_KNN",
    "DEFAULT_N_NEIGHBORS",
]
