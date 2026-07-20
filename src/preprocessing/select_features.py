# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/select_features.py
=============================================================================
ROLE :
    Selection des colonnes finales par categorie pour la modelisation/le
    clustering ("moins de colonnes, seulement celles qui comptent pour le
    prix"). Combine :

      1. Des drops STRUCTURELS -- faits sur le ROLE de la colonne, pas sur
         son pouvoir predictif : identifiants/metadonnees (gtin, url...),
         colonnes remplacees par une version derivee plus exploitable
         (processeur brut -> cpu_brand/serie/gen, os brut -> os_platform),
         et colonnes CONSTANTES (nunique<=1 -- ex: semaine=1 tant qu'une
         seule semaine de scrape existe, type_stockage quasi-constant/
         100% manquant selon la categorie). Ce dernier cas est verifie
         DYNAMIQUEMENT sur les donnees, pas code en dur : semaine
         redeviendra automatiquement une feature des que S2/S3 existeront.
      2. Un test statistique d'effet sur log(prix_tnd), colonne par
         colonne, PAR CATEGORIE : correlation de Spearman pour le
         numerique, Kruskal-Wallis pour le categoriel (non-parametrique,
         adapte a la distribution de prix asymetrique deja etablie dans
         l'EDA). Rapporte de facon transparente (jamais un filtrage
         silencieux) via compute_effect_sizes().

UTILISATION :
    from src.preprocessing.select_features import select_features_for_category
    kept_columns, report = select_features_for_category(df_pc_bureau, force_keep={"ram_go", "marque"})
=============================================================================
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("preprocessing.select_features")

# Colonnes toujours retirees des FEATURES -- metadonnees, identifiants, ou
# remplacees par une version derivee plus exploitable. Fait structurel sur
# le role de la colonne, independant de tout test statistique.
ALWAYS_DROP = {
    "gtin", "dedup_key", "categorie", "semaine", "date_collecte",
    "specs_brutes", "processeur", "connectivite", "os",
    # Flags d'audit : deja "consommes" au moment ou clean.py corrige les
    # valeurs hors bornes (elles deviennent la trace de ce qui a ete
    # corrige, pas une variable a modeliser) -- retires du jeu final.
    "prix_suspect", "ram_suspecte", "stockage_suspect", "ecran_suspect",
    "stockage_incoherent_nom_vs_specs",
}

# Colonnes toujours gardees, jamais soumises au test d'effet : identifiants
# de tracabilite (non consommes comme features par un modele) et cible.
ALWAYS_KEEP = {"url", "nom", "prix_tnd"}

NUMERIC_SPEARMAN_THRESHOLD = 0.10
CATEGORICAL_PVALUE_THRESHOLD = 0.05
MIN_OBSERVATIONS_FOR_TEST = 5


def _is_constant(series: pd.Series) -> bool:
    return series.dropna().nunique() <= 1


def compute_effect_sizes(
    df: pd.DataFrame,
    candidate_columns: list,
    target_col: str = "prix_tnd",
) -> pd.DataFrame:
    """
    Calcule un score d'effet de chaque colonne candidate sur
    log1p(target_col) :
      - numerique  -> |correlation de Spearman|, seuil NUMERIC_SPEARMAN_THRESHOLD
      - categoriel -> Kruskal-Wallis (p-value), seuil CATEGORICAL_PVALUE_THRESHOLD

    Returns: DataFrame indexe par nom de colonne, avec les colonnes
             type/effect/p_value/recommandation/raison -- toujours
             consultable pour comprendre POURQUOI une colonne a ete
             gardee/ecartee (jamais un filtrage silencieux).
    """
    rows = []
    log_target = np.log1p(df[target_col].astype(float))

    for col in candidate_columns:
        if col not in df.columns:
            continue
        series = df[col]

        if _is_constant(series):
            rows.append({
                "colonne": col, "type": "constant", "effect": 0.0, "p_value": None,
                "recommandation": "ecarter",
                "raison": "colonne constante (nunique<=1) -- aucune information",
            })
            continue

        if pd.api.types.is_numeric_dtype(series):
            valid = series.notna() & log_target.notna()
            if valid.sum() < MIN_OBSERVATIONS_FOR_TEST:
                rows.append({
                    "colonne": col, "type": "numerique", "effect": None, "p_value": None,
                    "recommandation": "ecarter",
                    "raison": f"moins de {MIN_OBSERVATIONS_FOR_TEST} valeur(s) non manquante(s)",
                })
                continue
            corr = series[valid].corr(log_target[valid], method="spearman")
            corr = 0.0 if pd.isna(corr) else float(corr)
            keep = abs(corr) >= NUMERIC_SPEARMAN_THRESHOLD
            rows.append({
                "colonne": col, "type": "numerique", "effect": round(corr, 3), "p_value": None,
                "recommandation": "garder" if keep else "ecarter",
                "raison": f"|Spearman|={abs(corr):.3f} {'>=' if keep else '<'} {NUMERIC_SPEARMAN_THRESHOLD}",
            })
        else:
            groups = [
                log_target[(series == val) & log_target.notna()]
                for val in series.dropna().unique()
            ]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) < 2:
                rows.append({
                    "colonne": col, "type": "categoriel", "effect": None, "p_value": None,
                    "recommandation": "ecarter",
                    "raison": "moins de 2 groupes exploitables (n>=2 chacun)",
                })
                continue
            try:
                h_stat, p_value = stats.kruskal(*groups)
            except ValueError as exc:
                rows.append({
                    "colonne": col, "type": "categoriel", "effect": None, "p_value": None,
                    "recommandation": "ecarter", "raison": f"test impossible ({exc})",
                })
                continue
            keep = p_value < CATEGORICAL_PVALUE_THRESHOLD
            rows.append({
                "colonne": col, "type": "categoriel", "effect": round(float(h_stat), 2),
                "p_value": round(float(p_value), 4),
                "recommandation": "garder" if keep else "ecarter",
                "raison": f"Kruskal-Wallis p={p_value:.4f} {'<' if keep else '>='} {CATEGORICAL_PVALUE_THRESHOLD}",
            })

    return pd.DataFrame(rows).set_index("colonne") if rows else pd.DataFrame(
        columns=["type", "effect", "p_value", "recommandation", "raison"]
    )


def select_features_for_category(
    df_category: pd.DataFrame,
    target_col: str = "prix_tnd",
    force_keep: set | None = None,
) -> tuple:
    """
    Determine les colonnes finales pour UNE categorie deja isolee
    (df_category ne doit contenir que les lignes de cette categorie).

    force_keep : colonnes gardees inconditionnellement meme si l'effet
    mesure est faible -- typiquement les variables hedoniques etablies
    par la theorie de Lancaster/Rosen et deja confirmees par l'EDA
    (ram_go, stockage_go, taille_ecran, marque). Le test d'effet sert
    surtout a reperer les colonnes structurellement inutiles (semaine,
    type_stockage) ou a arbitrer les colonnes DERIVEES optionnelles
    (has_4g, has_ethernet...), pas a remettre en cause les fondamentaux
    du modele hedonique.

    Returns: (colonnes_gardees: list[str], rapport: pd.DataFrame)
    """
    force_keep = force_keep or set()
    candidate_columns = [
        c for c in df_category.columns
        if c not in ALWAYS_DROP and c not in ALWAYS_KEEP
    ]

    report = compute_effect_sizes(df_category, candidate_columns, target_col=target_col)

    kept = [c for c in ALWAYS_KEEP if c in df_category.columns]
    for col in candidate_columns:
        if col in force_keep:
            kept.append(col)
            continue
        if col in report.index and report.loc[col, "recommandation"] == "garder":
            kept.append(col)

    return kept, report


__all__ = [
    "compute_effect_sizes",
    "select_features_for_category",
    "ALWAYS_DROP",
    "ALWAYS_KEEP",
    "NUMERIC_SPEARMAN_THRESHOLD",
    "CATEGORICAL_PVALUE_THRESHOLD",
]
