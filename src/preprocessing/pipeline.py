# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/pipeline.py
=============================================================================
ROLE :
    Orchestrateur qui produit, PAR CATEGORIE, un jeu de donnees pret pour
    le clustering et la modelisation hedonique :

      1. clean_products() (clean.py) -- dedup, marque normalisee, valeurs
         hors bornes corrigees (re-derivees depuis specs_brutes/nom) ou
         mises a None (prix).
      2. extract_cpu_features / extract_os_platform / extract_connectivity_flags
         (encode.py) -- caracteristiques structurees depuis processeur/os/
         connectivite (texte libre a haute cardinalite ou peu exploitable).
      3. Lignes sans prix fiable ECARTEES (pas de cible = pas de modele
         hedonique possible pour cette ligne).
      4. Par categorie : imputation des valeurs manquantes restantes
         (impute.py, cascade KNN/mode-des-voisins/mediane -- jamais de
         NaN residuel dans le CSV final) puis selection des colonnes
         (select_features.py, effet sur le prix + drops structurels).
      5. Ecriture de 2 fichiers par categorie dans data/processed/ :
         <categorie>_clean.csv   (tidy, categoriel en texte lisible)
         <categorie>_encoded.csv (numerique, one-hot via encode_for_ridge --
                                   pret pour KMeans/sklearn sans etape
                                   supplementaire)

UTILISATION :
    python -m src.preprocessing.pipeline
    python -m src.preprocessing.pipeline --raw-dir data/raw --out-dir data/processed
=============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .clean import load_raw_files, clean_products, VALIDITY_BOUNDS
from .encode import (
    extract_cpu_features,
    extract_os_platform,
    extract_connectivity_flags,
    encode_for_ridge,
)
from .impute import impute_numeric_cascade, impute_categorical_by_neighbors
from .select_features import select_features_for_category

logger = logging.getLogger("preprocessing.pipeline")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)

# Champs specifiques aux televiseurs (n'existent que pour cette categorie ;
# 100% NaN pour les autres, ce qui les fait ecarter automatiquement par le
# test de colonne constante dans select_features -- aucune exclusion
# manuelle necessaire pour les autres categories).
TV_SPECIFIC_NUMERIC = ["taux_rafraichissement"]
TV_SPECIFIC_CATEGORICAL = ["resolution_affichage", "technologie_dalle"]
TV_SPECIFIC_BOOLEAN_ALWAYS_PRESENT = ["hdr", "smart_tv"]  # jamais None par construction (parser.py)

_BOUNDS_KEY_TO_COLUMN = {"ram": "ram_go", "storage": "stockage_go", "screen": "taille_ecran"}


def _numeric_targets_for_category(category: str) -> list:
    """
    Derive dynamiquement, depuis VALIDITY_BOUNDS (source unique de verite,
    deja utilisee par clean.py), quelles colonnes numeriques communes sont
    structurellement pertinentes pour cette categorie -- un bounds a None
    (ex: ram/stockage pour televiseurs, ecran pour pc_bureau) signifie
    "pas un attribut standard de cette categorie", donc on ne tente meme
    pas de l'imputer/le garder (evite une colonne quasi-integralement
    imputee et presentee comme si elle etait une vraie caracteristique --
    ex: 86% des televiseurs n'ont pas de "RAM" au sens propre).
    """
    bounds = VALIDITY_BOUNDS.get(category, {})
    targets = [
        _BOUNDS_KEY_TO_COLUMN[key]
        for key, col in _BOUNDS_KEY_TO_COLUMN.items()
        if bounds.get(key) is not None
    ]
    if category == "televiseurs":
        targets = targets + TV_SPECIFIC_NUMERIC
    return targets


def _structurally_irrelevant_columns(category: str) -> list:
    """
    L'inverse de _numeric_targets_for_category : les colonnes
    ram_go/stockage_go/taille_ecran dont VALIDITY_BOUNDS dit explicitement
    qu'elles ne sont PAS pertinentes pour cette categorie (bounds=None).

    A retirer explicitement du DataFrame AVANT select_features_for_category
    -- sinon une colonne comme ram_go pour televiseurs (16/112 renseignes,
    par les seules Smart/Google TV -- qui sont aussi les plus cheres)
    peut passer le test d'effet sur un tout petit echantillon biaise
    ("a du RAM renseigne" ~= "est une Smart TV premium" ~= "prix eleve",
    cf. EDA §3.3) et se retrouver gardee puis grossierement remplie par
    la mediane globale -- alors meme que ce n'est structurellement pas
    un attribut standard de la categorie.
    """
    bounds = VALIDITY_BOUNDS.get(category, {})
    return [
        _BOUNDS_KEY_TO_COLUMN[key]
        for key, col in _BOUNDS_KEY_TO_COLUMN.items()
        if bounds.get(key) is None
    ]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applique les 3 extractions de encode.py sur le DataFrame unifie."""
    df = extract_cpu_features(df)
    df = extract_os_platform(df)
    df = extract_connectivity_flags(df)
    return df


def process_category(df_category: pd.DataFrame, category: str) -> tuple:
    """
    Impute puis selectionne les colonnes pour UNE categorie deja isolee.

    Returns: (df_clean_tidy, df_encoded, rapport_selection)
    """
    df_category = df_category.copy()

    irrelevant = [c for c in _structurally_irrelevant_columns(category) if c in df_category.columns]
    if irrelevant:
        df_category = df_category.drop(columns=irrelevant)
        logger.info(f"  [{category}] colonnes structurellement non pertinentes retirees : {irrelevant}")

    numeric_targets = _numeric_targets_for_category(category)
    if numeric_targets:
        df_category = impute_numeric_cascade(
            df_category, target_columns=numeric_targets, neighbor_columns=numeric_targets,
            category_col="categorie",
        )

    neighbor_cols_for_categorical = [c for c in numeric_targets if c in df_category.columns]
    categorical_targets = ["os_platform"]
    if category == "televiseurs":
        categorical_targets += TV_SPECIFIC_CATEGORICAL
    df_category = impute_categorical_by_neighbors(
        df_category, target_columns=categorical_targets,
        neighbor_columns=neighbor_cols_for_categorical, category_col="categorie",
    )

    force_keep = {"marque"} | set(numeric_targets)
    if category == "televiseurs":
        force_keep |= set(TV_SPECIFIC_CATEGORICAL) | set(TV_SPECIFIC_BOOLEAN_ALWAYS_PRESENT)

    kept_columns, report = select_features_for_category(
        df_category, target_col="prix_tnd", force_keep=force_keep,
    )

    df_tidy = df_category[kept_columns].reset_index(drop=True)

    # cpu_serie/cpu_gen : un NaN ici n'est PAS une donnee manquante au
    # meme titre que ram_go/taille_ecran -- c'est une famille de
    # processeur qui n'a structurellement pas de tier/generation au sens
    # Core iX / Ryzen X (Celeron, Pentium, Jasper Lake...). Y assigner
    # une mediane fabriquerait un tier fictif (ex: "tier 5" sur un
    # Celeron) -- on encode plutot explicitement "pas de tier" par 0,
    # en-dessous de la plage reelle 3/5/7/9, avant le filet de securite
    # generique ci-dessous.
    for col in ("cpu_serie", "cpu_gen"):
        if col in df_tidy.columns:
            df_tidy[col] = df_tidy[col].fillna(0)

    # Filet de securite : le coeur de ce pipeline est "zero valeur
    # manquante" -- si une colonne gardee a encore des NaN a ce stade,
    # c'est un bug de la cascade d'imputation, pas un etat acceptable a
    # laisser passer silencieusement.
    remaining_na = df_tidy.isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if not remaining_na.empty:
        logger.warning(
            f"  [{category}] valeurs manquantes residuelles apres imputation : "
            f"{remaining_na.to_dict()} -- repli mediane/mode de secours."
        )
        for col in remaining_na.index:
            if pd.api.types.is_numeric_dtype(df_tidy[col]):
                df_tidy[col] = df_tidy[col].fillna(df_tidy[col].median())
            else:
                mode = df_tidy[col].mode(dropna=True)
                df_tidy[col] = df_tidy[col].fillna(mode.iloc[0] if not mode.empty else "Inconnu")

    df_encoded = encode_for_ridge(df_tidy.drop(columns=["nom", "url"], errors="ignore"))

    return df_tidy, df_encoded, report


def build_processed_datasets(raw_dir: Path, out_dir: Path) -> dict:
    """
    Point d'entree principal : charge, nettoie, impute, selectionne et
    ecrit les CSV finaux pour chaque categorie presente dans les donnees.

    Returns: dict {categorie: {"clean_rows": int, "encoded_cols": int}}
    """
    raw_products = load_raw_files(raw_dir)
    if not raw_products:
        raise RuntimeError(f"Aucun produit charge depuis {raw_dir}.")

    df = clean_products(raw_products)
    df = engineer_features(df)

    before = len(df)
    df = df.dropna(subset=["prix_tnd"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        logger.info(
            f"{dropped} ligne(s) ecartee(s) faute de prix fiable "
            f"(manquant ou hors bornes) -- {len(df)} produit(s) restant(s)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for category in sorted(df["categorie"].dropna().unique()):
        sub = df[df["categorie"] == category]
        logger.info(f"{'─' * 60}\nCategorie : {category} ({len(sub)} produit(s))")

        df_tidy, df_encoded, report = process_category(sub, category)

        logger.info(f"  Rapport de selection des colonnes :\n{report.to_string()}")
        logger.info(f"  Colonnes finales ({len(df_tidy.columns)}) : {list(df_tidy.columns)}")

        clean_path = out_dir / f"{category}_clean.csv"
        encoded_path = out_dir / f"{category}_encoded.csv"
        df_tidy.to_csv(clean_path, index=False, encoding="utf-8-sig")
        df_encoded.to_csv(encoded_path, index=False, encoding="utf-8-sig")
        logger.info(f"  Ecrit : {clean_path.name} ({len(df_tidy)} lignes, {len(df_tidy.columns)} col.)")
        logger.info(f"  Ecrit : {encoded_path.name} ({len(df_encoded)} lignes, {len(df_encoded.columns)} col.)")

        summary[category] = {
            "clean_rows": len(df_tidy), "clean_cols": len(df_tidy.columns),
            "encoded_cols": len(df_encoded.columns),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Construit les jeux de donnees finaux (par categorie) "
                    "pour le clustering et la modelisation hedonique.",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        logger.error(f"Le repertoire {raw_dir} n'existe pas.")
        sys.exit(1)

    summary = build_processed_datasets(raw_dir, out_dir)

    logger.info(f"{'=' * 60}\nRESUME FINAL")
    for cat, info in summary.items():
        logger.info(
            f"  {cat:<24} : {info['clean_rows']:>4} lignes | "
            f"{info['clean_cols']} col. (clean) / {info['encoded_cols']} col. (encoded)"
        )


if __name__ == "__main__":
    main()
