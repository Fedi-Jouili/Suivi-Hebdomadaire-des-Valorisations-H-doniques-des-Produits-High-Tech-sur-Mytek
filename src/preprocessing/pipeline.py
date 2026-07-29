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

    SELECTION DE FEATURES STABLE (2026-07-28, audit methodologique
    reviewer 1 -- Major) : select_features_for_category (§4) est un test
    STATISTIQUE (Spearman/Kruskal-Wallis) -- l'appeler independamment sur
    chaque semaine (un echantillon parfois < 100 lignes) produit une
    recommandation bruitee, differente d'une semaine a l'autre pour la
    MEME variable (constate en pratique : cpu_brand/has_4g apparaissant/
    disparaissant, cf. l'ancienne _reconcile_pooled_schema de
    save_artifacts.py). compute_stable_feature_selection()/build_all_weeks()
    calculent desormais cette selection UNE SEULE FOIS, sur toutes les
    semaines actuellement disponibles POOLEES (plus de puissance
    statistique, plus de derive de schema inter-semaines) -- c'est le
    point d'entree RECOMMANDE (`--all`) pour reconstruire l'integralite du
    pipeline. build_processed_datasets()/process_category() restent
    utilisables seuls (une semaine a la fois, selection locale a cette
    semaine) pour un usage ponctuel/les tests -- jamais retire, seulement
    plus recommande par defaut.

UTILISATION :
    python -m src.preprocessing.pipeline --raw-dir data/raw/week_1 --out-dir data/processed/week_1
    python -m src.preprocessing.pipeline --all
        (traite TOUTES les semaines sous data/raw/, selection de features
        calculee une seule fois sur les donnees poolees)
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
from .split import discover_weeks

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


def _force_keep_for(category: str, numeric_targets: list) -> set:
    force_keep = {"marque"} | set(numeric_targets)
    if category == "televiseurs":
        force_keep |= set(TV_SPECIFIC_CATEGORICAL) | set(TV_SPECIFIC_BOOLEAN_ALWAYS_PRESENT)
    return force_keep


def _impute_category(df_category: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Retire les colonnes structurellement non pertinentes puis impute --
    etape commune a process_category() (donnees d'UNE semaine) et
    compute_stable_feature_selection() (donnees POOLEES sur toutes les
    semaines, cf. plus bas) : la MEME preparation doit preceder le test
    statistique de selection de features dans les deux cas, jamais deux
    logiques paralleles qui pourraient diverger.
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
    return df_category


def compute_stable_feature_selection(raw_root: Path | str = "data/raw", categories: list | None = None) -> dict:
    """
    Calcule, PAR CATEGORIE, une selection de features UNIQUE a partir de
    TOUTES les semaines actuellement disponibles sous raw_root (poolees),
    au lieu de la recalculer independamment a chaque semaine sur un petit
    echantillon (cf. §SELECTION DE FEATURES STABLE du docstring de module).

    Reproduit le meme nettoyage/imputation que build_processed_datasets/
    process_category (jamais une seconde logique de preparation), mais
    sur le pool complet -- seul le resultat de select_features_for_category
    (kept_columns, report) est retenu ici ; aucun fichier n'est ecrit,
    aucune ligne n'est imputee "pour de vrai" (chaque semaine sera
    re-imputee individuellement dans build_processed_datasets, cette
    imputation poolee ne sert qu'a donner au test statistique un
    DataFrame sans NaN sur lequel s'executer).

    Returns: {categorie: (kept_columns: list[str], report: pd.DataFrame)}
    """
    raw_root = Path(raw_root)
    weeks = discover_weeks(raw_root)
    if not weeks:
        raise FileNotFoundError(f"Aucune semaine trouvee sous {raw_root}")

    all_products = []
    for w in weeks:
        all_products.extend(load_raw_files(raw_root / f"week_{w}"))
    if not all_products:
        raise RuntimeError(f"Aucun produit charge depuis {raw_root} (semaines {weeks}).")

    df = clean_products(all_products)
    df = engineer_features(df)
    df = df.dropna(subset=["prix_tnd"]).reset_index(drop=True)

    cats = categories if categories is not None else sorted(df["categorie"].dropna().unique())
    result = {}
    for category in cats:
        sub = df[df["categorie"] == category]
        if sub.empty:
            logger.warning(f"[selection stable] aucune ligne pour '{category}' -- ignoree.")
            continue
        prepped = _impute_category(sub, category)
        force_keep = _force_keep_for(category, _numeric_targets_for_category(category))
        kept_columns, report = select_features_for_category(prepped, target_col="prix_tnd", force_keep=force_keep)
        result[category] = (kept_columns, report)
        logger.info(
            f"[selection stable, {len(weeks)} semaine(s) poolees, n={len(sub)}] "
            f"{category} : {len(kept_columns)} colonne(s) retenue(s) -- {kept_columns}"
        )
    return result


def process_category(df_category: pd.DataFrame, category: str, stable_columns: list | None = None) -> tuple:
    """
    Impute puis selectionne les colonnes pour UNE categorie deja isolee.

    stable_columns : si fourni (cf. compute_stable_feature_selection),
        remplace l'appel a select_features_for_category par cette liste
        deja calculee sur les donnees poolees -- la meme selection est
        alors utilisee pour TOUTES les semaines, plus de derive de schema.
        None (par defaut) : comportement inchange, selection recalculee
        localement sur df_category seul (une semaine).

    Returns: (df_clean_tidy, df_encoded, rapport_selection) -- rapport_selection
        est None quand stable_columns est fourni (le rapport a deja ete
        loggue une fois par compute_stable_feature_selection, pas la
        peine de le recalculer a chaque semaine).
    """
    df_category = _impute_category(df_category, category)

    if stable_columns is not None:
        kept_columns = [c for c in stable_columns if c in df_category.columns]
        missing = sorted(set(stable_columns) - set(kept_columns))
        if missing:
            logger.warning(
                f"  [{category}] colonne(s) de la selection stable absente(s) cette semaine : {missing}"
            )
        report = None
    else:
        numeric_targets = _numeric_targets_for_category(category)
        force_keep = _force_keep_for(category, numeric_targets)
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


def build_processed_datasets(raw_dir: Path, out_dir: Path, stable_columns_by_category: dict | None = None) -> dict:
    """
    Point d'entree principal (UNE semaine) : charge, nettoie, impute,
    selectionne et ecrit les CSV finaux pour chaque categorie presente
    dans les donnees.

    stable_columns_by_category : si fourni (cf. compute_stable_feature_selection,
        typiquement passe par build_all_weeks), chaque categorie utilise
        cette selection deja calculee sur toutes les semaines poolees au
        lieu de la recalculer localement (cf. process_category). None par
        defaut : comportement inchange (selection locale a raw_dir seul).

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

        stable_columns = stable_columns_by_category.get(category) if stable_columns_by_category else None
        df_tidy, df_encoded, report = process_category(sub, category, stable_columns=stable_columns)

        if report is not None:
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


def build_all_weeks(raw_root: Path | str = "data/raw", out_root: Path | str = "data/processed") -> dict:
    """
    Point d'entree RECOMMANDE pour reconstruire TOUTES les semaines a la
    fois : calcule la selection de features UNE SEULE FOIS sur les
    donnees poolees (compute_stable_feature_selection), puis reconstruit
    chaque semaine individuellement avec cette MEME selection -- elimine
    la derive de schema inter-semaines (cf. §SELECTION DE FEATURES STABLE
    du docstring de module). Remplace l'ancien usage "une semaine a la
    fois" pour un rebuild complet ; build_processed_datasets() reste
    disponible seul pour un usage ponctuel/les tests.

    Returns: {semaine: {categorie: {"clean_rows": int, ...}}}
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    stable_selection = compute_stable_feature_selection(raw_root=raw_root)
    stable_columns_by_category = {cat: cols for cat, (cols, _report) in stable_selection.items()}
    for cat, (_cols, report) in stable_selection.items():
        logger.info(f"{'=' * 60}\nSelection stable ({cat}, poolee sur toutes les semaines) :\n{report.to_string()}")

    weeks = discover_weeks(raw_root)
    summary = {}
    for w in weeks:
        logger.info(f"{'#' * 60}\nSemaine {w}\n{'#' * 60}")
        summary[w] = build_processed_datasets(
            raw_root / f"week_{w}", out_root / f"week_{w}",
            stable_columns_by_category=stable_columns_by_category,
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Construit les jeux de donnees finaux (par categorie) "
                    "pour le clustering et la modelisation hedonique.",
    )
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Repertoire des JSON bruts -- UNE semaine si --all n'est pas passe, "
                             "sinon la racine contenant week_1/, week_2/...")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument(
        "--all", action="store_true",
        help="Traite TOUTES les semaines decouvertes sous --raw-dir (week_1, week_2...), avec une "
             "selection de features calculee UNE SEULE FOIS sur les donnees poolees -- recommande "
             "pour reconstruire le pipeline complet plutot qu'une semaine isolee.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    if not raw_dir.exists():
        logger.error(f"Le repertoire {raw_dir} n'existe pas.")
        sys.exit(1)

    if args.all:
        summary_by_week = build_all_weeks(raw_root=raw_dir, out_root=out_dir)
        logger.info(f"{'=' * 60}\nRESUME FINAL (toutes semaines)")
        for w, summary in summary_by_week.items():
            for cat, info in summary.items():
                logger.info(
                    f"  S{w} {cat:<24} : {info['clean_rows']:>4} lignes | "
                    f"{info['clean_cols']} col. (clean) / {info['encoded_cols']} col. (encoded)"
                )
        return

    summary = build_processed_datasets(raw_dir, out_dir)

    logger.info(f"{'=' * 60}\nRESUME FINAL")
    for cat, info in summary.items():
        logger.info(
            f"  {cat:<24} : {info['clean_rows']:>4} lignes | "
            f"{info['clean_cols']} col. (clean) / {info['encoded_cols']} col. (encoded)"
        )


if __name__ == "__main__":
    main()
