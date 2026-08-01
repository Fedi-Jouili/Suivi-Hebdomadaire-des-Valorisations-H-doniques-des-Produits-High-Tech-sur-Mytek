# -*- coding: utf-8 -*-
"""
=============================================================================
Script : scripts/merge_segmentation_prix_semaine.py
=============================================================================
ROLE :
    Pour chaque categorie, fusionne le fichier de segmentation
    (outputs/labels/segmentation_<categorie>.csv -- cle_produit, categorie,
    marque, gamme, cluster_segmentation, un cluster STATIQUE par produit,
    cf. notebooks/Segmentation_Prix_Clustering_produits_technologiques.ipynb)
    avec le prix de CHAQUE semaine ou le produit apparait
    (data/processed/week_<n>/<categorie>_clean.csv -- url, prix_tnd).

    Un meme produit (cle_produit) vu a plusieurs semaines produit plusieurs
    LIGNES dans le fichier de sortie, une par semaine, avec le prix de cette
    semaine-la et le meme cluster_segmentation (le cluster ne varie pas dans
    le temps, seul le prix varie). Jointure INTERNE sur cle_produit/url :
    seuls les produits presents A LA FOIS dans la segmentation et dans le
    catalogue de la semaine consideree sont retenus -- un produit filtre par
    la segmentation (marque negligeable, cf. MIN_BRAND_COUNT) ou absent du
    catalogue cette semaine-la (pas encore ou plus en vente) n'a, par
    construction, pas de ligne pour cette semaine.

    Fichiers de sortie : outputs/labels/produits_prix_cluster_semaine_
    <categorie>.csv. Une fois les 5 fichiers ecrits avec succes, les 5
    fichiers de segmentation d'origine (outputs/labels/segmentation_
    <categorie>.csv) sont supprimes -- decision utilisateur du 2026-07-31 :
    ce nouveau fichier les remplace, plus besoin de les garder separement.

UTILISATION :
    python -m scripts.merge_segmentation_prix_semaine
=============================================================================
"""

import logging

import pandas as pd

from src.preprocessing.split import discover_weeks
from src.utils.config import CATEGORY_ORDER, DATA_PROCESSED_DIR, PROJECT_ROOT

logger = logging.getLogger("scripts.merge_segmentation_prix_semaine")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

LABELS_DIR = PROJECT_ROOT / "outputs" / "labels"


def _load_weekly_prices(category: str) -> pd.DataFrame:
    """Concatene (cle_produit, semaine, prix_tnd) de TOUTES les semaines
    decouvertes sous data/processed/ pour `category` -- une ligne par
    (produit, semaine) ou le produit apparait dans le catalogue nettoye."""
    weeks = discover_weeks(DATA_PROCESSED_DIR)
    frames = []
    for w in weeks:
        path = DATA_PROCESSED_DIR / f"week_{w}" / f"{category}_clean.csv"
        if not path.exists():
            logger.warning(f"  [{category}] semaine {w} absente ({path}), ignoree.")
            continue
        df_w = pd.read_csv(path, encoding="utf-8-sig")[["url", "prix_tnd"]].copy()
        df_w = df_w.rename(columns={"url": "cle_produit"})
        df_w["semaine"] = w
        frames.append(df_w)

    if not frames:
        raise FileNotFoundError(f"Aucun fichier clean trouve pour '{category}' sous {DATA_PROCESSED_DIR}")
    return pd.concat(frames, ignore_index=True)


def build_category_file(category: str) -> pd.DataFrame:
    """Fusionne la segmentation et les prix hebdomadaires pour `category` --
    une ligne par (produit, semaine), triee pour une lecture stable."""
    seg_path = LABELS_DIR / f"segmentation_{category}.csv"
    df_seg = pd.read_csv(seg_path, encoding="utf-8-sig")
    df_prices = _load_weekly_prices(category)

    df_merged = df_seg.merge(df_prices, on="cle_produit", how="inner")

    col_order = ["categorie", "marque", "gamme", "cluster_segmentation", "cle_produit", "semaine", "prix_tnd"]
    df_merged = df_merged[col_order].sort_values(
        ["marque", "gamme", "cluster_segmentation", "cle_produit", "semaine"]
    ).reset_index(drop=True)

    n_produits = df_seg["cle_produit"].nunique()
    n_couverts = df_merged["cle_produit"].nunique()
    logger.info(
        f"  [{category}] {n_produits} produits segmentes -> {n_couverts} retrouves dans au moins une semaine "
        f"-> {len(df_merged)} lignes (produit x semaine)."
    )
    return df_merged


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    written_paths = []

    for category in CATEGORY_ORDER:
        logger.info(f"{'=' * 70}\nCategorie : {category}")
        df_out = build_category_file(category)
        out_path = LABELS_DIR / f"produits_prix_cluster_semaine_{category}.csv"
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        written_paths.append(out_path)
        logger.info(f"  [{category}] ecrit -> {out_path}")

    logger.info(f"{'=' * 70}\n{len(written_paths)} fichiers ecrits sous {LABELS_DIR}. Suppression des fichiers de segmentation d'origine...")

    for category in CATEGORY_ORDER:
        seg_path = LABELS_DIR / f"segmentation_{category}.csv"
        if seg_path.exists():
            seg_path.unlink()
            logger.info(f"  supprime : {seg_path}")

    logger.info("Termine.")


if __name__ == "__main__":
    main()
