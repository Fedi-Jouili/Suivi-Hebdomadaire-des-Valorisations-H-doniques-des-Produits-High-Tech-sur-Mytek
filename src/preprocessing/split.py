# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/split.py
=============================================================================
ROLE :
    Genere, POUR CHAQUE SEMAINE disponible sous data/processed/week_<N>/,
    un jeu de donnees complet (les 5 categories reunies) et un split
    train/test 80/20 stratifie par categorie :

        data/processed/week_<N>/full_data.csv   (5 categories concatenees)
        data/processed/week_<N>/train.csv        (~80%, stratifie)
        data/processed/week_<N>/test.csv         (~20%, stratifie)

    Aucun nettoyage n'est refait ici : full_data.csv est construit en
    concatenant tel quel les <categorie>_clean.csv DEJA produits par
    src/preprocessing/pipeline.py (qui orchestre lui-meme clean.py pour le
    nettoyage et encode.py pour la derivation des caracteristiques
    cpu_serie/cpu_gen/cpu_brand/os_platform/has_* -- ces colonnes sont donc
    deja presentes dans <categorie>_clean.csv). Une colonne "categorie" est
    ajoutee (absente des fichiers par-categorie, ou elle etait implicite
    dans le nom de fichier).

OU CE MODULE S'INSERE DANS LE PIPELINE (clean -> encode -> split) :
    Le pipeline n'etait jusqu'ici pas explicitement "chaine" au sens ou
    pipeline.py produit deux sorties par categorie :
        <categorie>_clean.csv    -- tidy, features derivees par encode.py
                                     (cpu_*, os_platform, has_*) mais SANS
                                     encodage one-hot final.
        <categorie>_encoded.csv  -- one-hot (encode_for_ridge), specifique
                                     a un usage Ridge en particulier.
    split.py lit <categorie>_clean.csv, PAS <categorie>_encoded.csv : le
    split train/test doit rester utilisable par n'importe quel modele en
    aval (Ridge, Random Forest, regression hedonique statsmodels), chacun
    encodant a sa maniere une fois le split fixe -- encoder AVANT de
    splitter figerait un choix d'encodage (one-hot Ridge) qui ne convient
    pas forcement aux autres modeles. La chaine complete est donc :

        clean.py --> pipeline.py (encode.py inclus) --> split.py
        (data/raw/)   (data/processed/week_N/<cat>_clean.csv)   (data/processed/week_N/{full_data,train,test}.csv)

COHERENCE DE SCHEMA INTER-SEMAINES :
    select_features_for_category (appele par pipeline.py) choisit les
    colonnes gardees PAR SEMAINE, via un test statistique recalcule sur les
    donnees de cette semaine seule -- deux semaines peuvent donc legitimement
    ne pas avoir exactement les memes colonnes (constate en pratique :
    pc_portables a gagne cpu_brand/has_4g en semaine 3). check_schema_consistency
    detecte et SIGNALE (log.warning) cette divergence -- jamais silencieuse,
    jamais une exception qui bloquerait le split des autres semaines.

UTILISATION :
    python -m src.preprocessing.split
=============================================================================
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import CATEGORY_ORDER, DATA_PROCESSED_DIR, RANDOM_STATE, TEST_SIZE
from src.utils.logger import get_logger

logger = get_logger("preprocessing.split")


def discover_weeks(processed_dir: Path = DATA_PROCESSED_DIR) -> list:
    """
    Decouvre dynamiquement les semaines disponibles (sous-dossiers
    'week_<N>' de data/processed/) -- jamais un nombre code en dur : une
    semaine 4 collectee plus tard est prise en compte sans modification de
    code.
    """
    weeks = []
    for p in sorted(Path(processed_dir).glob("week_*")):
        if p.is_dir():
            suffix = p.name.split("_", 1)[-1]
            if suffix.isdigit():
                weeks.append(int(suffix))
    return sorted(weeks)


def load_full_data_for_week(
    week: int,
    processed_dir: Path = DATA_PROCESSED_DIR,
    categories: list = CATEGORY_ORDER,
) -> pd.DataFrame:
    """
    Concatene les <categorie>_clean.csv d'UNE semaine en un seul
    DataFrame, avec une colonne "categorie" ajoutee en premiere position
    (absente des fichiers par-categorie). Une categorie non encore
    scrapee cette semaine-la (ex: semaine 3 partielle) est ignoree avec un
    avertissement explicite, jamais une erreur silencieuse.
    """
    week_dir = Path(processed_dir) / f"week_{week}"
    frames = []
    for cat in categories:
        path = week_dir / f"{cat}_clean.csv"
        if not path.exists():
            logger.warning(f"[semaine {week}] fichier absent, catégorie ignorée : {path}")
            continue
        df_cat = pd.read_csv(path)
        df_cat.insert(0, "categorie", cat)
        frames.append(df_cat)

    if not frames:
        raise FileNotFoundError(f"Aucune donnée nettoyée trouvée pour la semaine {week} dans {week_dir}")

    return pd.concat(frames, ignore_index=True)


def check_schema_consistency(full_data_by_week: dict) -> bool:
    """
    Compare les colonnes de full_data entre semaines et SIGNALE (log)
    toute divergence -- ne leve pas d'exception (un modèle en aval peut
    choisir de se restreindre aux colonnes communes), mais rien n'est
    jamais tu.

    Returns: True si toutes les semaines ont des colonnes identiques,
        False sinon.
    """
    weeks = sorted(full_data_by_week)
    if len(weeks) < 2:
        logger.info("Une seule semaine disponible -- rien à comparer.")
        return True

    col_sets = {w: set(full_data_by_week[w].columns) for w in weeks}
    reference_week = weeks[0]
    reference_cols = col_sets[reference_week]

    consistent = True
    for w in weeks[1:]:
        missing = reference_cols - col_sets[w]
        extra = col_sets[w] - reference_cols
        if missing or extra:
            consistent = False
            logger.warning(
                f"Colonnes NON identiques entre semaine {reference_week} et semaine {w} : "
                f"absentes en S{w}={sorted(missing) or 'aucune'}, "
                f"en plus en S{w}={sorted(extra) or 'aucune'}"
            )
        else:
            logger.info(f"Semaine {w} : colonnes identiques à la semaine {reference_week}.")

    if consistent:
        logger.info("Schéma identique sur toutes les semaines : un modèle entraîné sur une semaine "
                     "peut prédire sur une autre sans adaptation de colonnes.")
    return consistent


def split_train_test(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    stratify_col: str = "categorie",
) -> tuple:
    """
    Split 80/20 déterministe (random_state), stratifié sur stratify_col
    pour que chaque catégorie garde ~80/20 dans train ET test (évite
    qu'une petite catégorie comme telephones_portables tombe entièrement
    d'un côté).

    Vérifie EXPLICITEMENT par assertion (jamais une hypothèse tacite) :
    aucune ligne perdue (train + test == full) et, si une colonne "url"
    existe, aucun chevauchement train/test.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[stratify_col],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    assert len(train_df) + len(test_df) == len(df), (
        f"train ({len(train_df)}) + test ({len(test_df)}) != full_data ({len(df)}) -- lignes perdues."
    )
    if "url" in df.columns:
        overlap = set(train_df["url"]) & set(test_df["url"])
        assert not overlap, f"Chevauchement train/test détecté sur {len(overlap)} url(s)."

    return train_df, test_df


def build_week_splits(
    week: int,
    processed_dir: Path = DATA_PROCESSED_DIR,
    categories: list = CATEGORY_ORDER,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Pipeline complet pour UNE semaine : charge les <categorie>_clean.csv,
    écrit full_data.csv, split 80/20 stratifié, écrit train.csv/test.csv.

    Idempotent : to_csv écrase proprement les fichiers existants, et le
    split est déterministe (random_state fixe) -- relancer produit des
    fichiers strictement identiques, jamais dupliqués ni corrompus.
    """
    full_df = load_full_data_for_week(week, processed_dir, categories)
    week_dir = Path(processed_dir) / f"week_{week}"
    week_dir.mkdir(parents=True, exist_ok=True)

    full_path = week_dir / "full_data.csv"
    full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"[semaine {week}] full_data.csv : {len(full_df)} lignes, {len(full_df.columns)} colonnes -> {full_path}"
    )

    train_df, test_df = split_train_test(full_df, test_size=test_size, random_state=random_state)

    train_path = week_dir / "train.csv"
    test_path = week_dir / "test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    train_repartition = train_df["categorie"].value_counts().to_dict()
    test_repartition = test_df["categorie"].value_counts().to_dict()
    logger.info(f"[semaine {week}] train.csv : {len(train_df)} lignes -> {train_path} | répartition : {train_repartition}")
    logger.info(f"[semaine {week}] test.csv  : {len(test_df)} lignes -> {test_path} | répartition : {test_repartition}")

    return {"full": full_df, "train": train_df, "test": test_df}


def build_all_weeks(
    processed_dir: Path = DATA_PROCESSED_DIR,
    categories: list = CATEGORY_ORDER,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Point d'entrée principal : découvre les semaines disponibles et
    construit full_data/train/test pour chacune, puis vérifie la
    cohérence de schéma inter-semaines."""
    weeks = discover_weeks(processed_dir)
    if not weeks:
        raise FileNotFoundError(f"Aucune semaine trouvée sous {processed_dir}")
    logger.info(f"Semaines détectées : {weeks}")

    results = {}
    for w in weeks:
        results[w] = build_week_splits(w, processed_dir, categories, test_size, random_state)

    check_schema_consistency({w: r["full"] for w, r in results.items()})
    return results


if __name__ == "__main__":
    build_all_weeks()
