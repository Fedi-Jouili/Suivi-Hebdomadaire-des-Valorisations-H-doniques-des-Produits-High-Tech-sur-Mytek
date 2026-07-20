# -*- coding: utf-8 -*-
"""
=============================================================================
Package : src/preprocessing
=============================================================================
ROLE :
    Pipeline de pretraitement des donnees du projet de modele hedonique
    Mytek.tn. Regroupe :

      - clean.py           : nettoyage des donnees brutes scrapees (JSON ->
                              DataFrame consolide, dedup, marque normalisee,
                              valeurs hors bornes corrigees).
      - encode.py           : caracteristiques structurees (cpu_brand/serie/
                              gen, os_platform, flags de connectivite) et
                              encodage numerique (one-hot / label encoding)
                              pour les modeles hedoniques.
      - bounds.py           : calibration empirique des bornes de plausibilite
                              utilisees par clean.py.
      - impute.py           : cascade d'imputation par similarite (KNN /
                              mode des voisins), par categorie.
      - select_features.py  : selection des colonnes finales par categorie
                              (effet mesure sur le prix + drops structurels).
      - pipeline.py          : orchestrateur -- produit les CSV finaux
                              (tidy + encode) prets pour clustering/modelisation.
      - transform.py         : transformations appliquees juste avant
                              modelisation (log-prix, cluster en ligne via
                              modele deja ajuste, termes d'interaction).
      - split.py             : genere, par semaine, full_data.csv (5
                              categories reunies) et un split train/test
                              80/20 stratifie par categorie.

    Ce fichier re-exporte les fonctions principales de chaque sous-module
    pour permettre des imports courts depuis le reste du projet, par ex. :

        from src.preprocessing import clean_products, load_raw_files

    plutot que :

        from src.preprocessing.clean import clean_products
=============================================================================
"""

from .clean import (
    clean_products,
    load_raw_files,
    extract_gtin,
    dedup_key,
    fix_brand,
    normalize_brand_column,
    fix_ram,
    fix_screen_size,
    fix_storage,
    fix_os,
    fix_hdr,
    fix_refresh_rate,
    parse_storage_from_text,
    flag_out_of_range,
    get_bounds,
    write_summary,
    VALIDITY_BOUNDS,
)
from .encode import (
    NOMINAL_COLUMNS,
    parse_cpu,
    extract_cpu_features,
    extract_os_platform,
    extract_connectivity_flags,
    encode_for_ridge,
    encode_for_rf,
    apply_rf_encoders,
)
from .impute import (
    impute_numeric_cascade,
    impute_categorical_by_neighbors,
)
from .select_features import (
    compute_effect_sizes,
    select_features_for_category,
)
from .pipeline import build_processed_datasets
from .transform import (
    apply_log_transform,
    add_cluster_labels_online,
    create_polynomial_features,
    prepare_final_features,
)
from .split import (
    discover_weeks,
    load_full_data_for_week,
    check_schema_consistency,
    split_train_test,
    build_week_splits,
    build_all_weeks,
)

__all__ = [
    # clean.py
    "clean_products",
    "load_raw_files",
    "extract_gtin",
    "dedup_key",
    "fix_brand",
    "normalize_brand_column",
    "fix_ram",
    "fix_screen_size",
    "fix_storage",
    "fix_os",
    "fix_hdr",
    "fix_refresh_rate",
    "parse_storage_from_text",
    "flag_out_of_range",
    "get_bounds",
    "write_summary",
    "VALIDITY_BOUNDS",
    # encode.py
    "NOMINAL_COLUMNS",
    "parse_cpu",
    "extract_cpu_features",
    "extract_os_platform",
    "extract_connectivity_flags",
    "encode_for_ridge",
    "encode_for_rf",
    "apply_rf_encoders",
    # impute.py
    "impute_numeric_cascade",
    "impute_categorical_by_neighbors",
    # select_features.py
    "compute_effect_sizes",
    "select_features_for_category",
    # pipeline.py
    "build_processed_datasets",
    # transform.py
    "apply_log_transform",
    "add_cluster_labels_online",
    "create_polynomial_features",
    "prepare_final_features",
    # split.py
    "discover_weeks",
    "load_full_data_for_week",
    "check_schema_consistency",
    "split_train_test",
    "build_week_splits",
    "build_all_weeks",
]