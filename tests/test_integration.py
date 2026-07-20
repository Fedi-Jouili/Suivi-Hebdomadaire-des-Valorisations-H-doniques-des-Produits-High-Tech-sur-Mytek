# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_integration.py
=============================================================================
ROLE :
    Test de bout en bout minimal : verifie que la CHAINE COMPLETE
    s'execute sans erreur, du DataFrame brut jusqu'a une prediction Ridge --
    clean_products -> extract_cpu_features/extract_os_platform/
    extract_connectivity_flags -> prepare_final_features (log-prix +
    cluster + interactions) -> RidgeModel.fit/predict.

    Jeu de donnees LOCAL (pas sample_raw_df) : RidgeModel.fit() utilise un
    GridSearchCV(cv=5) fixe (cf. src/models/ridge_model.py), qui exige au
    moins 5 echantillons d'ENTRAINEMENT -- sample_raw_df (5 lignes, 4 apres
    neutralisation du prix aberrant) n'en laisse que 3 une fois un holdout
    retire. On genere donc ici 10 produits (9 valides + 1 prix aberrant),
    pour rester au-dessus de ce seuil apres le split train/holdout, tout en
    gardant le meme schema et la meme logique que sample_raw_df.

    Ce test ne juge PAS la qualite du modele -- seulement l'absence
    d'exception sur l'ensemble de la chaine. L'assertion RMSE > 0 est
    volontairement faible (une prediction n'importe comment fausse la
    validerait aussi) ; on verifie EN PLUS que les predictions sont finies
    et de la bonne forme, pour un test un peu plus informatif qu'un simple
    "> 0".
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing.clean import clean_products
from src.preprocessing.encode import extract_cpu_features, extract_os_platform, extract_connectivity_flags
from src.preprocessing.transform import prepare_final_features
from src.models.ridge_model import RidgeModel

CLUSTER_FEATURES = ["ram_go", "stockage_go", "cpu_serie", "taille_ecran"]

MARQUES_ET_CPU = [
    ("ASUS", "Intel Core i5-1235U", 5), ("HP", "Intel Core i7-1255U", 7),
    ("DELL", "Intel Core i3-1115G4", 3), ("LENOVO", "AMD Ryzen 5 5500U", 5),
    ("MSI", "Intel Core i7-1360P", 7), ("ACER", "Intel Core i5-1240P", 5),
    ("GIGABYTE", "AMD Ryzen 7 5800H", 7), ("TOSHIBA", "Intel Core i3-1215U", 3),
    ("APPLE", "Intel Core i9-9880H", 9), ("SAMSUNG", "Intel Core i5-1135G7", 5),
    ("HUAWEI", "Intel Core i5-1155G7", 5), ("XIAOMI", "Intel Core i7-1165G7", 7),
    ("MEDION", "AMD Ryzen 3 5300U", 3), ("PACKARD-BELL", "Intel Core i3-1005G1", 3),
    ("VAIO", "Intel Core i7-10510U", 7),
]


def _generer_produits_bruts() -> list:
    """15 produits synthetiques (14 valides + 1 prix aberrant), memes
    champs que sample_raw_df -- volume choisi pour laisser >= 2
    echantillons par pli meme apres le split train/holdout (GridSearchCV
    a un cv=5 fixe dans RidgeModel ; avec trop peu d'echantillons par
    pli, le R² d'un pli a 1 seul point devient NaN, cf. warning sklearn
    "non-finite test scores" observe avec un jeu de donnees plus petit)."""
    produits = []
    for i, (marque, processeur, _cpu_serie) in enumerate(MARQUES_ET_CPU):
        prix = -50.0 if i == 0 else 1200.0 + i * 300.0  # 1 seul prix aberrant, en tete
        produits.append({
            "nom": f"PC Portable {marque} Modele{i} {processeur}",
            "marque": marque,
            "prix_tnd": prix,
            "processeur": processeur,
            "ram_go": float(8 + 4 * (i % 4)),
            "stockage_go": float(256 * (1 + i % 4)),
            "type_stockage": "SSD",
            "taille_ecran": 14.0 + (i % 3),
            "os": "Windows 11 Famille" if i % 2 == 0 else "FreeDos",
            "connectivite": "Wi-Fi; Bluetooth",
            "url": f"https://exemple.test/pc-{marque.lower()}-{i}.html",
            "categorie": "pc_portables",
            "semaine": 1,
            "date_collecte": "2026-07-01 10:00:00",
            "specs_brutes": {"Marque": marque},
        })
    return produits


def test_pipeline_complet_brut_vers_prediction(tmp_path):
    """Chaine complete : donnees brutes -> nettoyage -> encodage ->
    log-prix + cluster + interactions -> Ridge -> prediction."""

    # ── 1. Nettoyage ──────────────────────────────────────────────────────
    df = clean_products(_generer_produits_bruts())

    # Le prix aberrant a ete neutralise (None) par clean_products(), pas
    # supprime -- la suppression des lignes sans prix fiable est la
    # responsabilite de l'etape suivante (cf.
    # src/preprocessing/pipeline.py::build_processed_datasets), reproduite
    # ici explicitement.
    df = df.dropna(subset=["prix_tnd"]).reset_index(drop=True)
    assert len(df) == 14  # 15 produits - 1 (prix aberrant) = 14

    # ── 2. Caracteristiques structurees ──────────────────────────────────
    df = extract_cpu_features(df)
    df = extract_os_platform(df)
    df = extract_connectivity_flags(df)
    assert df[CLUSTER_FEATURES].isna().sum().sum() == 0  # aucune valeur manquante residuelle

    # ── 3. Modele de clustering "deja ajuste" (simule l'export du notebook) ──
    scaler = StandardScaler().fit(df[CLUSTER_FEATURES])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(scaler.transform(df[CLUSTER_FEATURES]))
    model_path = tmp_path / "kmeans.pkl"
    scaler_path = tmp_path / "scaler.pkl"
    joblib.dump(kmeans, model_path)
    joblib.dump(scaler, scaler_path)

    # ── 4. log-prix + cluster + interactions ─────────────────────────────
    X, y, poly = prepare_final_features(
        df, cluster_model_path=model_path, cluster_scaler_path=scaler_path,
        cluster_features=CLUSTER_FEATURES, add_polynomials=True,
    )
    X_numerique = X[CLUSTER_FEATURES + [c for c in X.columns if " " in c]]  # variables + interactions (pas "cluster", categoriel)

    # ── 5. Split train/holdout + Ridge (train >= 5 pour le cv=5 de RidgeModel) ──
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_numerique, y, test_size=2, random_state=42,
    )

    model = RidgeModel(degree=1, alphas=[1.0]).fit(X_train, y_train)
    predictions = model.predict(X_holdout)

    assert np.all(np.isfinite(predictions))
    assert len(predictions) == len(X_holdout)

    rmse = float(np.sqrt(np.mean((predictions - y_holdout.to_numpy()) ** 2)))
    assert rmse > 0  # verifie surtout l'absence d'erreur d'execution sur toute la chaine
