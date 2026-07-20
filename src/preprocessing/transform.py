# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/preprocessing/transform.py
=============================================================================
ROLE :
    Transformations numeriques appliquees juste avant modelisation/clustering
    (contrairement a clean.py/impute.py/select_features.py, qui produisent un
    CSV "tidy" reutilisable pour n'importe quel usage) :

      - apply_log_transform      : cible log-lineaire du modele hedonique
                                    (Rosen, 1974), convention deja utilisee
                                    dans l'EDA et le notebook de clustering.
      - add_cluster_labels_online : applique un clustering DEJA AJUSTE
                                    (jamais un nouveau fit ici) a n'importe
                                    quelle semaine de donnees.
      - create_polynomial_features : termes d'interaction entre
                                    caracteristiques numeriques.
      - prepare_final_features   : orchestrateur combinant les trois.

ORDRE D'EXECUTION ATTENDU (important -- ce module ne fonctionne pas seul) :
    1. Notebook "src/notebooks/Clustering_produits_technologiques.ipynb" :
       ajuste un KMeans + StandardScaler par categorie sur les donnees de
       la semaine 1 (S1), PUIS les exporte sur disque avec joblib :
           import joblib
           joblib.dump(kmeans, "models/clustering/kmeans.pkl")
           joblib.dump(scaler, "models/clustering/scaler.pkl")
       (Cette etape d'export n'existe pas encore dans le notebook actuel --
       a y ajouter avant de pouvoir utiliser add_cluster_labels_online.)
    2. Ce module (transform.py) est ensuite appelable sur N'IMPORTE QUELLE
       semaine (S1 pour verifier la coherence, puis S2/S3 au fur et a
       mesure de leur collecte) : il CHARGE les objets deja ajustes et se
       contente de .transform()/.predict() -- jamais de re-fit -- pour que
       le sens des clusters reste stable et comparable d'une semaine a
       l'autre (un "cluster 2" doit designer le meme segment en S1 et S3).

    Tant que l'etape 1 n'a pas ete faite, add_cluster_labels_online (et donc
    prepare_final_features) leve une FileNotFoundError explicite plutot que
    d'echouer silencieusement ou de fabriquer un clustering a la volee.

UTILISATION :
    from src.preprocessing.transform import prepare_final_features
    X, y, poly = prepare_final_features(df_semaine_2)
=============================================================================
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

logger = logging.getLogger("preprocessing.transform")

# Caracteristiques numeriques communes utilisees par defaut pour le
# clustering ET les termes polynomiaux -- correspond au schema reel produit
# par clean.py/pipeline.py (cf. data/processed/<categorie>_clean.csv).
DEFAULT_CLUSTER_FEATURES = ["ram_go", "stockage_go", "cpu_serie", "taille_ecran"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSFORMATION LOGARITHMIQUE DE LA CIBLE
# ─────────────────────────────────────────────────────────────────────────────

def apply_log_transform(df: pd.DataFrame, price_col: str = "prix_tnd") -> pd.DataFrame:
    """
    Ajoute log_prix = ln(prix) -- forme fonctionnelle log-lineaire du
    modele hedonique de Rosen (1974), deja utilisee dans l'EDA et le
    notebook de clustering (interpretation des coefficients comme
    variations en % plutot qu'en valeur absolue).

    Args:
        df: DataFrame contenant la colonne de prix.
        price_col: nom de la colonne de prix (Go/TND). Le schema reel du
            projet utilise "prix_tnd" (cf. data/processed/*.csv).

    Returns: copie de df avec la colonne "log_prix" ajoutee.
    """
    if price_col not in df.columns:
        raise KeyError(
            f"Colonne '{price_col}' absente du DataFrame. "
            f"Colonnes disponibles : {list(df.columns)}"
        )

    df = df.copy()
    n_non_positif = (df[price_col] <= 0).sum()
    if n_non_positif:
        logger.warning(
            f"{n_non_positif} valeur(s) de '{price_col}' <= 0 -- "
            f"log_prix sera NaN/-inf pour ces lignes (log indefini)."
        )
    df["log_prix"] = np.log(df[price_col])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLUSTERING EN LIGNE (MODELE DEJA AJUSTE, JAMAIS RE-FIT ICI)
# ─────────────────────────────────────────────────────────────────────────────

def add_cluster_labels_online(
    df: pd.DataFrame,
    model_path: str = "models/clustering/kmeans.pkl",
    scaler_path: str = "models/clustering/scaler.pkl",
    feature_cols: list | None = None,
) -> pd.DataFrame:
    """
    Applique un KMeans + StandardScaler DEJA AJUSTES (typiquement sur les
    donnees de la semaine 1, cf. le notebook de clustering) a df, sans
    jamais reentrainer de modele -- seulement .transform()/.predict().
    C'est ce qui permet d'appliquer EXACTEMENT le meme decoupage en
    segments aux semaines 2 et 3 a mesure qu'elles sont collectees, pour
    que les clusters restent comparables dans le temps.

    Args:
        df: DataFrame contenant feature_cols (numeriques, sans NaN).
        model_path: chemin vers le KMeans serialise (joblib).
        scaler_path: chemin vers le StandardScaler serialise (joblib).
        feature_cols: colonnes utilisees pour le clustering. Par defaut,
            DEFAULT_CLUSTER_FEATURES (ram_go, stockage_go, cpu_serie,
            taille_ecran) -- a adapter par categorie si necessaire (ex:
            pas de cpu_serie pertinent pour les televiseurs).

    Returns: copie de df avec la colonne "cluster" (dtype categoriel) ajoutee.

    Raises:
        FileNotFoundError: si model_path/scaler_path n'existent pas --
            avec des instructions explicites plutot qu'un stacktrace brut.
        KeyError: si une colonne de feature_cols est absente de df.
        TypeError: si une colonne de feature_cols n'est pas numerique.
        ValueError: si feature_cols contient des valeurs manquantes.
    """
    feature_cols = list(feature_cols) if feature_cols is not None else list(DEFAULT_CLUSTER_FEATURES)
    model_path = Path(model_path)
    scaler_path = Path(scaler_path)

    if not model_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            "Modele de clustering introuvable "
            f"(model_path={model_path} existe={model_path.exists()}, "
            f"scaler_path={scaler_path} existe={scaler_path.exists()}).\n"
            "Marche a suivre :\n"
            "  1. Executer le notebook "
            "'src/notebooks/Clustering_produits_technologiques.ipynb' "
            "jusqu'a l'ajustement du KMeans/StandardScaler pour la categorie voulue.\n"
            "  2. Exporter les objets ajustes avec joblib, par exemple :\n"
            "         import joblib\n"
            "         joblib.dump(kmeans, 'models/clustering/kmeans.pkl')\n"
            "         joblib.dump(scaler, 'models/clustering/scaler.pkl')\n"
            "  3. Relancer add_cluster_labels_online()."
        )

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Colonne(s) absente(s) du DataFrame pour le clustering : {missing_cols}")

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(
            f"Colonne(s) non numerique(s) parmi feature_cols : {non_numeric} -- "
            "le clustering (distance euclidienne) exige des variables numeriques."
        )

    X = df[feature_cols].astype(float)
    if X.isna().any().any():
        cols_avec_na = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"Valeur(s) manquante(s) dans {cols_avec_na} -- imputer au prealable "
            "(cf. src.preprocessing.impute) avant d'appeler add_cluster_labels_online."
        )

    scaler = joblib.load(scaler_path)
    kmeans = joblib.load(model_path)

    X_scaled = scaler.transform(X)
    labels = kmeans.predict(X_scaled)

    df = df.copy()
    df["cluster"] = pd.Categorical(labels)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARACTERISTIQUES POLYNOMIALES (TERMES D'INTERACTION)
# ─────────────────────────────────────────────────────────────────────────────

def create_polynomial_features(X: pd.DataFrame, degree: int = 2, interaction_only: bool = True):
    """
    Encapsule sklearn.preprocessing.PolynomialFeatures.

    interaction_only=True par defaut : ne genere QUE des produits croises
    entre variables distinctes (ex: ram_go * taille_ecran), jamais de
    termes au carre (ram_go^2) qui n'ont pas d'interpretation hedonique
    naturelle -- un produit croise capture un effet de complementarite
    entre deux caracteristiques (ex: la RAM compte-t-elle plus sur un
    grand ecran que sur un petit ?), ce qu'un carre isole ne fait pas.

    Args:
        X: DataFrame de variables numeriques uniquement (pas de colonnes
           categorielles/dummies -- cf. prepare_final_features qui les
           exclut avant d'appeler cette fonction).
        degree: degre polynomial maximal.
        interaction_only: si True, pas de termes au carre (cf. ci-dessus).

    Returns: (X_poly_df, poly) -- DataFrame des caracteristiques generees
        (noms de colonnes explicites via get_feature_names_out) et
        l'objet PolynomialFeatures ajuste (a reutiliser en .transform()
        sur de nouvelles donnees, jamais en re-fit).
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    return X_poly_df, poly


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────────────────────

def prepare_final_features(
    df: pd.DataFrame,
    cluster_model_path: str = "models/clustering/kmeans.pkl",
    cluster_scaler_path: str = "models/clustering/scaler.pkl",
    cluster_features: list | None = None,
    add_polynomials: bool = True,
    degree: int = 2,
    price_col: str = "prix_tnd",
):
    """
    Enchaine les 3 transformations pour produire (X, y) prets pour un
    modele hedonique ou un clustering downstream :

        1. log_prix = ln(prix_tnd)                         (§1)
        2. cluster = KMeans deja ajuste (S1), .predict()   (§2)
        3. termes d'interaction sur les specs numeriques    (§3),
           en excluant explicitement les dummies et "cluster"
           (une interaction entre deux indicatrices ou avec le
           cluster n'a pas le meme sens qu'entre deux specs continues).

    Args:
        df: DataFrame source (ex: un <categorie>_clean.csv, semaine
            S1/S2/S3).
        cluster_model_path, cluster_scaler_path: cf. add_cluster_labels_online.
        cluster_features: colonnes numeriques utilisees a la fois pour le
            clustering et (si add_polynomials) pour les interactions.
            Par defaut DEFAULT_CLUSTER_FEATURES.
        add_polynomials: si False, saute l'etape 3 (poly_obj retourne = None).
        degree: degre polynomial (cf. create_polynomial_features).
        price_col: nom de la colonne de prix source.

    Returns: (X, y, poly_obj)
        X : DataFrame des variables explicatives (prix_tnd/log_prix retires).
        y : Series log_prix (cible du modele hedonique).
        poly_obj : objet PolynomialFeatures ajuste, ou None si add_polynomials=False.
    """
    cluster_features = list(cluster_features) if cluster_features is not None else list(DEFAULT_CLUSTER_FEATURES)

    df = apply_log_transform(df, price_col=price_col)
    df = add_cluster_labels_online(
        df,
        model_path=cluster_model_path,
        scaler_path=cluster_scaler_path,
        feature_cols=cluster_features,
    )

    y = df["log_prix"]
    X = df.drop(columns=[price_col, "log_prix"])

    poly_obj = None
    if add_polynomials:
        # Seulement les specs numeriques d'origine -- exclut explicitement
        # "cluster" (categoriel, pas une grandeur continue) et toute
        # colonne dummy/one-hot presente dans X (ex: marque_ASUS), pour
        # lesquelles un produit croise n'a pas de sens hedonique direct.
        numeric_specs = [c for c in cluster_features if c in X.columns]
        X_poly, poly_obj = create_polynomial_features(X[numeric_specs], degree=degree, interaction_only=True)

        X_reste = X.drop(columns=numeric_specs).reset_index(drop=True)
        X = pd.concat([X_reste, X_poly.reset_index(drop=True)], axis=1)

    return X, y, poly_obj


# ─────────────────────────────────────────────────────────────────────────────
# TEST MANUEL (DataFrame factice -- ne necessite pas de modele de clustering
# pour les parties 1 et 3 ; la partie 2 est testee via la FileNotFoundError
# attendue tant qu'aucun modele n'a ete exporte par le notebook)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    dummy = pd.DataFrame({
        "prix_tnd": [999.0, 1499.0, 2499.0, 3999.0],
        "ram_go": [8.0, 16.0, 16.0, 32.0],
        "stockage_go": [256.0, 512.0, 512.0, 1024.0],
        "cpu_serie": [3.0, 5.0, 5.0, 7.0],
        "taille_ecran": [13.3, 15.6, 15.6, 17.3],
        "marque": ["ASUS", "HP", "DELL", "ASUS"],
    })

    print("--- 1. apply_log_transform ---")
    dummy_log = apply_log_transform(dummy)
    print(dummy_log[["prix_tnd", "log_prix"]])

    print("\n--- 2. add_cluster_labels_online (modele attendu absent -> FileNotFoundError) ---")
    try:
        add_cluster_labels_online(dummy_log)
    except FileNotFoundError as exc:
        print(f"OK, erreur attendue et explicite :\n{exc}")

    print("\n--- 3. create_polynomial_features ---")
    numeric_cols = ["ram_go", "stockage_go", "cpu_serie", "taille_ecran"]
    dummy_poly, poly = create_polynomial_features(dummy[numeric_cols])
    print("Colonnes generees :", list(dummy_poly.columns))
    print(dummy_poly.head())

    print("\n--- 4. prepare_final_features (echoue tant qu'aucun modele n'est exporte) ---")
    try:
        X, y, poly_obj = prepare_final_features(dummy, add_polynomials=True)
        print(X.head())
        print(y.head())
    except FileNotFoundError as exc:
        print(f"OK, erreur attendue et explicite :\n{exc}")
