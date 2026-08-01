# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/models/save_artifacts.py
=============================================================================
ROLE :
    Entraine et persiste (joblib/CSV/JSON) TOUS les artefacts necessaires
    au dashboard (src/dashboard/), qui doit rester STRICTEMENT lecture
    seule vis-a-vis de la modelisation -- jamais de fit() dans le
    dashboard, seulement des joblib.load()/pd.read_csv() sur ce que ce
    script produit. A relancer manuellement quand une nouvelle semaine de
    donnees est disponible.

    Pour chaque categorie (pc_bureau, pc_portables, smartphones,
    telephones_portables, televiseurs) :

      1. POOLING MULTI-SEMAINES (decision utilisateur du 2026-07-21) :
         charge <categorie>_clean.csv de TOUTES les semaines disponibles
         (data/processed/week_*/), les concatene. Split train/test 80/20
         GROUPE PAR PRODUIT (url, via GroupShuffleSplit) -- un meme
         produit vu a plusieurs semaines ne doit JAMAIS se retrouver a la
         fois en train et en test, sous peine de fuite (le modele aurait
         "vu" ce produit, a une semaine differente, pendant l'entrainement).
         C'est plus strict que le split par semaine de src.preprocessing.
         split (qui ne protege pas cette fuite inter-semaines -- non
         pertinent tant que chaque semaine etait evaluee separement, mais
         devient necessaire des qu'on poole).

      2. MATRICE DE DESIGN : construite via
         src.models.hedonic_model.build_design_matrix -- continues +
         categorielles incluant "marque" (effet fixe legitime, cf.
         correctif circularite 2026-07-21), JAMAIS cluster_id/gamme_prix
         (le garde-fou de circularite du module s'applique tel quel).
         Encodage one-hot calcule UNE FOIS sur train+test concatenes (cf.
         notebooks/Comparaison_Modeles_ML_Clustering.ipynb, meme
         raisonnement : garantit les memes colonnes des deux cotes, sans
         jamais laisser filtrer y_test dans le choix des modalites).

      3. 3 FAMILLES DE MODELES ajustees sur le train, evaluees hors
         echantillon sur le test : HedonicOLS (inference), RidgeModel
         (degree=1 -- cf. avertissement de ridge_model.py sur les
         interactions degenerees entre indicatrices one-hot),
         RandomForestModel. Metriques en log ET en TND retro-transforme
         (np.exp), SANS correction de biais Duan/Miller -- choix
         documente, pas un oubli (cf. reports/audit_code.md §3.3) : les
         RMSE/MAE en TND sous-estiment legerement le vrai niveau moyen
         d'erreur, la comparaison RELATIVE entre modeles reste valide.

      4. CLUSTERING N1 (technique pur, cf. notebooks/Clustering_produits_
         technologiques.ipynb) : KMeans + StandardScaler ajustes sur les
         SEULES caracteristiques techniques (jamais prix ni marque),
         persistes -- utilisables pour assigner un segment a un produit
         HYPOTHETIQUE (aucune dependance a un prix deja connu,
         contrairement a N2/gamme_prix).

      5. DONNEES ETIQUETEES (gamme_prix + cluster_id/N2 + cluster_direct/
         N1) persistees en CSV -- sert a la fois de table de "produits
         reels similaires" (page Prediction) et de source pour assigner
         une gamme/un segment N2 a un produit hypothetique en 2 temps
         (prix provisoire -> gamme -> segment N2, decision utilisateur du
         2026-07-21 -- cf. src/dashboard/pages/prediction.py).

         gamme_prix/cluster_id (N2) NE SONT JAMAIS calcules sur le pool
         multi-semaines (contrairement aux modeles de regression du point 1,
         un pooling legitime protege par le split groupe par produit) : un
         K-Means (et le qcut de compute_price_tiers) n'ont pas de notion de
         "periode" dans leurs features, melanger les semaines y traite un
         meme produit vu 4 fois comme 4 points independants, gonflant
         artificiellement l'effectif d'une unite marque x gamme et faisant
         apparaitre une fausse structure (decision utilisateur du
         2026-07-31). Les deux colonnes sont figees sur la SEULE semaine de
         reference (hedonic_model.n2_reference_week, la plus ancienne
         disponible) puis reappliquees par produit (url) a toutes les
         semaines ou il apparait -- cluster_direct/N1 reste, lui, ajuste sur
         le pool complet (modele persistable reutilise via .predict(), pas
         un decoupage en coupe, cf. point 4). Consequence assumee : un
         produit absent de la semaine de reference n'a AUCUN gamme_prix/
         cluster_id, dans aucune semaine (jamais invente), meme s'il passe
         par ailleurs le filtre MIN_BRAND_COUNT sur l'effectif poole.

UTILISATION :
    python -m src.models.save_artifacts
    python -m src.models.save_artifacts --category pc_bureau
=============================================================================
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.models.hedonic_model import (
    FORBIDDEN_REGRESSORS,
    HedonicOLS,
    _ID_COLUMNS,
    _choose_k,
    _classify_features,
    _min_cluster_size_for,
    _min_rows_required,
    build_design_matrix,
    compute_cluster_labels,
    compute_price_tiers,
    n2_reference_week,
)
from src.models.ridge_model import RidgeModel
from src.models.rf_model import RandomForestModel
from src.preprocessing.split import discover_weeks
from src.utils.config import CATEGORY_ORDER, DATA_PROCESSED_DIR, PROJECT_ROOT, RANDOM_STATE, TEST_SIZE

logger = logging.getLogger("models.save_artifacts")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

MODELS_DIR = PROJECT_ROOT / "models"

# Le clustering N1 (technique pur) suit la meme regle de garde-fou anti-
# degenerescence que les notebooks de clustering -- jamais de cluster < 5
# produits ou < 3% de l'echantillon retenu comme "structure reelle".
N1_MIN_K = 2


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT POOLE MULTI-SEMAINES + SPLIT GROUPE PAR PRODUIT
# ─────────────────────────────────────────────────────────────────────────────

def load_pooled_category(category: str, processed_root: Path = DATA_PROCESSED_DIR) -> pd.DataFrame:
    """Concatene <categorie>_clean.csv de TOUTES les semaines decouvertes
    sous processed_root (jamais une liste figee -- cf. discover_weeks),
    avec une colonne "semaine" ajoutee (absente des CSV par-categorie)."""
    weeks = discover_weeks(processed_root)
    if not weeks:
        raise FileNotFoundError(f"Aucune semaine trouvee sous {processed_root}")

    frames = []
    for w in weeks:
        path = Path(processed_root) / f"week_{w}" / f"{category}_clean.csv"
        if not path.exists():
            logger.warning(f"  [{category}] semaine {w} absente ({path}), ignoree.")
            continue
        df_w = pd.read_csv(path)
        df_w["semaine"] = w
        frames.append(df_w)

    if not frames:
        raise FileNotFoundError(f"Aucune donnee trouvee pour '{category}' sous {processed_root}")
    df_pooled = pd.concat(frames, ignore_index=True)
    logger.info(f"[{category}] {len(df_pooled)} lignes poolees sur {len(frames)} semaine(s) ({weeks}).")
    return _reconcile_pooled_schema(df_pooled, category)


# Prefixe des indicatrices de connectivite (encode.py::extract_connectivity_
# flags) -- toujours calculees pour CHAQUE ligne en amont (jamais NaN a la
# source), une absence de colonne pour une semaine ne peut donc venir que
# du filtrage statistique de select_features_for_category (colonne jugee
# constante ou sans effet mesure CETTE semaine-la), jamais d'une donnee
# reellement inconnue -- fillna(0) est donc la valeur EXACTE, pas une
# approximation (cf. encode.py : absence de connectivite -> flags a 0).
_BINARY_FLAG_PREFIX = "has_"


def _reconcile_pooled_schema(df_pooled: pd.DataFrame, category: str) -> pd.DataFrame:
    """
    Le schema de colonnes de <categorie>_clean.csv peut legerement varier
    d'une semaine a l'autre : select_features_for_category (§select_
    features.py) recalcule INDEPENDAMMENT, semaine par semaine, quelles
    colonnes ont un effet mesure sur le prix -- une colonne peut donc
    exister S4 et etre absente S1-S3 (constatee : has_4g pour televiseurs,
    cf. hedonic_model.py qui documente deja le meme phenomene pour
    cpu_brand/has_4g sur pc_portables). pd.concat() introduit alors des
    NaN pour les semaines ou la colonne est absente -- jamais silencieux
    ici : chaque colonne affectee et son nombre de lignes remplies sont
    loggues, jamais laisse tel quel (statsmodels leve sinon "exog contains
    inf or nans").

    Regle de reconciliation :
      - indicatrices has_* (toujours calculees en amont, cf. §ci-dessus) :
        NaN -> 0, valeur exacte (pas une approximation).
      - autres colonnes categorielles (object) : NaN -> "Manquant" (meme
        convention que encode_for_ridge, jamais une categorie silencieusement
        fusionnee avec une autre).
      - colonnes numeriques hors has_* (ne devrait pas arriver : ram_go/
        stockage_go/taille_ecran/taux_rafraichissement sont en force_keep
        dans pipeline.py, jamais sujettes a ce filtrage) : leve une erreur
        explicite plutot que de deviner une valeur -- une vraie lacune de
        donnees ici serait plus grave qu'un simple defaut de schema.
    """
    na_counts = df_pooled.isna().sum()
    affected = na_counts[na_counts > 0]
    if affected.empty:
        return df_pooled

    df_pooled = df_pooled.copy()
    for col, n_missing in affected.items():
        if col.startswith(_BINARY_FLAG_PREFIX):
            df_pooled[col] = df_pooled[col].fillna(0)
            logger.warning(
                f"  [{category}] '{col}' absente de certaines semaines (schema variable, "
                f"cf. select_features_for_category) -- {n_missing} ligne(s) remplie(s) a 0."
            )
        elif df_pooled[col].dtype == object:
            df_pooled[col] = df_pooled[col].fillna("Manquant")
            logger.warning(
                f"  [{category}] '{col}' absente de certaines semaines (schema variable) -- "
                f"{n_missing} ligne(s) remplie(s) a 'Manquant'."
            )
        else:
            raise ValueError(
                f"[{category}] '{col}' a {n_missing} valeur(s) manquante(s) apres pooling multi-semaines, "
                f"et n'est ni une indicatrice has_* ni une categorielle texte -- ceci ne devrait pas arriver "
                f"pour une colonne numerique structurelle (force_keep dans pipeline.py). A investiguer avant "
                f"de deviner une regle de remplissage."
            )
    return df_pooled


def group_split_by_product(df_pooled: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE):
    """Split train/test GROUPE PAR url (GroupShuffleSplit) -- jamais un
    meme produit des deux cotes, meme s'il apparait a plusieurs semaines
    (cf. docstring du module, §1). Plus strict qu'un split aleatoire simple
    des lors que les semaines sont poolees."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df_pooled, groups=df_pooled["url"]))
    df_train = df_pooled.iloc[train_idx].reset_index(drop=True)
    df_test = df_pooled.iloc[test_idx].reset_index(drop=True)

    overlap = set(df_train["url"]) & set(df_test["url"])
    assert not overlap, f"Chevauchement train/test detecte sur {len(overlap)} url(s) -- GroupShuffleSplit corrompu."
    return df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATRICE DE DESIGN COMMUNE (encodage identique train/test)
# ─────────────────────────────────────────────────────────────────────────────

def build_matrices(df_train: pd.DataFrame, df_test: pd.DataFrame, continuous_features: list, categorical_features: list):
    """Encodage one-hot calcule UNE SEULE FOIS sur train+test concatenes,
    puis re-separe -- garantit les memes colonnes (memes modalites, meme
    modalite de reference droppee) des deux cotes. Aucune fuite de la
    CIBLE : seules les modalites categorielles observees (jamais y_test)
    definissent l'encodage (meme raisonnement que Comparaison_Modeles_ML_
    Clustering.ipynb)."""
    combined = pd.concat(
        [df_train.assign(_split="train"), df_test.assign(_split="test")], ignore_index=True
    )
    X_all, y_all = build_design_matrix(combined, continuous_features, categorical_features)
    is_train = (combined["_split"] == "train").to_numpy()
    X_train = X_all.loc[is_train].reset_index(drop=True)
    y_train = y_all.loc[is_train].reset_index(drop=True)
    X_test = X_all.loc[~is_train].reset_index(drop=True)
    y_test = y_all.loc[~is_train].reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def evaluate_predictions(y_true_log, y_pred_log) -> dict:
    """Metriques hors-echantillon en log(prix) ET en prix TND retro-
    transforme (np.exp), SANS correction de biais de retransformation
    (Duan/Miller) -- choix documente (§docstring du module), pas un oubli."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    price_true = np.exp(y_true_log)
    price_pred = np.exp(y_pred_log)
    return {
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "rmse_tnd": float(np.sqrt(mean_squared_error(price_true, price_pred))),
        "mae_tnd": float(mean_absolute_error(price_true, price_pred)),
        "retransformation_bias_correction": None,
        "retransformation_note": (
            "rmse_tnd/mae_tnd sont calcules par simple exp() des predictions log, sans correction de "
            "biais de retransformation (Duan/Miller) -- sous-estiment legerement l'erreur moyenne reelle "
            "en TND. La comparaison RELATIVE entre modeles reste valide (cf. reports/audit_code.md §3.3)."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERING N1 (technique pur -- persistable, utilisable hors prix)
# ─────────────────────────────────────────────────────────────────────────────

def fit_n1_clustering(df_pooled: pd.DataFrame, continuous_features: list, categorical_features: list,
                       random_state: int = RANDOM_STATE):
    """
    KMeans + StandardScaler sur les SEULES caracteristiques techniques
    (jamais prix ni marque, meme logique que Clustering_produits_
    technologiques.ipynb) -- persistes pour etre reappliques (.transform/
    .predict, JAMAIS refit) a un produit hypothetique dont le prix est
    justement inconnu (contrairement a N2/gamme_prix, cf. docstring du
    module).

    Returns: (kmeans, scaler, feature_cols, labels: np.ndarray, k: int)
    """
    feature_cols = list(continuous_features) + list(categorical_features)
    X_numeric = df_pooled[continuous_features].astype(float) if continuous_features else pd.DataFrame(index=df_pooled.index)
    if categorical_features:
        X_categorical = pd.get_dummies(df_pooled[categorical_features].astype(str), columns=categorical_features)
    else:
        X_categorical = pd.DataFrame(index=df_pooled.index)
    X_raw = pd.concat([X_numeric, X_categorical], axis=1)

    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw)

    # _choose_k (importee de hedonic_model.py) prend directement la matrice
    # SCALEE et fait sa propre boucle interne sur k (silhouette + garde-fou
    # anti-degenerescence anti-collision KMeans, cf. son docstring) -- ne
    # pas reimplementer cette boucle ici.
    n = X_scaled.shape[0]
    min_size = _min_cluster_size_for(n)
    k = _choose_k(X_scaled, min_size)

    # Toujours ajuste (y compris n_clusters=1, trivialement valide) -- un
    # KMeans jamais fit() ne pourrait pas etre re-charge puis .predict() par
    # le dashboard (NotFittedError), y compris quand aucune structure n'est
    # retenue (k=1, cf. _choose_k).
    kmeans = KMeans(n_clusters=max(k, 1), random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, scaler, X_raw.columns.tolist(), labels, k


# ─────────────────────────────────────────────────────────────────────────────
# 3bis. ACCORD ENTRE MODELES (OLS / Ridge / RF) -- rigor upgrade
#    (2026-07-28, audit methodologique reviewer 1) : rien ne verifiait
#    jusqu'ici que les 3 modeles racontent une histoire coherente -- un
#    lecteur voyait "RF dit que la RAM compte le plus" sans aucun moyen de
#    savoir si OLS/Ridge le confirment ou le contredisent.
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_agreement(
    ols_coefficients: pd.DataFrame, ridge_coefficients: pd.DataFrame, rf_importances: pd.DataFrame,
) -> tuple:
    """
    Compare les 3 modeles sur les MEMES colonnes (OLS, Ridge en degree=1 et
    RF sont tous ajustes sur exactement la meme matrice one-hot, cf.
    process_category -- aucun realignement de noms necessaire).

    - OLS et Ridge ont chacun un coefficient SIGNE, directement comparables :
      `ols_ridge_signes_accordent` = le signe est-il le meme ?
    - RF n'a pas de signe (importance toujours positive) -- comparee par
      CORRELATION DE RANG (Spearman) entre son importance et |coefficient|
      OLS : RF et OLS s'accordent-ils sur QUELLES variables comptent, meme
      si RF ne dit rien du sens de l'effet ?

    Returns: (comparison_df, resume: dict) -- comparison_df =
        ["feature", "ols_coefficient", "ridge_coefficient",
        "ols_ridge_signes_accordent", "rf_importance"], triee par
        rf_importance decroissante.
    """
    ols = ols_coefficients[ols_coefficients["feature"] != "const"].set_index("feature")
    ridge = ridge_coefficients.set_index("feature")
    rf = rf_importances.set_index("feature")

    common = sorted(set(ols.index) & set(ridge.index) & set(rf.index))
    rows = []
    for feat in common:
        ols_coef = float(ols.loc[feat, "coefficient"])
        ridge_coef = float(ridge.loc[feat, "coefficient"])
        rows.append({
            "feature": feat,
            "ols_coefficient": ols_coef,
            "ridge_coefficient": ridge_coef,
            "ols_ridge_signes_accordent": (ols_coef > 0) == (ridge_coef > 0),
            "rf_importance": float(rf.loc[feat, "importance"]),
        })
    columns = ["feature", "ols_coefficient", "ridge_coefficient", "ols_ridge_signes_accordent", "rf_importance"]
    comparison_df = (
        pd.DataFrame(rows, columns=columns).sort_values("rf_importance", ascending=False).reset_index(drop=True)
        if rows else pd.DataFrame(columns=columns)
    )

    if comparison_df.empty:
        resume = {
            "n_features_comparees": 0, "pct_signes_ols_ridge_accordent": None,
            "rf_ols_spearman_rho": None, "rf_ols_spearman_p_value": None,
        }
    else:
        rho, p_value = spearmanr(comparison_df["rf_importance"], comparison_df["ols_coefficient"].abs())
        resume = {
            "n_features_comparees": len(comparison_df),
            "pct_signes_ols_ridge_accordent": round(float(comparison_df["ols_ridge_signes_accordent"].mean() * 100), 1),
            "rf_ols_spearman_rho": round(float(rho), 3) if pd.notna(rho) else None,
            "rf_ols_spearman_p_value": round(float(p_value), 4) if pd.notna(p_value) else None,
        }
    return comparison_df, resume


# ─────────────────────────────────────────────────────────────────────────────
# 3ter. MODELISATION PAR CLUSTER (N1 ET N2 SEPAREMENT) -- decision
#    utilisateur du 2026-08-01 : un seul modele Ridge/Hedonic/RF par
#    categorie entiere est trop general, la prediction doit etre affinee
#    par segment. Generalise le patron de fit_strategy_b (hedonic_model.py
#    -- regression par marque dominante, gardee par un seuil n/predicteurs,
#    marques ecartees toujours RAPPORTEES, jamais un filtrage silencieux) a
#    un segment ARBITRAIRE (N1 : cluster_direct, technique pur, toute la
#    categorie ; N2 : cluster_id, marque x gamme x sous-cluster) et aux 3
#    familles de modeles (fit_strategy_b est OLS seul).
#
#    Seuil MIN N INDEPENDANT PAR FAMILLE (meme regle empirique que
#    _min_rows_required, ratio differencie) : OLS exige le plus de degres
#    de liberte residuels pour une inference valide (erreurs-types,
#    p-values) ; Ridge (regularisation L2) et Random Forest (agregation
#    d'arbres, cf. leurs modules respectifs) tolerent un echantillon plus
#    modeste par construction -- un ratio unique pour les 3 aurait ete soit
#    trop laxiste pour OLS, soit prive Ridge/RF de segments qu'ils peuvent
#    pourtant gerer proprement.
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MIN_ROWS_RATIOS = {"hedonic_ols": 10, "ridge": 5, "random_forest": 5}


def _sanitize_segment_name(value) -> str:
    """Nom de segment (ex. cluster_id = "MSI::Premium::c0") -> nom de
    fichier/dossier sur (":" est invalide dans un chemin Windows)."""
    text = str(value)
    for bad, repl in ((":", "_"), ("/", "-"), ("\\", "-"), ("*", ""), ("?", ""),
                       ('"', ""), ("<", ""), (">", ""), ("|", "_"), (" ", "_")):
        text = text.replace(bad, repl)
    return text


def _drop_constant_categorical(df: pd.DataFrame, categorical_features: list) -> list:
    """Retire les colonnes categorielles devenues CONSTANTES dans `df`
    (aucune variance a exploiter) -- ex. `marque` a l'interieur d'un
    cluster N2 (constante par construction, un cluster est nichee dans une
    marque). Determine DYNAMIQUEMENT (nunique()<=1), jamais une liste
    figee : un cluster N1 (technique pur, traverse les marques) garde au
    contraire `marque` la plupart du temps -- meme fonction, deux resultats
    differents selon le segment, jamais suppose."""
    return [c for c in categorical_features if df[c].nunique(dropna=False) > 1]


def fit_models_per_segment(
    df_train: pd.DataFrame, df_test: pd.DataFrame, segment_col: str,
    continuous_features: list, categorical_features: list,
    category_models: dict | None = None, category_design_columns: list | None = None,
    min_rows_ratios: dict | None = None, random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Pour CHAQUE valeur de `segment_col` presente dans df_train (lignes ou
    segment_col est NaN ignorees -- produit non couvert par cette
    segmentation, cf. n2_reference_week/coverage_by_week), tente d'ajuster
    HedonicOLS/RidgeModel/RandomForestModel sur le sous-ensemble
    correspondant de df_train, evalue hors-echantillon sur le
    sous-ensemble correspondant de df_test (si non vide).

    Garde-fous, TOUJOURS rapportes (jamais un filtrage silencieux, meme
    principe que fit_strategy_b) :
      - moins de 2 produits DISTINCTS dans le segment (url.nunique()) :
        aucune validation croisee groupee possible (Ridge/RF) ni erreur-type
        groupee valide (OLS, cov_type="cluster") -- segment ecarte pour les
        3 familles.
      - sinon, par famille INDEPENDAMMENT (ratio different, cf.
        DEFAULT_MIN_ROWS_RATIOS) : n_lignes >= _min_rows_required(n_predicteurs,
        ratio), sinon cette famille (seule) est ecartee pour ce segment.
      - tout echec d'ajustement inattendu (ex. matrice degeneree) est
        capture et rapporte comme un rejet, jamais une exception qui
        interrompt le traitement des autres segments/familles.

    `marque`/`gamme_prix`/`cluster_id` restent interdits comme regresseurs
    (garde-fou de circularite herite de build_design_matrix) ; toute autre
    colonne categorielle devenue constante DANS LE SEGMENT est retiree
    dynamiquement (cf. _drop_constant_categorical) -- notamment `marque`
    pour un segment N2 (constante par construction), jamais pour un
    segment N1 (traverse les marques).

    "AJUSTE" NE VEUT PAS DIRE "MEILLEUR QUE LE MODELE CATEGORIE" (rigor
    upgrade -- mesure empirique sur pc_bureau, 2026-08-01) : passer le seuil
    _min_rows_required garantit une matrice estimable, PAS une bonne
    generalisation -- des R² hors-echantillon fortement negatifs (jusqu'a
    -700) ont ete observes sur des segments qui passent pourtant le seuil,
    un symptome classique de surapprentissage sur petit echantillon. Si
    `category_models`/`category_design_columns` sont fournis (modeles deja
    ajustes sur la categorie ENTIERE, cf. process_category), CHAQUE modele
    de segment ajuste est aussi COMPARE au modele categorie sur EXACTEMENT
    les memes lignes de test du segment (jamais une comparaison sur des
    echantillons differents) -- `retenu_pour_prediction` (colonne du
    summary) n'est True QUE si le modele de segment bat le modele categorie
    sur cette comparaison (ou si le segment n'a aucune ligne de test, auquel
    cas la comparaison est impossible et le modele est retenu par defaut
    avec `comparaison_possible=False`, jamais un rejet automatique faute de
    donnees). C'est cette colonne, jamais `ajuste` seule, que doivent lire
    weekly_report.py/prediction_utils.py pour decider d'utiliser un modele
    de segment plutot que le repli categorie.

    Returns:
        (fitted, summary) --
        fitted : dict {segment: {"models": {famille: modele ajuste, ...},
            "continuous_features": list, "categorical_features": list
            (post-exclusion des constantes), "design_columns": list,
            "n_train": int, "n_test": int, "retenu_pour_prediction":
            {famille: bool}}} -- seules les familles REELLEMENT ajustees
            apparaissent dans "models" ;
        summary : DataFrame, une ligne par (segment, famille) --
            [segment_col, segment, n_lignes, n_produits_distincts,
            n_predicteurs, famille, n_min_requis, ajuste, raison_rejet,
            n_test, r2_test, rmse_tnd_test, r2_test_categorie,
            rmse_tnd_test_categorie, comparaison_possible,
            retenu_pour_prediction].
    """
    ratios = dict(min_rows_ratios) if min_rows_ratios else dict(DEFAULT_MIN_ROWS_RATIOS)
    category_models = category_models or {}
    fitted = {}
    summary_rows = []

    segments = sorted(df_train[segment_col].dropna().unique(), key=str)
    for segment in segments:
        seg_train = df_train[df_train[segment_col] == segment].reset_index(drop=True)
        seg_test = df_test[df_test[segment_col] == segment].reset_index(drop=True)

        seg_categorical = _drop_constant_categorical(seg_train, categorical_features)
        X_train_seg, y_train_seg = build_design_matrix(seg_train, list(continuous_features), seg_categorical)
        n_predictors = X_train_seg.shape[1] + 1  # +1 pour la constante, meme convention que fit_strategy_b
        n_rows = len(seg_train)
        n_groups = seg_train["url"].nunique()

        X_test_seg, y_test_seg = (None, None)
        if len(seg_test) > 0:
            X_test_seg, y_test_seg = build_design_matrix(seg_test, list(continuous_features), seg_categorical)
            X_test_seg = X_test_seg.reindex(columns=X_train_seg.columns, fill_value=0.0)

        # Matrice de test sur le schema COMPLET du modele categorie (colonnes
        # potentiellement plus larges que seg_categorical, ex. marque, jamais
        # constante a l'echelle categorie) -- necessaire pour comparer le
        # modele de segment au modele categorie sur EXACTEMENT les memes
        # lignes de test, cf. note ci-dessus.
        X_test_cat = None
        if category_models and category_design_columns is not None and len(seg_test) > 0:
            X_test_cat, _ = build_design_matrix(seg_test, list(continuous_features), categorical_features)
            X_test_cat = X_test_cat.reindex(columns=category_design_columns, fill_value=0.0)

        segment_models = {}
        segment_retenu = {}
        for famille in ("hedonic_ols", "ridge", "random_forest"):
            min_required = _min_rows_required(n_predictors, ratios[famille])
            row = {
                "segment_col": segment_col, "segment": segment, "famille": famille,
                "n_lignes": n_rows, "n_produits_distincts": n_groups, "n_predicteurs": n_predictors,
                "n_min_requis": min_required, "ajuste": False, "raison_rejet": None,
                "n_test": len(seg_test), "r2_test": None, "rmse_tnd_test": None,
                "r2_test_categorie": None, "rmse_tnd_test_categorie": None,
                "comparaison_possible": False, "retenu_pour_prediction": False,
            }

            if n_groups < 2:
                row["raison_rejet"] = f"n_produits_distincts={n_groups} < 2 -- validation croisee groupee impossible"
                summary_rows.append(row)
                continue
            if n_rows < min_required:
                row["raison_rejet"] = f"n_lignes={n_rows} < n_min_requis={min_required} (ratio={ratios[famille]})"
                summary_rows.append(row)
                continue

            try:
                if famille == "hedonic_ols":
                    model = HedonicOLS(cov_type="cluster").fit(
                        X_train_seg, y_train_seg, continuous_cols=continuous_features,
                        cov_kwds={"groups": seg_train["url"]},
                    )
                elif famille == "ridge":
                    model = RidgeModel(degree=1).fit(X_train_seg, y_train_seg, groups=seg_train["url"])
                else:
                    model = RandomForestModel().fit(X_train_seg, y_train_seg, groups=seg_train["url"])
            except Exception as exc:
                row["raison_rejet"] = f"echec d'ajustement : {exc}"
                summary_rows.append(row)
                continue

            row["ajuste"] = True
            segment_models[famille] = model
            # Par defaut retenu (aucune comparaison possible faute de test --
            # jamais un rejet automatique pour ce seul motif, cf. note
            # ci-dessus) ; devient conditionnel des qu'une comparaison a lieu.
            retenu = True

            if X_test_seg is not None and len(y_test_seg) > 0:
                pred = model.predict(X_test_seg)
                metrics = evaluate_predictions(y_test_seg.to_numpy(), np.asarray(pred))
                row["r2_test"] = round(metrics["r2_log"], 4)
                row["rmse_tnd_test"] = round(metrics["rmse_tnd"], 2)

                if X_test_cat is not None and famille in category_models:
                    pred_cat = category_models[famille].predict(X_test_cat)
                    metrics_cat = evaluate_predictions(y_test_seg.to_numpy(), np.asarray(pred_cat))
                    row["r2_test_categorie"] = round(metrics_cat["r2_log"], 4)
                    row["rmse_tnd_test_categorie"] = round(metrics_cat["rmse_tnd"], 2)
                    row["comparaison_possible"] = True
                    # R2 (log) plus eleve = meilleur ajustement hors-echantillon
                    # -- seul critere retenu (coherent avec ols_metrics/
                    # ridge_metrics/rf_metrics du modele categorie, tous en R2 log).
                    retenu = metrics["r2_log"] > metrics_cat["r2_log"]

            row["retenu_pour_prediction"] = retenu
            segment_retenu[famille] = retenu
            summary_rows.append(row)

        if segment_models:
            fitted[segment] = {
                "models": segment_models,
                "continuous_features": list(continuous_features),
                "categorical_features": seg_categorical,
                "design_columns": list(X_train_seg.columns),
                "n_train": n_rows, "n_test": len(seg_test),
                "retenu_pour_prediction": segment_retenu,
            }

    columns = ["segment_col", "segment", "famille", "n_lignes", "n_produits_distincts", "n_predicteurs",
               "n_min_requis", "ajuste", "raison_rejet", "n_test", "r2_test", "rmse_tnd_test",
               "r2_test_categorie", "rmse_tnd_test_categorie", "comparaison_possible", "retenu_pour_prediction"]
    summary = pd.DataFrame(summary_rows, columns=columns) if summary_rows else pd.DataFrame(columns=columns)
    return fitted, summary


def persist_segment_models(fitted: dict, summary: pd.DataFrame, out_dir: Path, subdir: str) -> None:
    """Persiste les modeles ajustes PAR SEGMENT (cf. fit_models_per_segment)
    sous out_dir/<subdir>/<segment_assaini>/{ols,ridge,rf}.joblib +
    coefficients/importances CSV -- meme convention de fichiers que le
    modele categorie (out_dir/ols.joblib etc.), un niveau plus bas.
    `summary` (schema ajuste/ecarte par segment x famille) est ecrit sous
    out_dir/<subdir>_summary.csv -- table d'audit centrale (quels segments
    ont un modele dedie, pourquoi les autres n'en ont pas)."""
    root = out_dir / subdir
    root.mkdir(parents=True, exist_ok=True)

    for segment, info in fitted.items():
        seg_dir = root / _sanitize_segment_name(segment)
        seg_dir.mkdir(parents=True, exist_ok=True)
        models = info["models"]
        design_columns = info["design_columns"]

        if "hedonic_ols" in models:
            joblib.dump(models["hedonic_ols"], seg_dir / "ols.joblib")
            models["hedonic_ols"].get_coefficients().to_csv(
                seg_dir / "coefficients.csv", index=False, encoding="utf-8-sig")
        if "ridge" in models:
            joblib.dump(models["ridge"], seg_dir / "ridge.joblib")
            models["ridge"].get_coefficients(feature_names=design_columns).to_csv(
                seg_dir / "ridge_coefficients.csv", index=False, encoding="utf-8-sig")
        if "random_forest" in models:
            joblib.dump(models["random_forest"], seg_dir / "rf.joblib")
            models["random_forest"].get_importances(feature_names=design_columns).to_csv(
                seg_dir / "rf_importances.csv", index=False, encoding="utf-8-sig")

        with open(seg_dir / "feature_schema.json", "w", encoding="utf-8") as fh:
            json.dump({
                "segment": str(segment),
                "continuous_features": info["continuous_features"],
                "categorical_features": info["categorical_features"],
                "design_matrix_columns": design_columns,
                "n_train": info["n_train"], "n_test": info["n_test"],
                "familles_ajustees": sorted(models.keys()),
                "retenu_pour_prediction": info.get("retenu_pour_prediction", {}),
            }, fh, ensure_ascii=False, indent=2)

    summary.to_csv(out_dir / f"{subdir}_summary.csv", index=False, encoding="utf-8-sig")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ORCHESTRATEUR PAR CATEGORIE
# ─────────────────────────────────────────────────────────────────────────────

def process_category(category: str) -> dict:
    logger.info(f"{'=' * 70}\nCategorie : {category}")
    out_dir = MODELS_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)

    df_pooled = load_pooled_category(category)

    continuous_features, categorical_base = _classify_features(df_pooled, exclude=_ID_COLUMNS | {"semaine"})
    categorical_features = categorical_base + ["marque"]
    logger.info(f"  [{category}] continues={continuous_features}")
    logger.info(f"  [{category}] categorielles={categorical_features}")

    # ── Clustering N1 (technique pur, persistable) -- calcule AVANT le split
    # train/test : cluster_direct doit etre present dans df_train ET df_test
    # pour permettre un ajustement de modele PAR cluster N1 plus bas (§3ter).
    # Reutilise continuous_features/categorical_base ci-dessus (marque deja
    # exclue via _ID_COLUMNS) -- meme appel que l'ancien n1_continuous/
    # n1_categorical, jamais recalcule deux fois pour le meme resultat.
    kmeans_n1, scaler_n1, n1_feature_cols, labels_n1, k_n1 = fit_n1_clustering(
        df_pooled, continuous_features, categorical_base)
    joblib.dump(kmeans_n1, out_dir / "kmeans_n1.joblib")
    joblib.dump(scaler_n1, out_dir / "scaler_n1.joblib")
    with open(out_dir / "n1_feature_schema.json", "w", encoding="utf-8") as fh:
        json.dump({
            "continuous_features": continuous_features, "categorical_features": categorical_base,
            "design_columns": n1_feature_cols, "k": int(k_n1),
        }, fh, ensure_ascii=False, indent=2)
    logger.info(f"  [{category}] clustering N1 : k={k_n1}, tailles={pd.Series(labels_n1).value_counts().to_dict()}")

    # cluster_direct attache directement par position -- fit_n1_clustering a
    # ete ajuste sur df_pooled dans le MEME ordre, aucun filtrage de marque
    # ne s'applique a N1 (cf. son docstring).
    df_pooled = df_pooled.copy()
    df_pooled["cluster_direct"] = df_pooled.index.map(lambda i: int(labels_n1[i]))

    # Filtre "marque negligeable" (MIN_BRAND_COUNT) : applique sur l'effectif
    # POOLE (toutes semaines) -- un simple seuil de comptage, pas un calcul
    # geometrique/par distance, jamais concerne par le correctif ci-dessous.
    # brand_plan seul est retenu de cet appel (rapporte tel quel) ; gamme_prix
    # est recalcule plus bas sur la seule semaine de reference, jamais ici.
    # N'AFFECTE PAS df_pooled lui-meme (les marques negligeables restent
    # visibles avec cluster_direct, gamme_prix/cluster_id a NaN -- seule la
    # table pooled_labeled.csv finale les exclut encore, cf. plus bas,
    # comportement inchange depuis avant ce correctif).
    _, brand_plan = compute_price_tiers(df_pooled)

    # gamme_prix + cluster_id (N2) : ancres sur LA SEULE semaine de reference
    # (n2_reference_week -- la plus ancienne disponible), jamais sur le pool
    # complet -- decision utilisateur du 2026-07-31 : un K-Means (et le qcut
    # de compute_price_tiers) ne doivent jamais melanger des observations de
    # semaines differentes. Les deux colonnes sont ensuite reappliquees par
    # produit (url) a TOUTES les semaines ou il apparait -- cf.
    # n2_reference_week (hedonic_model.py) pour la justification complete.
    reference_week = n2_reference_week(df_pooled)
    df_reference = df_pooled[df_pooled["semaine"] == reference_week]
    df_tiers_reference, _ = compute_price_tiers(df_reference)
    df_labeled_reference, unit_summary = compute_cluster_labels(
        df_tiers_reference, category, continuous_features=continuous_features, categorical_features=categorical_base,
    )
    unit_summary.insert(0, "semaine_reference", reference_week)

    n2_labels = df_labeled_reference[["url", "gamme_prix", "cluster_id"]].drop_duplicates(subset="url")
    df_pooled = df_pooled.merge(n2_labels, on="url", how="left")

    # ── Split train/test GROUPE PAR PRODUIT -- df_train/df_test heritent
    # desormais de cluster_direct/gamme_prix/cluster_id (colonnes ordinaires,
    # simple sous-ensemble de lignes de df_pooled).
    df_train, df_test = group_split_by_product(df_pooled)
    logger.info(f"  [{category}] train={len(df_train)} lignes, test={len(df_test)} lignes (groupe par produit).")

    X_train, y_train, X_test, y_test = build_matrices(df_train, df_test, continuous_features, categorical_features)
    assert not (set(X_train.columns) & FORBIDDEN_REGRESSORS), "garde-fou de circularite viole -- ne devrait jamais arriver"

    # ── HedonicOLS (inference : coefficients, p-values) ─────────────────────
    # cov_type="cluster" (groupe par url) : df_train poole plusieurs semaines
    # (cf. group_split_by_product) -- un meme produit y apparait donc
    # plusieurs fois, ce ne sont PAS des observations independantes. HC3 seul
    # suppose l'independance (corrige seulement l'heteroscedasticite) et
    # sous-estimerait les erreurs-types -- meme convention deja utilisee par
    # fit_strategy_c_pooled_time (hedonic_model.py) pour la meme raison.
    ols_cov_kwds = {"groups": df_train["url"]}
    ols = HedonicOLS(cov_type="cluster").fit(
        X_train, y_train, continuous_cols=continuous_features, cov_kwds=ols_cov_kwds
    )
    ols_pred_test = ols.predict(X_test)
    ols_metrics = evaluate_predictions(y_test.to_numpy(), ols_pred_test.to_numpy())
    ols_metrics.update({"adj_r2_train": float(ols.rsquared_adj), "aic": float(ols.aic), "bic": float(ols.bic)})
    joblib.dump(ols, out_dir / "ols.joblib")
    ols.get_coefficients().to_csv(out_dir / "coefficients.csv", index=False, encoding="utf-8-sig")

    # ── Ridge (degree=1 -- one-hot present, cf. avertissement ridge_model.py) ─
    # groups=url : X_train poole plusieurs semaines (meme raison que le
    # cluster-robuste ci-dessus) -- sans grouper la CV interne de
    # GridSearchCV par produit, un meme produit vu a 2+ semaines pourrait
    # se retrouver a la fois dans le pli d'entrainement ET le pli de
    # validation d'une meme iteration, biaisant la selection d'alpha.
    ridge = RidgeModel(degree=1).fit(X_train, y_train, groups=df_train["url"])
    ridge_pred_test = ridge.predict(X_test)
    ridge_metrics = evaluate_predictions(y_test.to_numpy(), np.asarray(ridge_pred_test))
    ridge_metrics["best_alpha"] = float(ridge.get_best_alpha())
    joblib.dump(ridge, out_dir / "ridge.joblib")
    ridge.get_coefficients(feature_names=list(X_train.columns)).to_csv(
        out_dir / "ridge_coefficients.csv", index=False, encoding="utf-8-sig"
    )

    # ── Random Forest ────────────────────────────────────────────────────────
    # groups=url : meme raison que pour Ridge ci-dessus.
    rf = RandomForestModel().fit(X_train, y_train, groups=df_train["url"])
    rf_pred_test = rf.predict(X_test)
    rf_metrics = evaluate_predictions(y_test.to_numpy(), np.asarray(rf_pred_test))
    rf_metrics["best_params"] = rf.grid_search_.best_params_
    rf_metrics["importance_note"] = (
        "Le graphique d'importance des variables de la Random Forest donne un indice utile, mais il peut "
        "favoriser certaines variables plus que d'autres. Il faut donc le lire comme un repère, pas comme "
        "une preuve de cause à effet. Pour une lecture plus robuste, consulter aussi l'importance par "
        "permutation et la comparaison avec OLS/Ridge."
    )
    joblib.dump(rf, out_dir / "rf.joblib")
    rf_importances_df = rf.get_importances(feature_names=list(X_train.columns))
    rf_importances_df.to_csv(out_dir / "rf_importances.csv", index=False, encoding="utf-8-sig")

    # Importance par permutation (sur le TEST, jamais biaisee vers les
    # variables continues/forte cardinalite comme MDI ci-dessus) -- cf.
    # rf_model.py::get_permutation_importance.
    rf_perm_importance_df = rf.get_permutation_importance(X_test, y_test, feature_names=list(X_train.columns))
    rf_perm_importance_df.to_csv(out_dir / "rf_permutation_importance.csv", index=False, encoding="utf-8-sig")

    # Accord entre les 3 modeles -- cf. compute_model_agreement ci-dessus.
    agreement_df, agreement_resume = compute_model_agreement(
        ols.get_coefficients(), ridge.get_coefficients(feature_names=list(X_train.columns)), rf_importances_df,
    )
    agreement_df.to_csv(out_dir / "model_agreement.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "category": category, "n_pooled": len(df_pooled), "n_train": len(df_train), "n_test": len(df_test),
        "weeks": discover_weeks(DATA_PROCESSED_DIR),
        "continuous_features": continuous_features, "categorical_features": categorical_features,
        "design_matrix_columns": list(X_train.columns),
        "hedonic_ols": ols_metrics, "ridge": ridge_metrics, "random_forest": rf_metrics,
        "model_agreement": agreement_resume,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2, default=str)

    # ── Donnees etiquetees (gamme_prix + N2 + N1) -- table "produits reels" ──
    # pooled_labeled.csv reste restreint aux marques suffisantes (comportement
    # inchange depuis avant ce correctif) -- seul l'ENSEMBLE de marques
    # survivantes nous interesse ici (brand_plan deja calcule plus haut sur
    # df_pooled AVANT la fusion gamme_prix/cluster_id, cf. plus haut) ; on ne
    # relance pas compute_price_tiers ici pour ne pas ecraser gamme_prix
    # (deja la version correcte, ancree semaine de reference) par un qcut sur
    # le pool multi-semaines.
    marques_retenues = set(brand_plan["marque"])
    df_labeled = df_pooled[df_pooled["marque"].isin(marques_retenues)]

    keep_cols = [c for c in df_labeled.columns if c not in ("specs_brutes",)]
    df_labeled[keep_cols].to_csv(out_dir / "pooled_labeled.csv", index=False, encoding="utf-8-sig")
    brand_plan.to_csv(out_dir / "brand_plan.csv", index=False, encoding="utf-8-sig")
    unit_summary.to_csv(out_dir / "unit_summary.csv", index=False, encoding="utf-8-sig")

    # ── Modelisation PAR CLUSTER, N1 et N2 SEPAREMENT (§3ter) -- un seul
    # modele categorie entiere (ci-dessus) est trop general pour une
    # prediction fine par segment ; ajuste OLS/Ridge/RF par cluster QUAND
    # l'effectif le permet (garde-fou par famille) ET quand le resultat BAT
    # reellement le modele categorie sur le meme test (cf. docstring de
    # fit_models_per_segment -- passer le seuil de lignes ne garantit pas
    # une bonne generalisation, mesure empiriquement sur ce projet). Sinon
    # le segment reste couvert par le seul modele categorie (repli explicite
    # -- weekly_report.py/prediction_utils.py lisent `retenu_pour_prediction`,
    # jamais `ajuste` seule, pour decider).
    category_models = {"hedonic_ols": ols, "ridge": ridge, "random_forest": rf}
    category_design_columns = list(X_train.columns)

    n1_fitted, n1_summary = fit_models_per_segment(
        df_train, df_test, "cluster_direct", continuous_features, categorical_features,
        category_models=category_models, category_design_columns=category_design_columns,
    )
    persist_segment_models(n1_fitted, n1_summary, out_dir, "clusters_n1")
    n1_n_retenus = n1_summary.loc[n1_summary["retenu_pour_prediction"], "segment"].nunique() if len(n1_summary) else 0
    logger.info(f"  [{category}] modeles par cluster N1 : {n1_n_retenus}/{len(n1_summary['segment'].unique()) if len(n1_summary) else 0} clusters avec un modele dedie retenu (bat le modele categorie)")

    n2_fitted, n2_summary = fit_models_per_segment(
        df_train, df_test, "cluster_id", continuous_features, categorical_features,
        category_models=category_models, category_design_columns=category_design_columns,
    )
    persist_segment_models(n2_fitted, n2_summary, out_dir, "clusters_n2")
    n2_n_retenus = n2_summary.loc[n2_summary["retenu_pour_prediction"], "segment"].nunique() if len(n2_summary) else 0
    logger.info(f"  [{category}] modeles par cluster N2 : {n2_n_retenus}/{len(n2_summary['segment'].unique()) if len(n2_summary) else 0} clusters avec un modele dedie retenu (bat le modele categorie)")

    logger.info(
        f"  [{category}] OK -- OLS adj_R2={ols_metrics['adj_r2_train']:.3f} | "
        f"Ridge R2_test={ridge_metrics['r2_log']:.3f} | RF R2_test={rf_metrics['r2_log']:.3f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Entraine et persiste les artefacts (modeles + clustering + donnees etiquetees) pour le dashboard."
    )
    parser.add_argument("--category", type=str, default=None, choices=CATEGORY_ORDER,
                         help="Ne traiter qu'une seule categorie (defaut : toutes).")
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORY_ORDER
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}
    for category in categories:
        summary[category] = process_category(category)

    logger.info(f"\n{'=' * 70}\nRESUME FINAL -- artefacts ecrits sous {MODELS_DIR}")
    for cat, m in summary.items():
        logger.info(
            f"  {cat:<24} : n_train={m['n_train']:>4} n_test={m['n_test']:>4} | "
            f"OLS adj_R2={m['hedonic_ols']['adj_r2_train']:.3f} | "
            f"Ridge R2={m['ridge']['r2_log']:.3f} | RF R2={m['random_forest']['r2_log']:.3f}"
        )


if __name__ == "__main__":
    main()
