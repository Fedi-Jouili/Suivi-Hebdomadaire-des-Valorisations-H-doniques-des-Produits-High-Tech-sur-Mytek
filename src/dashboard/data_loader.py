# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/data_loader.py
=============================================================================
ROLE :
    Point d'acces UNIQUE aux donnees et artefacts pour tout le dashboard --
    STRICTEMENT LECTURE SEULE : pd.read_csv/json.load/joblib.load
    uniquement, jamais de fit()/entrainement. Les artefacts sont produits
    en amont par `python -m src.models.save_artifacts` (src/models/
    save_artifacts.py) ; ce module echoue explicitement (jamais un
    fallback silencieux qui entrainerait un modele a la volee) si un
    artefact attendu est absent.

    Chemins lus via src.utils.config (DATA_PROCESSED_DIR, PROJECT_ROOT) --
    jamais de chemin absolu code en dur, cf. convention du reste du projet.

    Toutes les fonctions sont cachees (functools.lru_cache) : les fichiers
    sources ne changent pas pendant la duree de vie du process dashboard
    (ils sont regeneres hors-ligne par la pipeline/save_artifacts), inutile
    de relire le disque a chaque callback.
=============================================================================
"""

import functools
import json
from pathlib import Path

import joblib
import pandas as pd

from src.models.save_artifacts import MODELS_DIR, load_pooled_category
from src.models.weekly_report import REPORTS_DIR
from src.preprocessing.split import discover_weeks
from src.utils.config import CATEGORY_LABELS, CATEGORY_ORDER, DATA_PROCESSED_DIR, DATA_RAW_DIR

# ─────────────────────────────────────────────────────────────────────────────
# SEMAINES / PROVENANCE
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def available_weeks() -> tuple:
    """Semaines decouvertes sous data/processed/ -- jamais figees en dur."""
    return tuple(discover_weeks(DATA_PROCESSED_DIR))


def last_n_weeks(n: int = 4) -> tuple:
    """Les n dernieres semaines disponibles (Page 1 : "4 dernieres semaines").
    Retourne moins de n si moins de n semaines existent -- jamais une
    erreur, la degradation est explicite (cf. bandeau de provenance)."""
    weeks = available_weeks()
    return weeks[-n:] if weeks else ()


@functools.lru_cache(maxsize=1)
def last_data_update() -> str | None:
    """Date de derniere modification (mtime) la plus recente parmi les CSV
    de data/processed/ -- affichee dans l'en-tete du dashboard ("Derniere
    mise a jour"). None si aucun fichier trouve (degrade proprement,
    l'en-tete omet alors la mention plutot que d'afficher une date fausse)."""
    csvs = list(Path(DATA_PROCESSED_DIR).glob("week_*/*_clean.csv"))
    if not csvs:
        return None
    latest = max(csvs, key=lambda p: p.stat().st_mtime)
    from datetime import datetime
    return datetime.fromtimestamp(latest.stat().st_mtime).strftime("%d/%m/%Y")


# ─────────────────────────────────────────────────────────────────────────────
# DONNEES TRAITEES PAR SEMAINE (pour la page descriptive)
# ─────────────────────────────────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def load_clean_category_week(category: str, week: int) -> pd.DataFrame:
    """<categorie>_clean.csv d'UNE semaine. DataFrame vide (pas d'exception)
    si le fichier n'existe pas -- une semaine peut manquer pour une
    categorie (cf. meme convention que src.models.hedonic_model)."""
    path = DATA_PROCESSED_DIR / f"week_{week}" / f"{category}_clean.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["semaine"] = week
    return df


@functools.lru_cache(maxsize=None)
def load_clean_category_recent(category: str, n_weeks: int = 4) -> pd.DataFrame:
    """Concatene <categorie>_clean.csv des n dernieres semaines disponibles,
    avec reconciliation de schema (cf. src.models.save_artifacts, meme
    fonction reutilisee -- une colonne comme has_4g peut n'exister que sur
    certaines semaines, jamais une fuite de NaN silencieuse ici non plus)."""
    weeks = last_n_weeks(n_weeks)
    frames = [load_clean_category_week(category, w) for w in weeks]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    from src.models.save_artifacts import _reconcile_pooled_schema
    return _reconcile_pooled_schema(pd.concat(frames, ignore_index=True), category)


@functools.lru_cache(maxsize=None)
def load_pooled_category_full(category: str) -> pd.DataFrame:
    """Toutes les semaines disponibles poolees (memes donnees que celles
    utilisees pour entrainer les modeles persistes, cf. save_artifacts)."""
    return load_pooled_category(category)


@functools.lru_cache(maxsize=1)
def _load_raw_vs_clean_counts_full() -> pd.DataFrame:
    path = REPORTS_DIR / "raw_vs_clean_counts_hebdo.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@functools.lru_cache(maxsize=None)
def raw_vs_clean_counts(category: str, week: int) -> dict:
    """Compare le nombre de produits SCRAPES (JSON brut) au nombre RETENU
    dans <categorie>_clean.csv pour une semaine -- rend visible l'ecart
    (produits sans prix fiable ou caracteristiques hors bornes ecartes en
    amont par le pipeline, cf. src/preprocessing/pipeline.py), jamais
    silencieux (principe du projet, cf. README)."""
    df_counts = _load_raw_vs_clean_counts_full()
    if not df_counts.empty:
        mask = (df_counts["categorie"] == category) & (df_counts["semaine"] == week)
        match = df_counts.loc[mask]
        if not match.empty:
            row = match.iloc[0]
            n_raw = int(row["n_raw"])
            n_clean = int(row["n_clean"])
            n_excluded = int(row["n_excluded"])
            return {"n_raw": n_raw, "n_clean": n_clean, "n_excluded": n_excluded}

    raw_dir = DATA_RAW_DIR / f"week_{week}"
    n_raw = 0
    if raw_dir.exists():
        candidates = [f for f in raw_dir.glob(f"produits_{category}_*.json") if "partial" not in f.name]
        for f in candidates:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    n_raw += len(json.load(fh))
            except (json.JSONDecodeError, OSError):
                continue
    n_clean = len(load_clean_category_week(category, week))
    return {"n_raw": n_raw, "n_clean": n_clean, "n_excluded": max(n_raw - n_clean, 0)}


# ─────────────────────────────────────────────────────────────────────────────
# ARTEFACTS (models/<categorie>/, produits par save_artifacts.py)
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactsMissingError(RuntimeError):
    """Levee quand un artefact attendu n'existe pas -- toujours avec un
    message expliquant quelle commande executer pour le produire, jamais
    un fallback silencieux qui entrainerait un modele a la volee dans le
    dashboard (interdit, cf. docstring du module)."""


def _artifact_path(category: str, filename: str) -> Path:
    path = MODELS_DIR / category / filename
    if not path.exists():
        raise ArtifactsMissingError(
            f"Artefact manquant : {path}\n"
            f"Executer d'abord : python -m src.models.save_artifacts --category {category} "
            f"(ou sans --category pour toutes les categories)."
        )
    return path


@functools.lru_cache(maxsize=None)
def artifacts_available(category: str) -> bool:
    return (MODELS_DIR / category / "metrics.json").exists()


@functools.lru_cache(maxsize=None)
def load_metrics(category: str) -> dict:
    with open(_artifact_path(category, "metrics.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def load_coefficients(category: str) -> pd.DataFrame:
    """Coefficients HedonicOLS (feature, coefficient, std_err, p_value, pct_effect)."""
    return pd.read_csv(_artifact_path(category, "coefficients.csv"))


@functools.lru_cache(maxsize=None)
def load_ridge_coefficients(category: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(category, "ridge_coefficients.csv"))


@functools.lru_cache(maxsize=None)
def load_rf_importances(category: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(category, "rf_importances.csv"))


@functools.lru_cache(maxsize=None)
def load_rf_permutation_importances(category: str) -> pd.DataFrame:
    """Importance par permutation (test set, non biaisee par cardinalite,
    contrairement a MDI/load_rf_importances) -- cf. rf_model.py::
    get_permutation_importance."""
    return pd.read_csv(_artifact_path(category, "rf_permutation_importance.csv"))


@functools.lru_cache(maxsize=None)
def load_model_agreement(category: str) -> pd.DataFrame:
    """Accord de signe OLS/Ridge + correlation de rang RF/OLS, par feature
    -- cf. save_artifacts.py::compute_model_agreement."""
    return pd.read_csv(_artifact_path(category, "model_agreement.csv"))


@functools.lru_cache(maxsize=None)
def load_pooled_labeled(category: str) -> pd.DataFrame:
    """Donnees poolees + gamme_prix + cluster_id (N2) + cluster_direct (N1)
    -- sert de table "produits reels" (page Prediction) et de source pour
    l'assignation de gamme/segment N2 en 2 temps."""
    return pd.read_csv(_artifact_path(category, "pooled_labeled.csv"))


@functools.lru_cache(maxsize=None)
def load_brand_plan(category: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(category, "brand_plan.csv"))


@functools.lru_cache(maxsize=None)
def load_unit_summary(category: str) -> pd.DataFrame:
    return pd.read_csv(_artifact_path(category, "unit_summary.csv"))


@functools.lru_cache(maxsize=None)
def load_n1_feature_schema(category: str) -> dict:
    with open(_artifact_path(category, "n1_feature_schema.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


@functools.lru_cache(maxsize=None)
def load_model_artifact(category: str, name: str):
    """name in {"ols", "ridge", "rf", "kmeans_n1", "scaler_n1"}."""
    return joblib.load(_artifact_path(category, f"{name}.joblib"))


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORTS HEBDOMADAIRES (reports/, produits par src.models.weekly_report --
# meme convention lecture-seule que les artefacts de save_artifacts.py)
# ─────────────────────────────────────────────────────────────────────────────

def _report_path(filename: str) -> Path:
    path = REPORTS_DIR / filename
    if not path.exists():
        raise ArtifactsMissingError(
            f"Rapport manquant : {path}\nExecuter d'abord : python -m src.models.weekly_report"
        )
    return path


@functools.lru_cache(maxsize=1)
def _load_clustering_coverage_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("couverture_clustering_hebdo.csv"))


def load_clustering_coverage(category: str | None = None) -> pd.DataFrame:
    """Couverture (N1/N2) par semaine -- cf. src.models.weekly_report.coverage_by_week."""
    df = _load_clustering_coverage_full()
    return df[df["categorie"] == category].reset_index(drop=True) if category else df


@functools.lru_cache(maxsize=1)
def _load_cluster_geometric_means_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("cluster_prix_geometrique.csv"))


def load_cluster_geometric_means(category: str) -> pd.DataFrame:
    """Moyenne geometrique du prix par cluster (N1 + N2) et par semaine."""
    df = _load_cluster_geometric_means_full()
    return df[df["categorie"] == category].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_weekly_estimates_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("estimations_prix_hebdo.csv"))


def load_weekly_estimates(category: str) -> pd.DataFrame:
    """Prix reel vs estime (Hedonic OLS / Ridge / RF) par semaine, + erreur."""
    df = _load_weekly_estimates_full()
    return df[df["categorie"] == category].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_hedonic_price_index_full() -> pd.DataFrame:
    path = REPORTS_DIR / "indice_prix_hedonique_hebdo.csv"
    if not path.exists():
        raise ArtifactsMissingError(
            f"Rapport manquant : {path}\nExecuter d'abord : python -m src.models.weekly_report"
        )
    return pd.read_csv(path)


def load_hedonic_price_index(category: str) -> pd.DataFrame | None:
    """Indice de prix hedonique (effet fixe semaine, net des caracteristiques) --
    None si la categorie est exclue du pooling temporel (cf.
    src.models.hedonic_model.POOLED_TIME_EXCLUDED_CATEGORIES), jamais une erreur :
    l'exclusion est un choix methodologique documente, pas une donnee manquante."""
    df = _load_hedonic_price_index_full()
    sub = df[df["categorie"] == category].reset_index(drop=True)
    return sub if not sub.empty else None


@functools.lru_cache(maxsize=1)
def _load_catalog_composition_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("composition_catalogue_hebdo.csv"))


def load_catalog_composition(category: str) -> pd.DataFrame:
    """Moyenne des caracteristiques techniques continues du catalogue, par
    semaine -- colonnes non pertinentes pour la categorie restent NaN
    (features propres a d'autres categories, cf. weekly_report.py)."""
    df = _load_catalog_composition_full()
    sub = df[df["categorie"] == category].reset_index(drop=True)
    return sub.dropna(axis=1, how="all")


@functools.lru_cache(maxsize=1)
def _load_marque_gamme_estimates_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("marque_gamme_estimations_hebdo.csv"))


def load_marque_gamme_estimates(category: str) -> pd.DataFrame:
    """Prix reel (moyenne geometrique) vs estime par Ridge/Hedonic/RF, PAR
    CLUSTER (marque x gamme x sous-cluster technique) et par semaine --
    cf. src.models.weekly_report.marque_gamme_model_estimates."""
    df = _load_marque_gamme_estimates_full()
    return df[df["categorie"] == category].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_cluster_transitions_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("transitions_cluster_hebdo.csv"))


def load_cluster_transitions(category: str) -> pd.DataFrame:
    """Classification semaine-a-semaine (grille prix reel x prix estime) de
    chaque cluster -- cf. src.models.weekly_report.cluster_transitions."""
    df = _load_cluster_transitions_full()
    return df[df["categorie"] == category].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_cluster_stability_n2_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("stabilite_clustering_n2.csv"))


def load_cluster_stability_n2(category: str) -> pd.DataFrame:
    """Stabilite bootstrap (ARI) du clustering N2 par unite marque x gamme --
    cf. src.models.weekly_report.cluster_stability_n2."""
    df = _load_cluster_stability_n2_full()
    return df[df["categorie"] == category].reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def _load_k_selection_justification_full() -> pd.DataFrame:
    return pd.read_csv(_report_path("justification_k_clustering.csv"))


def load_k_selection_justification(category: str) -> pd.DataFrame:
    """Silhouette de TOUS les k testes (N1 et N2), retenus ou non -- cf.
    src.models.weekly_report.k_selection_justification."""
    df = _load_k_selection_justification_full()
    return df[df["categorie"] == category].reset_index(drop=True)


@functools.lru_cache(maxsize=None)
def weekly_reports_available() -> bool:
    return (REPORTS_DIR / "couverture_clustering_hebdo.csv").exists()


# ─────────────────────────────────────────────────────────────────────────────
# PLAGES DE VALEURS POUR LE FORMULAIRE DE PREDICTION (page 3)
# ─────────────────────────────────────────────────────────────────────────────

# Caracteristiques "continues" mais en realite QUANTIFIEES par construction
# materielle (Go de RAM/stockage vendus par paliers -- 4/8/16/32... jamais
# une valeur arbitraire entre deux paliers) : un slider suggererait des
# valeurs qui n'existent dans aucun produit reel. Presentees en Select
# (options = valeurs REELLEMENT observees, cf. discret_options ci-dessous),
# jamais un NumberInput/slider -- decision utilisateur du 2026-07-25.
DISCRETE_OPTION_FEATURES = {"ram_go", "stockage_go"}

# Plancher physique supplementaire pour ram_go : le bug de confusion
# cache/RAM documente dans src/preprocessing/clean.py produit des valeurs
# proches de 0 (ex. 0,0039 Go) qui passent les bornes de plausibilite
# (VALIDITY_BOUNDS accepte "ram": (0.0, ...), 0 est une borne INCLUSE, pas
# suspecte) -- invisibles sur un slider p1-p99 (juste une position de
# curseur inutilisee), mais afficheraient une option absurde et
# selectionnable dans un Select. 1 Go est le plus petit palier RAM
# reellement commercialise dans ce catalogue -- jamais une grandeur
# continue, aucune ambiguite a arrondir vers le bas.
_DISCRETE_OPTION_FLOOR = {"ram_go": 1.0}


def get_feature_ranges(category: str) -> dict:
    """
    Plages PRATIQUES (percentiles 1-99, pas min/max bruts) pour chaque
    caracteristique continue, et modalites observees pour chaque
    caracteristique categorielle -- sert a construire des sliders/dropdowns
    avec des bornes sensees dans la page Prediction.

    Percentiles plutot que min/max : le catalogue contient des residus
    d'anomalies de donnees deja documentees (ex. ram_go proche de 0 sur
    certains PC portables/telephones -- bug de confusion cache processeur
    corrige a la source pour les futurs scrapes mais dont quelques lignes
    historiques subsistent, cf. src/preprocessing/clean.py). Un slider
    borne sur le minimum brut serait inutilisable (0 a 64 Go de RAM avec
    l'essentiel du curseur invalide) -- p1/p99 restent honnetes sur la
    distribution reelle sans se laisser dicter par un point aberrant isole.

    DISCRETE_OPTION_FEATURES (ram_go, stockage_go) : memes bornes p1-p99
    pour ecarter les memes anomalies, mais la liste des valeurs DISTINCTES
    observees dans cette plage est retournee (ranges["discrete_options"])
    plutot qu'un min/max continu -- ce sont des paliers materiels, pas une
    grandeur continue.
    """
    df = load_pooled_labeled(category)
    schema = load_metrics(category)
    continuous = schema["continuous_features"]
    categorical = [c for c in schema["categorical_features"] if c != "marque"]

    ranges = {"continuous": {}, "discrete_options": {}, "categorical": {}, "marque": []}
    for col in continuous:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        p1, p99 = float(series.quantile(0.01)), float(series.quantile(0.99))
        if col in DISCRETE_OPTION_FEATURES:
            floor = _DISCRETE_OPTION_FLOOR.get(col, p1)
            values = sorted(v for v in series.unique().tolist() if max(p1, floor) <= v <= p99)
            if values:
                ranges["discrete_options"][col] = values
            continue
        ranges["continuous"][col] = {
            "min": p1,
            "max": p99,
            "median": float(series.median()),
            "step": _sensible_step(series),
        }
    for col in categorical:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 2:
            continue  # indicatrice has_* -- traitee comme checkbox, pas dropdown
        values = sorted(v for v in df[col].dropna().unique().tolist())
        ranges["categorical"][col] = values

    ranges["marque"] = sorted(load_brand_plan(category)["marque"].tolist())
    ranges["binary_flags"] = [
        c for c in categorical
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 2
    ]
    return ranges


def _sensible_step(series: pd.Series) -> float:
    """Pas de slider raisonnable : 1 pour des valeurs entieres courantes
    (Go de RAM/stockage), 0.1 sinon (ex. taille d'ecran en pouces)."""
    if (series.dropna() % 1 == 0).mean() > 0.9:
        return 1.0
    return 0.1


# ─────────────────────────────────────────────────────────────────────────────
# LIBELLES
# ─────────────────────────────────────────────────────────────────────────────

def category_label(category: str) -> str:
    return CATEGORY_LABELS.get(category, category)


__all__ = [
    "available_weeks", "last_n_weeks", "last_data_update",
    "load_clean_category_week", "load_clean_category_recent", "load_pooled_category_full",
    "raw_vs_clean_counts",
    "ArtifactsMissingError", "artifacts_available",
    "load_metrics", "load_coefficients", "load_ridge_coefficients", "load_rf_importances",
    "load_rf_permutation_importances", "load_model_agreement",
    "load_pooled_labeled", "load_brand_plan", "load_unit_summary", "load_n1_feature_schema",
    "load_model_artifact", "get_feature_ranges", "DISCRETE_OPTION_FEATURES", "category_label",
    "load_clustering_coverage", "load_cluster_geometric_means", "load_weekly_estimates",
    "load_hedonic_price_index", "load_catalog_composition",
    "load_marque_gamme_estimates", "load_cluster_transitions", "weekly_reports_available",
    "load_cluster_stability_n2", "load_k_selection_justification",
    "CATEGORY_ORDER", "CATEGORY_LABELS",
]
