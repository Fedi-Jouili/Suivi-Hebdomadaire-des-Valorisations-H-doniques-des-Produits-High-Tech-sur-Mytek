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


@functools.lru_cache(maxsize=None)
def raw_vs_clean_counts(category: str, week: int) -> dict:
    """Compare le nombre de produits SCRAPES (JSON brut) au nombre RETENU
    dans <categorie>_clean.csv pour une semaine -- rend visible l'ecart
    (produits sans prix fiable ou caracteristiques hors bornes ecartes en
    amont par le pipeline, cf. src/preprocessing/pipeline.py), jamais
    silencieux (principe du projet, cf. README)."""
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
# PLAGES DE VALEURS POUR LE FORMULAIRE DE PREDICTION (page 3)
# ─────────────────────────────────────────────────────────────────────────────

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
    """
    df = load_pooled_labeled(category)
    schema = load_metrics(category)
    continuous = schema["continuous_features"]
    categorical = [c for c in schema["categorical_features"] if c != "marque"]

    ranges = {"continuous": {}, "categorical": {}, "marque": []}
    for col in continuous:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        ranges["continuous"][col] = {
            "min": float(series.quantile(0.01)),
            "max": float(series.quantile(0.99)),
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
    "available_weeks", "last_n_weeks",
    "load_clean_category_week", "load_clean_category_recent", "load_pooled_category_full",
    "raw_vs_clean_counts",
    "ArtifactsMissingError", "artifacts_available",
    "load_metrics", "load_coefficients", "load_ridge_coefficients", "load_rf_importances",
    "load_pooled_labeled", "load_brand_plan", "load_unit_summary", "load_n1_feature_schema",
    "load_model_artifact", "get_feature_ranges", "category_label",
    "CATEGORY_ORDER", "CATEGORY_LABELS",
]
