# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/prediction_utils.py
=============================================================================
ROLE :
    Logique de la page Prediction : encoder un produit HYPOTHETIQUE compose
    par l'utilisateur exactement comme la matrice de design d'entrainement
    (src/models/save_artifacts.py), predire son prix (retro-transforme
    correctement, cf. note de biais), l'assigner a un segment N1/N2, et
    retrouver des produits reels similaires. Strictement LECTURE SEULE sur
    les artefacts (joblib.load/pd.read_csv) -- jamais de fit() ici.

    Assignation N2 EN DEUX TEMPS (decision utilisateur du 2026-07-21) : N2
    (marque x gamme_prix) ne peut PAS etre calculee directement pour un
    produit hypothetique, puisque gamme_prix est un quantile du PRIX --
    justement ce qu'on cherche a predire (cf. reports/audit_code.md §3.1,
    meme raisonnement que le correctif de hedonic_model.py). On procede
    donc en 2 temps :
      1. Prix PROVISOIRE = prediction du modele choisi (OLS/Ridge/RF,
         AUCUN d'entre eux n'utilise gamme_prix/cluster_id comme
         regresseur -- seulement "marque", exogene, cf. correctif
         hedonic_model.py).
      2. Ce prix provisoire est situe dans les bornes de gamme DEJA
         OBSERVEES pour la marque choisie (min/max de prix_tnd par gamme,
         directement lues depuis pooled_labeled.csv -- pas de nouveau
         qcut, juste une lecture des bornes existantes) -> gamme estimee
         -> segment N2 assigne = celui du produit reel le plus proche
         (distance technique) parmi (marque, gamme estimee).
=============================================================================
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.neighbors import NearestNeighbors

from src.dashboard.data_loader import (
    load_metrics,
    load_model_artifact,
    load_n1_feature_schema,
    load_pooled_labeled,
)


# ─────────────────────────────────────────────────────────────────────────────
# ENCODAGE D'UNE LIGNE HYPOTHETIQUE (identique a build_design_matrix /
# fit_n1_clustering, sans reimplementer pandas.get_dummies sur 1 seule ligne
# -- get_dummies sur une ligne unique ne produirait qu'UNE colonne, jamais
# l'ensemble des colonnes vues a l'entrainement)
# ─────────────────────────────────────────────────────────────────────────────

def encode_hypothetical_row(design_columns: list, continuous_features: list, categorical_features: list,
                             values: dict) -> pd.DataFrame:
    """Construit une ligne (1, len(design_columns)) alignee EXACTEMENT sur
    les colonnes vues a l'entrainement (drop_first ou non, peu importe --
    fonctionne dans les deux cas : une modalite absente de design_columns,
    ex. la modalite de reference droppee, laisse simplement tous ses
    dummies a 0, ce qui est la representation correcte)."""
    row = {}
    for col in design_columns:
        if col in continuous_features:
            row[col] = float(values.get(col, 0.0))
            continue
        if col in categorical_features:
            # indicatrice deja binaire (has_*) passee telle quelle
            row[col] = float(values.get(col, 0))
            continue
        matched_cat = max(
            (c for c in categorical_features if col.startswith(f"{c}_")),
            key=len, default=None,
        )
        if matched_cat is None:
            row[col] = 0.0
            continue
        chosen = str(values.get(matched_cat, ""))
        suffix = col[len(matched_cat) + 1:]
        row[col] = 1.0 if chosen == suffix else 0.0
    return pd.DataFrame([row], columns=design_columns)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION DE PRIX
# ─────────────────────────────────────────────────────────────────────────────

MODEL_LABELS = {"ols": "Hedonic OLS", "ridge": "Ridge", "rf": "Random Forest"}


def predict_price(category: str, model_name: str, values: dict) -> dict:
    """
    Predit le prix d'un produit hypothetique.

    Retro-transformation : log(prix) -> prix par np.exp(), SANS correction
    de biais de retransformation (Duan/Miller) -- meme choix documente que
    src/models/save_artifacts.py::evaluate_predictions (sous-estime
    legerement le prix moyen "vrai", explicite dans le resultat retourne).

    Intervalle : disponible UNIQUEMENT pour Hedonic OLS (statsmodels
    .get_prediction(), intervalle de prediction a 95% sur log(prix) puis
    retro-transforme) -- Ridge/RF n'offrent pas nativement d'intervalle
    (ni l'un ni l'autre ne sont bayesiens/quantile), signale comme tel.

    Returns: dict {price, price_lower, price_upper (ou None), log_price,
        model_label, has_interval, note}
    """
    metrics = load_metrics(category)
    design_columns = metrics["design_matrix_columns"]
    continuous_features = metrics["continuous_features"]
    categorical_features = metrics["categorical_features"]

    X_row = encode_hypothetical_row(design_columns, continuous_features, categorical_features, values)
    model = load_model_artifact(category, model_name)

    if model_name == "ols":
        log_pred = float(model.predict(X_row).iloc[0])
        X_aug = sm.add_constant(X_row, has_constant="add").astype(float)
        X_aug = X_aug.reindex(columns=model.exog_.columns, fill_value=0.0)
        pred_summary = model.result_.get_prediction(X_aug).summary_frame(alpha=0.05)
        log_lo = float(pred_summary["obs_ci_lower"].iloc[0])
        log_hi = float(pred_summary["obs_ci_upper"].iloc[0])
        price_lower, price_upper, has_interval = float(np.exp(log_lo)), float(np.exp(log_hi)), True
    elif model_name == "ridge":
        log_pred = float(np.asarray(model.predict(X_row))[0])
        price_lower = price_upper = None
        has_interval = False
    elif model_name == "rf":
        log_pred = float(np.asarray(model.predict(X_row))[0])
        price_lower = price_upper = None
        has_interval = False
    else:
        raise ValueError(f"Modele inconnu : {model_name}")

    price = float(np.exp(log_pred))
    return {
        "price": price, "price_lower": price_lower, "price_upper": price_upper,
        "log_price": log_pred, "model_label": MODEL_LABELS[model_name], "has_interval": has_interval,
        "note": (
            "Intervalle de prédiction à 95% (statsmodels), rétro-transformé de log(prix) — pas d'ajustement "
            "supplémentaire au-delà de exp()." if has_interval else
            f"{MODEL_LABELS[model_name]} ne fournit pas nativement d'intervalle de prédiction."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION N1 (technique, calculable pour un produit hypothetique)
# ─────────────────────────────────────────────────────────────────────────────

def assign_n1_cluster(category: str, values: dict) -> int:
    schema = load_n1_feature_schema(category)
    row = encode_hypothetical_row(
        schema["design_columns"], schema["continuous_features"], schema["categorical_features"], values,
    )
    scaler = load_model_artifact(category, "scaler_n1")
    kmeans = load_model_artifact(category, "kmeans_n1")
    X_scaled = scaler.transform(row)
    return int(kmeans.predict(X_scaled)[0])


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION N2 EN 2 TEMPS (prix provisoire -> gamme -> segment reel le
# plus proche) -- cf. docstring du module
# ─────────────────────────────────────────────────────────────────────────────

def assign_n2_segment(category: str, marque: str, predicted_price: float, values: dict) -> dict:
    """
    Retourne {gamme_estimee, cluster_id, n_comparables, distance} ou un
    dict avec gamme_estimee=None si la marque n'a pas de gamme definie
    (ne devrait pas arriver : le formulaire ne propose que des marques
    retenues par compute_price_tiers, cf. get_feature_ranges)."""
    df = load_pooled_labeled(category)
    df_brand = df[df["marque"] == marque]
    if df_brand.empty:
        return {"gamme_estimee": None, "cluster_id": None, "n_comparables": 0}

    bounds = df_brand.groupby("gamme_prix")["prix_tnd"].agg(["min", "max"]).reset_index()
    # Gamme dont la plage [min, max] deja observee CONTIENT le prix
    # provisoire ; a defaut (prix hors de toutes les plages -- produit
    # hypothetique plus extreme que tout l'historique de la marque), la
    # gamme dont la borne la plus proche est la plus proche du prix
    # provisoire (jamais une erreur, un produit hypothetique peut
    # legitimement depasser l'historique).
    contained = bounds[(bounds["min"] <= predicted_price) & (predicted_price <= bounds["max"])]
    if not contained.empty:
        gamme = contained.iloc[0]["gamme_prix"]
    else:
        bounds["dist"] = bounds.apply(
            lambda r: min(abs(predicted_price - r["min"]), abs(predicted_price - r["max"])), axis=1
        )
        gamme = bounds.sort_values("dist").iloc[0]["gamme_prix"]

    df_unit = df_brand[df_brand["gamme_prix"] == gamme]
    if df_unit.empty:
        return {"gamme_estimee": gamme, "cluster_id": None, "n_comparables": 0}

    schema = load_n1_feature_schema(category)  # memes caracteristiques techniques que N1
    tech_cols = [c for c in schema["continuous_features"] if c in df_unit.columns]
    if not tech_cols:
        nearest = df_unit.iloc[0]
        return {"gamme_estimee": gamme, "cluster_id": nearest["cluster_id"], "n_comparables": len(df_unit)}

    row_vals = np.array([[float(values.get(c, 0.0)) for c in tech_cols]])
    unit_vals = df_unit[tech_cols].astype(float).to_numpy()
    nn = NearestNeighbors(n_neighbors=1).fit(unit_vals)
    dist, idx = nn.kneighbors(row_vals)
    nearest_row = df_unit.iloc[idx[0][0]]
    return {
        "gamme_estimee": gamme, "cluster_id": nearest_row["cluster_id"],
        "n_comparables": len(df_unit), "distance": float(dist[0][0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRODUITS REELS SIMILAIRES (distance technique dans l'espace standardise N1)
# ─────────────────────────────────────────────────────────────────────────────

def find_similar_products(category: str, values: dict, top_n: int = 10) -> pd.DataFrame:
    """Plus proches voisins REELS (memes caracteristiques standardisees que
    le clustering N1, cf. scaler_n1) -- sert de verification de bon sens
    pour la prediction (page 3)."""
    schema = load_n1_feature_schema(category)
    df = load_pooled_labeled(category)
    scaler = load_model_artifact(category, "scaler_n1")

    row = encode_hypothetical_row(
        schema["design_columns"], schema["continuous_features"], schema["categorical_features"], values,
    )

    cont = [c for c in schema["continuous_features"] if c in df.columns]
    cat = [c for c in schema["categorical_features"] if c in df.columns]
    X_numeric = df[cont].astype(float) if cont else pd.DataFrame(index=df.index)
    X_categorical = pd.get_dummies(df[cat].astype(str), columns=cat) if cat else pd.DataFrame(index=df.index)
    X_corpus = pd.concat([X_numeric, X_categorical], axis=1).reindex(columns=schema["design_columns"], fill_value=0.0)

    X_corpus_scaled = scaler.transform(X_corpus)
    X_row_scaled = scaler.transform(row)

    n_neighbors = min(top_n, len(df))
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_corpus_scaled)
    distances, indices = nn.kneighbors(X_row_scaled)

    result = df.iloc[indices[0]].copy()
    result["distance"] = distances[0]
    return result.sort_values("distance").reset_index(drop=True)
