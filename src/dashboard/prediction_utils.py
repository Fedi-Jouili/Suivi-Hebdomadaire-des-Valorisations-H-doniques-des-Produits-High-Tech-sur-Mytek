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
    load_hedonic_price_index,
    load_metrics,
    load_model_artifact,
    load_n1_feature_schema,
    load_pooled_labeled,
    load_retained_cluster_models,
)

# model_name (page Prediction, "ols"/"ridge"/"rf") -> famille (cf.
# save_artifacts.fit_models_per_segment, "hedonic_ols"/"ridge"/"random_forest")
# -- deux vocabulaires historiquement distincts (le premier prefixe les noms
# de fichiers .joblib, le second suit metrics.json/marque_gamme_estimations_
# hebdo.csv) -- jamais unifies pour ne pas casser les artefacts deja
# persistes, cette table de correspondance est la SEULE traduction.
_FAMILLE_BY_MODEL_NAME = {"ols": "hedonic_ols", "ridge": "ridge", "rf": "random_forest"}


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


def _week_adjustment_pct(category: str, week) -> tuple:
    """
    Ajustement (%) a appliquer au prix de base pour "situer" la prediction
    a une semaine donnee, a partir de l'indice de prix hedonique deja
    persiste (reports/indice_prix_hedonique_hebdo.csv, cf.
    src.models.weekly_report.hedonic_price_index) -- jamais recalcule ici
    (le dashboard ne fit() jamais, cf. docstring du module).

    Les 3 modeles persistes (OLS/Ridge/RF) sont entraines sur les donnees
    POOLEES sans variable "semaine" -- leur prediction represente donc
    implicitement une moyenne sur toutes les semaines poolees, pas la
    semaine 1 (la reference 0% de l'indice). L'ajustement est donc calcule
    RELATIVEMENT A LA MOYENNE de l'indice sur les semaines disponibles
    (indice_semaine - moyenne(indice)), pas relativement a la semaine 1 --
    sans quoi toutes les predictions seraient systematiquement biaisees
    dans le meme sens par rapport a ce que le modele a reellement appris.

    Returns: (adjustment_pct: float | None, weeks_used: list[int]) --
        None si la categorie est exclue de l'indice temporel (cf.
        POOLED_TIME_EXCLUDED_CATEGORIES) ou si la semaine demandee n'y
        figure pas -- jamais une erreur, un ajustement simplement absent.
    """
    if week is None:
        return None, []
    idx = load_hedonic_price_index(category)
    if idx is None or idx.empty:
        return None, []
    row = idx[idx["semaine"] == int(week)]
    if row.empty:
        return None, sorted(idx["semaine"].unique().tolist())
    mean_idx = float(idx["indice_prix_ajuste_qualite_pct"].mean())
    week_idx = float(row["indice_prix_ajuste_qualite_pct"].iloc[0])
    return week_idx - mean_idx, sorted(idx["semaine"].unique().tolist())


def _predict_with_model(model, model_name: str, design_columns: list, continuous_features: list,
                         categorical_features: list, values: dict, week, category: str) -> dict:
    """Coeur de la prediction, factorise pour etre reutilise SUR N'IMPORTE
    QUEL modele (categorie entiere, ou par cluster N1/N2 -- meme classe
    HedonicOLS/RidgeModel/RandomForestModel dans les 3 cas, cf.
    save_artifacts.fit_models_per_segment) -- jamais deux implementations
    paralleles de la retro-transformation/l'intervalle/l'ajustement semaine.
    Voir predict_price pour la description complete du resultat retourne
    (sans "price_source", ajoute par l'appelant cluster-aware)."""
    X_row = encode_hypothetical_row(design_columns, continuous_features, categorical_features, values)

    if model_name == "ols":
        log_pred = float(model.predict(X_row).iloc[0])
        X_aug = sm.add_constant(X_row, has_constant="add").astype(float)
        X_aug = X_aug.reindex(columns=model.exog_.columns, fill_value=0.0)
        pred_summary = model.result_.get_prediction(X_aug).summary_frame(alpha=0.05)
        log_lo = float(pred_summary["obs_ci_lower"].iloc[0])
        log_hi = float(pred_summary["obs_ci_upper"].iloc[0])
        price_lower, price_upper, has_interval = float(np.exp(log_lo)), float(np.exp(log_hi)), True
    elif model_name in ("ridge", "rf"):
        log_pred = float(np.asarray(model.predict(X_row))[0])
        price_lower = price_upper = None
        has_interval = False
    else:
        raise ValueError(f"Modele inconnu : {model_name}")

    price = float(np.exp(log_pred))

    adjustment_pct, weeks_used = _week_adjustment_pct(category, week)
    if adjustment_pct is not None:
        factor = 1 + adjustment_pct / 100
        price *= factor
        if price_lower is not None:
            price_lower *= factor
            price_upper *= factor
        week_note = (
            f" Ajusté pour S{int(week)} : indice hédonique {adjustment_pct:+.2f} % par rapport à la moyenne "
            f"poolée (semaines {min(weeks_used)}–{max(weeks_used)}, cf. page Évolution hebdomadaire)."
        )
    elif week is not None:
        week_note = (
            f" Aucun ajustement disponible pour S{int(week)} sur « {category} » (indice de prix hédonique non "
            f"calculé pour cette catégorie/semaine) — prix non ajusté, moyenne poolée sur toutes les semaines."
        )
    else:
        week_note = ""

    return {
        "price": price, "price_lower": price_lower, "price_upper": price_upper,
        "log_price": log_pred, "model_label": MODEL_LABELS[model_name], "has_interval": has_interval,
        "week_adjustment_pct": adjustment_pct,
        "note": (
            "Intervalle de prédiction à 95% (statsmodels), rétro-transformé de log(prix) — pas d'ajustement "
            "supplémentaire au-delà de exp()." if has_interval else
            f"{MODEL_LABELS[model_name]} ne fournit pas nativement d'intervalle de prédiction."
        ) + week_note,
    }


def predict_price(category: str, model_name: str, values: dict, week=None) -> dict:
    """
    Predit le prix d'un produit hypothetique avec le modele CATEGORIE
    ENTIERE (cf. predict_price_cluster_aware pour la version qui prefere un
    modele de cluster N1/N2 quand un est retenu -- decision utilisateur du
    2026-08-01, page Prediction).

    Retro-transformation : log(prix) -> prix par np.exp(), SANS correction
    de biais de retransformation (Duan/Miller) -- meme choix documente que
    src/models/save_artifacts.py::evaluate_predictions (sous-estime
    legerement le prix moyen "vrai", explicite dans le resultat retourne).

    Intervalle : disponible UNIQUEMENT pour Hedonic OLS (statsmodels
    .get_prediction(), intervalle de prediction a 95% sur log(prix) puis
    retro-transforme) -- Ridge/RF n'offrent pas nativement d'intervalle
    (ni l'un ni l'autre ne sont bayesiens/quantile), signale comme tel.

    week : semaine choisie dans le formulaire -- applique l'ajustement de
        _week_adjustment_pct (indice hedonique deja persiste) au prix ET a
        l'intervalle, si disponible pour cette categorie/semaine.

    Returns: dict {price, price_lower, price_upper (ou None), log_price,
        model_label, has_interval, week_adjustment_pct (ou None), note}
    """
    metrics = load_metrics(category)
    model = load_model_artifact(category, model_name)
    return _predict_with_model(
        model, model_name, metrics["design_matrix_columns"],
        metrics["continuous_features"], metrics["categorical_features"], values, week, category,
    )


def _resolve_n2_model_with_fallback(cluster_id: str | None, n2_models: dict, famille: str):
    """
    Recherche HIERARCHIQUE d'un modele N2 retenu pour la famille demandee,
    du plus specifique au plus general (decision utilisateur du 2026-08-03) :

        1. Modele dedie du cluster N2 exact  (cluster_id tel quel)
        2. Modele dedie marque x gamme       (n'importe quel cluster
           "MARQUE::GAMME::*" retenu pour cette famille)
        3. Modele dedie marque               (n'importe quel cluster
           "MARQUE::*" retenu pour cette famille)
        4. Aucun modele N2 utilisable        -> repli categorie (appelant)

    cluster_id est au format "MARQUE::GAMME::cN" (cf.
    hedonic_model.compute_cluster_labels). Si le format est inattendu
    (moins de 3 parties apres split sur "::"), les niveaux intermediaires
    sont simplement ignores (jamais une erreur -- on tombe directement sur
    le repli categorie, comportement le plus conservateur).

    Args:
        cluster_id: identifiant du cluster N2 assigne (ou None).
        n2_models: dict retourne par load_retained_cluster_models("clusters_n2")
            -- {segment_str: {famille: {model, design_columns, ...}}}.
        famille: cle de famille dans _FAMILLE_BY_MODEL_NAME.values()
            ("hedonic_ols"/"ridge"/"random_forest").

    Returns:
        (info, price_source) -- info est le dict {model, design_columns, ...}
        du modele retenu, ou None si aucun niveau n'a abouti ; price_source
        est le niveau utilise ("n2", "n2_marque_gamme", "n2_marque") ou None.
    """
    if cluster_id is None or not n2_models:
        return None, None

    cluster_id_str = str(cluster_id)

    # ── Niveau 1 : cluster N2 exact ──────────────────────────────────────
    info = n2_models.get(cluster_id_str, {}).get(famille)
    if info is not None:
        return info, "n2"

    # Decompose cluster_id = "MARQUE::GAMME::cN"
    parts = cluster_id_str.split("::")
    if len(parts) < 3:
        # Format inattendu -- aucun niveau intermediaire deductible.
        return None, None

    marque_part = parts[0]
    gamme_part = parts[1]
    prefix_marque_gamme = f"{marque_part}::{gamme_part}::"

    # ── Niveau 2 : n'importe quel cluster du meme couple marque x gamme ──
    for segment, segment_families in sorted(n2_models.items()):
        if segment.startswith(prefix_marque_gamme) and famille in segment_families:
            return segment_families[famille], "n2_marque_gamme"

    # ── Niveau 3 : n'importe quel cluster de la meme marque ─────────────
    prefix_marque = f"{marque_part}::"
    for segment, segment_families in sorted(n2_models.items()):
        if segment.startswith(prefix_marque) and famille in segment_families:
            return segment_families[famille], "n2_marque"

    # ── Niveau 4 : aucun modele N2 utilisable ────────────────────────────
    return None, None


def predict_price_cluster_aware(category: str, model_name: str, values: dict, week=None,
                                 segmentation: str = "n2") -> dict:
    """
    Meme resultat que predict_price, mais prefere un modele de CLUSTER au
    modele categorie entiere des qu'un est RETENU pour la famille demandee
    (bat le modele categorie sur son propre test hors-echantillon, cf.
    save_artifacts.fit_models_per_segment) -- decision utilisateur du
    2026-08-01 : un seul modele par categorie est trop general.

    `segmentation` ("n1" ou "n2", cf. le selecteur "Type de segmentation" de
    la page Prediction) determine QUEL NIVEAU produit le resultat FINAL --
    correctif du 2026-08-01bis : auparavant, la fonction essayait TOUJOURS
    N2 puis retombait sur N1 puis categorie, quel que soit ce selecteur
    (qui ne changeait alors que l'affichage du "segment assigne", jamais le
    PRIX -- deux segmentations differentes affichaient donc silencieusement
    le meme prix, signale par l'utilisateur comme un bug). Desormais :
      - segmentation="n1" : resultat FINAL = modele N1 (cluster_direct) si
        retenu pour cette famille, sinon modele categorie -- le cluster N2
        n'entre JAMAIS dans le resultat affiche.
      - segmentation="n2" : resultat FINAL avec FALLBACK HIERARCHIQUE
        (decision utilisateur du 2026-08-03, cf. _resolve_n2_model_with_
        fallback) : modele dedie du cluster N2 exact, sinon modele dedie
        marque x gamme, sinon modele dedie marque, sinon modele categorie.
        Le cluster N1 n'est utilise qu'en INTERNE, pour le prix PROVISOIRE
        necessaire a assigner la gamme (cf. section 2 temps en tete de
        module), jamais comme resultat final meme s'il a lui-meme un modele
        retenu : un utilisateur qui choisit explicitement "N2" doit voir
        soit un resultat N2 (a l'un des 3 niveaux de granularite), soit un
        repli categorie explicite, jamais un N1 substitue silencieusement.
    Les deux selections peuvent donc legitimement produire des prix
    DIFFERENTS (chacune son propre niveau de modele) -- ou le MEME prix si
    aucun des deux clusters n'a de modele retenu (repli categorie commun
    aux deux, alors visible via price_source="categorie" dans les deux cas).

    Returns: meme dict que predict_price, + "price_source"
        ("n2"/"n2_marque_gamme"/"n2_marque"/"n1"/"categorie") -- jamais
        ambigu sur quel modele a produit le prix retourne.
    """
    famille = _FAMILLE_BY_MODEL_NAME[model_name]
    metrics = load_metrics(category)
    category_model = load_model_artifact(category, model_name)
    category_design_columns = metrics["design_matrix_columns"]
    category_continuous = metrics["continuous_features"]
    category_categorical = metrics["categorical_features"]

    def _predict_category():
        result = _predict_with_model(
            category_model, model_name, category_design_columns,
            category_continuous, category_categorical, values, week, category,
        )
        result["price_source"] = "categorie"
        return result

    # ── Prix PROVISOIRE : modele N1 retenu si disponible, sinon categorie --
    # sert UNIQUEMENT a situer la gamme (etape suivante) ; devient aussi le
    # resultat FINAL si segmentation="n1".
    n1_cluster = assign_n1_cluster(category, values)
    n1_models = load_retained_cluster_models(category, "clusters_n1")
    n1_info = n1_models.get(str(n1_cluster), {}).get(famille)

    if n1_info is not None:
        provisional = _predict_with_model(
            n1_info["model"], model_name, n1_info["design_columns"],
            n1_info["continuous_features"], n1_info["categorical_features"], values, week, category,
        )
        provisional["price_source"] = "n1"
    else:
        provisional = _predict_category()

    if segmentation == "n1":
        result = provisional
    else:
        # ── Segment N2 a partir du prix provisoire, puis resultat FINAL
        # avec fallback hierarchique (cluster exact -> marque x gamme ->
        # marque -> categorie, cf. _resolve_n2_model_with_fallback) ──
        marque = values.get("marque")
        n2_assignment = assign_n2_segment(category, marque, provisional["price"], values) if marque else {"cluster_id": None}
        cluster_id = n2_assignment.get("cluster_id")

        n2_models = load_retained_cluster_models(category, "clusters_n2")
        resolved_info, resolved_source = _resolve_n2_model_with_fallback(
            cluster_id, n2_models, famille,
        )

        if resolved_info is not None:
            result = _predict_with_model(
                resolved_info["model"], model_name, resolved_info["design_columns"],
                resolved_info["continuous_features"], resolved_info["categorical_features"],
                values, week, category,
            )
            result["price_source"] = resolved_source
        else:
            # Jamais un repli silencieux sur N1 ici : l'utilisateur a choisi
            # "N2" explicitement, cf. docstring ci-dessus.
            result = _predict_category()

    result["note"] += {
        "n2": " Estimation par le modèle dédié à ce cluster marque × gamme × profil technique (plus précis que le modèle catégorie sur ce segment).",
        "n2_marque_gamme": " Aucun modèle dédié pour ce sous-cluster — repli sur un modèle du même couple marque × gamme (retenu comme plus précis que le modèle catégorie).",
        "n2_marque": " Aucun modèle dédié pour ce couple marque × gamme — repli sur un modèle de la même marque (retenu comme plus précis que le modèle catégorie).",
        "n1": " Estimation par le modèle dédié à ce cluster technique (plus précis que le modèle catégorie sur ce segment).",
        "categorie": " Aucun modèle dédié retenu pour ce segment — modèle catégorie utilisé.",
    }[result["price_source"]]
    return result


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
    dict avec gamme_estimee=None si la marque n'a pas de gamme definie.

    PEUT arriver (correctif du 2026-08-01bis -- IndexError observe sur
    televiseurs/ALMATV) : le formulaire propose les marques de
    brand_plan.csv (effectif POOLE, cf. get_feature_ranges), mais
    gamme_prix/cluster_id (N2) sont ancres sur la seule semaine de
    reference (cf. hedonic_model.n2_reference_week) -- une marque
    suffisante sur l'effectif poole peut n'avoir AUCUNE ligne couverte par
    cette semaine de reference (gamme_prix NaN partout), auquel cas
    `bounds` (groupby qui ignore les cles NaN, comportement pandas par
    defaut) est vide. Jamais un plantage : meme repli que df_brand vide."""
    df = load_pooled_labeled(category)
    df_brand = df[df["marque"] == marque]
    if df_brand.empty:
        return {"gamme_estimee": None, "cluster_id": None, "n_comparables": 0}

    bounds = df_brand.groupby("gamme_prix")["prix_tnd"].agg(["min", "max"]).reset_index()
    if bounds.empty:
        return {"gamme_estimee": None, "cluster_id": None, "n_comparables": 0}
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

def find_similar_products(category: str, values: dict, top_n: int = 10, week=None) -> pd.DataFrame:
    """Plus proches voisins REELS (memes caracteristiques standardisees que
    le clustering N1, cf. scaler_n1) -- sert de verification de bon sens
    pour la prediction (page 3). week : si fourni, restreint les
    comparables a cette seule semaine (produits reellement en vente a ce
    moment-la) -- retombe sur toutes les semaines poolees si la semaine
    demandee n'a aucun produit pour cette categorie (jamais une liste vide
    silencieuse)."""
    schema = load_n1_feature_schema(category)
    df = load_pooled_labeled(category)
    if week is not None:
        df_week = df[df["semaine"] == int(week)]
        if not df_week.empty:
            df = df_week
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
