# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/models/weekly_report.py
=============================================================================
ROLE :
    Rapports hebdomadaires transverses, persistes sous reports/ pour le
    dashboard (STRICTEMENT lecture seule, cf. src/dashboard/data_loader.py --
    jamais de fit() ici pour Ridge/RF/KMeans, seulement joblib.load() +
    .predict() sur les artefacts de src/models/save_artifacts.py). Seul
    fit_strategy_c_pooled_time (HedonicOLS, deja concu pour etre reajuste
    a la demande, cf. hedonic_model.py) est effectivement (re)ajuste ici --
    jamais dans le dashboard lui-meme, d'ou sa persistance en CSV.

    1. couverture_clustering_hebdo.csv : verifie que CHAQUE semaine, pour
       CHAQUE categorie, est bien couverte par les deux approches de
       clustering (N1 technique / N2 marque x gamme) -- jamais suppose,
       toujours mesure. Distingue explicitement deux causes de non-
       couverture N2 (jamais confondues) :
         - marque negligeable sur l'effectif POOLE (< MIN_BRAND_COUNT,
           filtree AVANT tout essai de clustering, cf. compute_price_tiers) --
           la ligne n'apparait alors PAS DU TOUT dans pooled_labeled.csv ;
         - marque/gamme absente ou sous MIN_BRAND_COUNT sur la SEULE semaine
           de reference (cf. hedonic_model.n2_reference_week) alors qu'elle
           passe le seuil poole -- gamme_prix/cluster_id restent NaN pour ce
           produit, dans TOUTES les semaines, mais la ligne reste presente
           (cluster_direct/N1 toujours renseigne). gamme_prix/cluster_id ne
           sont JAMAIS calcules sur le pool multi-semaines (decision
           utilisateur du 2026-07-31 -- un K-Means/qcut ne doit jamais
           melanger des observations de semaines differentes, cf.
           n2_reference_week) ; ils sont figes sur cette semaine puis
           reappliques par produit (url) partout ou il apparait. A
           l'interieur d'une unite couverte, un effectif encore insuffisant
           pour une STRUCTURE K-Means (< CLUSTERING_MIN_N sur la semaine de
           reference) ne produit PAS de NaN mais un groupe local trivial
           unique ("c0") -- a distinguer via unit_summary.csv (colonne
           outcome), jamais via la nullite de cluster_id.
       N1 est construit sur la totalite du catalogue poole (jamais de
       filtrage de marque) -- sa couverture est structurellement de 100%.

    2. cluster_prix_geometrique.csv : moyenne GEOMETRIQUE du prix par
       cluster (N1 et N2), par semaine -- la moyenne geometrique est la
       statistique coherente avec un modele log-lineaire (exp(moyenne des
       log-prix)), jamais tiree vers le haut par les quelques produits
       haut de gamme d'un cluster comme le serait une moyenne arithmetique.

    3. estimations_prix_hebdo.csv : prix reel (moyenne geometrique) vs
       prix ESTIME par chacun des 3 modeles persistes (HedonicOLS, Ridge,
       Random Forest), par semaine, + erreur. Deux mesures d'erreur,
       jamais confondues :
         - erreur_pct : ecart RELATIF entre moyennes geometriques
           (reel vs estime) -- un biais AGREGE par semaine ;
         - mape_pct : erreur absolue moyenne en %, PRODUIT PAR PRODUIT --
           l'erreur TYPIQUE, peut etre elevee meme si erreur_pct est
           proche de 0 (des erreurs positives/negatives qui se compensent).
       Predictions calculees sur la totalite des donnees poolees
       (train + test confondus, cf. save_artifacts.build_matrices) : un
       choix deliberement DESCRIPTIF (in-sample), pas une reevaluation de
       la generalisation hors-echantillon (deja mesuree par metrics.json,
       cf. save_artifacts.py) -- jamais presente comme telle.

    4. indice_prix_hedonique_hebdo.csv : effet de l'indicatrice "semaine"
       d'un modele hedonique POOLE a pentes communes (fit_strategy_c_
       pooled_time, cf. hedonic_model.py) -- la variation de prix "pure",
       NETTE des caracteristiques techniques et de la marque (methode
       standard des indices de prix par regression a variable muette
       temporelle, Triplett 2004 / Rosen 1974). Compare a l'evolution du
       prix BRUT (cluster_prix_geometrique.csv / estimations_prix_hebdo.csv),
       l'ecart entre les deux permet de distinguer un changement de PRIX
       "toutes choses egales" (promotion, cout, marge) d'un changement de
       COMPOSITION du catalogue (produits plus/moins haut de gamme d'une
       semaine a l'autre) -- cf. composition_catalogue_hebdo.csv pour le
       second volet de cette lecture. Categories de
       POOLED_TIME_EXCLUDED_CATEGORIES omises (coefficients hedoniques deja
       mesures comme instables entre semaines, cf. hedonic_model.py) --
       jamais silencieusement, une ligne de log l'indique explicitement.

    5. composition_catalogue_hebdo.csv : moyenne des caracteristiques
       techniques continues du catalogue, par semaine -- sert a juger si
       une hausse de prix brut s'accompagne (ou non) d'une hausse des
       specifications moyennes (RAM, stockage...), signe d'un changement
       de COMPOSITION/QUALITE plutot que d'un changement de PRIX pur.

    6. marque_gamme_estimations_hebdo.csv : meme logique que le point 3,
       mais au niveau du CLUSTER N2 (marque x gamme x sous-cluster
       technique), pas de la categorie entiere -- decision utilisateur du
       2026-07-25. Colonnes : categorie, marque, gamme, cluster, semaine,
       moyenne_geometrique (prix reel), moyenne_estimee_ridge/_hedonic/_rf,
       erreur_ridge_pct/_hedonic_pct/_rf_pct (ajoutees le 2026-07-29, page
       Telechargements -- ecart relatif estime vs reel, positif = le
       modele surestime), n_produits, source_ridge/_hedonic/_rf (ajoutees
       le 2026-08-01). Base de l'etude de transitions (point 7), du
       notebook notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb, et
       des exports CSV par categorie de la page Telechargements du
       dashboard.

       MODELISATION PAR CLUSTER (2026-08-01, cf. save_artifacts.py
       §3ter/fit_models_per_segment) : moyenne_estimee_* n'utilise plus
       SYSTEMATIQUEMENT le modele categorie entiere -- chaque produit est
       estime par le modele le plus specifique RETENU pour sa famille
       (N2 > N1 > categorie, cf. _predict_all_models_cluster_aware),
       "retenu" signifiant demontre MEILLEUR que le modele categorie sur le
       meme test hors-echantillon (jamais un modele de cluster simplement
       "ajustable", cf. fit_models_per_segment -- mesure empirique sur ce
       projet : passer le seuil d'effectif ne garantit pas une bonne
       generalisation, des R² hors-echantillon fortement negatifs ont ete
       observes sur des segments qui passaient pourtant le seuil). Colonnes
       source_ridge/_hedonic/_rf ("n2"/"n1"/"categorie") tracent EXPLICITEMENT
       quel modele a produit quelle estimation, jamais ambigu.

    7. transitions_cluster_hebdo.csv : pour chaque cluster et chaque paire
       de semaines consecutives, classe la transition dans une grille 3x3
       (direction du prix REEL x direction du prix ESTIME/implicite) --
       distingue un changement de PRIX pur d'un changement de VALEUR
       TECHNIQUE du mix vendu au sein du cluster, et mesure l'ecart
       residuel entre les deux (proxy de marge/politique commerciale non
       expliquee par les caracteristiques). Repond explicitement a la
       question posee par l'utilisateur (2026-07-25) : que signifie une
       moyenne geometrique STABLE entre deux semaines accompagnee d'une
       moyenne ESTIMEE en BAISSE ? (cf. section 3ter pour la grille
       complete et son interpretation). Colonnes bootstrap ajoutees le
       2026-07-25 (upgrade demande par l'utilisateur, cf.
       _bootstrap_ecart_residuel) : `ecart_residuel_ic_bas`/`_ic_haut`
       (intervalle de confiance a 95%, percentile, sur 1000 repliques
       reechantillonnees produit par produit) et `ecart_residuel_p_value` --
       remplacent le seuil de materialite FIXE (MATERIALITY_THRESHOLD_PCT)
       par une inference statistique quand l'effectif le permet
       (`bootstrap_possible`, n >= 2 des deux cotes de la transition).
       `ecart_residuel_significatif` (IC excluant 0) est le signal le plus
       fiable de ce tableau pour distinguer un vrai ecart d'un artefact
       d'echantillonnage. Colonne `cause_principale` ajoutee le 2026-07-26
       (decision utilisateur : la question du projet est "prix, qualite,
       OU nombre de produits ?", pas seulement "prix ou qualite ?") --
       {"effectif","aucune","prix","qualite","prix_et_qualite"} : si
       l'effectif du cluster a change (composition_stable=False), la cause
       est TOUJOURS "effectif" en priorite (la grille prix/qualite serait
       trompeuse sur un catalogue qui a change de composition, cf. section
       3ter) ; sinon, la cause est deduite de la grille QUADRANT_LABELS
       (_CAUSE_BY_DIRECTIONS). C'est cette colonne, pas `classification`
       (texte detaille), qu'il faut grouper/filtrer pour une conclusion
       agregee du type "X% des transitions sont dues a un effet de
       composition, Y% a un vrai changement de prix, Z% a la qualite".

    8. stabilite_clustering_n2.csv (ajoute le 2026-07-26, rigor upgrade) :
       pour chaque unite marque x gamme clusterisee, un score de stabilite
       bootstrap (ARI moyen + ecart-type sur 100 repliques, cf.
       cluster_stability_n2) -- le clustering N2 lui-meme, utilise ensuite
       par TOUS les rapports ci-dessus (2 a 7), n'avait jusqu'ici aucune
       mesure de robustesse a l'echantillonnage. Un ARI moyen bas (< 0.5)
       signale une segmentation dont les frontieres dependent fortement de
       quels produits precis sont observes -- a lire avec la meme prudence
       qu'un `ecart_residuel` non confirme par bootstrap (point 7).

    9. justification_k_clustering.csv (ajoute le 2026-07-27, rigor
       upgrade) : pour CHAQUE clustering K-Means du projet (N1 technique --
       une ligne par k teste, categorie entiere -- et N2 marque x gamme --
       une ligne par k teste, PAR unite), la silhouette de TOUS les k
       essayes (`k_retenu`=False inclus), avec la raison de rejet explicite
       quand applicable (n_clusters_effectif<2 : effondrement KMeans sur
       points dupliques ; taille_min_cluster sous le seuil). _choose_k
       (hedonic_model.py) ne retournait jusqu'ici QUE le k final -- ce
       choix restait invisible, jamais compare aux k voisins qu'il a
       ecartes. cf. cluster_stability_n2 pour la question complementaire
       ("ce k tiendrait-il avec un echantillon different ?") ; ce rapport-
       ci repond a une question distincte ("pourquoi CE k plutot qu'un
       autre, sur les donnees telles qu'observees ?").

       LIMITE CONNUE, documentee mais non corrigee (hors perimetre de ce
       rapport) : le K-Means sous-jacent (N1 et N2) est ajuste sur une
       matrice standardisee melangeant variables CONTINUES (RAM, stockage,
       apres StandardScaler -- distance euclidienne pleinement adaptee) et
       variables CATEGORIELLES one-hot (processeur, marque -- distance
       euclidienne entre indicatrices 0/1, moins directement interpretable
       qu'une vraie distance categorielle type Gower ou K-Prototypes). Le
       poids relatif de chaque groupe de variables dans la distance totale
       depend donc partiellement du NOMBRE de modalites categorielles
       encodees (plus une variable a de modalites, plus elle pese de
       "colonnes" dans l'espace euclidien) -- un artefact du one-hot
       encoding, pas un choix delibere de ponderation. Cf. RAPPORT_
       SYNTHESE.md pour la discussion complete et les pistes d'amelioration
       (K-Prototypes, poids explicites par groupe de variables).

UTILISATION :
    python -m src.models.weekly_report
    python -m src.models.weekly_report --category smartphones
=============================================================================
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.models.hedonic_model import (
    CLUSTERING_MIN_N,
    POOLED_TIME_EXCLUDED_CATEGORIES,
    _ID_COLUMNS,
    _build_feature_matrix,
    _choose_k,
    _classify_features,
    _min_cluster_size_for,
    build_design_matrix,
    compute_price_tiers,
    fit_strategy_c_pooled_time,
    n2_reference_week,
)
from src.models.save_artifacts import MODELS_DIR, load_pooled_category
from src.utils.cluster_names import n1_cluster_name
from src.utils.config import CATEGORY_ORDER, PROJECT_ROOT, RANDOM_STATE

logger = logging.getLogger("models.weekly_report")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")

REPORTS_DIR = PROJECT_ROOT / "reports"


def _geo_mean(prices) -> float:
    """Moyenne geometrique = exp(moyenne des log-prix) -- coherente avec un
    modele log-lineaire, contrairement a la moyenne arithmetique (biaisee
    vers le haut par les quelques prix eleves d'une distribution asymetrique,
    Jensen). np.nan si aucune valeur positive."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(arr))))


# ─────────────────────────────────────────────────────────────────────────────
# 1. COUVERTURE DU CLUSTERING PAR SEMAINE (les 2 approches)
# ─────────────────────────────────────────────────────────────────────────────

def coverage_by_week(category: str) -> pd.DataFrame:
    df_pooled = load_pooled_category(category)
    df_labeled = pd.read_csv(MODELS_DIR / category / "pooled_labeled.csv")

    n_pooled_by_week = df_pooled.groupby("semaine").size()
    n_retenus_by_week = df_labeled.groupby("semaine").size()
    n1_by_week = df_labeled.groupby("semaine")["cluster_direct"].apply(lambda s: s.notna().sum())
    n2_by_week = df_labeled.groupby("semaine")["cluster_id"].apply(lambda s: s.notna().sum())

    rows = []
    for week in sorted(n_pooled_by_week.index):
        n_pooled = int(n_pooled_by_week.get(week, 0))
        n_retenus = int(n_retenus_by_week.get(week, 0))
        n1 = int(n1_by_week.get(week, 0))
        n2 = int(n2_by_week.get(week, 0))
        rows.append({
            "categorie": category, "semaine": int(week),
            "n_produits_poole": n_pooled,
            "n_retenus_marque_suffisante": n_retenus,
            "pct_retenus_marque_suffisante": round(100 * n_retenus / n_pooled, 1) if n_pooled else float("nan"),
            "n1_couverts": n1,
            "n1_pct_couverture": round(100 * n1 / n_retenus, 1) if n_retenus else float("nan"),
            "n2_couverts": n2,
            "n2_pct_couverture": round(100 * n2 / n_retenus, 1) if n_retenus else float("nan"),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOYENNE GEOMETRIQUE DU PRIX PAR CLUSTER (N1 et N2), PAR SEMAINE
# ─────────────────────────────────────────────────────────────────────────────

def cluster_geometric_means(category: str) -> pd.DataFrame:
    df = pd.read_csv(MODELS_DIR / category / "pooled_labeled.csv")
    rows = []

    for (cluster, week), g in df.groupby(["cluster_direct", "semaine"]):
        rows.append({
            "categorie": category, "approche": "N1_technique", "cluster": n1_cluster_name(category, int(cluster)),
            "marque": None, "gamme_prix": None, "semaine": int(week),
            "n_produits": len(g), "prix_geometrique_tnd": round(_geo_mean(g["prix_tnd"]), 2),
        })

    df_n2 = df[df["cluster_id"].notna()]
    for (marque, gamme, cluster, week), g in df_n2.groupby(["marque", "gamme_prix", "cluster_id", "semaine"]):
        rows.append({
            "categorie": category, "approche": "N2_marque_gamme", "cluster": cluster,
            "marque": marque, "gamme_prix": gamme, "semaine": int(week),
            "n_produits": len(g), "prix_geometrique_tnd": round(_geo_mean(g["prix_tnd"]), 2),
        })

    return pd.DataFrame(rows).sort_values(["approche", "cluster", "semaine"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRIX ESTIME PAR MODELE (Ridge / RF / Hedonic OLS), PAR SEMAINE + ERREUR
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_FILES = {"hedonic_ols": "ols.joblib", "ridge": "ridge.joblib", "random_forest": "rf.joblib"}


def _predict_all_models(category: str, df: pd.DataFrame) -> tuple:
    """
    Predit log(prix) -> prix (np.exp(), meme choix documente que
    save_artifacts.evaluate_predictions, sans correction de biais Duan/
    Miller) avec les 3 modeles persistes, pour CHAQUE ligne de `df` (doit
    contenir au moins les colonnes de continuous_features/categorical_
    features + prix_tnd -- df_pooled ET pooled_labeled.csv conviennent
    tous les deux, meme schema de features). Reutilise par
    weekly_model_estimates (toutes les lignes) ET
    marque_gamme_model_estimates (lignes N2-couvertes seulement) -- jamais
    deux implementations paralleles du meme calcul.

    Returns: (price_true: np.ndarray, price_pred: dict[str, np.ndarray])
        alignes sur l'ordre des lignes de df (index reset).
    """
    with open(MODELS_DIR / category / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    continuous_features = metrics["continuous_features"]
    categorical_features = metrics["categorical_features"]
    design_cols = metrics["design_matrix_columns"]

    df = df.reset_index(drop=True)
    X, y = build_design_matrix(df, continuous_features, categorical_features)
    X = X.reindex(columns=design_cols, fill_value=0.0).astype(float)

    price_pred = {}
    for model_name, filename in _MODEL_FILES.items():
        model = joblib.load(MODELS_DIR / category / filename)
        log_pred = np.asarray(model.predict(X), dtype=float)
        price_pred[model_name] = np.exp(log_pred)

    price_true = np.exp(y.to_numpy())
    return price_true, price_pred


def _sanitize_segment_name(value) -> str:
    """Identique a save_artifacts._sanitize_segment_name (":" invalide dans
    un chemin Windows, cluster_id contient "::") -- reproduit ici plutot
    qu'importe pour eviter une dependance circulaire (save_artifacts importe
    deja des symboles de hedonic_model, pas de weekly_report)."""
    text = str(value)
    for bad, repl in ((":", "_"), ("/", "-"), ("\\", "-"), ("*", ""), ("?", ""),
                       ('"', ""), ("<", ""), (">", ""), ("|", "_"), (" ", "_")):
        text = text.replace(bad, repl)
    return text


def _load_retained_segment_models(category: str, subdir: str) -> dict:
    """Charge, pour clusters_n1 ou clusters_n2 (cf. save_artifacts.
    persist_segment_models), UNIQUEMENT les modeles marques
    retenu_pour_prediction=True dans <subdir>_summary.csv (bat le modele
    categorie sur le meme test, cf. sa docstring -- jamais un modele
    simplement "ajuste" mais pire que le repli categorie).

    Returns: {segment (str): {famille: {"model": objet charge,
        "continuous_features": list, "categorical_features": list,
        "design_columns": list}}} -- dict vide si le sous-dossier ou le
    summary sont absents (categorie pas encore regeneree avec ce
    correctif, ou aucun cluster retenu)."""
    root = MODELS_DIR / category / subdir
    summary_path = MODELS_DIR / category / f"{subdir}_summary.csv"
    if not root.exists() or not summary_path.exists():
        return {}

    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    retained = summary[summary["retenu_pour_prediction"] == True]  # noqa: E712 (comparaison explicite, pas "is True")
    if retained.empty:
        return {}

    files = {"hedonic_ols": "ols.joblib", "ridge": "ridge.joblib", "random_forest": "rf.joblib"}
    result = {}
    for segment, famille in zip(retained["segment"], retained["famille"]):
        seg_dir = root / _sanitize_segment_name(segment)
        model_path = seg_dir / files[famille]
        schema_path = seg_dir / "feature_schema.json"
        if not model_path.exists() or not schema_path.exists():
            continue
        with open(schema_path, "r", encoding="utf-8") as fh:
            schema = json.load(fh)
        result.setdefault(str(segment), {})[famille] = {
            "model": joblib.load(model_path),
            "continuous_features": schema["continuous_features"],
            "categorical_features": schema["categorical_features"],
            "design_columns": schema["design_matrix_columns"],
        }
    return result


def _predict_all_models_cluster_aware(category: str, df: pd.DataFrame) -> tuple:
    """
    Meme resultat que _predict_all_models (price_true, price_pred -- prix
    reel et estime par les 3 familles, alignes sur df.reset_index()), mais
    utilise, POUR CHAQUE LIGNE, le modele le plus specifique disponible et
    RETENU (bat le modele categorie sur son propre test, cf.
    _load_retained_segment_models) :
        1. modele N2 (cluster_id) si retenu pour cette famille ;
        2. sinon modele N1 (cluster_direct) si retenu pour cette famille ;
        3. sinon le modele categorie entiere (repli, cf. _predict_all_models).
    Decision utilisateur du 2026-08-01 : un seul modele par categorie est
    trop general -- mais jamais au prix d'utiliser un modele de cluster
    DEMONTRE moins bon que le repli (cf. retenu_pour_prediction).

    `df` DOIT contenir cluster_direct et cluster_id (cf. pooled_labeled.csv
    -- pas load_pooled_category, qui ne les fournit pas) -- sinon
    equivalent a _predict_all_models (aucune ligne couverte par un cluster).

    Returns: (price_true, price_pred, source) -- source :
        {famille: np.ndarray[str]} valant "n2"/"n1"/"categorie" par ligne,
        JAMAIS ambigu sur quel modele a produit quelle prediction (decision
        utilisateur du 2026-08-01).
    """
    price_true_cat, price_pred_cat = _predict_all_models(category, df)
    df = df.reset_index(drop=True)

    has_cluster_cols = "cluster_id" in df.columns and "cluster_direct" in df.columns
    source = {famille: np.array(["categorie"] * len(df), dtype=object) for famille in price_pred_cat}
    if not has_cluster_cols:
        return price_true_cat, price_pred_cat, source

    n1_models = _load_retained_segment_models(category, "clusters_n1")
    n2_models = _load_retained_segment_models(category, "clusters_n2")
    if not n1_models and not n2_models:
        return price_true_cat, price_pred_cat, source

    price_pred = {famille: arr.copy() for famille, arr in price_pred_cat.items()}

    for famille in price_pred:
        for segment_col, segment_models, label in (
            ("cluster_direct", n1_models, "n1"), ("cluster_id", n2_models, "n2"),
        ):
            for segment_str, by_famille in segment_models.items():
                if famille not in by_famille:
                    continue
                # cluster_direct est un entier (cf. pooled_labeled.csv) --
                # segment_str (cle du summary, toujours str) doit etre
                # compare comme tel, jamais suppose deja du bon type.
                mask = df[segment_col].astype(str) == segment_str
                if not mask.any():
                    continue
                info = by_famille[famille]
                X_seg, _ = build_design_matrix(df.loc[mask], info["continuous_features"], info["categorical_features"])
                X_seg = X_seg.reindex(columns=info["design_columns"], fill_value=0.0).astype(float)
                log_pred = np.asarray(info["model"].predict(X_seg), dtype=float)
                price_pred[famille][mask.to_numpy()] = np.exp(log_pred)
                source[famille][mask.to_numpy()] = label

    return price_true_cat, price_pred, source


def weekly_model_estimates(category: str) -> pd.DataFrame:
    df_pooled = load_pooled_category(category)
    price_true, price_pred = _predict_all_models(category, df_pooled)
    semaine = df_pooled["semaine"].to_numpy()

    rows = []
    for week in sorted(pd.unique(semaine)):
        mask = semaine == week
        n = int(mask.sum())
        geo_actual = _geo_mean(price_true[mask])
        for model_name, price_pred_arr in price_pred.items():
            price_pred_week = price_pred_arr[mask]
            geo_pred = _geo_mean(price_pred_week)
            mape = float(np.mean(np.abs(price_pred_week - price_true[mask]) / price_true[mask]) * 100)
            erreur_pct = (geo_pred - geo_actual) / geo_actual * 100 if geo_actual else float("nan")
            rows.append({
                "categorie": category, "semaine": int(week), "n_produits": n, "modele": model_name,
                "prix_reel_geometrique_tnd": round(geo_actual, 2),
                "prix_estime_geometrique_tnd": round(geo_pred, 2),
                "erreur_pct": round(erreur_pct, 2),
                "mape_pct": round(mape, 2),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3bis. PRIX ESTIME PAR MODELE, AU NIVEAU DU CLUSTER N2 (marque x gamme)
# ─────────────────────────────────────────────────────────────────────────────

def _marque_gamme_product_level(category: str) -> pd.DataFrame:
    """
    Base COMMUNE a marque_gamme_model_estimates (agregee en moyennes
    geometriques) ET cluster_transitions (reechantillonnage bootstrap
    produit par produit, cf. §3ter) -- une ligne par PRODUIT (pas encore
    agregee par cluster x semaine), prix reel + estime par les 3 modeles.
    Jamais deux implementations paralleles de la meme extraction/
    prediction.

    Restreint aux lignes N2-couvertes (cluster_id non NaN, cf.
    couverture_clustering_hebdo.csv) -- une ligne sans cluster_id n'a, par
    construction, aucun (marque, gamme, cluster) a lui attribuer.

    Predictions VIA _predict_all_models_cluster_aware (decision utilisateur
    du 2026-08-01) : chaque produit est estime par le modele le plus
    specifique RETENU (N2 > N1 > categorie, cf. sa docstring) -- colonnes
    <famille>_source ("n2"/"n1"/"categorie") ajoutees pour tracer, jamais
    ambigu, quel modele a produit quelle estimation.
    """
    df = pd.read_csv(MODELS_DIR / category / "pooled_labeled.csv")
    df = df[df["cluster_id"].notna()].reset_index(drop=True)

    price_true, price_pred, source = _predict_all_models_cluster_aware(category, df)

    work = df[["marque", "gamme_prix", "cluster_id", "semaine"]].copy()
    work["cluster"] = work["cluster_id"].str.split("::").str[-1]
    work = work.drop(columns=["cluster_id"]).rename(columns={"gamme_prix": "gamme"})
    work["prix_reel"] = price_true
    for model_name, arr in price_pred.items():
        work[model_name] = arr
        work[f"{model_name}_source"] = source[model_name]
    work["semaine"] = work["semaine"].astype(int)
    return work


def marque_gamme_model_estimates(category: str, product_level: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Meme principe que weekly_model_estimates, mais agrege par (marque,
    gamme, cluster, semaine) plutot que par (semaine) seule -- decision
    utilisateur du 2026-07-25 : colonnes exactement categorie, marque,
    gamme, cluster, semaine, moyenne_geometrique (prix reel), et moyenne
    ESTIMEE par chacun des 3 modeles, + n_produits (taille de l'echantillon
    derriere chaque moyenne -- indispensable pour juger la fiabilite d'une
    moyenne calculee sur parfois tres peu de produits).

    Colonnes erreur_<modele>_pct ajoutees le 2026-07-29 (page Telechargements
    du dashboard) : ecart relatif entre la moyenne ESTIMEE et la moyenne
    REELLE -- (estime - reel) / reel * 100, meme formule/convention que
    erreur_pct dans weekly_model_estimates (positif = le modele SURESTIME
    ce cluster cette semaine-la, negatif = il le SOUS-ESTIME).

    Colonnes source_<modele> ajoutees le 2026-08-01 (modelisation par
    cluster N1/N2) : "n2"/"n1"/"categorie" -- quel modele a REELLEMENT
    produit moyenne_estimee_<modele> pour cette ligne (cf.
    _predict_all_models_cluster_aware). Toutes les lignes d'un meme
    (marque, gamme, cluster) partagent la MEME source pour une famille
    donnee (le modele retenu d'un cluster s'applique a tout le cluster,
    jamais produit par produit) -- premiere valeur du groupe suffit,
    jamais un mode/vote necessaire.
    """
    work = product_level if product_level is not None else _marque_gamme_product_level(category)

    grouped = work.groupby(["marque", "gamme", "cluster", "semaine"]).agg(
        n_produits=("prix_reel", "size"),
        moyenne_geometrique=("prix_reel", _geo_mean),
        moyenne_estimee_ridge=("ridge", _geo_mean),
        moyenne_estimee_hedonic=("hedonic_ols", _geo_mean),
        moyenne_estimee_rf=("random_forest", _geo_mean),
        source_ridge=("ridge_source", "first"),
        source_hedonic=("hedonic_ols_source", "first"),
        source_rf=("random_forest_source", "first"),
    ).reset_index()

    grouped.insert(0, "categorie", category)
    for c in ["moyenne_geometrique", "moyenne_estimee_ridge", "moyenne_estimee_hedonic", "moyenne_estimee_rf"]:
        grouped[c] = grouped[c].round(2)
    grouped["semaine"] = grouped["semaine"].astype(int)

    for model_col, erreur_col in [
        ("moyenne_estimee_ridge", "erreur_ridge_pct"),
        ("moyenne_estimee_hedonic", "erreur_hedonic_pct"),
        ("moyenne_estimee_rf", "erreur_rf_pct"),
    ]:
        grouped[erreur_col] = (
            (grouped[model_col] - grouped["moyenne_geometrique"]) / grouped["moyenne_geometrique"] * 100
        ).round(2)

    col_order = ["categorie", "marque", "gamme", "cluster", "semaine",
                 "moyenne_geometrique", "moyenne_estimee_ridge", "moyenne_estimee_hedonic", "moyenne_estimee_rf",
                 "erreur_ridge_pct", "erreur_hedonic_pct", "erreur_rf_pct", "n_produits",
                 "source_ridge", "source_hedonic", "source_rf"]
    return grouped[col_order].sort_values(["marque", "gamme", "cluster", "semaine"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3ter. TRANSITIONS SEMAINE A SEMAINE PAR CLUSTER -- prix reel vs prix
#       "implicite" (estime par les caracteristiques) : ce que signifie
#       leur DIVERGENCE (decision utilisateur du 2026-07-25)
# ─────────────────────────────────────────────────────────────────────────────
#
# Cadre de lecture (grille 3x3, direction du prix reel x direction du prix
# estime/implicite -- Rosen 1974 / Triplett 2004, meme logique que l'indice
# hedonique de hedonic_price_index) :
#
#   moyenne_geometrique = ce que le marche a REELLEMENT factures cette
#     semaine-la pour ce cluster (marque x gamme x sous-cluster technique).
#   moyenne_estimee_* = ce que le modele hedonique aurait "du" facturer
#     ETANT DONNE les caracteristiques techniques des produits REELLEMENT
#     presents dans le cluster cette semaine-la (le mix produit peut
#     changer d'une semaine a l'autre au sein du meme cluster -- un
#     cluster n'est pas garanti contenir les memes SKU exacts).
#
#   Si les deux bougent ENSEMBLE (meme sens, ampleur proche) : le
#   changement de prix reel s'explique par un changement de la valeur
#   technique du mix vendu -- PAS un changement de politique de prix.
#
#   Si le prix REEL bouge mais pas l'estime (ou l'inverse) : le marche
#   facture quelque chose que les caracteristiques observees n'expliquent
#   PAS -- ecart residuel = signal de marge/politique commerciale (ou de
#   caracteristiques non observees -- cf. limites documentees dans le
#   notebook d'etude, notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb).
#
#   Cas explicitement souleve par l'utilisateur (2026-07-25) : moyenne
#   geometrique EGALE entre S(t) et S(t+1) mais moyenne ESTIMEE en BAISSE
#   -> case ("stable", "baisse") ci-dessous : le prix affiche est resté
#   inchange alors que la valeur technique implicite du mix a BAISSE --
#   l'ecart entre prix facture et valeur technique (la "marge implicite")
#   s'est donc CREUSE, meme si aucun des deux prix pris isolement n'a
#   "bouge" de facon spectaculaire.

MATERIALITY_THRESHOLD_PCT = 3.0
# Seuil en dessous duquel une variation est classee "stable" -- convention
# documentee (pas un test statistique formel : les effectifs par cluster x
# semaine sont souvent trop petits, cf. n_produits, pour un test de
# difference de moyennes fiable). 3% est delibérément plus large que le
# bruit d'arrondi mais assez fin pour capter un vrai mouvement sur des prix
# de plusieurs centaines/milliers de TND -- cf. discussion de sensibilite
# au seuil dans le notebook d'etude.
MIN_RELIABLE_N = 3
# Sous ce seuil de part et d'autre de la transition, n_produits est trop
# faible pour distinguer un vrai mouvement de prix d'un simple changement
# des quelques unites en catalogue -- transitions marquees
# fiabilite_limitee=True, jamais exclues silencieusement (l'utilisateur
# tranche, cf. README/principe "jamais de filtrage silencieux").

QUADRANT_LABELS = {
    ("stable", "stable"): "Stabilité réelle (prix et caractéristiques inchangés)",
    ("stable", "hausse"): "Amélioration technique non répercutée sur le prix (gain caché pour l'acheteur)",
    ("stable", "baisse"): "Prix maintenu malgré une baisse de la valeur technique implicite (marge implicite en hausse)",
    ("hausse", "stable"): "Hausse de prix non justifiée par les caractéristiques (majoration pure)",
    ("hausse", "hausse"): "Hausse de prix cohérente avec une montée en gamme technique",
    ("hausse", "baisse"): "Hausse de prix ET dégradation technique — écart maximal (majoration forte)",
    ("baisse", "stable"): "Baisse de prix non liée aux caractéristiques (promotion / remise réelle)",
    ("baisse", "hausse"): "Baisse de prix malgré une montée en gamme technique — remise maximale pour l'acheteur",
    ("baisse", "baisse"): "Baisse de prix cohérente avec une baisse de gamme technique",
}

# ─────────────────────────────────────────────────────────────────────────────
# CAUSE PRINCIPALE -- decision utilisateur du 2026-07-26 : la question
# posee par le projet n'est pas seulement "prix ou qualite ?" (grille
# ci-dessus) mais bien "prix, qualite, OU nombre de produits ?" -- une
# 3e cause structurelle qui doit etre isolee AVANT toute lecture prix/
# qualite, pas apres. Si l'effectif du cluster change d'une semaine a
# l'autre (composition_stable=False), la grille QUADRANT_LABELS devient
# TROMPEUSE : un "prix en hausse cohérent avec une montée en gamme
# technique" peut n'etre qu'un cluster qui a perdu ses produits d'entree
# de gamme et gagne des produits premium -- ce n'est PAS le meme produit
# qui est monte en gamme, c'est le CATALOGUE qui a change de composition.
# cause_principale rend cette priorite explicite et REQUETABLE (pas
# seulement lisible dans le texte de `classification`) :
#   "effectif" -- le nombre de produits du cluster a change (cause
#                 prioritaire, cf. ci-dessus -- la grille prix/qualite
#                 n'est calculee/affichee QUE si cette cause est ecartee).
#   "aucune"   -- ni le prix ni la valeur technique implicite ne bougent.
#   "prix"     -- le prix reel bouge, la valeur technique implicite non :
#                 changement de politique commerciale (cout, marge, promo).
#   "qualite"  -- la valeur technique implicite bouge (les DEUX bougent
#                 ensemble ou seule elle bouge) : le mouvement de prix,
#                 s'il y en a un, est explique par un changement de
#                 caracteristiques du mix, pas par une decision de prix.
#   "prix_et_qualite" -- prix et valeur technique implicite bougent en
#                 SENS CONTRAIRES : aucune cause unique ne domine, ecart
#                 maximal (cf. QUADRANT_LABELS).
CAUSE_EFFECTIF = "effectif"
CAUSE_AUCUNE = "aucune"
CAUSE_PRIX = "prix"
CAUSE_QUALITE = "qualite"
CAUSE_PRIX_ET_QUALITE = "prix_et_qualite"

_CAUSE_BY_DIRECTIONS = {
    ("stable", "stable"): CAUSE_AUCUNE,
    ("stable", "hausse"): CAUSE_QUALITE,
    ("stable", "baisse"): CAUSE_QUALITE,
    ("hausse", "stable"): CAUSE_PRIX,
    ("baisse", "stable"): CAUSE_PRIX,
    ("hausse", "hausse"): CAUSE_QUALITE,
    ("baisse", "baisse"): CAUSE_QUALITE,
    ("hausse", "baisse"): CAUSE_PRIX_ET_QUALITE,
    ("baisse", "hausse"): CAUSE_PRIX_ET_QUALITE,
}


def _composition_change_label(n_t: int, n_t1: int) -> str:
    return (
        f"Effectif du cluster modifié ({n_t} → {n_t1} produits) — lecture prix/qualité non isolable "
        f"(cf. cause_principale='{CAUSE_EFFECTIF}')"
    )


def _pct_change(v0, v1) -> float:
    if v0 is None or v1 is None or pd.isna(v0) or pd.isna(v1) or v0 == 0:
        return float("nan")
    return (v1 - v0) / v0 * 100


def _direction(delta_pct: float, threshold: float = MATERIALITY_THRESHOLD_PCT) -> str:
    if pd.isna(delta_pct):
        return "indéterminé"
    if delta_pct > threshold:
        return "hausse"
    if delta_pct < -threshold:
        return "baisse"
    return "stable"


DEFAULT_N_BOOT = 1000
# Repliques bootstrap par transition -- assez pour un intervalle a 95% stable
# (percentiles 2.5/97.5 sur 1000 valeurs restent peu bruites) sans cout de
# calcul prohibitif (~800 transitions x 1000 replique vectorisees en numpy,
# quelques secondes au total, cf. tests de performance dans
# tests/test_models/test_weekly_report.py).


def _bootstrap_ecart_residuel(prix_reel_t, prix_reel_t1, prix_est_t, prix_est_t1,
                               n_boot: int, rng: np.random.Generator, ci: float = 0.95) -> tuple:
    """
    Reechantillonne (avec remise) les prix PRODUIT PAR PRODUIT de chaque
    semaine, independamment pour le prix REEL et le prix ESTIME (meme
    tirage d'index pour les deux -- un produit reechantillonne "compte"
    a la fois pour son prix reel et son prix estime, cf. les tableaux
    alignes produit par produit de _marque_gamme_product_level), recalcule
    delta_prix_reel_pct - delta_estime_hedonic_pct sur chaque replique, et
    en derive un intervalle de confiance (percentile) + une p-value
    bootstrap bilaterale.

    Remplace le seuil de materialite FIXE (MATERIALITY_THRESHOLD_PCT,
    convention documentee mais pas un test statistique, cf. §1.5 du
    notebook d'etude) par une inference sur l'ecart residuel LUI-MEME,
    quand l'effectif le permet (cf. appelant : n < 2 d'un cote ou l'autre
    -> reechantillonnage degenere, pas tente).

    p-value bootstrap : 2 * min(P(ecart >= 0), P(ecart <= 0)), plafonnee a
    1 -- estimateur standard de Davison & Hinkley (1997, §4.2) pour un test
    bilateral base sur la proportion de repliques de part et d'autre de 0.

    Returns: (ic_bas, ic_haut, p_value) -- tous None si n_boot <= 0.
    """
    if n_boot <= 0:
        return None, None, None
    n_t, n_t1 = len(prix_reel_t), len(prix_reel_t1)
    idx_t = rng.integers(0, n_t, size=(n_boot, n_t))
    idx_t1 = rng.integers(0, n_t1, size=(n_boot, n_t1))

    geo_reel_t = np.exp(np.log(prix_reel_t)[idx_t].mean(axis=1))
    geo_reel_t1 = np.exp(np.log(prix_reel_t1)[idx_t1].mean(axis=1))
    geo_est_t = np.exp(np.log(prix_est_t)[idx_t].mean(axis=1))
    geo_est_t1 = np.exp(np.log(prix_est_t1)[idx_t1].mean(axis=1))

    delta_reel = (geo_reel_t1 - geo_reel_t) / geo_reel_t * 100
    delta_est = (geo_est_t1 - geo_est_t) / geo_est_t * 100
    ecart = delta_reel - delta_est

    alpha = (1 - ci) / 2
    ic_bas, ic_haut = np.quantile(ecart, [alpha, 1 - alpha])
    p_pos = float((ecart >= 0).mean())
    p_value = float(min(1.0, 2 * min(p_pos, 1 - p_pos)))
    return round(float(ic_bas), 2), round(float(ic_haut), 2), round(p_value, 4)


def cluster_transitions(category: str, estimates: pd.DataFrame | None = None,
                         product_level: pd.DataFrame | None = None,
                         n_boot: int = DEFAULT_N_BOOT, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Pour chaque cluster (marque x gamme x sous-cluster) et chaque paire de
    semaines CONSECUTIVES disponibles, classe la transition dans la grille
    3x3 (direction du prix reel x direction du prix estime) -- cf. note de
    section ci-dessus. `accord_modeles` : les 3 modeles (Hedonic OLS,
    Ridge, RF) s'accordent-ils sur le SIGNE de la variation du prix
    estime -- un desaccord de signe entre modeles est un signal de
    fragilite du diagnostic "estime", pas seulement du prix reel.

    n_boot : nombre de repliques bootstrap pour l'intervalle de confiance/
    p-value de l'ecart residuel (cf. _bootstrap_ecart_residuel) -- 0 pour
    desactiver entierement (tests rapides). Necessite les prix PRODUIT PAR
    PRODUIT (product_level, cf. _marque_gamme_product_level) -- calcules a
    la volee si absents, jamais recalcules deux fois si l'appelant les a
    deja (cf. main()).
    """
    df = estimates if estimates is not None else marque_gamme_model_estimates(category)
    if n_boot > 0 and product_level is None:
        product_level = _marque_gamme_product_level(category)
    rng = np.random.default_rng(random_state)

    # Groupe une fois pour toutes (marque, gamme, cluster, semaine) -> lignes
    # produit -- evite un filtrage booleen repete de product_level (potentiellement
    # des milliers de lignes) a chaque transition dans la boucle ci-dessous.
    product_groups = {}
    if n_boot > 0 and product_level is not None:
        for pkeys, pg in product_level.groupby(["marque", "gamme", "cluster", "semaine"]):
            product_groups[pkeys] = pg

    rows = []
    for keys, g in df.groupby(["categorie", "marque", "gamme", "cluster"]):
        g = g.sort_values("semaine")
        for i in range(len(g) - 1):
            row_t, row_t1 = g.iloc[i], g.iloc[i + 1]

            delta_raw = _pct_change(row_t["moyenne_geometrique"], row_t1["moyenne_geometrique"])
            delta_hed = _pct_change(row_t["moyenne_estimee_hedonic"], row_t1["moyenne_estimee_hedonic"])
            delta_ridge = _pct_change(row_t["moyenne_estimee_ridge"], row_t1["moyenne_estimee_ridge"])
            delta_rf = _pct_change(row_t["moyenne_estimee_rf"], row_t1["moyenne_estimee_rf"])

            dir_raw, dir_hed = _direction(delta_raw), _direction(delta_hed)
            n_t, n_t1 = int(row_t["n_produits"]), int(row_t1["n_produits"])
            composition_stable = n_t == n_t1

            # cause_principale/classification : l'effectif est teste EN
            # PREMIER (cf. note de section) -- la grille prix/qualite n'est
            # appliquee que si la composition du cluster n'a pas change.
            if not composition_stable:
                cause_principale = CAUSE_EFFECTIF
                label = _composition_change_label(n_t, n_t1)
            else:
                cause_principale = _CAUSE_BY_DIRECTIONS.get((dir_raw, dir_hed), CAUSE_AUCUNE)
                label = QUADRANT_LABELS.get((dir_raw, dir_hed), f"Cas non classé ({dir_raw} / {dir_hed})")

            est_signs = {np.sign(round(d, 6)) for d in (delta_hed, delta_ridge, delta_rf)
                         if pd.notna(d) and abs(d) > 0.5}
            accord_modeles = len(est_signs) <= 1
            ic_bas = ic_haut = p_value = None
            bootstrap_possible = False
            if n_boot > 0 and n_t >= 2 and n_t1 >= 2:
                pg_t = product_groups.get((keys[1], keys[2], keys[3], int(row_t["semaine"])))
                pg_t1 = product_groups.get((keys[1], keys[2], keys[3], int(row_t1["semaine"])))
                if pg_t is not None and pg_t1 is not None:
                    ic_bas, ic_haut, p_value = _bootstrap_ecart_residuel(
                        pg_t["prix_reel"].to_numpy(), pg_t1["prix_reel"].to_numpy(),
                        pg_t["hedonic_ols"].to_numpy(), pg_t1["hedonic_ols"].to_numpy(),
                        n_boot=n_boot, rng=rng,
                    )
                    bootstrap_possible = True

            rows.append({
                "categorie": keys[0], "marque": keys[1], "gamme": keys[2], "cluster": keys[3],
                "semaine_t": int(row_t["semaine"]), "semaine_t1": int(row_t1["semaine"]),
                "n_produits_t": int(row_t["n_produits"]), "n_produits_t1": int(row_t1["n_produits"]),
                "delta_prix_reel_pct": round(delta_raw, 2) if pd.notna(delta_raw) else None,
                "delta_estime_hedonic_pct": round(delta_hed, 2) if pd.notna(delta_hed) else None,
                "delta_estime_ridge_pct": round(delta_ridge, 2) if pd.notna(delta_ridge) else None,
                "delta_estime_rf_pct": round(delta_rf, 2) if pd.notna(delta_rf) else None,
                "ecart_residuel_pct": round(delta_raw - delta_hed, 2) if pd.notna(delta_raw) and pd.notna(delta_hed) else None,
                "classification": label,
                # cause_principale : reponse COMPACTE et REQUETABLE a la
                # question "prix, qualite, ou nombre de produits ?" (cf.
                # note de section) -- {"effectif","aucune","prix","qualite",
                # "prix_et_qualite"}. `classification` reste le texte
                # detaille pour la lecture humaine ; cause_principale est
                # ce qu'on groupe/filtre pour une conclusion agregee.
                "cause_principale": cause_principale,
                "accord_modeles": accord_modeles,
                "fiabilite_limitee": bool(row_t["n_produits"] < MIN_RELIABLE_N or row_t1["n_produits"] < MIN_RELIABLE_N),
                # n_produits_t == n_produits_t1 est une condition NECESSAIRE
                # (jamais suffisante -- les produits pourraient avoir change
                # tout en gardant le meme effectif) pour que la transition
                # porte sur le MEME ensemble de produits plutot que sur un
                # effectif de cluster qui a change (entree/sortie de
                # catalogue) -- un effectif differe => la composition a
                # CERTAINEMENT change, jamais une lecture "meme produits"
                # possible. Desormais la cause PRIORITAIRE (cause_principale
                # == "effectif"), pas seulement un indice de fiabilite.
                "composition_stable": composition_stable,
                # Inference bootstrap sur l'ecart residuel (cf.
                # _bootstrap_ecart_residuel) -- remplace le seuil de
                # materialite FIXE (MATERIALITY_THRESHOLD_PCT, utilise pour
                # `classification` ci-dessus) par un intervalle de confiance
                # et une p-value quand l'effectif le permet (n >= 2 des deux
                # cotes). `ecart_residuel_significatif` est le signal le
                # PLUS fiable de ce tableau pour juger si un ecart residuel
                # est un vrai signal ou du bruit d'echantillonnage -- a
                # preferer a `classification` (seuil fixe) des que
                # `bootstrap_possible` est True.
                "bootstrap_possible": bootstrap_possible,
                "ecart_residuel_ic_bas": ic_bas,
                "ecart_residuel_ic_haut": ic_haut,
                "ecart_residuel_p_value": p_value,
                "ecart_residuel_significatif": bool(
                    ic_bas is not None and ic_haut is not None and (ic_bas > 0 or ic_haut < 0)
                ),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. INDICE DE PRIX HEDONIQUE (effet fixe "semaine", net des caracteristiques)
# ─────────────────────────────────────────────────────────────────────────────

def hedonic_price_index(category: str) -> pd.DataFrame | None:
    if category in POOLED_TIME_EXCLUDED_CATEGORIES:
        logger.warning(
            f"  [{category}] exclue du pooling temporel (coefficients hedoniques instables entre semaines, "
            f"cf. hedonic_model.py) -- indice de prix hedonique non calcule pour cette categorie."
        )
        return None

    model, X, y, df_pooled = fit_strategy_c_pooled_time(category)
    coefs = model.get_coefficients()
    weeks = sorted(df_pooled["semaine"].unique())
    ref_week = weeks[0]

    rows = [{
        "categorie": category, "semaine": int(ref_week), "semaine_reference": True,
        "indice_prix_ajuste_qualite_pct": 0.0, "p_value": float("nan"),
    }]
    for w in weeks[1:]:
        feat = f"semaine_{w}"
        match = coefs[coefs["feature"] == feat]
        if match.empty:
            logger.warning(f"  [{category}] coefficient '{feat}' introuvable dans le modele poole -- semaine ignoree.")
            continue
        rows.append({
            "categorie": category, "semaine": int(w), "semaine_reference": False,
            "indice_prix_ajuste_qualite_pct": round(float(match["pct_effect"].iloc[0]), 2),
            "p_value": round(float(match["p_value"].iloc[0]), 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPOSITION DU CATALOGUE PAR SEMAINE (moyenne des specs continues)
# ─────────────────────────────────────────────────────────────────────────────

def catalog_composition_by_week(category: str) -> pd.DataFrame:
    with open(MODELS_DIR / category / "metrics.json", "r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    continuous_features = [c for c in metrics["continuous_features"]]

    df = load_pooled_category(category)
    agg = df.groupby("semaine")[continuous_features].mean().round(2)
    agg["prix_geometrique_tnd"] = df.groupby("semaine")["prix_tnd"].apply(_geo_mean).round(2)
    agg["n_produits"] = df.groupby("semaine").size()
    agg = agg.reset_index().rename(columns={"semaine": "semaine"})
    agg.insert(0, "categorie", category)
    agg["semaine"] = agg["semaine"].astype(int)
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# 6. STABILITE DU CLUSTERING N2 (bootstrap sur l'ARI) -- rigor upgrade
#    demande par l'utilisateur (2026-07-26) : le clustering lui-meme,
#    utilise ensuite par cluster_transitions, n'avait jusqu'ici aucune
#    mesure de robustesse -- seul l'ECART RESIDUEL qui en derive en avait
#    une (cf. _bootstrap_ecart_residuel). Meme logique appliquee ici en
#    amont : cette segmentation tiendrait-elle avec un echantillon
#    legerement different de produits ?
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_N_BOOT_STABILITE = 100
# Plus petit que DEFAULT_N_BOOT (transitions) : chaque replique ici REFIT
# un KMeans complet (cout non-vectorisable, contrairement au bootstrap sur
# des moyennes geometriques) -- 100 repliques par unite marque x gamme
# clusterisee reste suffisant pour un ARI moyen stable (l'ecart-type
# inter-repliques est rapporte explicitement pour en juger).
_N_INIT_BOOT = 3
# n_init reduit (vs 10 pour le clustering persiste/"officiel", cf.
# labels_original ci-dessous, qui DOIT rester fidele a ce que
# save_artifacts.py produit reellement) -- une replique bootstrap sert a
# mesurer une TENDANCE de stabilite sur ~100 tirages, pas a retrouver le
# minimum global a chaque fois ; n_init=3 est le compromis standard des
# methodes de stabilite par bootstrap (cf. clusterboot, Hennig 2007) entre
# fidelite et cout -- sans cette reduction, 100 x n_init=10 KMeans par
# unite rend le rapport hebdomadaire complet impraticable (mesure : ~200
# repliques x n_init=10 prend 40-60s PAR UNITE clusterisee).


def cluster_stability_n2(category: str, n_boot: int = DEFAULT_N_BOOT_STABILITE,
                          random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Pour chaque unite marque x gamme atteignant CLUSTERING_MIN_N (cf.
    hedonic_model.py), mesure la STABILITE du K-Means N2 par bootstrap :
    reechantillonne les produits de l'unite (avec remise, meme effectif),
    reajuste un K-Means (meme k) sur l'echantillon, et compare par ARI
    (Adjusted Rand Index, Hubert & Arabie 1985) les labels obtenus sur les
    donnees ORIGINALES (via .predict(), jamais reajuste sur les donnees
    originales elles-memes) a ceux du clustering persiste par
    save_artifacts.py. ARI proche de 1 = partition robuste au bruit
    d'echantillonnage (quels produits precis sont observes) ; proche de 0
    = structure fragile, largement due au hasard de l'echantillon plutot
    qu'a une vraie segmentation.

    Repond a une question laissee ouverte par le choix de k
    (_choose_k : silhouette maximale sous contrainte de taille minimale,
    jamais valide par ailleurs) -- une segmentation avec un k "optimal"
    au sens silhouette peut neanmoins etre instable si l'unite est petite.

    N'affecte JAMAIS le clustering persiste (aucun re-fit des artefacts de
    save_artifacts.py) -- purement diagnostique, cf. reports/
    stabilite_clustering_n2.csv.

    Restreint a la SEULE semaine de reference (n2_reference_week -- meme
    semaine que celle figee par save_artifacts.process_category) : audite le
    clustering REELLEMENT persiste, jamais un clustering fictif ajuste sur le
    pool multi-semaines (qui gonflerait artificiellement l'effectif de
    chaque unite et testerait la stabilite d'un K-Means qui n'est plus celui
    produit en amont, cf. decision utilisateur du 2026-07-31).
    """
    df_pooled = load_pooled_category(category)
    reference_week = n2_reference_week(df_pooled)
    df_reference = df_pooled[df_pooled["semaine"] == reference_week]
    df_tiers, _ = compute_price_tiers(df_reference)
    continuous_features, categorical_features = _classify_features(df_tiers, exclude=_ID_COLUMNS | {"semaine"})
    rng = np.random.default_rng(random_state)

    rows = []
    for (marque, gamme), df_unit in df_tiers.groupby(["marque", "gamme_prix"], observed=True):
        n = len(df_unit)
        row = {"categorie": category, "marque": marque, "gamme": gamme, "semaine_reference": reference_week, "n": n}

        if n < CLUSTERING_MIN_N:
            rows.append({**row, "k": 1, "outcome": "too_small",
                         "ari_moyen": None, "ari_ecart_type": None, "n_repliques_valides": 0})
            continue

        X = _build_feature_matrix(df_unit, continuous_features, categorical_features)
        k = _choose_k(X, _min_cluster_size_for(n))
        if k <= 1:
            rows.append({**row, "k": k, "outcome": "no_structure",
                         "ari_moyen": None, "ari_ecart_type": None, "n_repliques_valides": 0})
            continue

        labels_original = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit_predict(X)

        aris = []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            # Meme garde-fou anti-effondrement que _choose_k (cf. son
            # docstring) : un echantillon bootstrap peut par hasard ne
            # couvrir qu'un sous-ensemble degenere des clusters d'origine.
            if pd.Series(labels_original[idx]).nunique() < 2:
                continue
            try:
                km_boot = KMeans(n_clusters=k, random_state=int(rng.integers(0, 2**31 - 1)), n_init=_N_INIT_BOOT)
                km_boot.fit(X[idx])
                labels_predicted = km_boot.predict(X)
            except ValueError:
                continue
            aris.append(adjusted_rand_score(labels_original, labels_predicted))

        rows.append({
            **row, "k": k, "outcome": "clustered",
            "ari_moyen": round(float(np.mean(aris)), 3) if aris else None,
            "ari_ecart_type": round(float(np.std(aris)), 3) if aris else None,
            "n_repliques_valides": len(aris),
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7. JUSTIFICATION DU CHOIX DE k (silhouette de TOUS les k testes) -- rigor
#    upgrade demande par l'utilisateur (2026-07-26) : _choose_k ne
#    retournait jusqu'ici QUE le k final, un choix invisible/non audite.
#    Couvre les DEUX clustering K-Means du projet (N1 technique, N2
#    marque x gamme).
# ─────────────────────────────────────────────────────────────────────────────

def k_selection_justification(category: str) -> pd.DataFrame:
    """
    Reconstruit (jamais reajuste ni persiste -- lecture seule) la MEME
    matrice standardisee que chaque clustering K-Means du projet et
    appelle _choose_k(..., return_trace=True) pour EXPOSER la silhouette
    de chaque k teste (retenu ou non), avec la raison de rejet quand
    applicable (n_clusters_effectif < 2 : effondrement KMeans sur points
    dupliques, cf. _choose_k ; taille_min_cluster < seuil : cluster trop
    petit). Sans cela, le k final n'etait justifie par rien d'observable --
    seul son resultat apparaissait dans les artefacts
    (n1_feature_schema.json / pooled_labeled.csv), jamais la comparaison
    aux k voisins qui l'ont ecarte.

    Deux "approches" dans le tableau retourne (colonne `approche`, meme
    convention que cluster_geometric_means) :
      - N1_technique : UN SEUL choix de k par categorie (marque/gamme =
        None) -- reproduit fit_n1_clustering (save_artifacts.py) a
        l'identique.
      - N2_marque_gamme : un choix de k PAR unite marque x gamme
        atteignant CLUSTERING_MIN_N sur la SEULE semaine de reference
        (n2_reference_week) -- reproduit la boucle de cluster_stability_n2
        et la logique de save_artifacts.process_category (jamais un k choisi
        sur le pool multi-semaines, cf. decision utilisateur du 2026-07-31).

    N'affecte JAMAIS les artefacts persistes (aucun impact sur
    save_artifacts.py) -- purement diagnostique, cf. reports/
    justification_k_clustering.csv.
    """
    df_pooled = load_pooled_category(category)

    # ── N1 : un seul k, categorie entiere (reproduit fit_n1_clustering) ──
    n1_continuous, n1_categorical = _classify_features(df_pooled, exclude=_ID_COLUMNS | {"semaine"})
    X_numeric = df_pooled[n1_continuous].astype(float) if n1_continuous else pd.DataFrame(index=df_pooled.index)
    X_categorical = (
        pd.get_dummies(df_pooled[n1_categorical].astype(str), columns=n1_categorical)
        if n1_categorical else pd.DataFrame(index=df_pooled.index)
    )
    X_n1 = StandardScaler().fit_transform(pd.concat([X_numeric, X_categorical], axis=1))
    _, trace_n1 = _choose_k(X_n1, _min_cluster_size_for(X_n1.shape[0]), return_trace=True)
    trace_n1.insert(0, "categorie", category)
    trace_n1.insert(1, "approche", "N1_technique")
    trace_n1.insert(2, "marque", None)
    trace_n1.insert(3, "gamme", None)

    # ── N2 : un k par unite marque x gamme, sur la SEULE semaine de reference
    # (reproduit cluster_stability_n2 / save_artifacts.process_category --
    # jamais le pool multi-semaines, cf. n2_reference_week) ──
    reference_week = n2_reference_week(df_pooled)
    df_reference = df_pooled[df_pooled["semaine"] == reference_week]
    df_tiers, _ = compute_price_tiers(df_reference)
    n2_continuous, n2_categorical = _classify_features(df_tiers, exclude=_ID_COLUMNS | {"semaine"})
    n2_frames = []
    for (marque, gamme), df_unit in df_tiers.groupby(["marque", "gamme_prix"], observed=True):
        n = len(df_unit)
        if n < CLUSTERING_MIN_N:
            continue
        X_unit = _build_feature_matrix(df_unit, n2_continuous, n2_categorical)
        _, trace_unit = _choose_k(X_unit, _min_cluster_size_for(n), return_trace=True)
        trace_unit.insert(0, "categorie", category)
        trace_unit.insert(1, "approche", "N2_marque_gamme")
        trace_unit.insert(2, "marque", marque)
        trace_unit.insert(3, "gamme", gamme)
        n2_frames.append(trace_unit)

    return pd.concat([trace_n1, *n2_frames], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# 8. ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rapports hebdomadaires (couverture clustering, moyennes geometriques par cluster, "
                     "estimations par modele + erreur (globales et par cluster marque x gamme), transitions "
                     "semaine a semaine par cluster, indice de prix hedonique, composition du catalogue)."
    )
    parser.add_argument("--category", type=str, default=None, choices=CATEGORY_ORDER,
                         help="Ne traiter qu'une seule categorie (defaut : toutes).")
    args = parser.parse_args()

    categories = [args.category] if args.category else CATEGORY_ORDER
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    coverage_frames, cluster_frames, estimates_frames, composition_frames, index_frames = [], [], [], [], []
    mg_estimates_frames, transitions_frames, stability_frames, k_justification_frames = [], [], [], []

    for category in categories:
        logger.info(f"{'=' * 70}\nCategorie : {category}")
        coverage_frames.append(coverage_by_week(category))
        cluster_frames.append(cluster_geometric_means(category))
        estimates_frames.append(weekly_model_estimates(category))
        composition_frames.append(catalog_composition_by_week(category))
        idx = hedonic_price_index(category)
        if idx is not None:
            index_frames.append(idx)

        product_level = _marque_gamme_product_level(category)
        mg_estimates = marque_gamme_model_estimates(category, product_level=product_level)
        mg_estimates_frames.append(mg_estimates)
        transitions_frames.append(
            cluster_transitions(category, estimates=mg_estimates, product_level=product_level)
        )
        stability_frames.append(cluster_stability_n2(category))
        k_justification_frames.append(k_selection_justification(category))
        logger.info(f"  [{category}] OK")

    pd.concat(coverage_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "couverture_clustering_hebdo.csv", index=False, encoding="utf-8-sig")
    pd.concat(cluster_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "cluster_prix_geometrique.csv", index=False, encoding="utf-8-sig")
    pd.concat(estimates_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "estimations_prix_hebdo.csv", index=False, encoding="utf-8-sig")
    pd.concat(composition_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "composition_catalogue_hebdo.csv", index=False, encoding="utf-8-sig")
    pd.concat(mg_estimates_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "marque_gamme_estimations_hebdo.csv", index=False, encoding="utf-8-sig")
    pd.concat(transitions_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "transitions_cluster_hebdo.csv", index=False, encoding="utf-8-sig")
    pd.concat(stability_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "stabilite_clustering_n2.csv", index=False, encoding="utf-8-sig")
    pd.concat(k_justification_frames, ignore_index=True).to_csv(
        REPORTS_DIR / "justification_k_clustering.csv", index=False, encoding="utf-8-sig")
    if index_frames:
        pd.concat(index_frames, ignore_index=True).to_csv(
            REPORTS_DIR / "indice_prix_hedonique_hebdo.csv", index=False, encoding="utf-8-sig")

    logger.info(f"\n{'=' * 70}\nRapports hebdomadaires ecrits sous {REPORTS_DIR}")


if __name__ == "__main__":
    main()
