# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/models/hedonic_model.py
=============================================================================
ROLE :
    Regression hedonique OLS (statsmodels, PAS sklearn -- on a besoin
    d'inference : coefficients, erreurs-types robustes, p-values, tests F)
    sur log(prix_tnd), en complement de RidgeModel/RandomForestModel
    (feature_importance.py) qui n'offrent aucune inference statistique.

    Pipeline complet :
      1. compute_price_tiers    : filtre les marques negligeables et assigne
                                  gamme_prix (Economique/Milieu/Premium,
                                  1 a 3 gammes selon l'effectif de la
                                  marque) -- MEME logique que
                                  src/notebooks/Segmentation_Prix_
                                  Clustering_produits_technologiques.ipynb.
      2. compute_cluster_labels : recalcule les clusters techniques
                                  (K-Means par marque x gamme, meme
                                  RANDOM_STATE=42) et les attache a chaque
                                  produit via cluster_id.

                                  ATTENTION -- CE PROJET NE PERSISTE NULLE
                                  PART UN MAPPING produit -> cluster (verifie
                                  exhaustivement : aucune colonne "cluster"
                                  dans data/processed/, aucun to_csv/to_json
                                  de labels dans les 2 notebooks de
                                  clustering). Ce module reconstruit donc les
                                  labels A LA VOLEE plutot que de les
                                  supposer disponibles sur disque -- cf.
                                  decision utilisateur du 2026-07-11.
      3. HedonicOLS             : classe d'estimation (fit/predict/summary/
                                  get_coefficients), erreurs-types robustes
                                  HC3 partout.
      4. fit_strategy_a / fit_strategy_b : les deux strategies de
                                  modelisation demandees (pentes communes
                                  vs pentes libres par marque dominante).
      5. compare_strategies     : test de Chow (interactions marque x
                                  caracteristiques + test F emboîte).
      6. check_tier_monotonicity, run_diagnostics : verifications post-
                                  estimation.
      7. fit_strategy_c_pooled_time : pooling multi-semaines (pentes
                                  communes + effet fixe "semaine"),
                                  indice de prix hedonique a la
                                  Triplett/Rosen -- erreurs-types
                                  groupees par produit (meme produit vu
                                  plusieurs semaines). Exclut les
                                  categories a coefficients instables
                                  entre semaines (cf.
                                  POOLED_TIME_EXCLUDED_CATEGORIES).

GARDE-FOU DE CIRCULARITE (critique) :
    prix_tnd et gamme_prix (et toute variable derivee du prix) ne peuvent
    JAMAIS atteindre la matrice de design -- gamme_prix a ete construite
    A PARTIR du prix (§ compute_price_tiers), l'utiliser comme regresseur
    "expliquerait" le prix par lui-meme. Ce garde-fou est applique par
    _check_no_circularity(), qui leve CircularityError des qu'une colonne
    interdite apparait dans les colonnes passees a build_design_matrix
    OU dans la matrice assemblee finale (double verification).

    CORRECTIF (2026-07-21, cf. reports/audit_code.md §3.1) -- cluster_id
    EST DESORMAIS INTERDIT COMME REGRESSEUR, comme gamme_prix :
    cluster_id est construit comme "marque::gamme::sous-cluster" (§
    compute_cluster_labels) -- la sous-chaine "gamme" y est LITTERALEMENT
    la valeur de gamme_prix. Le garde-fou original ne comparait que des
    NOMS de colonnes a une liste interdite ; il ne pouvait donc jamais
    detecter qu'une colonne nommee "cluster_id" encode, dans ses VALEURS,
    l'information meme qu'il est cense bannir. Consequence mesuree :
    utiliser cluster_id comme effet fixe categoriel gonflait
    artificiellement R²/adj-R² (ex. jusqu'a 0.978 avec seulement 2
    variables continues, cf. Evolution_Temporelle_Marche_Mytek.ipynb) --
    le modele "expliquait" en partie le prix par une version discretisee
    du prix lui-meme, pas uniquement par les caracteristiques hedoniques.
    "cluster_id" est donc desormais dans FORBIDDEN_REGRESSORS au meme
    titre que "gamme_prix". compute_price_tiers/compute_cluster_labels
    restent utilisables tels quels pour leur usage DESCRIPTIF legitime
    (segmentation marque x gamme x profil technique, cf. notebook de
    segmentation) -- seule leur reutilisation comme regresseur d'un
    modele de PRIX est desormais bloquee.

NOTE DE CONCEPTION -- effet fixe de MARQUE (pas cluster_id) :
    fit_strategy_a/fit_strategy_c_pooled_time/compare_strategies ajoutent
    desormais "marque" comme categorielle (effet fixe), pas cluster_id :
    la marque est une caracteristique du produit exogene a SON PROPRE prix
    (ce n'est pas une variable calculee a partir de prix_tnd de la ligne
    courante, contrairement a gamme_prix/cluster_id) -- un effet fixe de
    marque est une pratique standard et non circulaire en econometrie
    hedonique. fit_strategy_b n'a pas besoin de cet ajout : la marque est
    deja constante au sein de chaque regression par marque.

INTERPRETATION DES COEFFICIENTS (Halvorsen & Palmquist, 1980) :
    y = log(prix). Pour une variable CONTINUE, coefficient * 100 est
    l'approximation usuelle de la semi-elasticite (cf. ridge_model.py,
    meme convention). Pour une variable CATEGORIELLE/INDICATRICE (marque,
    OS, tier CPU, connectivite...), l'approximation devient trompeuse des
    que le coefficient n'est plus tres proche de 0 -- get_coefficients()
    utilise donc la formule EXACTE (exp(coefficient) - 1) * 100 pour toute
    variable qui n'est pas dans `continuous_cols`, jamais le coefficient
    brut.

UTILISATION :
    from src.models.hedonic_model import (
        load_category_data, compute_price_tiers, compute_cluster_labels,
        fit_strategy_a, fit_strategy_b, compare_strategies,
        check_tier_monotonicity, run_diagnostics,
    )
    df = load_category_data("pc_bureau")
    df, plan = compute_price_tiers(df)
    df, units = compute_cluster_labels(df, "pc_bureau")
    model_a, X_a, y_a = fit_strategy_a(df, continuous_features=["ram_go", "stockage_go"])
    print(model_a.summary())

    Voir src/models/demo_hedonic.py pour un exemple complet execute sur
    des donnees reelles (categorie pc_bureau).

    STRATEGIE C -- pooling multi-semaines avec indice de prix hedonique
    (effet fixe "semaine", cf. fit_strategy_c_pooled_time) :
        from src.models.hedonic_model import fit_strategy_c_pooled_time
        model_c, X_c, y_c, df_pooled = fit_strategy_c_pooled_time("pc_bureau")
        print(model_c.get_coefficients())  # dummies semaine_2/semaine_3 = indice de prix vs S1

    Voir src/models/demo_hedonic_pooled_time.py pour un exemple complet
    (compare le modele poole aux 3 modeles separes par semaine, sur
    donnees reelles, pour toutes les categories non exclues).
=============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.graphics.gofplots import qqplot
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.exceptions import NotFittedError

from src.preprocessing.split import discover_weeks
from src.utils.config import RANDOM_STATE

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "week_1"
# Instantane a une seule observation par produit (pas le panel semaine 1 + 2
# fusionne de data/processed/ a la racine) -- une regression hedonique
# suppose des observations independantes ; le panel compterait chaque
# produit vu aux 2 semaines comme 2 lignes, biaisant les erreurs-types.

# RANDOM_STATE importe de src.utils.config (source UNIQUE, cf. son docstring)
# -- auparavant redefini ici independamment (meme valeur 42, mais sans lien :
# changer l'un n'aurait jamais propage a l'autre, audit methodologique
# reviewer 4).
MIN_BRAND_COUNT = 5     # seuil de marque negligeable, identique aux notebooks
CLUSTERING_MIN_N = 10   # seuil sous lequel le clustering n'est pas tente

GAMME_ORDER_3 = ["Économique", "Milieu de gamme", "Premium"]
GAMME_ORDER_2 = ["Économique", "Premium"]

# Variables de prix / derivees du prix -- JAMAIS des regresseurs (cf. garde-fou
# de circularite dans le docstring du module). "cluster_id" y figure au meme
# titre que "gamme_prix" depuis le correctif du 2026-07-21 : cluster_id =
# "marque::gamme::sous-cluster" (compute_cluster_labels) encode litteralement
# gamme_prix dans ses valeurs -- un nom de colonne different ne le rend pas
# moins derive du prix. Reste utilisable comme sortie DESCRIPTIVE (segmentation),
# desormais interdit comme entree d'un modele de prix.
FORBIDDEN_REGRESSORS = frozenset({"prix_tnd", "gamme_prix", "log_prix_tnd", "cluster_id"})

_ID_COLUMNS = {"url", "nom", "prix_tnd", "marque", "gamme_prix", "cluster_id"}


class CircularityError(ValueError):
    """Levee quand une variable de prix ou derivee du prix (gamme_prix,
    cluster_id) atteint la matrice de design d'un modele hedonique."""


def _check_no_circularity(columns, blocklist=FORBIDDEN_REGRESSORS):
    """Leve CircularityError si une colonne interdite est presente.
    Appelee a la fois sur la liste de features demandee et sur la matrice
    de design assemblee (double verification, §garde-fou)."""
    forbidden_found = set(columns) & set(blocklist)
    if forbidden_found:
        raise CircularityError(
            f"Variable(s) interdite(s) dans la matrice de design (prix ou "
            f"derivee du prix -- jamais une entree du modele hedonique) : "
            f"{sorted(forbidden_found)}. cluster_id/gamme_prix encodent le "
            f"positionnement de prix du produit lui-meme (jamais autorises) ; "
            f"pour un effet fixe de marque, utiliser 'marque' directement."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHARGEMENT DES DONNEES
# ─────────────────────────────────────────────────────────────────────────────

def load_category_data(category: str, processed_dir: Path | str | None = None) -> pd.DataFrame:
    """Charge <categorie>_clean.csv (data/processed/week_1/ par defaut --
    instantane a une seule observation par produit, cf. DEFAULT_PROCESSED_DIR)."""
    processed_dir = Path(processed_dir) if processed_dir is not None else DEFAULT_PROCESSED_DIR
    return pd.read_csv(processed_dir / f"{category}_clean.csv")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GAMMES DE PRIX (marque x prix uniquement -- AUCUNE caracteristique
#    technique n'intervient ici, exactement comme dans le notebook de
#    segmentation)
# ─────────────────────────────────────────────────────────────────────────────

def _n_gammes_for(n: int) -> int:
    """n >= 15 -> 3 gammes (terciles) ; 10 <= n < 15 -> 2 (mediane) ;
    n < 10 -> 1 (aucune coupure fiable). Identique au notebook."""
    if n >= 15:
        return 3
    if n >= 10:
        return 2
    return 1


def n2_reference_week(df_pooled: pd.DataFrame) -> int:
    """Semaine de reference pour le clustering N2 (marque x gamme) -- TOUJOURS
    la plus ancienne semaine disponible dans `df_pooled`, jamais recalculee a
    chaque appel/a chaque nouvelle semaine ajoutee (meme convention que
    `ref_week = weeks[0]` dans fit_strategy_c_pooled_time/hedonic_price_index).

    gamme_prix et cluster_id (N2) sont figes sur CETTE seule semaine
    (compute_price_tiers + compute_cluster_labels appeles sur le sous-ensemble
    df_pooled[semaine == n2_reference_week(df_pooled)] uniquement) puis
    reappliques par produit (url) a toutes les semaines ou il apparait --
    decision utilisateur du 2026-07-31 : un cluster K-Means ne doit JAMAIS
    melanger des observations de semaines differentes (un meme produit vu 4
    fois, une fois par semaine, n'est pas 4 points independants -- gonfle
    artificiellement l'effectif d'une unite marque x gamme au-dela de
    CLUSTERING_MIN_N et fait apparaitre une structure qui n'existe pas a
    l'echelle d'une seule semaine). Different du pooling multi-semaines des
    modeles de regression (legitime, decision utilisateur du 2026-07-21,
    protege par un split train/test groupe par produit) : le K-Means n'a pas
    de notion de "periode" dans ses features, mixer les semaines y cree des
    doublons de points, pas de la donnee supplementaire.

    Consequence assumee : un produit absent de la semaine de reference
    (nouvelle arrivee dans une semaine ulterieure, ou marque alors sous
    MIN_BRAND_COUNT) n'a AUCUN gamme_prix/cluster_id, dans aucune semaine, tant
    que le pipeline n'est pas relance apres mise a jour de la semaine de
    reference -- jamais invente silencieusement (cf. coverage_by_week, qui
    continue de rapporter le taux de couverture N2 reel).

    Seule source de verite pour save_artifacts.process_category,
    weekly_report.cluster_stability_n2 et weekly_report.k_selection_justification
    -- jamais de logique dupliquee/divergente entre ces 3 appelants."""
    return int(sorted(df_pooled["semaine"].unique())[0])


def compute_price_tiers(df: pd.DataFrame, min_brand_count: int = MIN_BRAND_COUNT):
    """
    Filtre les marques a effectif insuffisant (< min_brand_count) et assigne
    gamme_prix a chaque produit restant, calculee INDEPENDAMMENT par marque
    (le positionnement "Premium" est relatif a la marque, pas au marche
    entier -- cf. §2.1 du notebook de segmentation, meme motivation ici).

    Returns:
        (df_out, brand_plan) -- df_out : DataFrame filtre + colonne
        gamme_prix (str) ; brand_plan : DataFrame [marque, n,
        n_gammes_planifie, n_gammes_effectif] (l'effectif peut etre
        inferieur au planifie si des prix trop souvent identiques ont
        fusionne des paliers, qcut(duplicates="drop")).
    """
    counts = df["marque"].value_counts()
    kept_counts = counts[counts >= min_brand_count]

    tiered_parts = []
    plan_rows = []
    for brand, n in kept_counts.items():
        n = int(n)
        n_gammes_planned = _n_gammes_for(n)
        df_brand = df[df["marque"] == brand].copy()

        if n_gammes_planned == 1:
            df_brand["gamme_prix"] = "Gamme unique"
            n_gammes_effective = 1
        else:
            order = GAMME_ORDER_2 if n_gammes_planned == 2 else GAMME_ORDER_3
            codes = pd.qcut(df_brand["prix_tnd"], n_gammes_planned, duplicates="drop")
            labels = order[: len(codes.cat.categories)]
            df_brand["gamme_prix"] = codes.cat.rename_categories(labels).astype(str)
            n_gammes_effective = len(labels)

        tiered_parts.append(df_brand)
        plan_rows.append({
            "marque": brand, "n": n,
            "n_gammes_planifie": n_gammes_planned,
            "n_gammes_effectif": n_gammes_effective,
        })

    df_out = pd.concat(tiered_parts, ignore_index=True) if tiered_parts else df.iloc[0:0].copy()
    brand_plan = pd.DataFrame(plan_rows).sort_values("n", ascending=False).reset_index(drop=True)
    return df_out, brand_plan


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLUSTERS TECHNIQUES (recalcules a la volee -- aucun mapping persiste,
#    cf. ATTENTION dans le docstring du module)
# ─────────────────────────────────────────────────────────────────────────────


# Roles EXPLICITES demandes (RAM/stockage/ecran continus ; tier CPU/OS/
# connectivite categoriels) -- pas une heuristique de cardinalite : ram_go
# ne prend que 5-6 valeurs distinctes dans ce catalogue (8/16/24/32/64 Go)
# mais reste une quantite CONTINUE (un Go de plus a un sens marginal
# constant), a la difference de cpu_serie (0/3/5/7/9) dont les paliers ne
# sont PAS equidistants en valeur economique (l'ecart Core i5->i7 n'a
# aucune raison d'egaler i7->i9) -- d'ou la demande explicite de traiter
# le tier CPU comme categoriel malgre son encodage numerique.
_ALWAYS_CONTINUOUS = {"ram_go", "stockage_go", "taille_ecran", "taux_rafraichissement"}
_ALWAYS_CATEGORICAL = {
    "cpu_serie", "cpu_gen", "cpu_brand", "os_platform",
    "resolution_affichage", "technologie_dalle", "hdr", "smart_tv",
}


def _classify_features(df: pd.DataFrame, exclude=_ID_COLUMNS, max_unique_for_categorical: int = 6):
    """Classe les colonnes restantes en continues vs categorielles.
    Priorite a une liste EXPLICITE de roles (RAM/stockage/ecran/taux de
    rafraichissement toujours continus ; tier CPU/OS/dalle/resolution
    toujours categoriels, cf. _ALWAYS_CONTINUOUS/_ALWAYS_CATEGORICAL) --
    ne PAS se fier a la seule cardinalite (ram_go n'a que 5-6 valeurs
    distinctes dans ce catalogue mais reste continue). Les colonnes non
    reconnues (has_4g, has_5g... deja 0/1, ou une colonne future) tombent
    sur une regle de repli par cardinalite : > max_unique_for_categorical
    valeurs distinctes -> continue, sinon categorielle."""
    continuous, categorical = [], []
    for col in df.columns:
        if col in exclude:
            continue
        if col in _ALWAYS_CONTINUOUS:
            continuous.append(col)
        elif col in _ALWAYS_CATEGORICAL:
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > max_unique_for_categorical:
            continuous.append(col)
        else:
            categorical.append(col)
    return continuous, categorical


def _max_k_for(n_rows: int) -> int:
    return max(3, min(8, n_rows // 10))


def _min_cluster_size_for(n_rows: int) -> int:
    return max(5, round(0.03 * n_rows))


def _build_feature_matrix(df_unit: pd.DataFrame, continuous_features, categorical_features):
    """Matrice standardisee pour K-Means -- indicatrices SANS suppression
    de modalite (une methode de distance a besoin que chaque modalite
    soit equidistante des autres), a la difference de build_design_matrix
    (regression) qui doit au contraire supprimer une modalite de reference."""
    X_numeric = df_unit[continuous_features].astype(float) if continuous_features else pd.DataFrame(index=df_unit.index)
    if categorical_features:
        X_categorical = pd.get_dummies(df_unit[categorical_features].astype(str), columns=categorical_features)
    else:
        X_categorical = pd.DataFrame(index=df_unit.index)
    X_raw = pd.concat([X_numeric, X_categorical], axis=1)
    return StandardScaler().fit_transform(X_raw)


def _choose_k(X, min_size: int, return_trace: bool = False):
    """k a silhouette maximale PARMI ceux dont tous les clusters comptent
    au moins min_size produits. Si aucun k teste ne satisfait cette
    contrainte, retourne 1 (AUCUNE structure retenue) plutot qu'un k>=2
    degenere -- correction appliquee dans le notebook de segmentation,
    reprise ici a l'identique (cf. §2.6ter).

    GARDE-FOU EFFONDREMENT KMeans (bug reel, decouvert en poolant plusieurs
    semaines -- src/models/save_artifacts.py) : avec des points dupliques
    (meme produit technique vu a plusieurs semaines), KMeans(n_clusters=k)
    peut converger vers STRICTEMENT MOINS de k clusters distincts que
    demande (ConvergenceWarning "Number of distinct clusters found smaller
    than n_clusters"). Si l'effondrement est total (1 seul label malgre
    k>=2 demande), silhouette_score() leve ValueError ("Number of labels
    is 1") -- ce k est alors ignore (comme s'il ne passait pas le
    garde-fou min_size), jamais un plantage.

    return_trace : si True, retourne (k, trace) au lieu de k seul -- trace
    est un DataFrame listant CHAQUE k teste (retenu ou non) avec sa
    silhouette et la raison de rejet, pour AUDITER le choix au lieu de le
    laisser opaque (rigor upgrade, 2026-07-27). N'affecte jamais la
    decision elle-meme : a return_trace=False, le calcul est strictement
    identique a avant (aucun cout supplementaire sur le chemin de
    production -- save_artifacts.py, compute_cluster_labels)."""
    n = X.shape[0]
    best_k, best_sil = None, -1.0
    trace_rows = [] if return_trace else None
    for k in range(2, _max_k_for(n) + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        n_effectif = int(pd.Series(labels).nunique())
        if n_effectif < 2:
            if return_trace:
                trace_rows.append({"k": k, "n_clusters_effectif": n_effectif,
                                    "taille_min_cluster": 0, "silhouette": None, "valide": False})
            continue
        taille_min = int(pd.Series(labels).value_counts().min())
        valide = taille_min >= min_size
        sil = silhouette_score(X, labels) if (valide or return_trace) else None
        if return_trace:
            trace_rows.append({"k": k, "n_clusters_effectif": n_effectif,
                                "taille_min_cluster": taille_min, "silhouette": sil, "valide": bool(valide)})
        if valide and sil > best_sil:
            best_sil, best_k = sil, k
    k_final = best_k if best_k is not None else 1
    if return_trace:
        trace = pd.DataFrame(trace_rows)
        trace["k_retenu"] = trace["k"] == k_final
        return k_final, trace
    return k_final


def compute_cluster_labels(
    df: pd.DataFrame, category: str,
    continuous_features: list | None = None, categorical_features: list | None = None,
    clustering_min_n: int = CLUSTERING_MIN_N, random_state: int = RANDOM_STATE,
):
    """
    Recalcule les clusters techniques (K-Means par couple marque x gamme,
    meme methodologie et meme graine que le notebook de segmentation) et
    les attache a chaque produit via une colonne cluster_id.

    df DOIT deja avoir une colonne gamme_prix (cf. compute_price_tiers).

    cluster_id est une chaine "marque::gamme::c<label>" -- imbriquee par
    construction (cf. NOTE DE CONCEPTION dans le docstring du module) :
    - Si l'unite atteint clustering_min_n ET qu'un k>=2 valide est trouve :
      label = le label K-Means local (0..k-1).
    - Sinon (n < clustering_min_n, OU aucun k valide -- garde-fou
      declenche) : label = 0 pour tous les produits de l'unite (un seul
      groupe "trivial", qui est alors exactement le couple marque x gamme
      lui-meme).
    Dans les deux cas, cluster_id est TOUJOURS defini (jamais de valeur
    manquante) et toujours globalement unique.

    Args:
        continuous_features, categorical_features: si None, deduits
            automatiquement via _classify_features (cf. sa docstring).

    Returns:
        (df_out, unit_summary) -- df_out : df + colonne cluster_id ;
        unit_summary : DataFrame [marque, gamme_prix, n, k, outcome] avec
        outcome in {"clustered", "no_structure", "too_small"} (meme
        vocabulaire que le notebook de segmentation).
    """
    if "gamme_prix" not in df.columns:
        raise ValueError(
            "compute_cluster_labels requiert une colonne 'gamme_prix' -- "
            "appeler compute_price_tiers(df) d'abord."
        )

    auto_continuous, auto_categorical = _classify_features(df)
    continuous_features = continuous_features if continuous_features is not None else auto_continuous
    categorical_features = categorical_features if categorical_features is not None else auto_categorical

    cluster_ids = pd.Series(index=df.index, dtype=object)
    summary_rows = []

    for (brand, gamme), df_unit in df.groupby(["marque", "gamme_prix"], observed=True):
        n = len(df_unit)
        if n >= clustering_min_n:
            X = _build_feature_matrix(df_unit, continuous_features, categorical_features)
            k = _choose_k(X, _min_cluster_size_for(n))
            if k > 1:
                km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                local_labels = km.fit_predict(X)
                outcome = "clustered"
            else:
                local_labels = np.zeros(n, dtype=int)
                outcome = "no_structure"
        else:
            local_labels = np.zeros(n, dtype=int)
            k = 1
            outcome = "too_small"

        cluster_ids.loc[df_unit.index] = [f"{brand}::{gamme}::c{lbl}" for lbl in local_labels]
        summary_rows.append({"marque": brand, "gamme_prix": gamme, "n": n, "k": k, "outcome": outcome})

    df_out = df.copy()
    df_out["cluster_id"] = cluster_ids
    unit_summary = pd.DataFrame(summary_rows).sort_values(["marque", "gamme_prix"]).reset_index(drop=True)
    return df_out, unit_summary


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATRICE DE DESIGN (regression -- suppression de modalite de reference,
#    a la difference de _build_feature_matrix ci-dessus)
# ─────────────────────────────────────────────────────────────────────────────

def build_design_matrix(df: pd.DataFrame, continuous_features: list, categorical_features: list, drop_first: bool = True):
    """
    Construit (X, y) pour HedonicOLS. y = log(prix_tnd). X : continues
    telles quelles, categorielles en indicatrices (drop_first=True --
    une modalite de reference doit etre supprimee pour eviter la
    colinearite parfaite avec la constante, a la difference du clustering
    K-Means qui n'a pas de constante). Les colonnes deja binaires 0/1
    (has_5g, hdr...) ne sont PAS reencodees (deja directement utilisables).

    Verifie l'absence de circularite AVANT (sur les listes de features
    demandees) ET APRES (sur les colonnes assemblees, defense en profondeur).
    """
    _check_no_circularity(list(continuous_features) + list(categorical_features))

    X_numeric = df[continuous_features].astype(float) if continuous_features else pd.DataFrame(index=df.index)

    already_binary = [
        c for c in categorical_features
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 2
    ]
    to_encode = [c for c in categorical_features if c not in already_binary]

    X_binary = df[already_binary].astype(float) if already_binary else pd.DataFrame(index=df.index)
    if to_encode:
        X_categorical = pd.get_dummies(df[to_encode].astype(str), columns=to_encode, drop_first=drop_first).astype(float)
    else:
        X_categorical = pd.DataFrame(index=df.index)

    X = pd.concat([X_numeric, X_binary, X_categorical], axis=1)
    _check_no_circularity(X.columns)

    y = np.log(df["prix_tnd"].astype(float))
    y.name = "log_prix_tnd"
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 5. HedonicOLS -- estimation (statsmodels, erreurs-types robustes)
# ─────────────────────────────────────────────────────────────────────────────

class HedonicOLS:
    """
    OLS hedonique (statsmodels.OLS) sur log(prix), erreurs-types robustes
    a l'heteroscedasticite (HC3 par defaut) -- fournit l'inference que
    RidgeModel/RandomForestModel (sklearn) n'offrent pas : coefficients,
    p-values, tests F, AIC/BIC.
    """

    def __init__(self, cov_type: str = "HC3"):
        """
        Args:
            cov_type: type d'erreurs-types robustes passe a
                statsmodels' .fit(cov_type=...). "HC3" par defaut (le plus
                conservateur des estimateurs de White, recommande sur les
                petits echantillons -- cf. MacKinnon & White, 1985).
        """
        self.cov_type = cov_type
        self.result_ = None
        self.continuous_cols_ = None
        self.exog_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, continuous_cols: list | None = None,
            blocklist=FORBIDDEN_REGRESSORS, cov_kwds: dict | None = None):
        """Ajuste une constante + X par OLS. continuous_cols identifie
        les colonnes CONTINUES parmi X (le reste est traite comme
        categoriel/indicatrice pour l'interpretation des effets, cf.
        get_coefficients).

        cov_kwds : arguments supplementaires pour statsmodels'
            .fit(cov_type=...) -- necessaire pour cov_type="cluster"
            (ex: {"groups": df["url"]}), qui EXIGE un argument "groups"
            en plus du nom du type. Ignore (None) pour les cov_type
            robustes usuels (HC3...) qui n'en ont pas besoin. Utilise par
            fit_strategy_c_pooled_time pour grouper les erreurs-types par
            produit (url) quand plusieurs semaines d'un meme produit sont
            poolees -- des observations repetees, pas independantes."""
        _check_no_circularity(X.columns, blocklist)
        self.continuous_cols_ = set(continuous_cols) if continuous_cols is not None else set()

        X_ = sm.add_constant(X, has_constant="add").astype(float)
        y_ = y.astype(float)
        fit_kwargs = {"cov_type": self.cov_type}
        if cov_kwds:
            fit_kwargs["cov_kwds"] = cov_kwds
        self.result_ = sm.OLS(y_, X_).fit(**fit_kwargs)
        self.exog_ = X_
        return self

    def _check_fitted(self):
        if self.result_ is None:
            raise NotFittedError("HedonicOLS n'est pas encore ajuste -- appeler fit(X, y) d'abord.")

    def predict(self, X: pd.DataFrame):
        """Predit log(prix) pour de nouvelles observations."""
        self._check_fitted()
        X_ = sm.add_constant(X, has_constant="add").astype(float)
        X_ = X_.reindex(columns=self.exog_.columns, fill_value=0.0)
        return self.result_.predict(X_)

    def summary(self):
        """Retourne le resume statsmodels complet (coefficients, R2, F-stat...)."""
        self._check_fitted()
        return self.result_.summary()

    def get_coefficients(self) -> pd.DataFrame:
        """
        Coefficients avec erreur-type robuste, p-value, et effet en %
        (Halvorsen & Palmquist, 1980) :
        - variable CONTINUE (dans continuous_cols) : pct_effect =
          coefficient * 100 (semi-elasticite approximative, meme
          convention que RidgeModel.get_coefficients).
        - variable CATEGORIELLE/INDICATRICE (tout le reste, ex. marque,
          os_platform, tier CPU) : pct_effect =
          (exp(coefficient) - 1) * 100 -- formule EXACTE, jamais le
          coefficient brut (l'approximation lineaire est mauvaise des
          que |coefficient| s'eloigne de 0, frequent pour des effets de
          marque/cluster).

        Returns: DataFrame ["feature", "coefficient", "std_err",
            "p_value", "pct_effect"], trie par |coefficient| decroissant.
        """
        self._check_fitted()
        params, bse, pvalues = self.result_.params, self.result_.bse, self.result_.pvalues

        rows = []
        for name in params.index:
            beta = params[name]
            if name == "const":
                pct_effect = np.nan
            elif name in self.continuous_cols_:
                pct_effect = beta * 100
            else:
                pct_effect = (np.exp(beta) - 1) * 100
            rows.append({
                "feature": name, "coefficient": beta,
                "std_err": bse[name], "p_value": pvalues[name],
                "pct_effect": pct_effect,
            })
        df = pd.DataFrame(rows)
        return df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    @property
    def rsquared_adj(self):
        self._check_fitted()
        return self.result_.rsquared_adj

    @property
    def aic(self):
        self._check_fitted()
        return self.result_.aic

    @property
    def bic(self):
        self._check_fitted()
        return self.result_.bic

    @property
    def nobs(self):
        self._check_fitted()
        return self.result_.nobs

    @property
    def ssr(self):
        self._check_fitted()
        return self.result_.ssr

    @property
    def df_resid(self):
        self._check_fitted()
        return self.result_.df_resid


# ─────────────────────────────────────────────────────────────────────────────
# 6. STRATEGIE A -- un modele par categorie, pentes communes
# ─────────────────────────────────────────────────────────────────────────────

def fit_strategy_a(df: pd.DataFrame, continuous_features: list, categorical_features: list | None = None,
                    cov_type: str = "HC3"):
    """
    Strategie A : UN modele pour toute la categorie -- pentes communes sur
    les caracteristiques continues, "marque" (effet fixe) decale
    l'intercept.

    Historique (correctif 2026-07-21, cf. reports/audit_code.md §3.1) :
    cette fonction ajoutait auparavant cluster_id (imbrique marque x gamme
    x sous-cluster) plutot que marque seule -- mais gamme_prix, encodee
    dans cluster_id, est derivee du prix de la ligne elle-meme : un effet
    fixe qui "sait" dans quel tercile de prix se trouve deja le produit
    gonfle artificiellement R²/adj-R² (verifie empiriquement : jusqu'a
    0.978 avec seulement 2 variables continues). "marque" seule n'a pas ce
    probleme (une caracteristique du produit exogene a son propre prix) et
    reste un effet fixe standard en econometrie hedonique.

    Args:
        categorical_features: categorielles A AJOUTER en plus de "marque"
            (ex: os_platform, has_5g...). "marque" est toujours incluse
            automatiquement -- ne pas la repeter ici.

    Returns: (model: HedonicOLS deja ajuste, X, y)
    """
    categorical_features = list(categorical_features) if categorical_features else []
    if "marque" not in categorical_features:
        categorical_features = categorical_features + ["marque"]

    X, y = build_design_matrix(df, list(continuous_features), categorical_features)
    model = HedonicOLS(cov_type=cov_type).fit(X, y, continuous_cols=continuous_features)
    return model, X, y


# ─────────────────────────────────────────────────────────────────────────────
# 7. STRATEGIE B -- regressions separees par segment (pentes libres),
#    gardees par un seuil n/predicteurs (meme esprit que CLUSTERING_MIN_N)
# ─────────────────────────────────────────────────────────────────────────────

def _min_rows_required(n_predictors: int, ratio: int = 10) -> int:
    """Regle empirique : au moins `ratio` observations par predicteur
    (10 par defaut) -- meme logique de garde-fou que CLUSTERING_MIN_N,
    generalisee a un nombre de predicteurs variable selon le segment."""
    return ratio * n_predictors


def fit_strategy_b(df: pd.DataFrame, continuous_features: list, categorical_features: list | None = None,
                    top_n_brands: int = 3, min_rows_ratio: int = 10, cov_type: str = "HC3"):
    """
    Strategie B : une regression SEPAREE par marque dominante (pentes
    libres -- chaque marque peut avoir son propre effet RAM/stockage/...),
    UNIQUEMENT pour les `top_n_brands` marques les plus representees, et
    UNIQUEMENT si son effectif est suffisant relativement au nombre de
    predicteurs de son propre modele (guarde par _min_rows_required,
    ratio configurable). Les marques ecartees sont rapportees, jamais
    silencieusement.

    Returns:
        (results, skipped) -- results : dict {marque: {"model", "X", "y",
        "n"}} ; skipped : DataFrame [marque, n, n_predicteurs,
        n_min_requis] pour les marques ecartees.
    """
    categorical_features = list(categorical_features) if categorical_features else []
    # Pas d'ajout automatique ici (contrairement a fit_strategy_a) : chaque
    # regression est deja limitee a UNE marque, "marque" y serait constante
    # (aucune variance a exploiter) -- et cluster_id/gamme_prix restent de
    # toute facon interdits (§garde-fou de circularite).

    results, skipped_rows = {}, []
    top_brands = df["marque"].value_counts().head(top_n_brands).index.tolist()

    for brand in top_brands:
        df_brand = df[df["marque"] == brand].reset_index(drop=True)
        X_b, y_b = build_design_matrix(df_brand, list(continuous_features), categorical_features)
        n_predictors = X_b.shape[1] + 1  # +1 pour la constante
        min_required = _min_rows_required(n_predictors, min_rows_ratio)

        if len(df_brand) >= min_required:
            model = HedonicOLS(cov_type=cov_type).fit(X_b, y_b, continuous_cols=continuous_features)
            results[brand] = {"model": model, "X": X_b, "y": y_b, "n": len(df_brand)}
        else:
            skipped_rows.append({
                "marque": brand, "n": len(df_brand),
                "n_predicteurs": n_predictors, "n_min_requis": min_required,
            })

    skipped = pd.DataFrame(skipped_rows) if skipped_rows else pd.DataFrame(
        columns=["marque", "n", "n_predicteurs", "n_min_requis"])
    return results, skipped


# ─────────────────────────────────────────────────────────────────────────────
# 7bis. STRATEGIE C -- pooling multi-semaines avec effet fixe temps
#       (indice de prix hedonique a la Triplett/Rosen)
# ─────────────────────────────────────────────────────────────────────────────

# Categories exclues du pooling temporel : Evolution_Temporelle_Marche_Mytek.ipynb
# (pont hedonique, S1->S3) a deja mesure des coefficients INSTABLES pour
# telephones_portables (ex: prix implicite du stockage +14.47%/Go en S1 ->
# quasi nul en S3) -- probablement du bruit d'echantillon (tres peu d'unites
# marque x gamme clusterisees dans cette petite categorie), mais poolable
# quand meme serait statistiquement malhonnete : un effet fixe semaine
# suppose des pentes COMMUNES entre semaines, l'exact contraire de ce qui a
# ete observe ici. Les 4 autres categories ont toutes montre des coefficients
# stables (RAM/stockage/ecran) sur le meme pont hedonique.
POOLED_TIME_EXCLUDED_CATEGORIES = frozenset({"telephones_portables"})


def fit_strategy_c_pooled_time(
    category: str, weeks: list | None = None, processed_root: Path | str | None = None,
    continuous_features: list | None = None, categorical_features: list | None = None,
    cov_type: str = "cluster",
):
    """
    Strategie C : POOLE plusieurs semaines d'une categorie en un seul
    modele a pentes communes (RAM/stockage/... -- comme la Strategie A),
    plus une indicatrice "semaine" (S1 = reference) dont les coefficients
    fournissent directement un INDICE DE PRIX HEDONIQUE dans le temps
    (methode standard des indices de prix par regression a variables
    muettes temporelles, Triplett 2004 / Rosen 1974) : la variation de
    prix "pure", nette des caracteristiques/de la composition du
    catalogue -- exactement la question que l'analyse en coupe ne peut
    pas trancher (cf. Evolution_Temporelle_Marche_Mytek.ipynb, §0).

    cluster_id est recalcule INDEPENDAMMENT pour chaque semaine (meme
    fonction compute_cluster_labels qu'ailleurs, pas de dependance aux
    labels persistes dans outputs/labels/ qui ne couvrent que S1) et reste
    disponible en colonne de df_pooled pour un usage DESCRIPTIF (ex.
    tableau de bord) -- mais n'entre plus dans la matrice de design
    (§garde-fou de circularite : cluster_id encode gamme_prix). L'effet
    fixe categoriel utilise ici est "marque", pas cluster_id.

    ERREURS-TYPES GROUPEES PAR PRODUIT (cov_type="cluster", groups=url) :
    un meme produit apparait plusieurs fois (une fois par semaine, cf.
    dedup_key de clean.py qui garde deliberement ces observations
    distinctes) -- ce ne sont PAS des observations independantes, grouper
    par url evite de sous-estimer les erreurs-types comme le ferait HC3
    seul (qui suppose l'independance, seulement l'heteroscedasticite).

    Leve ValueError si `category` est dans POOLED_TIME_EXCLUDED_CATEGORIES
    (coefficients hedoniques deja mesures comme instables entre semaines
    -- pooler avec des pentes communes serait errone).

    Args:
        weeks: semaines a pooler. Si None (par defaut), decouvre
            dynamiquement TOUTES les semaines disponibles sous
            data/processed/ (meme mecanisme que
            src.preprocessing.split.discover_weeks) -- jamais fige a
            (1, 2, 3) : une semaine 4 ajoutee au depot est incluse sans
            modifier ce fichier.

    Returns: (model: HedonicOLS ajuste, X, y, df_pooled)
    """
    if category in POOLED_TIME_EXCLUDED_CATEGORIES:
        raise ValueError(
            f"'{category}' est exclue du pooling temporel : Evolution_Temporelle_Marche_Mytek.ipynb a mesure "
            f"des coefficients hedoniques instables entre semaines pour cette categorie (pentes communes non "
            f"justifiees). Utiliser fit_strategy_a semaine par semaine a la place."
        )

    processed_root_path = Path(processed_root) if processed_root is not None else DEFAULT_PROCESSED_DIR.parent
    if weeks is None:
        weeks = discover_weeks(processed_root_path)
        if not weeks:
            raise FileNotFoundError(f"Aucune semaine trouvee sous {processed_root_path}")

    week_frames = []
    for w in weeks:
        path = processed_root_path / f"week_{w}" / f"{category}_clean.csv"
        if not path.exists():
            continue
        df_w = pd.read_csv(path)
        df_w, _ = compute_price_tiers(df_w)
        auto_continuous, auto_categorical = _classify_features(df_w)
        cont_w = continuous_features if continuous_features is not None else auto_continuous
        cat_w = categorical_features if categorical_features is not None else auto_categorical
        df_w, _ = compute_cluster_labels(df_w, category, continuous_features=cont_w, categorical_features=cat_w)
        df_w["semaine"] = w
        week_frames.append(df_w)

    if not week_frames:
        raise FileNotFoundError(f"Aucune semaine trouvee pour '{category}'.")
    df_pooled = pd.concat(week_frames, ignore_index=True)

    continuous_features = continuous_features if continuous_features is not None else _classify_features(df_pooled)[0]
    categorical_features = list(categorical_features) if categorical_features else []
    if "marque" not in categorical_features:
        categorical_features = categorical_features + ["marque"]
    categorical_features = categorical_features + ["semaine"]

    X, y = build_design_matrix(df_pooled, list(continuous_features), categorical_features)
    cov_kwds = {"groups": df_pooled["url"]} if cov_type == "cluster" else None
    model = HedonicOLS(cov_type=cov_type).fit(X, y, continuous_cols=continuous_features, cov_kwds=cov_kwds)
    return model, X, y, df_pooled


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPARAISON A vs B -- test de Chow (interactions + test F emboîte)
# ─────────────────────────────────────────────────────────────────────────────

def compare_strategies(df: pd.DataFrame, continuous_features: list, categorical_features: list | None = None,
                        top_n_brands: int = 3, cov_type: str = "HC3"):
    """
    Test de Chow formel entre strategie A (pentes communes) et strategie B
    (pentes libres par marque dominante), via un modele groupe avec
    interactions marque x caracteristique :

        restreint (= strategie A)     : y ~ continues + C(marque)
        non-restreint (= strategie B) : restreint + (marque_dominante x
                                          caracteristique) pour chaque
                                          marque dominante et chaque
                                          caracteristique continue

    Les deux modeles sont EMBOITES (le non-restreint contient toutes les
    colonnes du restreint) et ajustes sur le MEME echantillon (toute la
    categorie) -- comparables par test F direct
    (RegressionResults.compare_f_test), l'implementation standard d'un
    test de Chow via variables muettes d'interaction.

    H0 : les pentes sont communes a toutes les marques (A suffit).
    H1 : les marques dominantes ont des pentes propres (B necessaire).

    Returns: (comparison, chow, restricted_model, unrestricted_model)
        comparison : DataFrame [modele, n_predicteurs, adj_r2, aic, bic]
        chow : dict {F, p_value, df_diff, conclusion}
    """
    categorical_features = list(categorical_features) if categorical_features else []
    if "marque" not in categorical_features:
        categorical_features = categorical_features + ["marque"]

    dominant_brands = df["marque"].value_counts().head(top_n_brands).index.tolist()

    X_restricted, y = build_design_matrix(df, list(continuous_features), categorical_features)
    X_unrestricted = X_restricted.copy()
    for brand in dominant_brands:
        is_brand = (df["marque"] == brand).astype(float).to_numpy()
        for feat in continuous_features:
            X_unrestricted[f"{brand}_x_{feat}"] = is_brand * df[feat].astype(float).to_numpy()

    restricted_model = HedonicOLS(cov_type=cov_type).fit(X_restricted, y, continuous_cols=continuous_features)
    unrestricted_model = HedonicOLS(cov_type=cov_type).fit(X_unrestricted, y, continuous_cols=continuous_features)

    fvalue, pvalue, df_diff = unrestricted_model.result_.compare_f_test(restricted_model.result_)

    comparison = pd.DataFrame([
        {"modele": "A -- pentes communes", "n_predicteurs": X_restricted.shape[1] + 1,
         "adj_r2": restricted_model.rsquared_adj, "aic": restricted_model.aic, "bic": restricted_model.bic},
        {"modele": "B -- pentes libres (marques dominantes)", "n_predicteurs": X_unrestricted.shape[1] + 1,
         "adj_r2": unrestricted_model.rsquared_adj, "aic": unrestricted_model.aic, "bic": unrestricted_model.bic},
    ])

    conclusion = (
        "Les pentes different significativement par marque (p<0.05) -- la strategie B est justifiee."
        if pvalue < 0.05 else
        "Pentes communes non rejetees (p>=0.05) -- la strategie A (plus simple, plus parcimonieuse) suffit."
    )
    chow = {"F": float(fvalue), "p_value": float(pvalue), "df_diff": df_diff, "conclusion": conclusion}
    return comparison, chow, restricted_model, unrestricted_model


# ─────────────────────────────────────────────────────────────────────────────
# 9. MONOTONIE DES GAMMES -- "Premium" est-il hedoniquement plus cher,
#    une fois les caracteristiques controlees ?
# ─────────────────────────────────────────────────────────────────────────────

def check_tier_monotonicity(df: pd.DataFrame, model: HedonicOLS) -> pd.DataFrame:
    """
    Utilise les prix PREDITS par `model` (a partir des seules
    caracteristiques techniques + marque -- ni gamme_prix ni cluster_id ne
    sont jamais une entree du modele, §garde-fou) pour verifier que le
    prix predit croit bien Economique -> Milieu de gamme -> Premium AU
    SEIN de chaque marque. Ce test n'a de sens QUE parce que `model` ne
    connait pas deja la gamme (sans quoi il "predirait" trivialement une
    monotonie parfaite, cf. correctif 2026-07-21 -- avant que cluster_id
    ne soit retire des regresseurs, ce test etait quasi tautologique).
    Une marque non-monotone revele que son etiquette "Premium" (fondee
    UNIQUEMENT sur le prix observe, cf. compute_price_tiers) n'est PAS
    corroboree par les caracteristiques une fois celles-ci controlees --
    une majoration de marque pure, pas une vraie difference technique.

    IMPORTANT : df doit etre le MEME DataFrame (meme index) que celui
    utilise pour construire X/y de `model` (l'alignement des valeurs
    predites se fait par index pandas, pas positionnellement).

    Returns: DataFrame [marque, monotone, Économique, Milieu de gamme,
        Premium] (colonnes de gammes absentes omises pour une marque a
        1-2 gammes), triee marques non-monotones en premier.
    """
    model._check_fitted()
    predicted_price = np.exp(model.result_.fittedvalues)

    check_df = df[["marque", "gamme_prix"]].copy()
    check_df["predicted_price"] = predicted_price

    rows = []
    for brand, g in check_df.groupby("marque", observed=True):
        means = g.groupby("gamme_prix", observed=True)["predicted_price"].mean()
        order = [t for t in GAMME_ORDER_3 if t in means.index]
        if len(order) < 2:
            continue
        ordered_means = means.reindex(order)
        row = {"marque": brand, "monotone": bool(ordered_means.is_monotonic_increasing)}
        row.update(ordered_means.to_dict())
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["monotone", "marque"]).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 10. DIAGNOSTICS -- VIF, test de White, distance de Cook, QQ-plot
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostics(model: HedonicOLS) -> dict:
    """
    Bloc de diagnostics post-estimation :
        - VIF (variance_inflation_factor) : multicolinearite entre
          regresseurs (constante exclue, son VIF n'est pas interpretable).
        - Test de White (het_white) : heteroscedasticite -- les
          erreurs-types robustes HC3 sont deja utilisees dans HedonicOLS
          quel que soit le resultat de ce test ; il sert a QUANTIFIER
          l'ampleur de l'heteroscedasticite, pas a decider s'il faut des
          SE robustes (deja le cas partout).
        - Distance de Cook (OLSInfluence) : points influents, seuil usuel
          4/n.
        - QQ-plot des residus (normalite) -- Figure matplotlib RETOURNEE,
          jamais affichee directement (pas d'effet de bord, cf. contrainte
          du package) ; a l'appelant de faire .show() ou .savefig().

    Returns: dict {vif, white_test, cooks_distance, n_influential_points,
        n_singular_leverage_points, cooks_threshold, qq_fig}.
    """
    model._check_fitted()
    resid = model.result_.resid
    exog = model.exog_

    vif_cols = [c for c in exog.columns if c != "const"]
    exog_values = exog[["const"] + vif_cols].to_numpy()
    vif_rows = [
        {"feature": col, "VIF": variance_inflation_factor(exog_values, i)}
        for i, col in enumerate(vif_cols, start=1)
    ]
    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)

    lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(resid, exog.to_numpy())
    white_test = {
        "LM_stat": float(lm_stat), "LM_p_value": float(lm_pvalue),
        "F_stat": float(f_stat), "F_p_value": float(f_pvalue),
        "conclusion": (
            "Heteroscedasticite detectee (p<0.05) -- corrobore l'usage de SE robustes HC3."
            if lm_pvalue < 0.05 else
            "Pas d'heteroscedasticite significative detectee."
        ),
    }

    cooks_d, _ = OLSInfluence(model.result_).cooks_distance
    threshold = 4 / model.nobs
    cooks_df = pd.DataFrame({"index": exog.index, "cooks_d": cooks_d})
    cooks_df["influential"] = cooks_df["cooks_d"] > threshold
    # La distance de Cook est NaN quand le levier vaut exactement 1 (un
    # point qui EST son propre parametre -- typiquement une modalite de
    # cluster_id/categorielle portee par une seule observation, frequente
    # avec des effets fixes tres granulaires sur un petit echantillon).
    # C'est le cas le PLUS influent possible, jamais un cas sans probleme
    # -- un simple NaN > seuil vaudrait silencieusement False, masquant
    # exactement le point qu'il faudrait signaler.
    cooks_df.loc[cooks_df["cooks_d"].isna(), "influential"] = True

    qq_fig = qqplot(resid, line="45", fit=True)
    qq_fig.suptitle("QQ-plot des résidus (normalité)")

    return {
        "vif": vif_df,
        "white_test": white_test,
        "cooks_distance": cooks_df,
        "n_influential_points": int(cooks_df["influential"].sum()),
        "n_singular_leverage_points": int(cooks_df["cooks_d"].isna().sum()),
        "cooks_threshold": threshold,
        "qq_fig": qq_fig,
    }


if __name__ == "__main__":
    # ── Exemple jouet minimal (sanity check), meme esprit que ridge_model.py
    # et rf_model.py -- ne depend pas des vraies donnees du projet. Pour un
    # exemple complet sur des donnees reelles (une categorie de bout en
    # bout), voir src/models/demo_hedonic.py. ─────────────────────────────
    rng = np.random.default_rng(RANDOM_STATE)
    n = 120
    marque = rng.choice(["MSI", "DELL", "HP"], size=n)
    ram_go = rng.choice([8, 16, 32], size=n).astype(float)
    stockage_go = rng.choice([256, 512, 1024], size=n).astype(float)
    cluster_id = rng.choice(["MSI::Économique::c0", "DELL::Économique::c0", "HP::Économique::c0"], size=n)

    log_prix = (
        6.5 + 0.02 * ram_go + 0.0005 * stockage_go
        + np.where(marque == "MSI", 0.1, 0.0) + rng.normal(0, 0.05, size=n)
    )

    toy_df = pd.DataFrame({
        "marque": marque, "ram_go": ram_go, "stockage_go": stockage_go,
        # cluster_id present dans le DataFrame source (comme dans un vrai
        # df_clustered) mais PAS passe a build_design_matrix ci-dessous --
        # seule "marque" sert d'effet fixe categoriel (cf. correctif
        # 2026-07-21). Garde ce champ pour la demonstration du garde-fou
        # plus bas.
        "cluster_id": cluster_id, "prix_tnd": np.exp(log_prix),
    })

    X_toy, y_toy = build_design_matrix(toy_df, ["ram_go", "stockage_go"], ["marque"])
    toy_model = HedonicOLS().fit(X_toy, y_toy, continuous_cols=["ram_go", "stockage_go"])
    print("R2 ajuste :", toy_model.rsquared_adj)
    print("\nCoefficients :")
    print(toy_model.get_coefficients())

    # Verifie que le garde-fou de circularite se declenche bien, pour
    # gamme_prix ET pour cluster_id (qui l'encode -- correctif 2026-07-21).
    for label, cat_features, extra_cols in [
        ("gamme_prix", ["marque"], {"gamme_prix": "Premium"}),
        ("cluster_id", ["cluster_id"], {}),
    ]:
        try:
            candidate_continuous = ["ram_go", "gamme_prix"] if label == "gamme_prix" else ["ram_go", "stockage_go"]
            build_design_matrix(toy_df.assign(**extra_cols), candidate_continuous, cat_features)
            raise AssertionError(f"CircularityError attendue mais non levee pour {label} !")
        except CircularityError as exc:
            print(f"\nGarde-fou de circularite OK ({label}) :", exc)
