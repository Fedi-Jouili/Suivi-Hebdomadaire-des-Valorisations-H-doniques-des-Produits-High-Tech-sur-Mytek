# -*- coding: utf-8 -*-
"""
Page 2 -- Modeles hedoniques & clustering, par categorie. Explique la
methodologie, affiche les metriques deja calculees (JAMAIS de reentrainement
ici, cf. src/models/save_artifacts.py) et les caracteristiques des clusters.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from dash_iconify import DashIconify

from src.dashboard.components import category_selector, kpi_card, kpi_row, provenance_strip, section_header
from src.dashboard.data_loader import (
    ArtifactsMissingError,
    artifacts_available,
    available_weeks,
    category_label,
    load_cluster_models_summary,
    load_cluster_stability_n2,
    load_coefficients,
    load_k_selection_justification,
    load_metrics,
    load_model_agreement,
    load_pooled_labeled,
    load_rf_importances,
    load_rf_permutation_importances,
    load_ridge_coefficients,
    load_unit_summary,
)
from src.dashboard.format_utils import fmt_number, fmt_pct_effect, fmt_price
from src.dashboard.theme import CATEGORY_COLORS, GRAPH_CONFIG, TEXT_MUT
from src.utils.cluster_names import n1_cluster_name

dash.register_page(__name__, path="/modeles", name="Modèles & clustering")

_METHOD_TEXT = dmc.List(
    [
        dmc.ListItem(
            dmc.Text([
                dmc.Text("Décomposition hédonique (Ridge / OLS).", span=True, fw=600),
                " Le prix est modélisé en ", dmc.Text("log(prix)", span=True, ff="monospace"),
                " en fonction des caractéristiques techniques (RAM, stockage, écran…) et de la marque — "
                "théorie de Lancaster (1966) / Rosen (1974) : le prix observé est la somme des prix implicites "
                "de chaque caractéristique. Un coefficient log-linéaire s'interprète comme un ",
                dmc.Text("effet en pourcentage du prix", span=True, fw=600),
                ", jamais un montant en dinars — cf. formule ci-dessous.",
            ], size="sm"),
        ),
        dmc.ListItem(
            dmc.Text([
                dmc.Text("Random Forest.", span=True, fw=600),
                " Alternative non linéaire, capture des effets de seuil/interactions qu'une régression "
                "log-linéaire ne peut représenter que via des termes explicites. Le graphique d'importance "
                "aide à voir quelles variables pèsent le plus, mais il doit être lu avec prudence.",
            ], size="sm"),
        ),
        dmc.ListItem(
            dmc.Text([
                dmc.Text("Clustering technique (N1).", span=True, fw=600),
                " K-Means sur les seules caractéristiques techniques (jamais le prix ni la marque) — révèle des "
                "profils naturels, comparés a posteriori au prix pour valider (ou non) l'hypothèse hédonique.",
            ], size="sm"),
        ),
        dmc.ListItem(
            dmc.Text([
                dmc.Text("Segmentation marque × gamme (N2).", span=True, fw=600),
                " Chaque marque est d'abord partagée en gammes de prix (Économique / Milieu / Premium, propres "
                "à la marque), puis un clustering technique est tenté au sein de chaque couple — une lecture "
                "commerciale directement actionnable, complémentaire à N1.",
            ], size="sm"),
        ),
    ],
    size="sm", spacing="xs",
)


def layout():
    return dmc.Stack(
        [
            section_header("Modèles & clustering", "Décomposition hédonique du prix et segmentation, par catégorie."),
            category_selector("models-category"),
            dmc.LoadingOverlay(
                visible=False, id="models-loading", zIndex=10,
                loaderProps={"type": "dots", "color": "blue"},
            ),
            html.Div(id="models-content"),
        ],
        pos="relative",
        className="page-fade",
    )


def _missing_artifacts_alert(category: str):
    return dmc.Alert(
        [
            dmc.Text(f"Aucun artefact entraîné trouvé pour « {category_label(category)} »."),
            dmc.Code("python -m src.models.save_artifacts --category " + category, block=True, mt="xs"),
        ],
        title="Modèles non disponibles",
        color="yellow",
        icon=DashIconify(icon="tabler:alert-triangle"),
    )


def _r2_comparison_chart(metrics: dict, color: str):
    rows = [
        {"modèle": "Hedonic OLS", "R² (test, log)": metrics["hedonic_ols"]["r2_log"]},
        {"modèle": "Ridge", "R² (test, log)": metrics["ridge"]["r2_log"]},
        {"modèle": "Random Forest", "R² (test, log)": metrics["random_forest"]["r2_log"]},
    ]
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="modèle", y="R² (test, log)", text="R² (test, log)", color_discrete_sequence=[color])
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        title="Qualité de prédiction hors-échantillon (R², échelle log)",
        yaxis=dict(range=[0, 1.05], title="R² (test)"), xaxis_title="", height=340,
    )
    return fig


def _coefficients_chart(coefs: pd.DataFrame, color: str, top_n: int = 15):
    """Coefficients OLS en effet %, avec barre d'erreur approximee par la
    methode delta (meme formule que Evolution_Temporelle_Marche_Mytek.ipynb
    ::plot_coefficient_stability) -- pas d'inference pour Ridge/RF."""
    df = coefs[coefs["feature"] != "const"].copy()
    df = df.reindex(df["pct_effect"].abs().sort_values(ascending=False).index).head(top_n)
    df["se_pct"] = np.exp(df["coefficient"]) * 100 * df["std_err"]
    df = df.sort_values("pct_effect")
    fig = px.bar(
        df, x="pct_effect", y="feature", orientation="h", error_x="se_pct",
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        title=f"Coefficients hédoniques (OLS) — effet sur le prix, top {top_n}",
        xaxis_title="Effet sur le prix (%)", yaxis_title="", height=max(340, 28 * len(df)),
        yaxis=dict(automargin=True), margin=dict(l=10),
    )
    fig.add_vline(x=0, line_color=TEXT_MUT, line_width=1)
    return fig


def _rf_importance_chart(imp: pd.DataFrame, color: str, top_n: int = 15):
    df = imp.head(top_n).sort_values("importance")
    fig = px.bar(df, x="importance", y="feature", orientation="h", color_discrete_sequence=[color])
    fig.update_layout(
        title=f"Variables les plus utiles selon la Random Forest, top {top_n}",
        xaxis_title="Importance relative", yaxis_title="", height=max(340, 28 * len(df)),
        yaxis=dict(automargin=True), margin=dict(l=10),
    )
    return fig


def _rf_permutation_importance_chart(imp: pd.DataFrame, color: str, top_n: int = 15):
    """Meme lecture que _rf_importance_chart, mais calculee sur le TEST par
    permutation -- utile pour vérifier si une variable apporte vraiment
    quelque chose au modèle, barre d'erreur = ecart-type sur les répliques
    de permutation (importance_std)."""
    df = imp.head(top_n).sort_values("importance_mean")
    fig = px.bar(
        df, x="importance_mean", y="feature", orientation="h", error_x="importance_std",
        color_discrete_sequence=[color],
    )
    fig.update_layout(
        title=f"Vérification des variables par permutation, top {top_n}",
        xaxis_title="Impact sur la qualité du modèle", yaxis_title="",
        height=max(340, 28 * len(df)), yaxis=dict(automargin=True), margin=dict(l=10),
    )
    fig.add_vline(x=0, line_color=TEXT_MUT, line_width=1)
    return fig


def _model_agreement_alert(agreement_resume: dict):
    """Resume textuel de compute_model_agreement (save_artifacts.py) --
    ajoute le 2026-07-28 : jusqu'ici rien ne verifiait que les 3 modeles
    racontent une histoire coherente."""
    pct_signs = agreement_resume.get("pct_signes_ols_ridge_accordent")
    rho = agreement_resume.get("rf_ols_spearman_rho")
    p_value = agreement_resume.get("rf_ols_spearman_p_value")
    n = agreement_resume.get("n_features_comparees")

    if pct_signs is None or rho is None:
        return dmc.Alert(
            "Comparaison entre modèles non disponible. Relancer la génération des résultats.",
            color="gray", variant="light", icon=DashIconify(icon="tabler:info-circle"), mb="sm",
        )

    signs_color = "green" if pct_signs >= 90 else ("orange" if pct_signs >= 70 else "red")
    rho_significant = p_value is not None and p_value < 0.05
    rho_color = "green" if (rho_significant and rho > 0) else ("red" if (rho_significant and rho < 0) else "gray")

    return dmc.Alert(
        dmc.Text([
            f"OLS et Ridge donnent souvent le même sens pour ",
            dmc.Text(f"{pct_signs:.0f} %", span=True, fw=700, c=signs_color),
            f" des {n} caractéristiques communes. Comparaison entre l'importance de la Random Forest et la taille "
            f"des coefficients OLS : ",
            dmc.Text(f"ρ = {rho:.2f}", span=True, fw=700, c=rho_color),
            f" ({'significatif' if rho_significant else 'non significatif'}, p = {p_value:.3f})" if p_value is not None else "",
            ". Si ce lien est faible, cela veut juste dire que les modèles ne mettent pas l'accent sur les "
            "mêmes variables.",
        ], size="sm"),
        color="blue", variant="light", icon=DashIconify(icon="tabler:git-compare"), mb="sm",
    )


def _n1_cluster_profile(category: str) -> pd.DataFrame:
    df = load_pooled_labeled(category)
    metrics = load_metrics(category)
    numeric_cols = [c for c in metrics["continuous_features"] if c in df.columns]
    profile = df.groupby("cluster_direct").agg(
        n=("prix_tnd", "count"), prix_median=("prix_tnd", "median"), **{c: (c, "mean") for c in numeric_cols}
    ).reset_index()
    profile["cluster_name"] = profile["cluster_direct"].map(lambda v: n1_cluster_name(category, v))
    profile = profile.sort_values("n", ascending=False)
    for c in numeric_cols:
        profile[c] = profile[c].round(1)
    profile["prix_median"] = profile["prix_median"].round(0)
    return profile


def _cluster_models_summary_grid(summary: pd.DataFrame) -> dag.AgGrid | dmc.Alert:
    """Table d'audit (cf. save_artifacts.fit_models_per_segment) -- une
    ligne par (segment, famille), colonne retenu_pour_prediction TOUJOURS
    lue plutot que ajuste seule (bat le modele categorie sur son propre
    test, pas seulement "estimable", cf. sa docstring)."""
    if summary.empty:
        return dmc.Alert(
            "Modèles par cluster non disponibles pour cette catégorie -- exécuter "
            "`python -m src.models.save_artifacts` (version 2026-08-01 ou plus récente).",
            color="gray", variant="light", icon=DashIconify(icon="tabler:info-circle"),
        )
    display = summary.copy()
    display["famille"] = display["famille"].map({
        "hedonic_ols": "Hedonic OLS", "ridge": "Ridge", "random_forest": "Random Forest",
    })
    display["statut"] = display.apply(
        lambda r: "Retenu (bat le modèle catégorie)" if r["retenu_pour_prediction"]
        else ("Ajusté mais pas meilleur" if r["ajuste"] else "Écarté"), axis=1,
    )
    for c in ("r2_test", "r2_test_categorie"):
        display[c] = display[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
    display["raison_rejet"] = display["raison_rejet"].fillna("—")

    return dag.AgGrid(
        className="ag-theme-quartz-dark",
        rowData=display.to_dict("records"),
        columnDefs=[
            {"field": "segment", "headerName": "Segment", "flex": 2},
            {"field": "famille", "headerName": "Famille", "flex": 2},
            {"field": "n_lignes", "headerName": "n (train)", "type": "rightAligned", "flex": 1},
            {
                "field": "statut", "headerName": "Statut", "flex": 2,
                "cellClassRules": {"ag-status-positive": "value.indexOf('Retenu') === 0"},
            },
            {"field": "r2_test", "headerName": "R² test (cluster)", "type": "rightAligned", "flex": 1},
            {"field": "r2_test_categorie", "headerName": "R² test (catégorie)", "type": "rightAligned", "flex": 1},
            {"field": "raison_rejet", "headerName": "Raison si écarté", "flex": 3},
        ],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        style={"height": "420px"}, columnSize="sizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


def _n2_summary_with_diagnostics(n2_summary: pd.DataFrame, stability: pd.DataFrame,
                                  k_justification: pd.DataFrame) -> pd.DataFrame:
    """Enrichit le tableau des unites N2 (marque x gamme) avec 2 diagnostics
    de rigueur ajoutes le 2026-07-27, jamais disponibles auparavant :
      - ari_moyen/ari_ecart_type (stabilite bootstrap du clustering, cf.
        cluster_stability_n2) -- un k "optimal" au sens silhouette peut
        neanmoins etre instable si l'unite est petite ;
      - silhouette_retenue (cf. k_selection_justification) -- la silhouette
        du k EFFECTIVEMENT choisi, pour juger sa qualite dans l'absolu (pas
        seulement relative aux autres k, deja arbitree en amont).
    LEFT JOIN : une unite sans structure retenue (k=1) n'a simplement aucune
    valeur a merger (NaN), jamais une erreur."""
    merged = n2_summary.merge(
        stability[["marque", "gamme", "ari_moyen", "ari_ecart_type"]],
        left_on=["marque", "gamme_prix"], right_on=["marque", "gamme"], how="left",
    ).drop(columns=["gamme"], errors="ignore")

    retenus = k_justification[
        (k_justification.get("approche") == "N2_marque_gamme") & (k_justification.get("k_retenu") == True)  # noqa: E712
    ] if not k_justification.empty else k_justification
    if not retenus.empty:
        merged = merged.merge(
            retenus[["marque", "gamme", "silhouette"]].rename(columns={"silhouette": "silhouette_retenue"}),
            left_on=["marque", "gamme_prix"], right_on=["marque", "gamme"], how="left",
        ).drop(columns=["gamme"], errors="ignore")
    else:
        merged["silhouette_retenue"] = np.nan

    # Formate en chaine ("—" si absent) AVANT le passage en dict pour la
    # grille -- un NaN brut serialise en JSON invalide (litteral `NaN`,
    # que JSON.parse cote navigateur rejette), meme convention deja
    # utilisee pour ecart_residuel_ic_bas (cf. evolution.py::_notable_cases_grid).
    merged["ari_moyen_num"] = merged["ari_moyen"]
    merged["ari_moyen"] = merged["ari_moyen"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    merged["silhouette_retenue"] = merged["silhouette_retenue"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    return merged


@callback(
    Output("models-content", "children"),
    Output("models-loading", "visible"),
    Input("models-category", "value"),
)
def render_models(category):
    weeks = available_weeks()
    if not artifacts_available(category):
        return _missing_artifacts_alert(category), False

    try:
        metrics = load_metrics(category)
        coefs = load_coefficients(category)
        ridge_coefs = load_ridge_coefficients(category)
        rf_imp = load_rf_importances(category)
        unit_summary = load_unit_summary(category)
    except ArtifactsMissingError:
        return _missing_artifacts_alert(category), False

    # Rapports de rigueur ajoutes le 2026-07-27 (stabilite bootstrap + choix
    # de k) -- optionnels : absents tant que `python -m src.models.
    # weekly_report` n'a pas ete rejoue apres cette mise a jour, jamais un
    # blocage de la page (meme logique de resilience que les autres pages).
    try:
        stability = load_cluster_stability_n2(category)
    except ArtifactsMissingError:
        stability = pd.DataFrame(columns=["marque", "gamme", "ari_moyen", "ari_ecart_type"])
    try:
        k_justification = load_k_selection_justification(category)
    except ArtifactsMissingError:
        k_justification = pd.DataFrame(columns=["approche", "marque", "gamme", "silhouette", "k_retenu"])

    # Accord entre modeles + importance par permutation, ajoutes le
    # 2026-07-28 (audit methodologique) -- meme logique de resilience :
    # optionnels tant que save_artifacts.py n'a pas ete rejoue.
    try:
        rf_perm_imp = load_rf_permutation_importances(category)
    except ArtifactsMissingError:
        rf_perm_imp = pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
    try:
        agreement = load_model_agreement(category)
    except ArtifactsMissingError:
        agreement = pd.DataFrame(columns=["feature", "ols_coefficient", "ridge_coefficient",
                                           "ols_ridge_signes_accordent", "rf_importance"])

    # Modeles PAR CLUSTER (N1/N2), ajoutes le 2026-08-01 -- DataFrame vide
    # (jamais une erreur) tant que save_artifacts.py n'a pas ete rejoue avec
    # ce correctif, meme logique de resilience que stability/k_justification.
    n1_models_summary = load_cluster_models_summary(category, "clusters_n1")
    n2_models_summary = load_cluster_models_summary(category, "clusters_n2")

    color = CATEGORY_COLORS.get(category, "#2a78d6")

    kpis = kpi_row([
        kpi_card(
            "Adj-R² (OLS, train)", f"{metrics['hedonic_ols']['adj_r2_train']:.3f}", "tabler:chart-line", color=color,
            raw_value=metrics["hedonic_ols"]["adj_r2_train"], decimals=3,
        ),
        kpi_card(
            "R² Ridge (test)", f"{metrics['ridge']['r2_log']:.3f}", "tabler:chart-line", color=color,
            raw_value=metrics["ridge"]["r2_log"], decimals=3,
        ),
        kpi_card(
            "R² Random Forest (test)", f"{metrics['random_forest']['r2_log']:.3f}", "tabler:chart-line", color=color,
            raw_value=metrics["random_forest"]["r2_log"], decimals=3,
        ),
        kpi_card(
            "RMSE Ridge (test)",
            fmt_price(metrics["ridge"]["rmse_tnd"]),
            "tabler:ruler-2", color=color, note="rétro-transformé, sans correction de biais",
            raw_value=metrics["ridge"]["rmse_tnd"], suffix=" TND",
        ),
    ])

    metrics_note = dmc.Alert(
        [
            dmc.Text(
                "Toutes les métriques ci-dessus sont calculées HORS-ÉCHANTILLON (jeu de test jamais vu à "
                "l'entraînement). Le R² est en échelle log(prix) ; le RMSE/MAE en TND est obtenu par simple "
                "exp() des prédictions log, SANS correction de biais de retransformation (Duan/Miller) — "
                "sous-estime légèrement l'erreur moyenne réelle, mais la comparaison relative entre modèles "
                "reste valide.",
                size="xs",
            ),
        ],
        color="blue", variant="light", icon=DashIconify(icon="tabler:info-circle"), mb="md",
    )

    circularity_note = dmc.Alert(
        dmc.Text(
            "Les 3 modèles utilisent « marque » comme effet fixe, jamais une variable dérivée du prix "
            "(gamme de prix, segment marque × gamme) — un garde-fou de circularité explicite "
            "(CircularityError) empêche toute fuite de la cible dans les régresseurs, cf. "
            "src/models/hedonic_model.py.",
            size="xs",
        ),
        color="gray", variant="light", icon=DashIconify(icon="tabler:shield-check"), mb="md",
    )

    hedonic_tab = dmc.Stack([
        metrics_note,
        circularity_note,
        dcc.Graph(figure=_r2_comparison_chart(metrics, color), config=GRAPH_CONFIG),
        dmc.Grid([
            dmc.GridCol(dcc.Graph(figure=_coefficients_chart(coefs, color), config=GRAPH_CONFIG), span={"base": 12, "lg": 6}),
            dmc.GridCol(dcc.Graph(figure=_rf_importance_chart(rf_imp, color), config=GRAPH_CONFIG), span={"base": 12, "lg": 6}),
        ]),
        dmc.Alert(
            metrics["random_forest"]["importance_note"], color="orange", variant="light",
            icon=DashIconify(icon="tabler:alert-triangle"), mt="sm",
        ),
        section_header("Accord entre modèles", order=5,
                        subtitle="OLS et Ridge s'accordent-ils sur le sens de l'effet ? Random Forest confirme-t-il, en "
                                  "rang, les mêmes variables que les modèles linéaires ?"),
        _model_agreement_alert(metrics.get("model_agreement", {})),
        dcc.Graph(figure=_rf_permutation_importance_chart(rf_perm_imp, color), config=GRAPH_CONFIG)
        if not rf_perm_imp.empty else dmc.Alert(
            "Importance par permutation non disponible -- exécuter `python -m src.models.save_artifacts`.",
            color="gray", variant="light", icon=DashIconify(icon="tabler:info-circle"),
        ),
        section_header("Coefficients Ridge (échelle d'origine)", order=5,
                        subtitle="Semi-élasticités re-exprimées sur l'échelle des variables, sans inférence (p-values non disponibles pour Ridge)."),
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            rowData=ridge_coefs.head(20).round(4).to_dict("records"),
            columnDefs=[
                {"field": "feature", "headerName": "Caractéristique", "flex": 2},
                {"field": "coefficient", "headerName": "Coefficient", "type": "rightAligned", "flex": 1},
            ],
            defaultColDef={"sortable": True, "resizable": True},
            style={"height": "320px"}, columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ),
    ])

    n1_profile = _n1_cluster_profile(category)
    n2_summary = unit_summary.copy()
    n2_summary["outcome"] = n2_summary["outcome"].map({
        "clustered": "Clusterisé", "no_structure": "Sans structure", "too_small": "Trop petit",
    })
    n2_summary = _n2_summary_with_diagnostics(n2_summary, stability, k_justification)
    ari_valides = n2_summary["ari_moyen_num"].dropna()

    stability_note = dmc.Alert(
        dmc.Text([
            dmc.Text("ARI stabilité", span=True, fw=600),
            " (Adjusted Rand Index, 100 réplications bootstrap, cf. cluster_stability_n2) : le clustering "
            "N2 tiendrait-il avec un échantillon légèrement différent de produits ? Proche de 1 = partition "
            "robuste ; proche de 0 = frontières largement dues au hasard de l'échantillon, à lire avec prudence. "
            "« Silhouette (k retenu) » indique la cohésion du k effectivement choisi — le détail complet de "
            "tous les k comparés (retenus ou non) est dans reports/justification_k_clustering.csv.",
        ], size="xs"),
        color="blue", variant="light", icon=DashIconify(icon="tabler:info-circle"), mb="sm",
    )
    mixed_distance_note = dmc.Alert(
        dmc.Text(
            "Limite connue du K-Means (N1 et N2) : la distance euclidienne mélange variables continues "
            "standardisées et variables catégorielles one-hot — le poids relatif de chaque groupe dans la "
            "distance dépend en partie du nombre de modalités encodées (artefact du one-hot encoding, pas "
            "une pondération délibérée). Alternative plus rigoureuse non implémentée ici : K-Prototypes / "
            "distance de Gower.",
            size="xs",
        ),
        color="orange", variant="light", icon=DashIconify(icon="tabler:alert-triangle"), mb="md",
    )

    clustering_tab = dmc.Stack([
        kpi_row([
            kpi_card(
                "Segments techniques (N1)", fmt_number(n1_profile["cluster_direct"].nunique()), "tabler:chart-dots",
                color=color, raw_value=int(n1_profile["cluster_direct"].nunique()),
            ),
            kpi_card(
                "Unités marque×gamme (N2)", fmt_number(len(n2_summary)), "tabler:layout-grid", color=color,
                raw_value=len(n2_summary),
            ),
            kpi_card(
                "Unités N2 clusterisées",
                fmt_number((n2_summary["outcome"] == "Clusterisé").sum()),
                "tabler:circle-check", color=color,
                raw_value=int((n2_summary["outcome"] == "Clusterisé").sum()),
            ),
            kpi_card(
                "ARI moyen (stabilité N2)",
                f"{ari_valides.mean():.2f}" if not ari_valides.empty else "n/d",
                "tabler:shield-check", color=color,
                note="100 réplications bootstrap — proche de 1 = robuste" if not ari_valides.empty else "rapport non disponible",
                raw_value=float(ari_valides.mean()) if not ari_valides.empty else None, decimals=2,
            ),
        ]),
        section_header("Clustering technique (N1) — profil par cluster", order=5,
                        subtitle="Construit sans prix ni marque ; prix médian affiché a posteriori, pour validation."),
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            rowData=n1_profile.to_dict("records"),
            columnDefs=[{"field": "cluster_name", "headerName": "Segment technique", "flex": 1}] +
                       [{"field": "n", "headerName": "n produits", "type": "rightAligned", "flex": 1}] +
                       [{"field": "prix_median", "headerName": "Prix médian (TND)", "type": "rightAligned", "flex": 1}] +
                       [{"field": c, "headerName": c.replace("_", " "), "type": "rightAligned", "flex": 1}
                        for c in n1_profile.columns if c not in ("cluster_direct", "cluster_name", "n", "prix_median")],
            defaultColDef={"sortable": True, "resizable": True},
            style={"height": "260px"}, columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ),
        section_header("Segmentation marque × gamme (N2) — unités", order=5,
                        subtitle="Une unité en dessous de l'effectif minimal reste un profil descriptif (jamais un clustering forcé)."),
        stability_note,
        mixed_distance_note,
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            rowData=n2_summary.to_dict("records"),
            columnDefs=[
                {"field": "marque", "headerName": "Marque", "flex": 2},
                {"field": "gamme_prix", "headerName": "Gamme", "flex": 2},
                {"field": "n", "headerName": "n produits", "type": "rightAligned", "flex": 1},
                {"field": "k", "headerName": "k retenu", "type": "rightAligned", "flex": 1},
                {
                    "field": "outcome", "headerName": "Résultat", "flex": 2,
                    "cellClassRules": {"ag-status-positive": "value == 'Clusterisé'"},
                },
                {
                    "field": "silhouette_retenue", "headerName": "Silhouette (k retenu)", "type": "rightAligned", "flex": 2,
                },
                {
                    "field": "ari_moyen", "headerName": "ARI stabilité (bootstrap)", "type": "rightAligned", "flex": 2,
                    "cellClassRules": {"ag-status-negative": "Number(value) < 0.5"},
                },
            ],
            defaultColDef={"sortable": True, "resizable": True, "filter": True},
            style={"height": "360px"}, columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ),
    ])

    n1_retenus = int(n1_models_summary["retenu_pour_prediction"].sum()) if not n1_models_summary.empty else 0
    n1_segments = int(n1_models_summary["segment"].nunique()) if not n1_models_summary.empty else 0
    n2_retenus_count = (
        n2_models_summary.loc[n2_models_summary["retenu_pour_prediction"], "segment"].nunique()
        if not n2_models_summary.empty else 0
    )
    n2_segments = int(n2_models_summary["segment"].nunique()) if not n2_models_summary.empty else 0

    cluster_models_tab = dmc.Stack([
        dmc.Alert(
            dmc.Text([
                "Décision du 2026-08-01 : en plus du modèle catégorie entière (onglet précédent), un OLS/Ridge/"
                "Random Forest est ajusté ", dmc.Text("par cluster", span=True, fw=600),
                " (N1 technique et N2 marque × gamme séparément) quand l'effectif le permet — ", dmc.Text("et", span=True, fw=600, fs="italic"),
                " seulement s'il bat démontrablement le modèle catégorie sur son propre test hors-échantillon "
                "(colonne « Statut » ci-dessous). Passer le seuil d'effectif ne suffit pas : mesuré empiriquement "
                "sur ce projet, plusieurs clusters ajustables produisaient un R² hors-échantillon fortement "
                "négatif (surapprentissage sur petit échantillon) — jamais utilisés pour une prédiction malgré "
                "un ajustement techniquement réussi.",
            ], size="sm"),
            color="blue", variant="light", icon=DashIconify(icon="tabler:info-circle"), mb="sm",
        ),
        kpi_row([
            kpi_card("Clusters N1 avec modèle retenu", f"{n1_retenus}/{n1_segments}" if n1_segments else "n/d",
                      "tabler:chart-dots", color=color),
            kpi_card("Clusters N2 avec modèle retenu", f"{n2_retenus_count}/{n2_segments}" if n2_segments else "n/d",
                      "tabler:layout-grid", color=color),
        ]),
        section_header("Clusters N2 (marque × gamme × profil technique)", order=5,
                        subtitle="Une ligne par (segment, famille) — le modèle catégorie reste utilisé pour toute ligne non « Retenu »."),
        _cluster_models_summary_grid(n2_models_summary),
        section_header("Clusters N1 (technique, toute la catégorie)", order=5),
        _cluster_models_summary_grid(n1_models_summary),
    ])

    content = dmc.Stack([
        provenance_strip(weeks, extra=f"catégorie : {category_label(category)} · modèles entraînés sur {metrics['n_pooled']} produits poolés"),
        dmc.Paper(_METHOD_TEXT, p="md", withBorder=True, mb="sm"),
        kpis,
        dmc.Tabs(
            [
                dmc.TabsList([
                    dmc.TabsTab("Décomposition hédonique", value="hedonic", leftSection=DashIconify(icon="tabler:chart-line")),
                    dmc.TabsTab("Clustering & segmentation", value="clustering", leftSection=DashIconify(icon="tabler:chart-dots-3")),
                    dmc.TabsTab("Modèles par cluster", value="cluster_models", leftSection=DashIconify(icon="tabler:apps")),
                ]),
                dmc.TabsPanel(hedonic_tab, value="hedonic", pt="md"),
                dmc.TabsPanel(clustering_tab, value="clustering", pt="md"),
                dmc.TabsPanel(cluster_models_tab, value="cluster_models", pt="md"),
            ],
            value="hedonic", mt="md",
        ),
    ])
    return content, False
