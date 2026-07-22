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
    load_coefficients,
    load_metrics,
    load_pooled_labeled,
    load_rf_importances,
    load_ridge_coefficients,
    load_unit_summary,
)
from src.dashboard.format_utils import fmt_number, fmt_pct_effect, fmt_price
from src.dashboard.theme import CATEGORY_COLORS, GRAPH_CONFIG, TEXT_MUT

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
                "log-linéaire ne peut représenter que via des termes explicites. Les importances de variables "
                "sont basées sur la réduction moyenne d'impureté (MDI) — biaisées en faveur des variables "
                "continues, cf. l'avertissement affiché plus bas.",
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
        title=f"Importance des variables (Random Forest, MDI), top {top_n}",
        xaxis_title="Importance (réduction moyenne d'impureté)", yaxis_title="", height=max(340, 28 * len(df)),
        yaxis=dict(automargin=True), margin=dict(l=10),
    )
    return fig


def _n1_cluster_profile(category: str) -> pd.DataFrame:
    df = load_pooled_labeled(category)
    metrics = load_metrics(category)
    numeric_cols = [c for c in metrics["continuous_features"] if c in df.columns]
    profile = df.groupby("cluster_direct").agg(
        n=("prix_tnd", "count"), prix_median=("prix_tnd", "median"), **{c: (c, "mean") for c in numeric_cols}
    ).reset_index()
    profile = profile.sort_values("n", ascending=False)
    for c in numeric_cols:
        profile[c] = profile[c].round(1)
    profile["prix_median"] = profile["prix_median"].round(0)
    return profile


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

    clustering_tab = dmc.Stack([
        kpi_row([
            kpi_card(
                "Clusters techniques (N1)", fmt_number(n1_profile["cluster_direct"].nunique()), "tabler:chart-dots",
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
        ]),
        section_header("Clustering technique (N1) — profil par cluster", order=5,
                        subtitle="Construit sans prix ni marque ; prix médian affiché a posteriori, pour validation."),
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            rowData=n1_profile.to_dict("records"),
            columnDefs=[{"field": "cluster_direct", "headerName": "Cluster", "flex": 1}] +
                       [{"field": "n", "headerName": "n produits", "type": "rightAligned", "flex": 1}] +
                       [{"field": "prix_median", "headerName": "Prix médian (TND)", "type": "rightAligned", "flex": 1}] +
                       [{"field": c, "headerName": c.replace("_", " "), "type": "rightAligned", "flex": 1}
                        for c in n1_profile.columns if c not in ("cluster_direct", "n", "prix_median")],
            defaultColDef={"sortable": True, "resizable": True},
            style={"height": "260px"}, columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ),
        section_header("Segmentation marque × gamme (N2) — unités", order=5,
                        subtitle="Une unité en dessous de l'effectif minimal reste un profil descriptif (jamais un clustering forcé)."),
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
            ],
            defaultColDef={"sortable": True, "resizable": True, "filter": True},
            style={"height": "360px"}, columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ),
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
                ]),
                dmc.TabsPanel(hedonic_tab, value="hedonic", pt="md"),
                dmc.TabsPanel(clustering_tab, value="clustering", pt="md"),
            ],
            value="hedonic", mt="md",
        ),
    ])
    return content, False
