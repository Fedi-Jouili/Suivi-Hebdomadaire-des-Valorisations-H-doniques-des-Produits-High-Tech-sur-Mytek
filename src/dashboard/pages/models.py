# -*- coding: utf-8 -*-
"""
Page 2 -- Modeles hedoniques & clustering, par categorie. Explique la
methodologie, affiche les metriques deja calculees (JAMAIS de reentrainement
ici, cf. src/models/save_artifacts.py) et les caracteristiques des clusters.
"""

import functools

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
    load_cluster_products,
    load_cluster_segment_detail,
    load_cluster_stability_n2,
    load_coefficients,
    load_k_selection_justification,
    load_marque_gamme_estimates,
    load_metrics,
    load_model_agreement,
    load_n1_cluster_estimates,
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
        style={"height": "420px"}, columnSize="responsiveSizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


_FAMILLE_LABELS = {"hedonic_ols": "Hedonic OLS", "ridge": "Ridge", "random_forest": "Random Forest"}
_STATUT_COLORS = {"Retenu": "green", "Ajusté mais pas meilleur": "orange", "Écarté": "red", "n/d": "gray"}


def _format_linear_equation(lhs: str, intercept: float | None, terms: list) -> str:
    """Equation log-lineaire lisible : signe + coefficient + variable, triee
    par |coefficient| decroissant (deja fait par l'appelant). Sans
    intercept (cas Ridge, pas d'ordonnee a l'origine apres standardisation),
    le premier terme garde son propre signe au lieu d'un "+" de tete
    artificiel."""
    pieces = [f"{intercept:.3f}"] if intercept is not None else []
    for name, coef in terms:
        op = "+" if coef >= 0 else "−"
        pieces.append(f"{op} {abs(coef):.3f}·{name}")
    if not pieces:
        return f"{lhs} = —"
    expr = " ".join(pieces)
    if intercept is None and expr.startswith("+ "):
        expr = expr[2:]
    return f"{lhs} = {expr}"


def _ols_equation_text(coefs: pd.DataFrame, max_terms: int = 8) -> tuple:
    body = coefs[coefs["feature"] != "const"].copy()
    const_rows = coefs.loc[coefs["feature"] == "const", "coefficient"]
    intercept = float(const_rows.iloc[0]) if not const_rows.empty else None
    body = body.reindex(body["coefficient"].abs().sort_values(ascending=False).index)
    shown = body.head(max_terms)
    n_hidden = max(len(body) - max_terms, 0)
    equation = _format_linear_equation("log(prix)", intercept, list(zip(shown["feature"], shown["coefficient"])))
    return equation, n_hidden


def _ridge_equation_text(coefs: pd.DataFrame, max_terms: int = 8) -> tuple:
    body = coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index)
    shown = body.head(max_terms)
    n_hidden = max(len(body) - max_terms, 0)
    equation = _format_linear_equation(
        "log(prix) (coefficients ré-exprimés, sans ordonnée à l'origine)", None,
        list(zip(shown["feature"], shown["coefficient"])),
    )
    return equation, n_hidden


def _rf_importance_text(imp: pd.DataFrame, max_terms: int = 5) -> str:
    top = imp.head(max_terms)
    parts = [f"{row.feature} ({row.importance * 100:.1f} %)" for row in top.itertuples()]
    return (
        "Pas de formule fermée (ensemble d'arbres) — variables les plus influentes (importance MDI) : "
        + ", ".join(parts) + "."
    )


def _format_ols_table(coefs: pd.DataFrame) -> list:
    df = coefs.copy()
    df["coefficient"] = df["coefficient"].round(4)
    df["p_value"] = df["p_value"].map(lambda v: "<0.001" if pd.notna(v) and v < 0.001 else (f"{v:.3f}" if pd.notna(v) else "—"))
    df["pct_effect"] = df["pct_effect"].map(lambda v: f"{v:.2f}" if pd.notna(v) else "—")
    return df[["feature", "coefficient", "p_value", "pct_effect"]].to_dict("records")


def _segment_status_info(summary: pd.DataFrame, segment: str, famille: str) -> dict:
    """Statut + diagnostics (R² cluster vs R² catégorie, raison si écarté)
    d'UNE famille pour UN segment -- lu depuis <subdir>_summary.csv (cf.
    save_artifacts.fit_models_per_segment), jamais recalculé ici."""
    row = summary[(summary["segment"].astype(str) == str(segment)) & (summary["famille"] == famille)]
    if row.empty:
        return {"statut": "n/d", "raison": None, "r2_test": None, "r2_test_categorie": None}
    r = row.iloc[0]
    if bool(r["retenu_pour_prediction"]):
        statut = "Retenu"
    elif bool(r["ajuste"]):
        statut = "Ajusté mais pas meilleur"
    else:
        statut = "Écarté"
    raison = r.get("raison_rejet")
    r2_test = r.get("r2_test")
    r2_test_categorie = r.get("r2_test_categorie")
    return {
        "statut": statut,
        "raison": raison if pd.notna(raison) else None,
        "r2_test": float(r2_test) if pd.notna(r2_test) else None,
        "r2_test_categorie": float(r2_test_categorie) if pd.notna(r2_test_categorie) else None,
    }


def _famille_formula_card(famille: str, status: dict, coefs: pd.DataFrame | None) -> dmc.Paper:
    """Carte "formule" d'UNE famille de modele pour UN cluster -- equation
    lisible (OLS/Ridge) ou resume d'importance (RF, pas de forme close),
    plus le tableau COMPLET des coefficients/importances (jamais tronque
    silencieusement, cf. `n_hidden` dans l'equation) et le R² qui justifie
    le statut retenu/ecarte (transparence academique demandee explicitement
    par l'utilisateur -- montrer le travail, pas seulement le resultat)."""
    label = _FAMILLE_LABELS[famille]
    badge = dmc.Badge(status["statut"], color=_STATUT_COLORS.get(status["statut"], "gray"), variant="light", size="sm")

    extra = []
    if status["r2_test"] is not None:
        r2_text = f"R² test (ce cluster) = {status['r2_test']:.3f}"
        if status["r2_test_categorie"] is not None:
            r2_text += f"  vs  R² test (modèle catégorie) = {status['r2_test_categorie']:.3f}"
        extra.append(dmc.Text(r2_text, size="xs", c="dimmed"))
    if status["raison"]:
        extra.append(dmc.Text(f"Non ajusté : {status['raison']}", size="xs", c="orange"))

    if coefs is None or coefs.empty:
        return dmc.Paper(
            dmc.Stack([
                dmc.Group([dmc.Text(label, fw=600), badge], justify="space-between"),
                dmc.Text("Modèle non ajusté sur ce cluster (effectif insuffisant pour cette famille).",
                          size="sm", c="dimmed"),
                *extra,
            ], gap=4),
            p="sm", withBorder=True,
        )

    if famille == "hedonic_ols":
        equation, n_hidden = _ols_equation_text(coefs)
        table_rows = _format_ols_table(coefs)
        table_cols = [
            {"field": "feature", "headerName": "Variable", "flex": 2},
            {"field": "coefficient", "headerName": "Coefficient (log)", "type": "rightAligned", "flex": 1},
            {"field": "p_value", "headerName": "p-value", "type": "rightAligned", "flex": 1},
            {"field": "pct_effect", "headerName": "Effet (%)", "type": "rightAligned", "flex": 1},
        ]
    elif famille == "ridge":
        equation, n_hidden = _ridge_equation_text(coefs)
        table_rows = coefs.round(4).to_dict("records")
        table_cols = [
            {"field": "feature", "headerName": "Variable", "flex": 2},
            {"field": "coefficient", "headerName": "Coefficient (log)", "type": "rightAligned", "flex": 1},
        ]
    else:
        equation = _rf_importance_text(coefs)
        n_hidden = 0
        table_rows = coefs.round(4).to_dict("records")
        table_cols = [
            {"field": "feature", "headerName": "Variable", "flex": 2},
            {"field": "importance", "headerName": "Importance (MDI)", "type": "rightAligned", "flex": 1},
        ]

    if n_hidden:
        extra.insert(0, dmc.Text(f"+ {n_hidden} autre(s) terme(s) dans le tableau ci-dessous.", size="xs", c="dimmed"))

    return dmc.Paper(
        dmc.Stack([
            dmc.Group([dmc.Text(label, fw=600), badge], justify="space-between"),
            dmc.Code(equation, block=True),
            *extra,
            dag.AgGrid(
                className="ag-theme-quartz-dark",
                rowData=table_rows,
                columnDefs=table_cols,
                defaultColDef={"sortable": True, "resizable": True},
                style={"height": f"{min(260, 50 + 28 * len(table_rows))}px"}, columnSize="responsiveSizeToFit",
                dashGridOptions={"theme": "legacy"},
            ),
        ], gap=6),
        p="sm", withBorder=True,
    )


@functools.lru_cache(maxsize=None)
def _cluster_weekly_comparison(category: str, subdir: str, segment: str) -> pd.DataFrame:
    """Prix reel (moyenne geometrique) vs estime par semaine, pour UN
    cluster -- cf. n1_cluster_estimations_hebdo.csv (N1) / marque_gamme_
    estimations_hebdo.csv (N2, ou "cluster" n'est que le sous-cluster "c0"/
    "c1", jamais la cle complete "marque::gamme::sous-cluster").

    marque/gamme sont lus directement depuis un PRODUIT reel du cluster
    (load_cluster_products), jamais reconstruits en decoupant `segment` sur
    "::" -- meme principe que weekly_report.py::_marque_gamme_product_level
    (`cluster_id.str.split("::").str[-1]`, qui ne prend QUE le dernier
    element) : rien ne garantit que marque/gamme ne contiennent jamais
    "::", jamais une hypothese non verifiee sur le nombre de parties."""
    if subdir == "clusters_n1":
        df = load_n1_cluster_estimates(category)
        if df.empty:
            return df
        try:
            seg_val = int(segment)
        except (TypeError, ValueError):
            return pd.DataFrame()
        return df[df["cluster"] == seg_val].sort_values("semaine").reset_index(drop=True)

    df = load_marque_gamme_estimates(category)
    if df.empty:
        return df
    products = load_cluster_products(category, subdir, segment)
    if products.empty:
        return pd.DataFrame()
    marque = products["marque"].iloc[0]
    gamme = products["gamme_prix"].iloc[0]
    subcluster = str(segment).split("::")[-1]
    mask = (df["marque"] == marque) & (df["gamme"] == gamme) & (df["cluster"] == subcluster)
    return df[mask].sort_values("semaine").reset_index(drop=True)


def _weekly_comparison_grid(weekly: pd.DataFrame) -> dag.AgGrid:
    display = weekly.copy()
    display = display.sort_values("semaine").reset_index(drop=True)
    
    if not display.empty:
        display["var_hebdo_reel_pct"] = display["moyenne_geometrique"].pct_change() * 100
        for c, col_name in [("moyenne_estimee_ridge", "var_hebdo_ridge_pct"),
                            ("moyenne_estimee_hedonic", "var_hebdo_hedonic_pct"),
                            ("moyenne_estimee_rf", "var_hebdo_rf_pct")]:
            if c in display.columns:
                display[col_name] = display[c].pct_change() * 100

    for c in ("moyenne_geometrique", "moyenne_estimee_ridge", "moyenne_estimee_hedonic", "moyenne_estimee_rf"):
        if c in display.columns:
            display[c] = display[c].round(2)
            
    for c in ("var_hebdo_reel_pct", "var_hebdo_ridge_pct", "var_hebdo_hedonic_pct", "var_hebdo_rf_pct"):
        if c in display.columns:
            display[c] = display[c].round(2)

    cols = [
        {"field": "semaine", "headerName": "Sem.", "headerTooltip": "Semaine", "flex": 1, "minWidth": 70},
        {"field": "n_produits", "headerName": "n", "headerTooltip": "Nombre de produits", "type": "rightAligned", "flex": 1, "minWidth": 60},
        {"field": "moyenne_geometrique", "headerName": "Réel (TND)", "headerTooltip": "Prix réel moyen (moyenne géométrique, TND)", "type": "rightAligned", "flex": 2, "minWidth": 100},
        {"field": "var_hebdo_reel_pct", "headerName": "Δ Réel %", "headerTooltip": "Variation hebdomadaire du prix réel (%)", "type": "rightAligned", "flex": 1, "minWidth": 90},
        {"field": "moyenne_estimee_ridge", "headerName": "Ridge (TND)", "headerTooltip": "Prix estimé par le modèle Ridge (TND)", "type": "rightAligned", "flex": 2, "minWidth": 100},
        {"field": "var_hebdo_ridge_pct", "headerName": "Δ Ridge %", "headerTooltip": "Variation hebdomadaire de l'estimation Ridge (%)", "type": "rightAligned", "flex": 1, "minWidth": 90},
        {"field": "moyenne_estimee_hedonic", "headerName": "OLS (TND)", "headerTooltip": "Prix estimé par le modèle Hedonic OLS (TND)", "type": "rightAligned", "flex": 2, "minWidth": 100},
        {"field": "var_hebdo_hedonic_pct", "headerName": "Δ OLS %", "headerTooltip": "Variation hebdomadaire de l'estimation Hedonic OLS (%)", "type": "rightAligned", "flex": 1, "minWidth": 90},
        {"field": "moyenne_estimee_rf", "headerName": "RF (TND)", "headerTooltip": "Prix estimé par le modèle Random Forest (TND)", "type": "rightAligned", "flex": 2, "minWidth": 100},
        {"field": "var_hebdo_rf_pct", "headerName": "Δ RF %", "headerTooltip": "Variation hebdomadaire de l'estimation Random Forest (%)", "type": "rightAligned", "flex": 1, "minWidth": 90},
        {"field": "erreur_ridge_pct", "headerName": "Err. Ridge %", "headerTooltip": "Écart entre le prix estimé Ridge et le prix réel (%)", "type": "rightAligned", "flex": 1.5, "minWidth": 100},
        {"field": "erreur_hedonic_pct", "headerName": "Err. OLS %", "headerTooltip": "Écart entre le prix estimé Hedonic OLS et le prix réel (%)", "type": "rightAligned", "flex": 1.5, "minWidth": 100},
        {"field": "erreur_rf_pct", "headerName": "Err. RF %", "headerTooltip": "Écart entre le prix estimé Random Forest et le prix réel (%)", "type": "rightAligned", "flex": 1.5, "minWidth": 100},
    ]
    return dag.AgGrid(
        className="ag-theme-quartz-dark",
        rowData=display.to_dict("records"),
        columnDefs=cols,
        defaultColDef={"sortable": True, "resizable": True},
        style={"height": f"{min(360, 90 + 32 * len(display))}px"}, columnSize="responsiveSizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


def _products_grid(products: pd.DataFrame) -> dag.AgGrid:
    display = products.sort_values(["semaine", "prix_tnd"], ascending=[False, True]).round(2)
    base_cols = [
        {"field": "semaine", "headerName": "Semaine", "flex": 1},
        {"field": "nom", "headerName": "Produit", "flex": 3},
        {"field": "marque", "headerName": "Marque", "flex": 1},
        {"field": "prix_tnd", "headerName": "Prix (TND)", "type": "rightAligned", "flex": 1},
    ]
    excluded = {"semaine", "nom", "marque", "prix_tnd", "url", "gamme_prix", "cluster_id", "cluster_direct"}
    feature_cols = [{"field": c, "headerName": c.replace("_", " "), "flex": 1}
                     for c in display.columns if c not in excluded]
    return dag.AgGrid(
        className="ag-theme-quartz-dark",
        rowData=display.to_dict("records"),
        columnDefs=base_cols + feature_cols,
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        style={"height": "420px"}, columnSize="responsiveSizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


def _cluster_overview_rows(category: str, subdir: str, summary: pd.DataFrame) -> list:
    """Une ligne par CLUSTER (pas par famille, contrairement a
    _cluster_models_summary_grid) -- vue compacte utilisee a la fois pour
    la grille d'ensemble et les options du selecteur de detail."""
    if summary.empty:
        return []
    rows = []
    for segment, grp in summary.groupby("segment", sort=False):
        display = n1_cluster_name(category, segment) if subdir == "clusters_n1" else str(segment).replace("::", " → ")
        row = {
            "segment": str(segment),
            "cluster_display": display,
            "n_lignes": int(grp["n_lignes"].iloc[0]),
            "n_produits_distincts": int(grp["n_produits_distincts"].iloc[0]) if "n_produits_distincts" in grp else None,
        }
        for famille in ("hedonic_ols", "ridge", "random_forest"):
            row[f"statut_{famille}"] = _segment_status_info(summary, segment, famille)["statut"]
        rows.append(row)
    rows.sort(key=lambda r: r["n_lignes"], reverse=True)
    return rows


def _cluster_overview_grid(rows: list) -> dag.AgGrid:
    status_rules = {"ag-status-positive": "value == 'Retenu'", "ag-status-negative": "value == 'Écarté'"}
    return dag.AgGrid(
        className="ag-theme-quartz-dark",
        rowData=rows,
        columnDefs=[
            {"field": "cluster_display", "headerName": "Cluster", "flex": 2},
            {"field": "n_lignes", "headerName": "n (train, poolé)", "type": "rightAligned", "flex": 1},
            {"field": "n_produits_distincts", "headerName": "n produits distincts", "type": "rightAligned", "flex": 1},
            {"field": "statut_hedonic_ols", "headerName": "Hedonic OLS", "flex": 1, "cellClassRules": status_rules},
            {"field": "statut_ridge", "headerName": "Ridge", "flex": 1, "cellClassRules": status_rules},
            {"field": "statut_random_forest", "headerName": "Random Forest", "flex": 1, "cellClassRules": status_rules},
        ],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        style={"height": f"{min(400, 90 + 32 * len(rows))}px"}, columnSize="responsiveSizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


def _cluster_detail_panel(category: str, subdir: str, segment: str | None) -> dmc.Accordion | dmc.Alert:
    """Detail complet d'UN cluster choisi via le selecteur -- 3 sections
    repliables (boutons AccordionControl, cf. demande explicite d'optimiser
    l'espace) : formules des 3 familles, prix reel vs estime par semaine,
    produits du cluster par semaine avec leurs caracteristiques."""
    if not segment:
        return dmc.Alert(
            "Aucun cluster disponible pour cette segmentation -- exécuter "
            "`python -m src.models.save_artifacts` (version 2026-08-01 ou plus récente).",
            color="gray", variant="light", icon=DashIconify(icon="tabler:info-circle"),
        )

    summary = load_cluster_models_summary(category, subdir)
    detail = load_cluster_segment_detail(category, subdir, segment)
    coefficients = detail.get("coefficients", {})

    formula_cards = [
        _famille_formula_card(famille, _segment_status_info(summary, segment, famille), coefficients.get(famille))
        for famille in ("hedonic_ols", "ridge", "random_forest")
    ]

    weekly = _cluster_weekly_comparison(category, subdir, segment)
    weekly_section = _weekly_comparison_grid(weekly) if not weekly.empty else dmc.Alert(
        "Comparaison hebdomadaire non disponible pour ce cluster.", color="gray", variant="light",
        icon=DashIconify(icon="tabler:info-circle"),
    )

    products = load_cluster_products(category, subdir, segment)
    products_section = _products_grid(products) if not products.empty else dmc.Alert(
        "Aucun produit trouvé pour ce cluster.", color="gray", variant="light",
        icon=DashIconify(icon="tabler:info-circle"),
    )

    return dmc.Accordion(
        [
            dmc.AccordionItem([
                dmc.AccordionControl("Formules des modèles (Hedonic OLS / Ridge / Random Forest)"),
                dmc.AccordionPanel(dmc.Stack(formula_cards, gap="sm")),
            ], value="formulas"),
            dmc.AccordionItem([
                dmc.AccordionControl("Prix réel (moyenne géométrique) vs estimé, par semaine"),
                dmc.AccordionPanel(weekly_section),
            ], value="weekly"),
            dmc.AccordionItem([
                dmc.AccordionControl("Produits du cluster, par semaine (prix et caractéristiques)"),
                dmc.AccordionPanel(products_section),
            ], value="products"),
        ],
        value=["formulas"], multiple=True, variant="separated",
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
            style={"height": "320px"}, columnSize="responsiveSizeToFit",
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
            style={"height": "260px"}, columnSize="responsiveSizeToFit",
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
            style={"height": "360px"}, columnSize="responsiveSizeToFit",
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

    n1_rows = _cluster_overview_rows(category, "clusters_n1", n1_models_summary)
    n2_rows = _cluster_overview_rows(category, "clusters_n2", n2_models_summary)
    
    def _make_dropdown_label(r):
        has_model = any(r.get(f"statut_{f}") == "Retenu" for f in ("hedonic_ols", "ridge", "random_forest"))
        suffix = " (Modèle dédié retenu)" if has_model else " (Aucun modèle dédié — sans repli)"
        return f"{r['cluster_display']} — n={r['n_lignes']} lignes poolées{suffix}"
        
    n1_select_data = [{"value": r["segment"], "label": _make_dropdown_label(r)} for r in n1_rows]
    n2_select_data = [{"value": r["segment"], "label": _make_dropdown_label(r)} for r in n2_rows]

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

        section_header("Clusters N2 (marque × gamme × profil technique) — vue d'ensemble", order=5,
                        subtitle="Une ligne par cluster. Choisir un cluster ci-dessous pour son détail complet (formules, prix par semaine, produits)."),
        _cluster_overview_grid(n2_rows) if n2_rows else dmc.Alert(
            "Modèles par cluster N2 non disponibles pour cette catégorie -- exécuter "
            "`python -m src.models.save_artifacts` (version 2026-08-01 ou plus récente).",
            color="gray", variant="light", icon=DashIconify(icon="tabler:info-circle"),
        ),
        dmc.Select(
            id="models-n2-cluster-select", label="Cluster N2 à examiner en détail",
            data=n2_select_data, value=n2_select_data[0]["value"] if n2_select_data else None,
            disabled=not n2_select_data, mt="sm", mb="sm", maw=520, clearable=False,
        ),
        html.Div(id="models-n2-cluster-detail"),
        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl("Voir le tableau d'audit complet N2 (tous les clusters × familles)"),
                dmc.AccordionPanel(_cluster_models_summary_grid(n2_models_summary)),
            ], value="n2-audit"),
        ], mt="sm", mb="md"),

        section_header("Clusters N1 (technique, toute la catégorie) — vue d'ensemble", order=5,
                        subtitle="Une ligne par cluster. Choisir un cluster ci-dessous pour son détail complet (formules, prix par semaine, produits)."),
        _cluster_overview_grid(n1_rows) if n1_rows else dmc.Alert(
            "Modèles par cluster N1 non disponibles pour cette catégorie -- exécuter "
            "`python -m src.models.save_artifacts` (version 2026-08-01 ou plus récente).",
            color="gray", variant="light", icon=DashIconify(icon="tabler:info-circle"),
        ),
        dmc.Select(
            id="models-n1-cluster-select", label="Cluster N1 à examiner en détail",
            data=n1_select_data, value=n1_select_data[0]["value"] if n1_select_data else None,
            disabled=not n1_select_data, mt="sm", mb="sm", maw=520, clearable=False,
        ),
        html.Div(id="models-n1-cluster-detail"),
        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl("Voir le tableau d'audit complet N1 (tous les clusters × familles)"),
                dmc.AccordionPanel(_cluster_models_summary_grid(n1_models_summary)),
            ], value="n1-audit"),
        ], mt="sm"),
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


@callback(
    Output("models-n1-cluster-detail", "children"),
    Input("models-category", "value"),
    Input("models-n1-cluster-select", "value"),
)
def _render_n1_cluster_detail(category, segment):
    """Callback chaîné (cf. render_form/on_predict de prediction.py pour le
    même patron) : le Select N1 est recréé par render_models a chaque
    changement de catégorie, ce qui redéclenche naturellement ce callback
    (suppress_callback_exceptions=True, composants insérés dynamiquement)."""
    if not artifacts_available(category):
        return dash.no_update
    return _cluster_detail_panel(category, "clusters_n1", segment)


@callback(
    Output("models-n2-cluster-detail", "children"),
    Input("models-category", "value"),
    Input("models-n2-cluster-select", "value"),
)
def _render_n2_cluster_detail(category, segment):
    if not artifacts_available(category):
        return dash.no_update
    return _cluster_detail_panel(category, "clusters_n2", segment)
