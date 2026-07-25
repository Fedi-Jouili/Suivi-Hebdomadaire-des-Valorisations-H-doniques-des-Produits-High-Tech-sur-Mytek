# -*- coding: utf-8 -*-
"""
Page 1 -- Statistiques descriptives (4 dernieres semaines), par categorie.
Volumes, prix, marques, caracteristiques techniques, valeurs manquantes,
evolution hebdomadaire. Jamais de filtrage silencieux (cf. README projet).
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
import plotly.express as px
from dash import Input, Output, callback, dcc, html
from dash_iconify import DashIconify

from src.dashboard.components import category_selector, kpi_card, kpi_row, provenance_strip, section_header
from src.dashboard.data_loader import (
    available_weeks,
    category_label,
    load_clean_category_recent,
    load_clean_category_week,
    load_metrics,
    raw_vs_clean_counts,
)
from src.dashboard.format_utils import fmt_number, fmt_price
from src.dashboard.theme import CATEGORY_COLORS, GRAPH_CONFIG, TEXT_MUT
from src.utils.config import CATEGORY_ORDER

dash.register_page(__name__, path="/descriptif", name="Statistiques descriptives")

TOP_N_BRANDS = 8


def layout():
    return dmc.Stack(
        [
            section_header(
                "Statistiques descriptives",
                "Volumes, prix, marques et caractéristiques techniques sur les 4 dernières semaines de collecte.",
            ),
            category_selector("descriptive-category"),
            dmc.LoadingOverlay(
                visible=False, id="descriptive-loading", zIndex=10,
                loaderProps={"type": "dots", "color": "blue"},
            ),
            html.Div(id="descriptive-content"),
        ],
        pos="relative",
        className="page-fade",
    )


def _empty_state(category: str):
    return dmc.Alert(
        f"Aucune donnée disponible pour « {category_label(category)} ». "
        f"Vérifier que le pipeline de prétraitement a bien été exécuté (data/processed/week_*/).",
        title="Données indisponibles",
        color="yellow",
        icon=DashIconify(icon="tabler:alert-triangle"),
    )


def _missing_data_panel(category: str, weeks: tuple):
    rows = []
    for w in weeks:
        counts = raw_vs_clean_counts(category, w)
        rows.append({
            "Semaine": f"S{w}",
            "Produits scrapés": counts["n_raw"],
            "Retenus (nettoyage)": counts["n_clean"],
            "Écartés": counts["n_excluded"],
            "% écarté": f"{100 * counts['n_excluded'] / counts['n_raw']:.1f} %" if counts["n_raw"] else "—",
        })
    df = pd.DataFrame(rows)
    return dmc.Stack([
        dmc.Text(
            "Un produit scrapé est écarté du jeu final s'il n'a aucun prix fiable après nettoyage "
            "(hors bornes de plausibilité et non ré-extractible depuis la fiche produit) — jamais silencieusement, "
            "cf. src/preprocessing/pipeline.py.",
            size="xs", c="dimmed", mb="xs",
        ),
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            id="descriptive-missing-grid",
            rowData=df.to_dict("records"),
            columnDefs=[
                {"field": "Semaine", "flex": 1},
                {"field": "Produits scrapés", "type": "rightAligned", "flex": 1},
                {"field": "Retenus (nettoyage)", "type": "rightAligned", "flex": 1},
                {"field": "Écartés", "type": "rightAligned", "flex": 1},
                {"field": "% écarté", "type": "rightAligned", "flex": 1},
            ],
            defaultColDef={"sortable": True, "resizable": True},
            style={"height": "180px"},
            dashGridOptions={"domLayout": "normal", "theme": "legacy"},
        ),
    ])


@callback(
    Output("descriptive-content", "children"),
    Output("descriptive-loading", "visible"),
    Input("descriptive-category", "value"),
)
def render_descriptive(category):
    weeks = available_weeks()[-4:]
    if not weeks:
        return _empty_state(category), False

    df_recent = load_clean_category_recent(category, 4)
    if df_recent.empty:
        return _empty_state(category), False

    latest_week = max(weeks)
    df_latest = load_clean_category_week(category, latest_week)
    color = CATEGORY_COLORS.get(category, "#2a78d6")

    # ── KPIs (semaine la plus récente = catalogue "actuel") ──────────────────
    median_price = float(df_latest["prix_tnd"].median())
    n_brands = int(df_latest["marque"].nunique())
    kpis = kpi_row([
        kpi_card(
            "Produits (S" + str(latest_week) + ")", fmt_number(len(df_latest)), "tabler:package", color=color,
            raw_value=len(df_latest),
        ),
        kpi_card(
            "Prix médian", fmt_price(median_price), "tabler:coin", color=color,
            raw_value=median_price, suffix=" TND",
        ),
        kpi_card(
            "Prix min / max",
            f"{fmt_price(df_latest['prix_tnd'].min())} / {fmt_price(df_latest['prix_tnd'].max())}",
            "tabler:arrows-vertical", color=color,
        ),
        kpi_card("Marques distinctes", fmt_number(n_brands), "tabler:tag", color=color, raw_value=n_brands),
    ])

    # ── Effectifs par semaine ─────────────────────────────────────────────────
    counts_by_week = df_recent.groupby("semaine").size().reindex(weeks, fill_value=0).reset_index(name="n")
    counts_by_week["semaine_label"] = counts_by_week["semaine"].apply(lambda w: f"S{w}")
    fig_counts = px.bar(counts_by_week, x="semaine_label", y="n", text="n", color_discrete_sequence=[color])
    fig_counts.update_traces(textposition="outside")
    fig_counts.update_layout(
        title="Produits distincts par semaine", xaxis_title="Semaine", yaxis_title="Nombre de produits",
        showlegend=False, height=340,
    )

    # ── Évolution du prix médian / moyen ──────────────────────────────────────
    price_evol = df_recent.groupby("semaine")["prix_tnd"].agg(["median", "mean"]).reindex(weeks).reset_index()
    price_evol["semaine_label"] = price_evol["semaine"].apply(lambda w: f"S{w}")
    fig_price_evol = px.line(
        price_evol, x="semaine_label", y=["median", "mean"], markers=True,
        color_discrete_sequence=[color, TEXT_MUT],
    )
    fig_price_evol.update_layout(
        title="Évolution du prix médian / moyen", xaxis_title="Semaine", yaxis_title="Prix (TND)",
        legend_title_text="", height=340,
    )
    fig_price_evol.for_each_trace(lambda t: t.update(name={"median": "Médiane", "mean": "Moyenne"}.get(t.name, t.name)))
    fig_price_evol.update_yaxes(rangemode="tozero")

    # ── Distribution des prix (semaine actuelle) ──────────────────────────────
    fig_price_dist = px.histogram(df_latest, x="prix_tnd", nbins=30, color_discrete_sequence=[color])
    fig_price_dist.update_layout(
        title=f"Distribution des prix — S{latest_week}", xaxis_title="Prix (TND)", yaxis_title="Nombre de produits",
        height=340,
    )

    # ── Prix par marque (top marques par effectif) ────────────────────────────
    top_brands = df_latest["marque"].value_counts().head(TOP_N_BRANDS).index.tolist()
    df_brands = df_latest[df_latest["marque"].isin(top_brands)]
    fig_price_brand = px.box(
        df_brands, x="marque", y="prix_tnd",
        category_orders={"marque": top_brands}, color_discrete_sequence=[color],
    )
    fig_price_brand.update_layout(
        title=f"Prix par marque — top {TOP_N_BRANDS} (S{latest_week})",
        xaxis_title="", yaxis_title="Prix (TND)", height=380,
    )

    # ── Caractéristiques techniques (top specs) ───────────────────────────────
    metrics = load_metrics(category) if _has_artifacts(category) else None
    spec_figs = []
    if metrics:
        for col in metrics["continuous_features"][:3]:
            if col not in df_latest.columns:
                continue
            f = px.histogram(df_latest, x=col, nbins=20, color_discrete_sequence=[color])
            f.update_layout(title=col.replace("_", " "), height=280, margin=dict(l=50, r=20, t=45, b=40))
            spec_figs.append(f)
        for col in [c for c in metrics["categorical_features"] if c != "marque"][:2]:
            if col not in df_latest.columns or df_latest[col].nunique() > 15:
                continue
            vc = df_latest[col].value_counts().reset_index()
            vc.columns = [col, "n"]
            f = px.bar(vc, x=col, y="n", color_discrete_sequence=[color])
            f.update_layout(title=col.replace("_", " "), height=280, margin=dict(l=50, r=20, t=45, b=40))
            spec_figs.append(f)

    # ── Répartition par marque (grille) ───────────────────────────────────────
    brand_table = (
        df_latest.groupby("marque")["prix_tnd"]
        .agg(n="count", median="median", min="min", max="max")
        .sort_values("n", ascending=False)
        .reset_index()
    )
    for c in ("median", "min", "max"):
        brand_table[c] = brand_table[c].round(0)

    content = dmc.Stack([
        provenance_strip(weeks, n=len(df_latest), extra=f"catégorie : {category_label(category)}"),
        kpis,
        dmc.Grid([
            dmc.GridCol(dcc.Graph(figure=fig_counts, config=GRAPH_CONFIG), span={"base": 12, "md": 6}),
            dmc.GridCol(dcc.Graph(figure=fig_price_evol, config=GRAPH_CONFIG), span={"base": 12, "md": 6}),
        ], mt="md"),
        dmc.Grid([
            dmc.GridCol(dcc.Graph(figure=fig_price_dist, config=GRAPH_CONFIG), span={"base": 12, "md": 6}),
            dmc.GridCol(dcc.Graph(figure=fig_price_brand, config=GRAPH_CONFIG), span={"base": 12, "md": 6}),
        ], mt="md"),
        section_header("Caractéristiques techniques", order=4) if spec_figs else None,
        dmc.SimpleGrid(cols={"base": 1, "sm": 2, "lg": len(spec_figs)}, spacing="md", children=[
            dcc.Graph(figure=f, config=GRAPH_CONFIG) for f in spec_figs
        ]) if spec_figs else None,
        section_header("Répartition par marque", order=4, subtitle=f"Semaine {latest_week}"),
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            id="descriptive-brand-grid",
            rowData=brand_table.to_dict("records"),
            columnDefs=[
                {"field": "marque", "headerName": "Marque", "flex": 2},
                {"field": "n", "headerName": "Produits", "type": "rightAligned", "flex": 1},
                {"field": "median", "headerName": "Prix médian (TND)", "type": "rightAligned", "flex": 1,
                 "valueFormatter": {"function": "params.value != null ? params.value.toLocaleString('fr-FR') : '—'"}},
                {"field": "min", "headerName": "Prix min (TND)", "type": "rightAligned", "flex": 1,
                 "valueFormatter": {"function": "params.value != null ? params.value.toLocaleString('fr-FR') : '—'"}},
                {"field": "max", "headerName": "Prix max (TND)", "type": "rightAligned", "flex": 1,
                 "valueFormatter": {"function": "params.value != null ? params.value.toLocaleString('fr-FR') : '—'"}},
            ],
            defaultColDef={"sortable": True, "resizable": True, "filter": True},
            style={"height": "360px"},
            columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ),
        section_header("Valeurs manquantes / produits écartés", order=4),
        _missing_data_panel(category, weeks),
    ])
    return content, False


def _has_artifacts(category: str) -> bool:
    from src.dashboard.data_loader import artifacts_available
    return artifacts_available(category)
