# -*- coding: utf-8 -*-
"""
Page -- Evolution hebdomadaire des prix, par categorie. STRICTEMENT lecture
seule (cf. src/dashboard/data_loader.py) sur les rapports produits par
`python -m src.models.weekly_report` : verification de la couverture du
clustering (N1 + N2) semaine par semaine, moyenne geometrique du prix par
cluster, prix reel vs estime par les 3 modeles (Hedonic OLS / Ridge / Random
Forest) + erreur, et decomposition prix "toutes choses egales" (indice
hedonique, effet fixe semaine) vs changement de composition/qualite du
catalogue -- jamais un texte fige, toujours recalcule a partir des chiffres
affiches sur la page.
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
    ArtifactsMissingError,
    available_weeks,
    category_label,
    load_catalog_composition,
    load_cluster_geometric_means,
    load_cluster_transitions,
    load_clustering_coverage,
    load_hedonic_price_index,
    load_marque_gamme_estimates,
    load_weekly_estimates,
    weekly_reports_available,
)
from src.dashboard.format_utils import fmt_number, fmt_pct_effect, fmt_price
from src.dashboard.theme import CATEGORY_COLORS, GRAPH_CONFIG, RED, GREEN, TEXT_MUT
from src.models.hedonic_model import POOLED_TIME_EXCLUDED_CATEGORIES
from src.models.weekly_report import MATERIALITY_THRESHOLD_PCT, MIN_RELIABLE_N, QUADRANT_LABELS

dash.register_page(__name__, path="/evolution", name="Évolution hebdomadaire")

_MODEL_LABELS = {"hedonic_ols": "Hedonic OLS", "ridge": "Ridge", "random_forest": "Random Forest"}


def layout():
    return dmc.Stack(
        [
            section_header(
                "Évolution hebdomadaire",
                "Couverture du clustering, prix par cluster, prix estimé par modèle et interprétation "
                "prix vs. qualité, semaine par semaine.",
            ),
            category_selector("evolution-category"),
            dmc.LoadingOverlay(
                visible=False, id="evolution-loading", zIndex=10,
                loaderProps={"type": "dots", "color": "blue"},
            ),
            html.Div(id="evolution-content"),
        ],
        pos="relative",
        className="page-fade",
    )


def _missing_reports_alert():
    return dmc.Alert(
        [
            dmc.Text("Aucun rapport hebdomadaire trouvé."),
            dmc.Code("python -m src.models.weekly_report", block=True, mt="xs"),
        ],
        title="Rapports non disponibles",
        color="yellow",
        icon=DashIconify(icon="tabler:alert-triangle"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. COUVERTURE DU CLUSTERING (verification -- les 2 approches, chaque semaine)
# ─────────────────────────────────────────────────────────────────────────────

def _coverage_section(coverage: pd.DataFrame, color: str):
    latest = coverage.sort_values("semaine").iloc[-1]
    all_n1 = (coverage["n1_pct_couverture"] >= 99.9).all()
    all_n2_ok = (coverage["n2_pct_couverture"] >= 90).all()

    banner = dmc.Alert(
        dmc.Text(
            f"Les {len(coverage)} semaines disponibles sont toutes clusterisées par l'approche N1 (technique) — "
            f"couverture 100 % structurelle (aucun filtrage de marque). Couverture N2 (marque × gamme) : "
            f"{coverage['n2_pct_couverture'].min():.1f} % à {coverage['n2_pct_couverture'].max():.1f} % selon la "
            f"semaine — les écarts proviennent d'unités marque × gamme trop petites pour un clustering "
            f"significatif (< 10 produits), jamais d'une semaine oubliée.",
            size="sm",
        ),
        color="green" if all_n1 and all_n2_ok else "orange",
        variant="light",
        icon=DashIconify(icon="tabler:circle-check" if all_n1 else "tabler:alert-triangle"),
        mb="sm",
    )

    grid = dag.AgGrid(
        className="ag-theme-quartz-dark",
        rowData=coverage.to_dict("records"),
        columnDefs=[
            {"field": "semaine", "headerName": "Semaine", "valueFormatter": {"function": "'S' + params.value"}, "flex": 1},
            {"field": "n_produits_poole", "headerName": "Produits poolés", "type": "rightAligned", "flex": 1},
            {"field": "n_retenus_marque_suffisante", "headerName": "Marque suffisante", "type": "rightAligned", "flex": 1},
            {"field": "n1_couverts", "headerName": "N1 clusterisés", "type": "rightAligned", "flex": 1},
            {"field": "n1_pct_couverture", "headerName": "N1 %", "type": "rightAligned", "flex": 1,
             "cellClassRules": {"ag-status-positive": "value >= 99.9"}},
            {"field": "n2_couverts", "headerName": "N2 clusterisés", "type": "rightAligned", "flex": 1},
            {"field": "n2_pct_couverture", "headerName": "N2 %", "type": "rightAligned", "flex": 1,
             "cellClassRules": {"ag-status-positive": "value >= 90"}},
        ],
        defaultColDef={"sortable": True, "resizable": True},
        style={"height": "200px"}, columnSize="sizeToFit",
        dashGridOptions={"theme": "legacy"},
    )
    return dmc.Stack([banner, grid])


# ─────────────────────────────────────────────────────────────────────────────
# 2. PRIX GEOMETRIQUE PAR CLUSTER (N1 / N2), PIVOTE SEMAINE EN COLONNES
# ─────────────────────────────────────────────────────────────────────────────

def _pivot_cluster_table(cluster_means: pd.DataFrame, approach: str) -> pd.DataFrame:
    df = cluster_means[cluster_means["approche"] == approach].copy()
    weeks = sorted(df["semaine"].unique())
    id_cols = ["cluster"] + (["marque", "gamme_prix"] if approach == "N2_marque_gamme" else [])

    pivot = df.pivot_table(index=id_cols, columns="semaine", values="prix_geometrique_tnd", aggfunc="first")
    pivot.columns = [f"S{w}" for w in pivot.columns]
    n_latest = df[df["semaine"] == weeks[-1]].set_index(id_cols)["n_produits"]
    pivot["n_produits"] = n_latest

    first_col, last_col = f"S{weeks[0]}", f"S{weeks[-1]}"
    pivot["variation_pct"] = ((pivot[last_col] - pivot[first_col]) / pivot[first_col] * 100).round(1)
    pivot = pivot.reset_index().sort_values("n_produits", ascending=False)
    return pivot, weeks


def _cluster_table_grid(cluster_means: pd.DataFrame, approach: str, grid_id: str):
    pivot, weeks = _pivot_cluster_table(cluster_means, approach)
    price_cols = [f"S{w}" for w in weeks]

    col_defs = [{"field": "cluster", "headerName": "Cluster", "flex": 2}]
    if approach == "N2_marque_gamme":
        col_defs += [
            {"field": "marque", "headerName": "Marque", "flex": 1},
            {"field": "gamme_prix", "headerName": "Gamme", "flex": 1},
        ]
    col_defs += [{"field": "n_produits", "headerName": "n (dernière semaine)", "type": "rightAligned", "flex": 1}]
    col_defs += [
        {"field": c, "headerName": f"Prix géo. {c} (TND)", "type": "rightAligned", "flex": 1,
         "valueFormatter": {"function": "params.value != null ? params.value.toLocaleString('fr-FR') : '—'"}}
        for c in price_cols
    ]
    col_defs += [
        {"field": "variation_pct", "headerName": f"Variation {price_cols[0]}→{price_cols[-1]} (%)",
         "type": "rightAligned", "flex": 1,
         "cellClassRules": {"ag-status-positive": "value < 0", "ag-status-negative": "value > 0"}},
    ]

    return dag.AgGrid(
        id=grid_id,
        className="ag-theme-quartz-dark",
        rowData=pivot.round(2).to_dict("records"),
        columnDefs=col_defs,
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        style={"height": "380px"}, columnSize="sizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. PRIX REEL VS ESTIME PAR MODELE, PAR SEMAINE + ERREUR
# ─────────────────────────────────────────────────────────────────────────────

def _estimates_chart(estimates: pd.DataFrame, color: str):
    weeks = sorted(estimates["semaine"].unique())
    actual = estimates.drop_duplicates("semaine").set_index("semaine")["prix_reel_geometrique_tnd"].reindex(weeks)

    fig = px.line()
    fig.add_scatter(
        x=[f"S{w}" for w in weeks], y=actual.values, mode="lines+markers", name="Prix réel (moy. géométrique)",
        line=dict(color=color, width=3), marker=dict(size=9),
    )
    palette = {"hedonic_ols": RED, "ridge": GREEN, "random_forest": TEXT_MUT}
    for model_name, label in _MODEL_LABELS.items():
        sub = estimates[estimates["modele"] == model_name].set_index("semaine")["prix_estime_geometrique_tnd"].reindex(weeks)
        fig.add_scatter(
            x=[f"S{w}" for w in weeks], y=sub.values, mode="lines+markers", name=f"Estimé — {label}",
            line=dict(color=palette[model_name], width=1.5, dash="dot"), marker=dict(size=6),
        )
    fig.update_layout(
        title="Prix réel vs. estimé par modèle (moyenne géométrique)",
        xaxis_title="Semaine", yaxis_title="Prix (TND)", height=380, legend_title_text="",
    )
    return fig


def _error_chart(estimates: pd.DataFrame):
    df = estimates.copy()
    df["modele_label"] = df["modele"].map(_MODEL_LABELS)
    df["semaine_label"] = df["semaine"].apply(lambda w: f"S{w}")
    fig = px.bar(
        df, x="semaine_label", y="mape_pct", color="modele_label", barmode="group",
        color_discrete_sequence=[RED, GREEN, TEXT_MUT],
    )
    fig.update_layout(
        title="Erreur typique par produit (MAPE) par modèle et semaine",
        xaxis_title="Semaine", yaxis_title="MAPE (%)", height=340, legend_title_text="",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. INTERPRETATION -- PRIX "TOUTES CHOSES EGALES" vs COMPOSITION/QUALITE
# ─────────────────────────────────────────────────────────────────────────────

_SPEC_LABELS = {
    "ram_go": "RAM moyenne", "stockage_go": "stockage moyen", "taille_ecran": "taille d'écran moyenne",
    "taux_rafraichissement": "taux de rafraîchissement moyen",
}


def _interpretation_section(category: str, composition: pd.DataFrame, price_index: pd.DataFrame | None):
    weeks = sorted(composition["semaine"].unique())
    first_week, last_week = weeks[0], weeks[-1]
    price_first = float(composition.loc[composition["semaine"] == first_week, "prix_geometrique_tnd"].iloc[0])
    price_last = float(composition.loc[composition["semaine"] == last_week, "prix_geometrique_tnd"].iloc[0])
    raw_change_pct = (price_last / price_first - 1) * 100 if price_first else float("nan")

    spec_cols = [c for c in composition.columns if c not in ("categorie", "semaine", "prix_geometrique_tnd", "n_produits")]
    spec_changes = {}
    for c in spec_cols:
        v0 = composition.loc[composition["semaine"] == first_week, c].iloc[0]
        v1 = composition.loc[composition["semaine"] == last_week, c].iloc[0]
        if pd.notna(v0) and pd.notna(v1) and v0:
            spec_changes[c] = (v1 / v0 - 1) * 100

    if category in POOLED_TIME_EXCLUDED_CATEGORIES or price_index is None:
        exclusion_note = dmc.Alert(
            dmc.Text(
                f"« {category_label(category)} » est exclue de l'indice de prix hédonique poolé : "
                f"Evolution_Temporelle_Marche_Mytek.ipynb a mesuré des coefficients hédoniques instables "
                f"entre semaines pour cette catégorie (échantillon réduit, pentes communes non justifiées). "
                f"La lecture ci-dessous reste donc DESCRIPTIVE (prix brut + composition du catalogue), sans "
                f"décomposition prix/qualité fiable — cf. src/models/hedonic_model.py.",
                size="sm",
            ),
            color="gray", variant="light", icon=DashIconify(icon="tabler:shield-off"), mb="sm",
        )
        spec_txt = _spec_change_sentence(spec_changes)
        conclusion = dmc.Text([
            f"Entre S{first_week} et S{last_week}, le prix affiché (moyenne géométrique) a varié de ",
            dmc.Text(fmt_pct_effect(raw_change_pct), span=True, fw=700,
                      c=RED if raw_change_pct > 0 else GREEN),
            f". {spec_txt} Sans indice hédonique fiable pour cette catégorie, il n'est pas possible de "
            f"trancher statistiquement entre un effet de coût/prix pur et un effet de composition du "
            f"catalogue — seule une lecture qualitative de ces deux évolutions conjointes est proposée ici.",
        ], size="sm")
        return dmc.Stack([exclusion_note, conclusion])

    row_last = price_index[price_index["semaine"] == last_week]
    quality_adj_pct = float(row_last["indice_prix_ajuste_qualite_pct"].iloc[0]) if not row_last.empty else None
    p_value = float(row_last["p_value"].iloc[0]) if not row_last.empty else None
    significant = p_value is not None and p_value < 0.05

    spec_txt = _spec_change_sentence(spec_changes)

    if quality_adj_pct is None:
        verdict = dmc.Text("Indice hédonique indisponible pour la dernière semaine.", size="sm", c="dimmed")
    elif significant:
        verdict = dmc.Text([
            "L'indice hédonique (net des caractéristiques techniques et de la marque) confirme un ",
            dmc.Text("changement de PRIX réel", span=True, fw=700, c=RED if quality_adj_pct > 0 else GREEN),
            f" de {fmt_pct_effect(quality_adj_pct)} « toutes choses égales » (p = {p_value:.3f} < 0,05, "
            f"statistiquement significatif) — cohérent avec un changement de politique de prix (coût, "
            f"marge, promotion), pas seulement avec le catalogue vendu.",
        ], size="sm")
    else:
        residual_pct = raw_change_pct - quality_adj_pct
        verdict = dmc.Text([
            "L'indice hédonique (net des caractéristiques) n'est PAS statistiquement significatif "
            f"({fmt_pct_effect(quality_adj_pct)}, p = {p_value:.3f} ≥ 0,05) : aucun changement de prix "
            "« toutes choses égales » ne peut être affirmé. La variation de prix brut observée "
            f"({fmt_pct_effect(raw_change_pct)}) s'explique donc surtout par un ",
            dmc.Text("changement de COMPOSITION/QUALITÉ du catalogue", span=True, fw=700, c=GREEN),
            f" (mix de produits différent d'une semaine à l'autre), pas par une politique de prix.",
        ], size="sm")

    conclusion = dmc.Stack([
        dmc.Text([
            f"Entre S{first_week} et S{last_week}, le prix affiché a varié de ",
            dmc.Text(fmt_pct_effect(raw_change_pct), span=True, fw=700),
            ".", " ", spec_txt,
        ], size="sm"),
        verdict,
    ], gap=6)

    return conclusion


def _spec_change_sentence(spec_changes: dict) -> str:
    if not spec_changes:
        return ""
    parts = []
    for col, pct in spec_changes.items():
        label = _SPEC_LABELS.get(col, col)
        direction = "en hausse" if pct > 0.5 else ("en baisse" if pct < -0.5 else "stable")
        parts.append(f"{label} {direction} ({fmt_pct_effect(pct)})")
    return "Sur la même période, la composition technique du catalogue : " + ", ".join(parts) + "."


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRANSITIONS PAR CLUSTER -- prix reel vs valeur technique implicite
# ─────────────────────────────────────────────────────────────────────────────
#
# Grille de lecture : QUADRANT_LABELS (importe de src.models.weekly_report,
# source UNIQUE du texte de classification -- jamais duplique ici) mappe
# (direction du prix reel, direction du prix estime) -> phrase academique.
# _severity() regroupe ces 9 cas en 4 niveaux visuels (neutre / explique /
# ecart partiel / ecart maximal), coherents avec la semantique rouge=alerte
# / vert=positif deja etablie dans theme.py -- jamais une palette ad hoc.

_LABEL_TO_DIRS = {v: k for k, v in QUADRANT_LABELS.items()}
_SEVERITY_ORDER = ["neutre", "explique", "ecart_partiel", "ecart_maximal"]
_SEVERITY_COLORS = {"neutre": TEXT_MUT, "explique": GREEN, "ecart_partiel": "#ef9fa4", "ecart_maximal": RED}
_SEVERITY_LABELS = {
    "neutre": "Neutre (rien ne bouge)",
    "explique": "Expliqué par les caractéristiques",
    "ecart_partiel": "Écart partiel",
    "ecart_maximal": "Écart maximal (signal fort)",
}
_SHORT_CLASSIFICATION_LABELS = {
    "Stabilité réelle (prix et caractéristiques inchangés)": "Stabilité réelle",
    "Amélioration technique non répercutée sur le prix (gain caché pour l'acheteur)": "Amélioration cachée (prix stable)",
    "Prix maintenu malgré une baisse de la valeur technique implicite (marge implicite en hausse)": "Marge en hausse (prix stable)",
    "Hausse de prix non justifiée par les caractéristiques (majoration pure)": "Majoration pure",
    "Hausse de prix cohérente avec une montée en gamme technique": "Montée en gamme (prix cohérent)",
    "Hausse de prix ET dégradation technique — écart maximal (majoration forte)": "Majoration forte + dégradation",
    "Baisse de prix non liée aux caractéristiques (promotion / remise réelle)": "Promotion réelle",
    "Baisse de prix malgré une montée en gamme technique — remise maximale pour l'acheteur": "Remise maximale",
    "Baisse de prix cohérente avec une baisse de gamme technique": "Baisse de gamme (prix cohérent)",
}


def _severity(classification: str) -> str:
    dirs = _LABEL_TO_DIRS.get(classification)
    if dirs is None:
        return "neutre"
    dir_raw, dir_hed = dirs
    if dir_raw == "stable" and dir_hed == "stable":
        return "neutre"
    if dir_raw == dir_hed:
        return "explique"
    if "stable" in (dir_raw, dir_hed):
        return "ecart_partiel"
    return "ecart_maximal"


def _severity_shares(transitions: pd.DataFrame) -> dict:
    if transitions.empty:
        return {k: 0.0 for k in _SEVERITY_ORDER}
    sev = transitions["classification"].map(_severity)
    share = (sev.value_counts(normalize=True) * 100)
    return {k: float(share.get(k, 0.0)) for k in _SEVERITY_ORDER}


def _transition_kpis(transitions: pd.DataFrame, color: str):
    shares = _severity_shares(transitions)
    pct_ok = shares["neutre"] + shares["explique"]
    pct_gap = shares["ecart_partiel"] + shares["ecart_maximal"]
    pct_accord = float(transitions["accord_modeles"].mean() * 100) if not transitions.empty else 0.0
    n_bootstrap = int(transitions["bootstrap_possible"].sum())
    pct_confirme = (
        float(transitions["ecart_residuel_significatif"].sum() / n_bootstrap * 100) if n_bootstrap else 0.0
    )
    return kpi_row([
        kpi_card(
            "Transitions étudiées", fmt_number(len(transitions)), "tabler:git-compare", color=color,
            raw_value=len(transitions), note="3 périodes × clusters marque × gamme",
        ),
        kpi_card(
            "Sans écart (stable ou expliqué)", f"{pct_ok:.0f} %", "tabler:circle-check", color=color,
            raw_value=pct_ok, suffix=" %",
        ),
        kpi_card(
            "Écart notable (seuil fixe ±3 %)", f"{pct_gap:.0f} %", "tabler:alert-triangle", color=color,
            raw_value=pct_gap, suffix=" %", note="lecture brute, cf. carte suivante",
        ),
        kpi_card(
            "Confirmé par bootstrap", f"{pct_confirme:.0f} %", "tabler:flask", color=color,
            raw_value=pct_confirme, suffix=" %",
            note=f"IC 95 % excluant 0, sur {n_bootstrap} testables — le signal le plus fiable",
        ),
        kpi_card(
            "Accord entre les 3 modèles", f"{pct_accord:.0f} %", "tabler:git-merge", color=color,
            raw_value=pct_accord, suffix=" %", note="même signe de variation estimée",
        ),
    ])


def _transition_summary_chart(transitions: pd.DataFrame):
    counts = transitions["classification"].value_counts().reset_index()
    counts.columns = ["classification", "n"]
    counts["severity"] = counts["classification"].map(_severity)
    counts["label_court"] = counts["classification"].map(lambda c: _SHORT_CLASSIFICATION_LABELS.get(c, c))
    counts = counts.sort_values("n")
    fig = px.bar(
        counts, x="n", y="label_court", orientation="h", color="severity",
        color_discrete_map=_SEVERITY_COLORS,
    )
    fig.update_layout(
        title="Répartition des transitions par type (3 périodes cumulées)",
        xaxis_title="Nombre de transitions (cluster × période)", yaxis_title="",
        height=max(320, 34 * len(counts)), showlegend=False, margin=dict(l=10),
        yaxis=dict(automargin=True),
    )
    return fig


def _quadrant_scatter(transitions: pd.DataFrame):
    df = transitions.copy()
    df["Lecture"] = df["classification"].map(_severity).map(_SEVERITY_LABELS)
    df["periode"] = "S" + df["semaine_t"].astype(str) + "→S" + df["semaine_t1"].astype(str)
    df["produit"] = df["marque"] + " · " + df["gamme"] + " · " + df["cluster"]

    fig = px.scatter(
        df, x="delta_estime_hedonic_pct", y="delta_prix_reel_pct", color="Lecture", facet_col="periode",
        color_discrete_map={_SEVERITY_LABELS[k]: v for k, v in _SEVERITY_COLORS.items()},
        category_orders={"Lecture": [_SEVERITY_LABELS[k] for k in _SEVERITY_ORDER]},
        hover_data={"produit": True, "classification": True, "n_produits_t": True, "n_produits_t1": True,
                    "Lecture": False, "periode": False},
    )
    fig.add_vline(x=MATERIALITY_THRESHOLD_PCT, line_dash="dot", line_color=TEXT_MUT, line_width=1)
    fig.add_vline(x=-MATERIALITY_THRESHOLD_PCT, line_dash="dot", line_color=TEXT_MUT, line_width=1)
    fig.add_hline(y=MATERIALITY_THRESHOLD_PCT, line_dash="dot", line_color=TEXT_MUT, line_width=1)
    fig.add_hline(y=-MATERIALITY_THRESHOLD_PCT, line_dash="dot", line_color=TEXT_MUT, line_width=1)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(title_text="Δ prix estimé (valeur technique, %)")
    fig.update_yaxes(title_text="Δ prix réel (%)", col=1)
    fig.update_layout(
        title="Transitions par cluster — prix réel vs. valeur technique implicite, par période",
        height=440, legend_title_text="",
    )
    return fig


def _raw_estimates_grid(mg_estimates: pd.DataFrame, grid_id: str):
    """Table BRUTE (une ligne par cluster x semaine) -- exactement le
    schema de reports/marque_gamme_estimations_hebdo.csv, pour inspecter
    les chiffres source derriere les transitions (jamais seulement le
    resultat agrege sans acces aux donnees qui le fondent)."""
    df = mg_estimates.rename(columns={
        "moyenne_geometrique": "Prix réel (moy. géo., TND)",
        "moyenne_estimee_ridge": "Estimé Ridge (TND)",
        "moyenne_estimee_hedonic": "Estimé Hedonic OLS (TND)",
        "moyenne_estimee_rf": "Estimé Random Forest (TND)",
    })
    cols = ["marque", "gamme", "cluster", "semaine", "Prix réel (moy. géo., TND)",
            "Estimé Ridge (TND)", "Estimé Hedonic OLS (TND)", "Estimé Random Forest (TND)", "n_produits"]
    return dag.AgGrid(
        id=grid_id,
        className="ag-theme-quartz-dark",
        rowData=df[cols].to_dict("records"),
        columnDefs=[
            {"field": "marque", "headerName": "Marque", "flex": 1},
            {"field": "gamme", "headerName": "Gamme", "flex": 1},
            {"field": "cluster", "headerName": "Cluster", "flex": 1},
            {"field": "semaine", "headerName": "Semaine", "type": "rightAligned", "flex": 1,
             "valueFormatter": {"function": "'S' + params.value"}},
        ] + [
            {"field": c, "headerName": c, "type": "rightAligned", "flex": 1,
             "valueFormatter": {"function": "params.value != null ? params.value.toLocaleString('fr-FR') : '—'"}}
            for c in ["Prix réel (moy. géo., TND)", "Estimé Ridge (TND)", "Estimé Hedonic OLS (TND)",
                      "Estimé Random Forest (TND)"]
        ] + [{"field": "n_produits", "headerName": "n produits", "type": "rightAligned", "flex": 1}],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        style={"height": "420px"}, columnSize="sizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


def _notable_cases_grid(transitions: pd.DataFrame, grid_id: str, top_n: int = 15):
    df = transitions.copy()
    df["abs_ecart"] = df["ecart_residuel_pct"].abs()
    # Tri PRIMAIRE sur la confirmation bootstrap (le signal fiable), pas
    # seulement sur l'amplitude du point estime -- un ecart de 20% sur 2
    # produits (IC tres large, non significatif) ne doit jamais passer
    # avant un ecart de 4% confirme sur un grand effectif.
    df = df.sort_values(["ecart_residuel_significatif", "abs_ecart"], ascending=[False, False]).head(top_n)
    df["periode"] = "S" + df["semaine_t"].astype(str) + "→S" + df["semaine_t1"].astype(str)
    df["classification_courte"] = df["classification"].map(lambda c: _SHORT_CLASSIFICATION_LABELS.get(c, c))
    df["ic_95pct"] = df.apply(
        lambda r: f"[{r['ecart_residuel_ic_bas']:.1f}, {r['ecart_residuel_ic_haut']:.1f}]"
        if pd.notna(r["ecart_residuel_ic_bas"]) else "—", axis=1,
    )

    cols = ["marque", "gamme", "cluster", "periode", "delta_prix_reel_pct", "delta_estime_hedonic_pct",
            "ecart_residuel_pct", "ic_95pct", "ecart_residuel_significatif", "classification_courte",
            "n_produits_t", "n_produits_t1", "composition_stable", "fiabilite_limitee"]
    return dag.AgGrid(
        id=grid_id,
        className="ag-theme-quartz-dark",
        rowData=df[cols].round(2).to_dict("records"),
        columnDefs=[
            {"field": "marque", "headerName": "Marque", "flex": 1},
            {"field": "gamme", "headerName": "Gamme", "flex": 1},
            {"field": "cluster", "headerName": "Cluster", "flex": 1},
            {"field": "periode", "headerName": "Période", "flex": 1},
            {"field": "delta_prix_reel_pct", "headerName": "Δ prix réel (%)", "type": "rightAligned", "flex": 1},
            {"field": "delta_estime_hedonic_pct", "headerName": "Δ estimé (%)", "type": "rightAligned", "flex": 1},
            {"field": "ecart_residuel_pct", "headerName": "Écart résiduel (%)", "type": "rightAligned", "flex": 1,
             "cellClassRules": {"ag-status-negative": "Math.abs(value) >= 5"}},
            {"field": "ic_95pct", "headerName": "IC bootstrap 95 %", "flex": 1},
            {"field": "ecart_residuel_significatif", "headerName": "Confirmé", "flex": 1,
             "cellClassRules": {"ag-status-positive": "value == true"}},
            {"field": "classification_courte", "headerName": "Classification", "flex": 2},
            {"field": "n_produits_t", "headerName": "n (t)", "type": "rightAligned", "flex": 1},
            {"field": "n_produits_t1", "headerName": "n (t+1)", "type": "rightAligned", "flex": 1},
            {"field": "composition_stable", "headerName": "Même effectif", "flex": 1},
            {"field": "fiabilite_limitee", "headerName": "Fiabilité limitée", "flex": 1,
             "cellClassRules": {"ag-status-negative": "value == true"}},
        ],
        defaultColDef={"sortable": True, "resizable": True, "filter": True},
        style={"height": "420px"}, columnSize="sizeToFit",
        dashGridOptions={"theme": "legacy"},
    )


def _transitions_narrative(category: str, transitions: pd.DataFrame):
    if transitions.empty:
        return dmc.Text("Pas assez de semaines consécutives pour étudier des transitions.", size="sm", c="dimmed")

    shares = _severity_shares(transitions)
    pct_ok = shares["neutre"] + shares["explique"]
    pct_gap = shares["ecart_partiel"] + shares["ecart_maximal"]
    n_bootstrap = int(transitions["bootstrap_possible"].sum())
    pct_confirme = float(transitions["ecart_residuel_significatif"].sum() / n_bootstrap * 100) if n_bootstrap else 0.0

    overview = dmc.Text([
        f"Sur {len(transitions)} transitions semaine-à-semaine observées pour « {category_label(category)} » "
        f"(tous clusters marque × gamme, 3 périodes cumulées), ",
        dmc.Text(f"{pct_ok:.0f} %", span=True, fw=700, c=GREEN),
        " ne montrent AUCUN écart entre prix réel et valeur technique implicite (stabilité ou variation de prix "
        "pleinement expliquée par les caractéristiques du mix vendu), tandis que ",
        dmc.Text(f"{pct_gap:.0f} %", span=True, fw=700, c=TEXT_MUT),
        " dépassent le seuil de matérialité fixe (±3 %, une convention, pas un test) — mais parmi les transitions "
        "où un test est possible (effectif ≥ 2 des deux côtés), seules ",
        dmc.Text(f"{pct_confirme:.0f} %", span=True, fw=700, c=(RED if pct_confirme > 15 else GREEN)),
        " résistent à un test bootstrap (intervalle de confiance à 95 % excluant 0, cf. §7 du notebook) — c'est ce "
        "chiffre-là, pas le premier, qui mesure un écart résiduel avec une base statistique plutôt qu'un seuil "
        "arbitraire.",
    ], size="sm")

    candidates = transitions[transitions["ecart_residuel_significatif"] & transitions["ecart_residuel_pct"].notna()]
    fallback_non_confirme = candidates.empty
    if candidates.empty:
        candidates = transitions[
            transitions["composition_stable"] & ~transitions["fiabilite_limitee"]
            & transitions["ecart_residuel_pct"].notna() & (transitions["ecart_residuel_pct"].abs() > 0.5)
        ]
    if candidates.empty:
        case_block = dmc.Text(
            f"Aucune transition suffisamment fiable (même effectif de produits des deux côtés de la transition, "
            f"échantillon ≥ {MIN_RELIABLE_N}) ne montre d'écart résiduel notable pour cette catégorie — cohérent "
            f"avec une tarification alignée sur les caractéristiques observées, dans la limite de ce que 4 "
            f"semaines de collecte permettent de conclure.",
            size="sm", c="dimmed",
        )
    else:
        case = candidates.loc[candidates["ecart_residuel_pct"].abs().idxmax()]
        case_severity = _severity(case["classification"])
        ic_txt = (
            f" (IC 95 % : [{case['ecart_residuel_ic_bas']:.1f}, {case['ecart_residuel_ic_haut']:.1f}])"
            if pd.notna(case.get("ecart_residuel_ic_bas")) else ""
        )
        titre = (
            "Cas le plus marquant — confirmé par bootstrap :" if not fallback_non_confirme else
            "Cas le plus marquant (point estimé seul, non confirmé par bootstrap — à lire avec prudence) :"
        )
        case_block = dmc.Stack([
            dmc.Text(titre, size="sm", fw=600),
            dmc.Text([
                f"{case['marque']} · {case['gamme']} · cluster {case['cluster']}, S{int(case['semaine_t'])} → "
                f"S{int(case['semaine_t1'])} ({int(case['n_produits_t'])} produit(s)) — prix réel : ",
                dmc.Text(fmt_pct_effect(case["delta_prix_reel_pct"]), span=True, fw=700),
                ", prix estimé (valeur technique implicite) : ",
                dmc.Text(fmt_pct_effect(case["delta_estime_hedonic_pct"]), span=True, fw=700),
                f" — écart résiduel de {fmt_pct_effect(case['ecart_residuel_pct'])}{ic_txt}. Lecture : ",
                dmc.Text(case["classification"], span=True, fw=700,
                          c=(RED if case_severity == "ecart_maximal" else TEXT_MUT)),
                ".",
            ], size="sm"),
        ], gap=4)

    example_note = dmc.Alert(
        dmc.Text(
            "Grille de lecture : une moyenne géométrique IDENTIQUE entre deux semaines mais une moyenne ESTIMÉE "
            "en baisse signifie que le prix affiché n'a pas bougé alors que la valeur technique implicite du mix "
            "vendu a diminué — l'écart entre ce que le marché facture et ce que les caractéristiques justifient "
            "s'est donc CREUSÉ, même si aucun des deux prix pris isolément n'a « bougé » de façon spectaculaire. "
            "Étude de cas détaillée, produit par produit : "
            "notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb.",
            size="xs",
        ),
        color="gray", variant="light", icon=DashIconify(icon="tabler:bulb"), mt="xs",
    )

    return dmc.Stack([overview, case_block, example_note])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output("evolution-content", "children"),
    Output("evolution-loading", "visible"),
    Input("evolution-category", "value"),
)
def render_evolution(category):
    if not weekly_reports_available():
        return _missing_reports_alert(), False

    weeks = available_weeks()
    color = CATEGORY_COLORS.get(category, "#2a78d6")

    try:
        coverage = load_clustering_coverage(category)
        cluster_means = load_cluster_geometric_means(category)
        estimates = load_weekly_estimates(category)
        composition = load_catalog_composition(category)
        price_index = load_hedonic_price_index(category)
        transitions = load_cluster_transitions(category)
        mg_estimates = load_marque_gamme_estimates(category)
    except ArtifactsMissingError:
        return _missing_reports_alert(), False

    last_week = int(coverage["semaine"].max())
    n1_pct = float(coverage.loc[coverage["semaine"] == last_week, "n1_pct_couverture"].iloc[0])
    n2_pct = float(coverage.loc[coverage["semaine"] == last_week, "n2_pct_couverture"].iloc[0])
    price_first = float(composition.sort_values("semaine")["prix_geometrique_tnd"].iloc[0])
    price_last = float(composition.sort_values("semaine")["prix_geometrique_tnd"].iloc[-1])
    raw_change_pct = (price_last / price_first - 1) * 100 if price_first else float("nan")
    best_model_mape = estimates[estimates["semaine"] == last_week].sort_values("mape_pct").iloc[0]

    kpis = kpi_row([
        kpi_card(
            "Couverture N1 (S" + str(last_week) + ")", f"{n1_pct:.0f} %", "tabler:chart-dots", color=color,
            raw_value=n1_pct, suffix=" %",
        ),
        kpi_card(
            "Couverture N2 (S" + str(last_week) + ")", f"{n2_pct:.0f} %", "tabler:layout-grid", color=color,
            raw_value=n2_pct, suffix=" %",
        ),
        kpi_card(
            f"Prix géo. S{last_week}", fmt_price(price_last), "tabler:coin", color=color,
            raw_value=price_last, suffix=" TND",
            note=f"{fmt_pct_effect(raw_change_pct)} vs S{int(composition['semaine'].min())}",
        ),
        kpi_card(
            "Meilleur modèle (MAPE, dernière semaine)",
            f"{_MODEL_LABELS[best_model_mape['modele']]} — {best_model_mape['mape_pct']:.1f} %",
            "tabler:target", color=color,
        ),
    ])

    content = dmc.Stack([
        provenance_strip(weeks, extra=f"catégorie : {category_label(category)} · rapports : python -m src.models.weekly_report"),
        kpis,
        section_header("1. Couverture du clustering — les deux approches", order=4,
                        subtitle="N1 (technique, tout le catalogue) et N2 (marque × gamme) — vérifié semaine par semaine, jamais supposé."),
        _coverage_section(coverage, color),
        section_header("2. Prix géométrique par cluster", order=4,
                        subtitle="Moyenne géométrique du prix — cohérente avec le modèle log-linéaire, par cluster et par semaine."),
        dmc.Tabs(
            [
                dmc.TabsList([
                    dmc.TabsTab("N1 — technique", value="n1", leftSection=DashIconify(icon="tabler:chart-dots")),
                    dmc.TabsTab("N2 — marque × gamme", value="n2", leftSection=DashIconify(icon="tabler:layout-grid")),
                ]),
                dmc.TabsPanel(_cluster_table_grid(cluster_means, "N1_technique", "evolution-n1-grid"), value="n1", pt="sm"),
                dmc.TabsPanel(_cluster_table_grid(cluster_means, "N2_marque_gamme", "evolution-n2-grid"), value="n2", pt="sm"),
            ],
            value="n1", mt="xs",
        ),
        section_header("3. Prix estimé par modèle et erreur", order=4,
                        subtitle="Hedonic OLS / Ridge / Random Forest, sur les données poolées (in-sample, cf. note ci-dessous)."),
        dmc.Alert(
            dmc.Text(
                "Prédictions calculées sur l'ensemble des données poolées (train + test), pas seulement le jeu de "
                "test — un choix descriptif (photographie du marché par semaine), pas une ré-évaluation de la "
                "généralisation hors-échantillon (déjà mesurée dans la page « Modèles & clustering »). "
                "erreur_pct = écart entre moyennes géométriques (biais agrégé) ; mape_pct = erreur absolue moyenne "
                "par produit (erreur typique).",
                size="xs",
            ),
            color="blue", variant="light", icon=DashIconify(icon="tabler:info-circle"), mb="sm",
        ),
        dmc.Grid([
            dmc.GridCol(dcc.Graph(figure=_estimates_chart(estimates, color), config=GRAPH_CONFIG), span={"base": 12, "lg": 7}),
            dmc.GridCol(dcc.Graph(figure=_error_chart(estimates), config=GRAPH_CONFIG), span={"base": 12, "lg": 5}),
        ]),
        section_header("4. Interprétation — prix ou qualité ?", order=4,
                        subtitle="Décomposition entre changement de prix « toutes choses égales » et changement de composition/qualité du catalogue."),
        dmc.Paper(_interpretation_section(category, composition, price_index), p="md", withBorder=True),
        section_header("5. Étude des transitions par cluster (marque × gamme)", order=4,
                        subtitle="Pour chaque cluster et chaque semaine consécutive : le prix réel a-t-il bougé pour une raison que les "
                                  "caractéristiques du mix vendu expliquent, ou pas ? Étude complète : notebooks/Etude_Transitions_Clusters_Marque_Gamme.ipynb."),
        _transition_kpis(transitions, color),
        dcc.Graph(figure=_quadrant_scatter(transitions), config=GRAPH_CONFIG),
        dmc.Grid([
            dmc.GridCol(dcc.Graph(figure=_transition_summary_chart(transitions), config=GRAPH_CONFIG), span={"base": 12, "lg": 6}),
            dmc.GridCol(
                dmc.Stack([
                    dmc.Text("Observations et interprétation", size="sm", fw=700, tt="uppercase", c="dimmed"),
                    _transitions_narrative(category, transitions),
                ]),
                span={"base": 12, "lg": 6},
            ),
        ], mt="sm"),
        section_header("Cas notables (plus grand écart résiduel)", order=5,
                        subtitle="Triés par |écart résiduel| décroissant — vérifier « Même effectif » et « Fiabilité limitée » avant toute lecture économique."),
        _notable_cases_grid(transitions, "evolution-notable-cases-grid"),
        dmc.Accordion([
            dmc.AccordionItem([
                dmc.AccordionControl("Voir les données source (marque × gamme × cluster × semaine)"),
                dmc.AccordionPanel(_raw_estimates_grid(mg_estimates, "evolution-mg-estimates-grid")),
            ], value="raw-data"),
        ], mt="sm"),
    ])
    return content, False
