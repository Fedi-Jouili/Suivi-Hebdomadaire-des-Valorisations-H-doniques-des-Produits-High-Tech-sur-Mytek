# -*- coding: utf-8 -*-
"""
Page 3 -- Prediction. L'utilisateur compose un produit hypothetique,
choisit un modele et un type de segmentation, obtient un prix predit
(retro-transforme, intervalle si disponible), un segment assigne, et une
liste de produits reels similaires pour verifier la coherence du resultat.
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import ALL, Input, Output, State, callback, dcc, html
from dash_iconify import DashIconify

from src.dashboard.components import category_selector, info_tip, provenance_strip, section_header
from src.dashboard.data_loader import (
    ArtifactsMissingError,
    artifacts_available,
    available_weeks,
    category_label,
    get_feature_ranges,
    load_unit_summary,
)
from src.dashboard.format_utils import fmt_number, fmt_pct_effect, fmt_price
from src.dashboard.prediction_utils import assign_n1_cluster, assign_n2_segment, find_similar_products, predict_price
from src.dashboard.theme import CATEGORY_COLORS, GRAPH_CONFIG

dash.register_page(__name__, path="/prediction", name="Prédiction")

_MODEL_DATA = [
    {"value": "ridge", "label": "Ridge"},
    {"value": "ols", "label": "Hedonic OLS"},
    {"value": "rf", "label": "Random Forest"},
]
_SEGMENTATION_DATA = [
    {"value": "n1", "label": "N1 — technique"},
    {"value": "n2", "label": "N2 — marque × gamme"},
]

_FEATURE_LABELS = {
    "ram_go": "RAM (Go)", "stockage_go": "Stockage (Go)", "taille_ecran": "Taille d'écran (pouces)",
    "taux_rafraichissement": "Taux de rafraîchissement (Hz)", "cpu_serie": "Tier CPU (Intel Core iX / AMD Ryzen X)",
    "cpu_brand": "Marque du processeur", "os_platform": "Plateforme OS", "resolution_affichage": "Résolution",
    "technologie_dalle": "Technologie de dalle", "hdr": "HDR", "smart_tv": "Smart TV",
}


def layout():
    return dmc.Stack(
        [
            section_header(
                "Prédiction",
                "Composez un produit hypothétique, choisissez un modèle et un type de segmentation, "
                "obtenez un prix estimé comparé à des produits réels similaires.",
            ),
            category_selector("prediction-category"),
            dmc.Grid([
                dmc.GridCol(
                    dmc.Select(
                        id="prediction-model",
                        label=dmc.Group([dmc.Text("Modèle de prédiction", size="sm", fw=500),
                                          info_tip("Modèle utilisé pour estimer le prix — Ridge et Random Forest "
                                                    "sont évalués hors-échantillon ; Hedonic OLS fournit en plus "
                                                    "un intervalle de prédiction.")], gap=0),
                        data=_MODEL_DATA, value="ridge",
                    ),
                    span={"base": 12, "sm": 4},
                ),
                dmc.GridCol(
                    dmc.Select(
                        id="prediction-segmentation",
                        label=dmc.Group([dmc.Text("Type de segmentation", size="sm", fw=500),
                                          info_tip("N1 : clustering technique pur, calculable directement pour un "
                                                    "produit hypothétique. N2 : marque × gamme de prix, estimée en "
                                                    "2 temps (prix provisoire → gamme) car la gamme dépend du prix.")], gap=0),
                        data=_SEGMENTATION_DATA, value="n1",
                    ),
                    span={"base": 12, "sm": 4},
                ),
                dmc.GridCol(
                    dmc.Select(
                        id="prediction-week",
                        label=dmc.Group([dmc.Text("Semaine", size="sm", fw=500),
                                          info_tip("Situe la prédiction à une semaine donnée, via l'indice de prix "
                                                    "hédonique déjà calculé (page Évolution hebdomadaire) — les 3 "
                                                    "modèles sont entraînés sur toutes les semaines poolées sans "
                                                    "distinction, l'ajustement vient donc de cet indice, pas du "
                                                    "modèle lui-même. Filtre aussi les produits réels comparables.")], gap=0),
                        data=[{"value": str(w), "label": f"S{w}"} for w in available_weeks()],
                        value=str(available_weeks()[-1]) if available_weeks() else None,
                    ),
                    span={"base": 12, "sm": 4},
                ),
            ], mb="md"),
            html.Div(id="prediction-form"),
            dmc.Button(
                "Prédire le prix", id="prediction-submit", leftSection=DashIconify(icon="tabler:calculator"),
                mt="md", size="md", color="red",
            ),
            dmc.LoadingOverlay(
                visible=False, id="prediction-loading", zIndex=10,
                loaderProps={"type": "dots", "color": "blue"},
            ),
            html.Div(id="prediction-results", style={"marginTop": "1.5rem"}),
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
        title="Prédiction indisponible", color="yellow", icon=DashIconify(icon="tabler:alert-triangle"),
    )


def _field_label(col: str) -> str:
    return _FEATURE_LABELS.get(col, col.replace("_", " ").capitalize())


def _fmt_capacity(v: float) -> str:
    """8.0 -> '8 Go' -- paliers materiels, jamais de decimale parasite."""
    return f"{v:.0f} Go" if float(v).is_integer() else f"{v:g} Go"


@callback(Output("prediction-form", "children"), Input("prediction-category", "value"))
def render_form(category):
    if not artifacts_available(category):
        return _missing_artifacts_alert(category)

    ranges = get_feature_ranges(category)
    fields = []

    fields.append(dmc.GridCol(
        dmc.Select(
            id={"type": "predict-input", "field": "marque"}, label="Marque",
            data=sorted(ranges["marque"]), value=(sorted(ranges["marque"])[0] if ranges["marque"] else None),
            searchable=True,
        ),
        span={"base": 12, "sm": 6, "md": 4},
    ))

    for col, values in ranges["discrete_options"].items():
        fields.append(dmc.GridCol(
            dmc.Select(
                id={"type": "predict-input", "field": col}, label=_field_label(col),
                data=[{"value": str(v), "label": _fmt_capacity(v)} for v in values],
                value=str(values[len(values) // 2]),
                description=f"{len(values)} paliers observés dans le catalogue",
            ),
            span={"base": 12, "sm": 6, "md": 4},
        ))

    for col, r in ranges["continuous"].items():
        fields.append(dmc.GridCol(
            dmc.NumberInput(
                id={"type": "predict-input", "field": col}, label=_field_label(col),
                value=round(r["median"], 2), min=r["min"], max=r["max"], step=r["step"],
                description=f"Plage observée : {r['min']:.1f} – {r['max']:.1f}",
            ),
            span={"base": 12, "sm": 6, "md": 4},
        ))

    for col, values in ranges["categorical"].items():
        fields.append(dmc.GridCol(
            dmc.Select(
                id={"type": "predict-input", "field": col}, label=_field_label(col),
                data=[str(v) for v in values], value=(str(values[0]) if values else None),
            ),
            span={"base": 12, "sm": 6, "md": 4},
        ))

    for col in ranges["binary_flags"]:
        fields.append(dmc.GridCol(
            dmc.Select(
                id={"type": "predict-input", "field": col}, label=_field_label(col),
                data=[{"value": "1", "label": "Oui"}, {"value": "0", "label": "Non"}], value="0",
            ),
            span={"base": 12, "sm": 6, "md": 4},
        ))

    return dmc.Paper(dmc.Grid(fields), p="md", withBorder=True)


def _price_result_card(pred: dict, color: str):
    header = [dmc.Text("Prix prédit", size="xs", tt="uppercase", c="dimmed", fw=500)]
    if pred.get("week_adjustment_pct") is not None:
        adj = pred["week_adjustment_pct"]
        header.append(dmc.Badge(
            f"ajustement semaine {fmt_pct_effect(adj)}", color=("red" if adj > 0 else "green"),
            variant="light", size="sm",
        ))
    body = [
        dmc.Group(header, gap="xs"),
        dmc.Text(fmt_price(pred["price"]), size="2rem", fw=700, style={"fontVariantNumeric": "tabular-nums"}),
    ]
    if pred["has_interval"]:
        body.append(dmc.Text(
            f"Intervalle 95% : {fmt_price(pred['price_lower'])} – {fmt_price(pred['price_upper'])}",
            size="sm", c="dimmed",
        ))
    body.append(dmc.Text(pred["note"], size="xs", c="dimmed", mt="xs"))
    return dmc.Card(body, p="lg", withBorder=True)


def _segment_result_card(segmentation: str, category: str, category_color: str, n1_cluster=None, n2_info=None):
    if segmentation == "n1":
        unit_summary_note = "Segment technique (N1) — construit sans prix ni marque."
        value = f"Cluster {n1_cluster}"
    else:
        if n2_info["gamme_estimee"] is None:
            value = "Non déterminable"
            unit_summary_note = "Marque sans gamme de prix définie."
        else:
            value = f"{n2_info['gamme_estimee']} — cluster {n2_info['cluster_id']}"
            unit_summary_note = (
                f"Gamme estimée à partir du prix provisoire, segment = celui du produit réel le plus proche "
                f"parmi {n2_info['n_comparables']} comparable(s) (marque × gamme)."
            )
    return dmc.Card(
        [
            dmc.Text("Segment assigné", size="xs", tt="uppercase", c="dimmed", fw=500),
            dmc.Text(value, size="xl", fw=700),
            dmc.Text(unit_summary_note, size="xs", c="dimmed", mt="xs"),
        ],
        p="lg", withBorder=True,
    )


@callback(
    Output("prediction-results", "children"),
    Output("prediction-loading", "visible"),
    Input("prediction-submit", "n_clicks"),
    State({"type": "predict-input", "field": ALL}, "value"),
    State({"type": "predict-input", "field": ALL}, "id"),
    State("prediction-category", "value"),
    State("prediction-model", "value"),
    State("prediction-segmentation", "value"),
    State("prediction-week", "value"),
    prevent_initial_call=True,
)
def on_predict(n_clicks, values, ids, category, model_name, segmentation, week):
    if not ids:
        return dmc.Alert("Formulaire indisponible — sélectionner une catégorie avec des artefacts entraînés.",
                          color="yellow"), False

    form_values = {}
    for id_dict, val in zip(ids, values):
        field = id_dict["field"]
        try:
            form_values[field] = float(val)
        except (TypeError, ValueError):
            form_values[field] = val

    try:
        pred = predict_price(category, model_name, form_values, week=week)
    except ArtifactsMissingError as exc:
        return dmc.Alert(str(exc), color="yellow", title="Artefacts manquants"), False

    color = CATEGORY_COLORS.get(category, "#2a78d6")
    marque = form_values.get("marque")

    if segmentation == "n1":
        n1_cluster = assign_n1_cluster(category, form_values)
        segment_card = _segment_result_card("n1", category, color, n1_cluster=n1_cluster)
    else:
        n2_info = assign_n2_segment(category, marque, pred["price"], form_values)
        segment_card = _segment_result_card("n2", category, color, n2_info=n2_info)

    similar = find_similar_products(category, form_values, top_n=10, week=week)
    display_cols = ["nom", "marque", "prix_tnd", "distance"]
    tech_cols = [c for c in similar.columns if c in form_values and c not in display_cols][:4]
    grid_cols = display_cols[:3] + tech_cols + ["distance"]
    grid_cols = list(dict.fromkeys(grid_cols))  # dedoublonne en gardant l'ordre

    col_defs = []
    for c in grid_cols:
        if c == "prix_tnd":
            col_defs.append({"field": c, "headerName": "Prix (TND)", "type": "rightAligned",
                              "valueFormatter": {"function": "params.value != null ? params.value.toLocaleString('fr-FR') : '—'"}})
        elif c == "distance":
            col_defs.append({"field": c, "headerName": "Distance (écart technique)", "type": "rightAligned",
                              "valueFormatter": {"function": "params.value != null ? params.value.toFixed(2) : '—'"}})
        elif c == "nom":
            col_defs.append({"field": c, "headerName": "Produit", "flex": 2})
        else:
            col_defs.append({"field": c, "headerName": _field_label(c), "flex": 1})

    results = dmc.Stack([
        dmc.SimpleGrid(cols={"base": 1, "sm": 2}, spacing="md", children=[
            _price_result_card(pred, color), segment_card,
        ]),
        section_header("Produits réels similaires", order=4,
                        subtitle="Plus proches voisins dans l'espace des caractéristiques techniques standardisées — pour vérifier la prédiction."),
        dag.AgGrid(
            className="ag-theme-quartz-dark",
            rowData=similar[grid_cols].round(2).to_dict("records") if not similar.empty else [],
            columnDefs=col_defs,
            defaultColDef={"sortable": True, "resizable": True},
            style={"height": "380px"}, columnSize="sizeToFit",
            dashGridOptions={"theme": "legacy"},
        ) if not similar.empty else dmc.Text("Aucun produit comparable trouvé.", c="dimmed"),
    ])
    return results, False
