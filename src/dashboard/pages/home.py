# -*- coding: utf-8 -*-
"""Page d'accueil -- presentation du projet + acces rapide aux 3 pages
d'analyse. Enregistree comme page racine ("/")."""

import dash
import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify

from src.dashboard.components import kpi_card, kpi_row
from src.dashboard.data_loader import available_weeks, load_pooled_category_full
from src.utils.config import CATEGORY_LABELS, CATEGORY_ORDER

dash.register_page(__name__, path="/", name="Accueil")

_CARDS = [
    {
        "title": "Statistiques descriptives",
        "text": "Volumes, prix, marques et caractéristiques techniques sur les 4 dernières semaines de collecte.",
        "icon": "tabler:chart-histogram",
        "href": "/descriptif",
    },
    {
        "title": "Modèles & clustering",
        "text": "Décomposition hédonique du prix (Ridge, Random Forest), segmentation par marque et gamme.",
        "icon": "tabler:brain",
        "href": "/modeles",
    },
    {
        "title": "Prédiction",
        "text": "Composez un produit hypothétique et estimez son prix, comparé à des produits réels similaires.",
        "icon": "tabler:target-arrow",
        "href": "/prediction",
    },
]


def _nav_card(item):
    return dcc.Link(
        dmc.Card(
            [
                dmc.ThemeIcon(DashIconify(icon=item["icon"], width=24), size=48, radius="md", variant="light"),
                dmc.Text(item["title"], fw=600, size="lg", mt="sm"),
                dmc.Text(item["text"], size="sm", c="dimmed", mt=4),
            ],
            p="lg",
            h="100%",
        ),
        href=item["href"],
        style={"textDecoration": "none", "color": "inherit"},
    )


def _category_counts():
    rows = []
    for cat in CATEGORY_ORDER:
        df = load_pooled_category_full(cat)
        rows.append({"categorie": CATEGORY_LABELS[cat], "n": len(df)})
    return rows


def layout():
    weeks = available_weeks()
    counts = _category_counts()
    total_n = sum(r["n"] for r in counts)

    return dmc.Stack(
        [
            dmc.Stack(
                [
                    dmc.Title("Décomposition hédonique des prix high-tech en Tunisie", order=1),
                    dmc.Text(
                        "Suivi hebdomadaire des prix de produits technologiques sur Mytek.tn, décomposés par "
                        "la méthode des prix hédoniques (Lancaster 1966 / Rosen 1974) — projet réalisé dans le "
                        "cadre d'un stage à l'Institut National de la Statistique (INS), Tunisie.",
                        size="md", c="dimmed", maw=780,
                    ),
                ],
                gap="xs",
                mb="lg",
            ),
            dmc.Box(
                kpi_row([
                    kpi_card(
                        "Semaines de collecte", f"{len(weeks)}", "tabler:calendar-stats",
                        note=f"S{min(weeks)} → S{max(weeks)}" if weeks else "—",
                        raw_value=len(weeks),
                    ),
                    kpi_card(
                        "Produits suivis (cumulé)", f"{total_n:,}".replace(",", " "), "tabler:package",
                        note="toutes semaines, 5 catégories", raw_value=total_n,
                    ),
                    kpi_card(
                        "Catégories analysées", f"{len(CATEGORY_ORDER)}", "tabler:category-2",
                        note="toujours séparément, jamais poolées", raw_value=len(CATEGORY_ORDER),
                    ),
                ]),
                mb="xl",
            ),
            dmc.Title("Explorer", order=3, mb="sm"),
            dmc.SimpleGrid(cols={"base": 1, "sm": 3}, spacing="md", children=[_nav_card(c) for c in _CARDS]),
        ],
        className="page-fade",
    )
