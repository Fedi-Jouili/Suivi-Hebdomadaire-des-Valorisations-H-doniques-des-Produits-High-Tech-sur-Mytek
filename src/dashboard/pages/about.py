# -*- coding: utf-8 -*-
"""Page 4 -- A propos du createur. Placeholders explicitement marques TODO
-- aucune donnee personnelle inventee, a completer par l'utilisateur."""

import dash
import dash_mantine_components as dmc
from dash_iconify import DashIconify

dash.register_page(__name__, path="/a-propos", name="À propos")

layout = dmc.Stack(
    [
        dmc.Center(
            dmc.Stack(
                [
                    dmc.Avatar(
                        DashIconify(icon="tabler:user", width=48), size=120, radius=120,
                        variant="light", color="blue", mx="auto",
                    ),
                    dmc.Title("TODO : Prénom Nom", order=2, ta="center"),
                    dmc.Text("TODO : intitulé du stage — Institut National de la Statistique (INS), Tunisie",
                             c="dimmed", ta="center"),
                ],
                gap="xs", align="center",
            ),
            mt="lg", mb="xl",
        ),
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 2},
            spacing="lg",
            children=[
                dmc.Card(
                    [
                        dmc.Group([DashIconify(icon="tabler:file-text", width=20), dmc.Text("À propos du projet", fw=600)]),
                        dmc.Text(
                            "TODO : décrire en 3-4 phrases le contexte du stage, l'objectif du projet "
                            "(décomposition hédonique des prix high-tech collectés sur Mytek.tn) et ce que "
                            "ce travail apporte à l'INS.",
                            size="sm", c="dimmed", mt="sm",
                        ),
                    ],
                    p="lg",
                ),
                dmc.Card(
                    [
                        dmc.Group([DashIconify(icon="tabler:address-book", width=20), dmc.Text("Contact", fw=600)]),
                        dmc.Stack(
                            [
                                dmc.Group([DashIconify(icon="tabler:mail", width=16), dmc.Text("TODO : email@example.com", size="sm")]),
                                dmc.Group([DashIconify(icon="tabler:brand-linkedin", width=16), dmc.Text("TODO : lien LinkedIn", size="sm")]),
                                dmc.Group([DashIconify(icon="tabler:brand-github", width=16), dmc.Text("TODO : lien GitHub", size="sm")]),
                            ],
                            gap="xs", mt="sm",
                        ),
                    ],
                    p="lg",
                ),
            ],
        ),
        dmc.Card(
            [
                dmc.Group([DashIconify(icon="tabler:tools", width=20), dmc.Text("Stack technique", fw=600)]),
                dmc.Text(
                    "Scraping (Scrapling), prétraitement (pandas, scikit-learn), modélisation hédonique "
                    "(statsmodels, Ridge, Random Forest), dashboard (Dash + Dash Mantine Components).",
                    size="sm", c="dimmed", mt="sm",
                ),
            ],
            p="lg", mt="lg",
        ),
    ],
    maw=780, mx="auto",
    className="page-fade",
)
