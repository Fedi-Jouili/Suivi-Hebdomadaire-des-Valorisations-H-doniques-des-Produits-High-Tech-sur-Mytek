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
                    dmc.Title("Fedi Jouili", order=2, ta="center"),
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
                            "Ce projet applique une approche hédonique pour décomposer les prix des produits "
                            "technologiques collectés sur Mytek.tn en fonction de leurs caractéristiques techniques. "
                            "En utilisant un modèle semi-logarithmique combinant une régression Ridge et des Forêts "
                            "Aléatoires, on extrait l'impact de chaque attribut (RAM, processeur, stockage, écran, etc.) "
                            "sur le prix. Cette méthodologie, basée sur la théorie de Lancaster (1966) et Rosen (1974), "
                            "apporte à l'INS une mesure fiable des prix hédoniques pour analyser l'évolution du marché "
                            "high-tech en Tunisie.",
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
                                dmc.Group([DashIconify(icon="tabler:mail", width=16), dmc.Text("fedi.jouili@essai.ucar.tn", size="sm")]),
                                dmc.Group([DashIconify(icon="tabler:brand-linkedin", width=16), dmc.Anchor("Fedi Jouili", href="https://www.linkedin.com/in/fedi-jouili-a91677321/", target="_blank", size="sm")]),
                                dmc.Group([DashIconify(icon="tabler:brand-github", width=16), dmc.Anchor("Fedi-Jouili", href="https://github.com/Fedi-Jouili", target="_blank", size="sm")]),
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
