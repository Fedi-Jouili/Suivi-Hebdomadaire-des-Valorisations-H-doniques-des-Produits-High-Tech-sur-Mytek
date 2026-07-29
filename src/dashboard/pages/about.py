# -*- coding: utf-8 -*-
"""Page 4 -- A propos du createur et des organismes lies au projet."""

import dash
import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify
from urllib.parse import quote

dash.register_page(__name__, path="/a-propos", name="À propos")

# Organismes lies au projet -- logo (img/), lien officiel, description courte
# (une ligne, cf. demande explicite "pas de longs paragraphes"). Fond clair
# systematique derriere chaque logo (dmc.Paper blanc) : les 3 fichiers ont
# des fonds d'origine differents (blanc, creme, rouge plein) qui trancheraient
# mal individuellement sur le theme sombre du dashboard sans ce cadre commun.
_ORGANISMES = [
    {
        "nom": "ESSAI",
        "logo": "ecole-superieure-de-la-statistique-et-de-lanalyse-de-linformation-e7bf10f6.webp",
        "description": "École d'ingénieurs en statistique, Université de Carthage",
        "href": "http://www.essai.rnu.tn",
    },
    {
        "nom": "INS",
        "logo": "INS.webp",
        "description": "Institut National de la Statistique — organisme d'accueil du stage",
        "href": "https://www.ins.tn",
    },
    {
        "nom": "Mytek.tn",
        "logo": "mytek.webp",
        "description": "Revendeur high-tech tunisien — source des données du projet",
        "href": "https://www.mytek.tn",
    },
]


def _organisme_card(org: dict):
    return dmc.Anchor(
        dmc.Card(
            [
                dmc.Paper(
                    html.Img(
                        src=f"/stage_img/{quote(org['logo'])}",
                        style={"height": "56px", "width": "auto", "maxWidth": "100%", "objectFit": "contain"},
                    ),
                    p="sm", radius="sm", withBorder=False,
                    style={"backgroundColor": "#f2f0e9", "display": "flex", "justifyContent": "center"},
                ),
                dmc.Group(
                    [
                        dmc.Text(org["nom"], fw=700, size="sm"),
                        DashIconify(icon="tabler:external-link", width=14, color="var(--mantine-color-dimmed)"),
                    ],
                    gap=4, mt="sm", justify="center",
                ),
                dmc.Text(org["description"], size="xs", c="dimmed", ta="center", mt=2),
            ],
            p="md", withBorder=True,
        ),
        href=org["href"], target="_blank", underline=False,
        style={"color": "inherit"},
    )


layout = dmc.Stack(
    [
        dmc.Center(
            dmc.Stack(
                [
                    dmc.Avatar(
                        src=f"/stage_img/{quote('Fedi Jouili.jpg')}",
                        size=120, radius=120,
                        variant="light", color="blue", mx="auto",
                    ),
                    dmc.Title("Fedi Jouili", order=2, ta="center"),
                    dmc.Text(
                        "Élève-ingénieur à l'ESSAI · Stage à l'Institut National de la Statistique (INS), Tunisie",
                        c="dimmed", ta="center",
                    ),
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
                            "Décomposition hédonique des prix de produits high-tech collectés sur Mytek.tn, "
                            "par catégorie technique.",
                            size="sm", c="dimmed", mt="sm",
                        ),
                        dmc.Text(
                            "Modèles Ridge et Random Forest, sur la base théorique de Lancaster (1966) et Rosen (1974).",
                            size="sm", c="dimmed", mt="xs",
                        ),
                        dmc.Text(
                            "Objectif : mesurer l'effet de chaque caractéristique (RAM, processeur, écran...) sur le prix.",
                            size="sm", c="dimmed", mt="xs",
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
        dmc.Divider(mt="xl", mb="md"),
        dmc.Text("Organismes liés au projet", fw=600, size="sm", tt="uppercase", c="dimmed", ta="center", mb="sm"),
        dmc.SimpleGrid(
            cols={"base": 1, "sm": 3},
            spacing="lg",
            children=[_organisme_card(org) for org in _ORGANISMES],
        ),
    ],
    maw=780, mx="auto",
    className="page-fade",
)
