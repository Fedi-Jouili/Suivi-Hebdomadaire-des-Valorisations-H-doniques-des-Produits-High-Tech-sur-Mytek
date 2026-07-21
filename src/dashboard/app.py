# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/app.py
=============================================================================
ROLE :
    Point d'entree du dashboard -- App shell (AppShell : en-tete + barre
    laterale + zone de contenu), MantineProvider avec le theme unique
    (theme.py), framework multi-pages Dash (dash.page_container).

UTILISATION :
    python -m src.dashboard.app
    -> http://127.0.0.1:8050

PREREQUIS :
    python -m src.models.save_artifacts   (produit models/, requis par les
    pages Modeles & clustering / Prediction -- la page Statistiques
    descriptives fonctionne sans, sur data/processed/ seul).
=============================================================================
"""

import dash
import dash_mantine_components as dmc
from dash import Dash, Input, Output, State, callback, dcc
from dash_iconify import DashIconify

from src.dashboard import theme as _theme  # noqa: F401 -- applique le template Plotly par defaut a l'import
from src.dashboard.components.nav import sidebar_links
from src.dashboard.theme import DMC_THEME

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Hédonique Mytek.tn — INS Tunisie",
)
server = app.server  # pour un futur deploiement WSGI (gunicorn/waitress)

_header = dmc.AppShellHeader(
    dmc.Group(
        [
            dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
            dmc.ThemeIcon(DashIconify(icon="tabler:chart-histogram", width=22), size=36, radius="md", variant="light"),
            dmc.Title("Hédonique Mytek.tn", order=4),
            dmc.Badge("INS Tunisie", variant="light", color="gray", ml="auto", visibleFrom="sm"),
        ],
        h="100%", px="md", wrap="nowrap",
    )
)

_navbar = dmc.AppShellNavbar(id="navbar-content", children=[], p="md")

_main = dmc.AppShellMain(dmc.Container(dash.page_container, fluid=True, px="md", py="md"))

_shell = dmc.AppShell(
    [_header, _navbar, _main],
    header={"height": 60},
    navbar={"width": 260, "breakpoint": "sm", "collapsed": {"mobile": True}},
    padding="md",
    id="appshell",
)

app.layout = dmc.MantineProvider(
    theme=DMC_THEME,
    children=[dcc.Location(id="url"), _shell],
)


@callback(
    Output("appshell", "navbar"),
    Input("burger", "opened"),
    State("appshell", "navbar"),
)
def _toggle_navbar(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


@callback(Output("navbar-content", "children"), Input("url", "pathname"))
def _update_nav(pathname):
    return sidebar_links(pathname)


if __name__ == "__main__":
    app.run(debug=True)
