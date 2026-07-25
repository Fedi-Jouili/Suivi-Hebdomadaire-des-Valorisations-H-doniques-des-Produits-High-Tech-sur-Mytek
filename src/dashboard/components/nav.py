# -*- coding: utf-8 -*-
"""Liens de navigation de la barre laterale (AppShellNavbar), construits
depuis dash.page_registry -- jamais une liste dupliquee a la main qui
pourrait diverger des pages reellement enregistrees."""

import dash
import dash_mantine_components as dmc
from dash import dcc
from dash_iconify import DashIconify

_ICONS = {
    "Accueil": "tabler:home",
    "Statistiques descriptives": "tabler:chart-histogram",
    "Modèles & clustering": "tabler:brain",
    "Évolution hebdomadaire": "tabler:trending-up",
    "Prédiction": "tabler:target-arrow",
    "À propos": "tabler:user-circle",
}


def sidebar_links(pathname: str | None = None):
    items = []
    for page in dash.page_registry.values():
        icon = _ICONS.get(page["name"], "tabler:point")
        active = pathname == page["relative_path"]
        items.append(
            dcc.Link(
                dmc.NavLink(
                    label=page["name"],
                    leftSection=DashIconify(icon=icon, width=18),
                    active=active,
                    variant="filled" if active else "subtle",
                    color="blue",
                ),
                href=page["relative_path"],
                style={"textDecoration": "none"},
            )
        )
    return dmc.Stack(items, gap=4)
