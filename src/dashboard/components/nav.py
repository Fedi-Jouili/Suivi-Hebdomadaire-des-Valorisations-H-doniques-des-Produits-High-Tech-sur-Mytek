# -*- coding: utf-8 -*-
"""Liens de navigation de la barre laterale (AppShellNavbar), construits
depuis dash.page_registry -- jamais une liste dupliquee a la main qui
pourrait diverger des pages reellement enregistrees.

Ordre d'affichage explicite (_PAGE_ORDER) : dash.page_registry est un dict
dont l'ordre d'insertion suit l'ordre d'IMPORT des modules de page (cf.
dash._pages._import_layouts_from_pages, os.walk non trie) -- fragile,
jamais garanti stable si un fichier de page est ajoute/renomme. _PAGE_ORDER
fixe l'ordre voulu independamment de cet artefact d'import ; toute page
enregistree mais absente de la liste (nouvelle page pas encore ajoutee ici)
degrade proprement en fin de liste, triee alphabetiquement, jamais omise."""

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
    "Téléchargements": "tabler:download",
    "À propos": "tabler:user-circle",
}

_PAGE_ORDER = [
    "Accueil",
    "Statistiques descriptives",
    "Modèles & clustering",
    "Évolution hebdomadaire",
    "Prédiction",
    "Téléchargements",
    "À propos",
]


def _sort_key(page):
    try:
        return (0, _PAGE_ORDER.index(page["name"]))
    except ValueError:
        return (1, page["name"])  # page inconnue de _PAGE_ORDER -- en fin de liste, alphabetique


def sidebar_links(pathname: str | None = None):
    pages = sorted(dash.page_registry.values(), key=_sort_key)
    items = []
    for page in pages:
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
