# -*- coding: utf-8 -*-
"""Carte KPI (chiffre-cle) reutilisable + ligne de cartes.

Animation de comptage (0 -> valeur) : purement cote client (MutationObserver
dans index_string, app.py) -- aucun aller-retour serveur. Declenchee des
qu'un noeud .kpi-value[data-target] apparait dans le DOM, donc fonctionne
aussi bien au premier chargement qu'apres un changement de categorie (le
contenu est re-rendu par callback)."""

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify


def kpi_card(
    label: str,
    value: str,
    icon: str = "tabler:chart-bar",
    note: str | None = None,
    color: str = "blue",
    raw_value: float | int | None = None,
    decimals: int = 0,
    suffix: str = "",
):
    """
    Une carte KPI unique : icone, valeur en grand, libelle, note optionnelle.

    Args:
        value: valeur DEJA formatee (francais), affichee telle quelle si
            `raw_value` n'est pas fourni -- comportement inchange par defaut.
        raw_value: si fourni, active l'animation de comptage cote client
            (0 -> raw_value sur ~900ms) ; `value` sert alors uniquement de
            repli pour les lecteurs sans JS (rendu initial cache par le JS).
        decimals, suffix: formatage du nombre anime (ex. decimals=1,
            suffix=" %").
    """
    if raw_value is not None:
        value_node = dmc.Text(
            html.Span(
                value, className="kpi-value",
                **{"data-target": raw_value, "data-decimals": decimals, "data-suffix": suffix},
            ),
            size="xl", fw=700, style={"fontVariantNumeric": "tabular-nums"},
        )
    else:
        value_node = dmc.Text(value, size="xl", fw=700, style={"fontVariantNumeric": "tabular-nums"})

    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.ThemeIcon(DashIconify(icon=icon, width=20), size=38, radius="md", color=color, variant="light"),
                    dmc.Stack(
                        [
                            dmc.Text(label, size="xs", c="dimmed", fw=500, tt="uppercase"),
                            value_node,
                        ],
                        gap=2,
                    ),
                ],
                gap="sm",
                align="center",
                wrap="nowrap",
            ),
            dmc.Text(note, size="xs", c="dimmed", mt=6) if note else None,
        ],
        p="md",
        className="kpi-card",
    )


def kpi_row(cards: list):
    """Ligne responsive de cartes KPI (dmc.SimpleGrid)."""
    return dmc.SimpleGrid(cols={"base": 1, "sm": 2, "lg": len(cards)}, spacing="md", children=cards)
