# -*- coding: utf-8 -*-
"""Carte KPI (chiffre-cle) reutilisable + ligne de cartes."""

import dash_mantine_components as dmc
from dash_iconify import DashIconify


def kpi_card(label: str, value: str, icon: str = "tabler:chart-bar", note: str | None = None, color: str = "blue"):
    """Une carte KPI unique : icone, valeur en grand, libelle, note optionnelle."""
    return dmc.Card(
        [
            dmc.Group(
                [
                    dmc.ThemeIcon(DashIconify(icon=icon, width=20), size=38, radius="md", color=color, variant="light"),
                    dmc.Stack(
                        [
                            dmc.Text(label, size="xs", c="dimmed", fw=500, tt="uppercase"),
                            dmc.Text(value, size="xl", fw=700, style={"fontVariantNumeric": "tabular-nums"}),
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
    )


def kpi_row(cards: list):
    """Ligne responsive de cartes KPI (dmc.SimpleGrid)."""
    return dmc.SimpleGrid(cols={"base": 1, "sm": 2, "lg": len(cards)}, spacing="md", children=cards)
