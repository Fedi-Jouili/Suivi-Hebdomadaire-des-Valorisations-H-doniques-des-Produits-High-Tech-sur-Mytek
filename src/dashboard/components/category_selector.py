# -*- coding: utf-8 -*-
"""Selecteur de categorie -- reutilise en haut de chaque page qui analyse
une categorie a la fois (le projet n'analyse JAMAIS plusieurs categories
regroupees, cf. README du projet)."""

import dash_mantine_components as dmc

from src.dashboard.components.info_tip import info_tip
from src.utils.config import CATEGORY_LABELS, CATEGORY_ORDER


def category_selector(component_id: str, value: str | None = None):
    return dmc.Stack(
        [
            dmc.Group(
                [
                    dmc.Text("Catégorie", size="xs", c="dimmed", fw=500, tt="uppercase"),
                    info_tip(
                        "Chaque catégorie a ses propres échelles de prix et caractéristiques pertinentes — "
                        "l'analyse ne regroupe jamais plusieurs catégories."
                    ),
                ],
                gap=0,
            ),
            dmc.SegmentedControl(
                id=component_id,
                value=value or CATEGORY_ORDER[0],
                data=[{"value": c, "label": CATEGORY_LABELS[c]} for c in CATEGORY_ORDER],
                fullWidth=True,
            ),
        ],
        gap=4,
        mb="md",
    )
