# -*- coding: utf-8 -*-
"""Selecteur de categorie -- reutilise en haut de chaque page qui analyse
une categorie a la fois (le projet n'analyse JAMAIS plusieurs categories
regroupees, cf. README du projet)."""

import dash_mantine_components as dmc

from src.utils.config import CATEGORY_LABELS, CATEGORY_ORDER


def category_selector(component_id: str, value: str | None = None):
    return dmc.SegmentedControl(
        id=component_id,
        value=value or CATEGORY_ORDER[0],
        data=[{"value": c, "label": CATEGORY_LABELS[c]} for c in CATEGORY_ORDER],
        fullWidth=True,
        mb="md",
    )
