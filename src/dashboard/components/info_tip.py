# -*- coding: utf-8 -*-
"""Icone d'info (ⓘ) declenchant une infobulle -- a placer a cote d'un
controle pour en expliquer le role en une ligne, sans alourdir le libelle."""

import dash_mantine_components as dmc
from dash import html


def info_tip(text: str, position: str = "right"):
    return dmc.Tooltip(
        label=text,
        position=position,
        withArrow=True,
        multiline=True,
        w=240,
        openDelay=200,
        children=html.Span(
            "ⓘ",
            style={"color": "var(--mantine-color-blue-6)", "cursor": "help", "marginLeft": "6px", "fontSize": "13px"},
        ),
    )
