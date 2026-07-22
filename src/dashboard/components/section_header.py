# -*- coding: utf-8 -*-
"""En-tete de section : titre affirmant le message-cle + sous-titre optionnel.

Leger desaxage postmoderne UNIQUEMENT sur le titre principal de page
(order=3) -- jamais sur les sous-titres de section (order>=4, souvent
immediatement suivis d'un graphique/tableau : la rotation reste un accent
de chrome, jamais un signal sur les donnees)."""

import dash_mantine_components as dmc


def section_header(title: str, subtitle: str | None = None, order: int = 3):
    return dmc.Stack(
        [
            dmc.Title(title, order=order, className="section-tilt" if order == 3 else None),
            dmc.Text(subtitle, size="sm", c="dimmed") if subtitle else None,
        ],
        gap=2,
        mb="sm",
    )
