# -*- coding: utf-8 -*-
"""En-tete de section : titre affirmant le message-cle + sous-titre optionnel."""

import dash_mantine_components as dmc


def section_header(title: str, subtitle: str | None = None, order: int = 3):
    return dmc.Stack(
        [
            dmc.Title(title, order=order),
            dmc.Text(subtitle, size="sm", c="dimmed") if subtitle else None,
        ],
        gap=2,
        mb="sm",
    )
