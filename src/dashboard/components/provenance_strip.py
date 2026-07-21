# -*- coding: utf-8 -*-
"""Bandeau de provenance/fraicheur des donnees -- visible sur CHAQUE page
(source, semaines couvertes, taille d'echantillon). Jamais omis : une
lecture sans savoir "sur quoi porte ce chiffre" est trompeuse."""

import dash_mantine_components as dmc
from dash_iconify import DashIconify


def provenance_strip(weeks: tuple, n: int | None = None, extra: str | None = None):
    if weeks:
        if len(weeks) == 1:
            weeks_txt = f"semaine {weeks[0]}"
        else:
            weeks_txt = f"semaines {min(weeks)} à {max(weeks)} ({len(weeks)} collectes)"
    else:
        weeks_txt = "aucune semaine disponible"

    parts = [f"Source : Mytek.tn — {weeks_txt}"]
    if n is not None:
        parts.append(f"n = {n:,}".replace(",", " ") + " produits")
    if extra:
        parts.append(extra)

    return dmc.Group(
        [
            DashIconify(icon="tabler:database", width=15, color="var(--mantine-color-dimmed)"),
            dmc.Text(" · ".join(parts), size="xs", c="dimmed"),
        ],
        gap=6,
        mb="md",
    )
