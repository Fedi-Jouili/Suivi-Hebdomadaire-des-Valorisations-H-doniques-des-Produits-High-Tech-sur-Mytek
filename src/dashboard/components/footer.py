# -*- coding: utf-8 -*-
"""Pied de page -- provenance et methodologie, visible sur chaque page
(complement du bandeau de provenance par page, cf. provenance_strip.py) :
ce que c'est un tableau de bord scientifique credible doit toujours dire
d'ou viennent ses chiffres, meme quand on ne regarde pas une page precise."""

from datetime import date

import dash_mantine_components as dmc


def app_footer():
    return dmc.Group(
        [
            dmc.Text(
                "Source : Mytek.tn (collecte hebdomadaire) · Méthodologie : prix hédoniques "
                "(Lancaster 1966 / Rosen 1974) · Institut National de la Statistique, Tunisie",
                size="xs",
            ),
            dmc.Text(f"Généré le {date.today().strftime('%d/%m/%Y')}", size="xs"),
        ],
        justify="space-between",
        wrap="wrap",
        className="app-footer",
    )
