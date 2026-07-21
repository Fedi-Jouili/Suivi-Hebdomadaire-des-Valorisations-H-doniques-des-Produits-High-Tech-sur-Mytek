# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/theme.py
=============================================================================
ROLE :
    Source UNIQUE du theme visuel du dashboard -- thema DMC (MantineProvider)
    ET template Plotly, definis UNE SEULE FOIS ici et importes partout
    ailleurs. Le theme DMC ne s'applique PAS aux figures Plotly (systemes de
    rendu independants) -- d'ou un template Plotly separe, construit pour
    rester visuellement cohérent avec le theme DMC (meme couleur d'accent,
    meme famille de police).

    Palette de categories reprise A L'IDENTIQUE de celle deja utilisee dans
    tous les notebooks du projet (EDA, Clustering, Segmentation...) --
    coherence visuelle entre notebooks et dashboard, pas une nouvelle
    palette inventee ici.

UTILISATION :
    from src.dashboard.theme import DMC_THEME, PLOTLY_TEMPLATE_NAME, CATEGORY_COLORS
    dmc.MantineProvider(theme=DMC_THEME, children=[...])
    fig.update_layout(template=PLOTLY_TEMPLATE_NAME)  # ou deja applique par defaut
=============================================================================
"""

import plotly.graph_objects as go
import plotly.io as pio

# ─────────────────────────────────────────────────────────────────────────────
# COULEURS -- reprises des notebooks (coherence visuelle projet entier)
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "pc_bureau": "#2a78d6",
    "pc_portables": "#1baf7a",
    "smartphones": "#eda100",
    "telephones_portables": "#4a3aa7",
    "televiseurs": "#e34948",
}

ACCENT = "#2a78d6"  # bleu -- couleur d'accent unique du dashboard (primaryColor DMC)
GRAY_SCALE = [
    "#f8f9fa", "#eef0f2", "#dfe3e6", "#c7ccd1", "#a3aab2",
    "#7c848d", "#5b636c", "#3f464e", "#292e33", "#16191c",
]

# ─────────────────────────────────────────────────────────────────────────────
# THEME DMC (MantineProvider) -- defini UNE SEULE FOIS
# ─────────────────────────────────────────────────────────────────────────────

DMC_THEME = {
    "primaryColor": "blue",
    "colors": {
        # Rampe bleue calibree sur ACCENT (#2a78d6), 10 teintes clair->fonce,
        # requises par Mantine pour toute couleur nommee dans "colors".
        "blue": [
            "#eaf2fc", "#cfe1f8", "#a3c6f2", "#75a9ec", "#4f90e6",
            "#2a78d6", "#2266ba", "#1a5296", "#123c6f", "#0a2748",
        ],
    },
    "fontFamily": "'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif",
    "fontFamilyMonospace": "'JetBrains Mono', 'Courier New', monospace",
    "defaultRadius": "md",
    "headings": {"fontFamily": "'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif", "fontWeight": "650"},
    "components": {
        "Card": {"defaultProps": {"withBorder": True, "radius": "md", "shadow": "xs"}},
        "Button": {"defaultProps": {"radius": "md"}},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE PLOTLY -- coherent avec le theme DMC, applique UNE SEULE FOIS
# (quasi-monochrome : un accent + gris, grille legere, marges/police stables)
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE_NAME = "mytek_dashboard"

_FONT = dict(family="Inter, -apple-system, 'Segoe UI', Roboto, sans-serif", size=13, color=GRAY_SCALE[7])

_layout = go.Layout(
    font=_FONT,
    title=dict(font=dict(size=16, weight=600, color=GRAY_SCALE[8]), x=0.0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=[ACCENT, "#eda100", "#1baf7a", "#e34948", "#4a3aa7", GRAY_SCALE[5]],
    margin=dict(l=60, r=30, t=60, b=50),
    xaxis=dict(
        showgrid=False, zeroline=True, zerolinecolor=GRAY_SCALE[2], zerolinewidth=1,
        showline=True, linecolor=GRAY_SCALE[3], ticks="outside", tickcolor=GRAY_SCALE[3],
        title=dict(font=dict(size=12, color=GRAY_SCALE[6])),
    ),
    yaxis=dict(
        showgrid=True, gridcolor=GRAY_SCALE[1], gridwidth=1, zeroline=True, zerolinecolor=GRAY_SCALE[2],
        showline=False, title=dict(font=dict(size=12, color=GRAY_SCALE[6])),
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=12)),
    hoverlabel=dict(bgcolor="white", font=dict(family=_FONT["family"], size=12), bordercolor=GRAY_SCALE[3]),
    hovermode="closest",
)

pio.templates[PLOTLY_TEMPLATE_NAME] = go.layout.Template(layout=_layout)
pio.templates.default = PLOTLY_TEMPLATE_NAME

# Config Plotly par defaut (modebar reduite -- cf. consignes dashboard) :
# a passer explicitement a chaque dcc.Graph(config=GRAPH_CONFIG).
GRAPH_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d", "lasso2d", "autoScale2d", "hoverCompareCartesian",
        "hoverClosestCartesian", "toggleSpikelines",
    ],
    "responsive": True,
}
