# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/theme.py
=============================================================================
ROLE :
    Source UNIQUE du theme visuel du dashboard -- theme DMC (MantineProvider)
    ET template Plotly, definis UNE SEULE FOIS ici et importes partout
    ailleurs. Le theme DMC ne s'applique PAS aux figures Plotly (systemes de
    rendu independants) -- d'ou un template Plotly separe, construit pour
    rester visuellement coherent avec le theme DMC (memes couleurs, meme
    esprit sombre).

    DIRECTION VISUELLE (2026-07-22) : postmoderniste, sombre, triade
    rouge/bleu/vert -- rupture volontaire avec le minimalisme clair du
    premier theme (Memphis Group / Weingart / deconstruction) applique au
    CHROME (cartes, en-tete, badges) ; les GRAPHIQUES restent lisibles et
    sobres (seule la palette change, jamais la structure des donnees).
    Signification semantique de la triade, coherente dans tout le
    dashboard (jamais arbitraire) :
      rouge  -> action principale / alerte / effet negatif
      bleu   -> chrome structurel (navigation, en-tete, liens)
      vert   -> statut positif / effet significatif / cluster valide

    Palette de categories (CATEGORY_COLORS) INCHANGEE : c'est un axe
    semantique different (identite visuelle de CHAQUE categorie de
    produit, reprise des notebooks) que la triade postmoderniste ne
    remplace pas -- seulement calibree pour rester lisible sur fond
    sombre.

UTILISATION :
    from src.dashboard.theme import DMC_THEME, PLOTLY_TEMPLATE_NAME, CATEGORY_COLORS
    dmc.MantineProvider(theme=DMC_THEME, forceColorScheme="dark", children=[...])
    fig.update_layout(template=PLOTLY_TEMPLATE_NAME)  # deja applique par defaut
=============================================================================
"""

import plotly.graph_objects as go
import plotly.io as pio

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE DE CATEGORIES -- reprise des notebooks (coherence projet entier),
# inchangee dans son principe -- axe semantique separe de la triade UI.
# ─────────────────────────────────────────────────────────────────────────────

CATEGORY_COLORS = {
    "pc_bureau": "#4f90e6",
    "pc_portables": "#2ecc71",
    "smartphones": "#eda100",
    "telephones_portables": "#8a6fe0",
    "televiseurs": "#e63946",
}

# ─────────────────────────────────────────────────────────────────────────────
# TRIADE POSTMODERNISTE -- rouge / bleu / vert, base sombre
# ─────────────────────────────────────────────────────────────────────────────

BG = "#0e0e12"
SURFACE = "#16161d"
SURFACE_2 = "#1e1e28"
BORDER = "#2a2a36"
TEXT = "#f2f0e9"
TEXT_MUT = "#a8a6b8"

RED = "#e63946"
BLUE = "#2d6cdf"
GREEN = "#2ecc71"

ACCENT = BLUE  # chrome structurel -- reste la couleur "primaire" DMC/plotly

# ─────────────────────────────────────────────────────────────────────────────
# THEME DMC (MantineProvider) -- defini UNE SEULE FOIS, force en mode sombre
# ─────────────────────────────────────────────────────────────────────────────

DMC_THEME = {
    "primaryColor": "blue",
    "colors": {
        # Rampes 10 teintes (clair -> fonce) requises par Mantine pour toute
        # couleur nommee dans "colors" -- generees a partir de RED/BLUE/GREEN.
        "blue": [
            "#e8eefc", "#c3d4f7", "#96b3f1", "#6992ea", "#4779e3",
            "#2d6cdf", "#2559c0", "#1e469f", "#17357d", "#0f2456",
        ],
        "red": [
            "#fbe9ea", "#f6c9cc", "#ef9fa4", "#e8747b", "#e14953",
            "#e63946", "#c92e3a", "#a92530", "#851d26", "#5e141b",
        ],
        "green": [
            "#e7faf0", "#c1f2d7", "#93e8ba", "#65de9d", "#43d489",
            "#2ecc71", "#25b563", "#1e9852", "#187b42", "#10542d",
        ],
    },
    "fontFamily": "'Space Mono', 'IBM Plex Mono', 'Courier New', monospace",
    "fontFamilyMonospace": "'Space Mono', 'IBM Plex Mono', 'Courier New', monospace",
    "defaultRadius": "sm",
    "headings": {"fontFamily": "'Archivo Black', 'Space Mono', sans-serif", "fontWeight": "400"},
    "components": {
        # Chrome volontairement heterogene (rayons/ombres) -- l'incoherence
        # controlee fait partie du langage postmoderniste (cf. custom.css
        # pour la rotation des couleurs/rayons par carte KPI, plus fine que
        # ce que defaultProps peut exprimer ici).
        "Card": {"defaultProps": {"withBorder": True, "radius": "sm", "shadow": "none"}},
        "Button": {"defaultProps": {"radius": "sm"}},
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE PLOTLY -- fond sombre coherent avec le theme DMC, triade en
# colorway par defaut (series categorielles sans couleur explicite),
# applique UNE SEULE FOIS. Les graphiques restent volontairement sobres :
# seule la palette change, jamais la structure (grille legere, police
# lisible, marges stables) -- l'expressivite postmoderniste vit dans le
# chrome (custom.css), pas dans les axes.
# ─────────────────────────────────────────────────────────────────────────────

PLOTLY_TEMPLATE_NAME = "mytek_dashboard"

_FONT = dict(family="'Space Mono', 'IBM Plex Mono', 'Courier New', monospace", size=13, color=TEXT_MUT)

_layout = go.Layout(
    font=_FONT,
    title=dict(font=dict(size=16, color=TEXT), x=0.0, xanchor="left"),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    # Triade d'abord, puis une teinte plus claire de chacune -- cf. §5 de la
    # direction visuelle (categorical series cycle through triad + tints).
    colorway=[RED, BLUE, GREEN, "#ef9fa4", "#96b3f1", "#93e8ba"],
    margin=dict(l=60, r=30, t=60, b=50),
    xaxis=dict(
        showgrid=False, zeroline=True, zerolinecolor=BORDER, zerolinewidth=1,
        showline=True, linecolor=BORDER, ticks="outside", tickcolor=BORDER,
        title=dict(font=dict(size=12, color=TEXT_MUT)), color=TEXT_MUT,
    ),
    yaxis=dict(
        showgrid=True, gridcolor=BORDER, gridwidth=1, zeroline=True, zerolinecolor=BORDER,
        showline=False, title=dict(font=dict(size=12, color=TEXT_MUT)), color=TEXT_MUT,
    ),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(size=12, color=TEXT_MUT)),
    hoverlabel=dict(bgcolor=SURFACE_2, font=dict(family=_FONT["family"], size=12, color=TEXT), bordercolor=BLUE),
    hovermode="closest",
    # Transition douce sur changement de donnees/relayout -- s'applique a
    # TOUTES les figures du dashboard sans y toucher individuellement,
    # puisque le template est le defaut global (pio.templates.default).
    transition=dict(duration=450, easing="cubic-in-out"),
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
