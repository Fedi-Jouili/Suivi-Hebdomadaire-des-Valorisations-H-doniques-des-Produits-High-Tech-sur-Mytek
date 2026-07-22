# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/app.py
=============================================================================
ROLE :
    Point d'entree du dashboard -- App shell (AppShell : en-tete + barre
    laterale + zone de contenu + pied de page), MantineProvider avec le
    theme unique (theme.py), framework multi-pages Dash (dash.page_container).

    En-tete : badge de provenance (semaines couvertes), date de derniere
    mise a jour des donnees, bouton d'export (modale listant les fichiers
    telechargeables par categorie -- donnees etiquetees + metriques). Pied
    de page : rappel de la source et de la methodologie sur chaque page.

UTILISATION :
    python -m src.dashboard.app
    -> http://127.0.0.1:8050

PREREQUIS :
    python -m src.models.save_artifacts   (produit models/, requis par les
    pages Modeles & clustering / Prediction -- la page Statistiques
    descriptives fonctionne sans, sur data/processed/ seul).
=============================================================================
"""

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import ALL, Dash, Input, Output, State, callback, ctx, dcc, no_update
from dash_iconify import DashIconify

from src.dashboard import theme as _theme  # noqa: F401 -- applique le template Plotly par defaut a l'import
from src.dashboard.components.footer import app_footer
from src.dashboard.components.nav import sidebar_links
from src.dashboard.data_loader import available_weeks, last_data_update
from src.dashboard.theme import DMC_THEME
from src.models.save_artifacts import MODELS_DIR
from src.utils.config import CATEGORY_LABELS, CATEGORY_ORDER

# ─────────────────────────────────────────────────────────────────────────────
# <head> : polices Google Fonts + scripts client purs (pas de round-trip Dash)
# ─────────────────────────────────────────────────────────────────────────────

# dash-ag-grid >= 32 (theming API AG Grid v33+) n'embarque plus le CSS des
# themes classiques (className="ag-theme-quartz-dark") dans son bundle JS --
# il faut le charger explicitement (CDN jsdelivr, URL fournie par le paquet
# lui-meme et alignee sur sa version exacte) + activer dashGridOptions=
# {"theme": "legacy"} sur chaque AgGrid (cf. pages/*.py) pour que le
# className soit pris en compte plutot qu'ignore par la nouvelle API JS.
_AG_GRID_THEME_LINKS = "\n        ".join(
    f'<link rel="stylesheet" href="{url}">' for url in (dag.themes.BASE, dag.themes.QUARTZ)
)

_INDEX_STRING = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Archivo+Black&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
        __AG_GRID_THEME_LINKS__
    </head>""".replace("__AG_GRID_THEME_LINKS__", _AG_GRID_THEME_LINKS) + """
    <body>
        <div id="scroll-progress-bar"></div>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <script>
            // Barre de progression de defilement -- purement visuel, aucun
            // callback Dash necessaire (round-trip serveur inutile pour ca).
            window.addEventListener('scroll', function () {
                var bar = document.getElementById('scroll-progress-bar');
                if (!bar) return;
                var h = document.documentElement;
                var scrolled = (h.scrollTop) / ((h.scrollHeight - h.clientHeight) || 1) * 100;
                bar.style.width = scrolled + '%';
            }, { passive: true });

            // Animation de comptage des cartes KPI (0 -> valeur cible) --
            // MutationObserver plutot qu'un clientside_callback Dash : le
            // contenu des cartes KPI est (re)rendu par des callbacks Python
            // differents selon la page (changement de categorie inclus),
            // un seul observateur global couvre tous les cas sans qu'il
            // faille cabler un id de sortie factice par page.
            function animateKpiValue(el) {
                var target = parseFloat(el.getAttribute('data-target'));
                if (isNaN(target)) return;
                var decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
                var suffix = el.getAttribute('data-suffix') || '';
                var start = null;
                var duration = 900;
                function step(ts) {
                    if (!start) start = ts;
                    var p = Math.min((ts - start) / duration, 1);
                    var eased = 1 - Math.pow(1 - p, 3);
                    var value = eased * target;
                    el.textContent = value.toLocaleString('fr-FR', {
                        minimumFractionDigits: decimals, maximumFractionDigits: decimals
                    }) + suffix;
                    if (p < 1) requestAnimationFrame(step);
                }
                requestAnimationFrame(step);
            }
            function scanForKpiValues(root) {
                (root.querySelectorAll ? root.querySelectorAll('.kpi-value[data-target]:not(.counted)') : [])
                    .forEach(function (el) {
                        el.classList.add('counted');
                        animateKpiValue(el);
                    });
            }
            document.addEventListener('DOMContentLoaded', function () {
                scanForKpiValues(document.body);
                new MutationObserver(function (mutations) {
                    mutations.forEach(function (m) {
                        m.addedNodes.forEach(function (node) {
                            if (node.nodeType === 1) scanForKpiValues(node);
                        });
                    });
                }).observe(document.body, { childList: true, subtree: true });
            });
        </script>
    </body>
</html>"""

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="Hédonique Mytek.tn — INS Tunisie",
)
app.index_string = _INDEX_STRING
server = app.server  # pour un futur deploiement WSGI (gunicorn/waitress)


# ─────────────────────────────────────────────────────────────────────────────
# EN-TETE : badge de provenance + derniere mise a jour + export
# ─────────────────────────────────────────────────────────────────────────────

def _provenance_badge():
    weeks = available_weeks()
    label = f"Mytek.tn · S{min(weeks)}–S{max(weeks)}" if weeks else "Mytek.tn"
    return dmc.Badge(label, variant="light", className="app-badge", radius="sm")


def _export_modal():
    rows = []
    for cat in CATEGORY_ORDER:
        rows.append(
            dmc.Group(
                [
                    dmc.Text(CATEGORY_LABELS[cat], style={"flex": 1}, fw=500),
                    dmc.Button(
                        "Données (CSV)", size="xs", variant="light",
                        leftSection=DashIconify(icon="tabler:table", width=14),
                        id={"type": "export-btn", "category": cat, "kind": "csv"},
                    ),
                    dmc.Button(
                        "Métriques (JSON)", size="xs", variant="light",
                        leftSection=DashIconify(icon="tabler:file-code", width=14),
                        id={"type": "export-btn", "category": cat, "kind": "json"},
                    ),
                ],
                justify="space-between", wrap="nowrap",
            )
        )
    return dmc.Modal(
        id="export-modal",
        title="Exporter les données",
        size="lg",
        children=[
            dmc.Text(
                "Données étiquetées (gamme, segments) et métriques des modèles, par catégorie — "
                "produites par src/models/save_artifacts.py.",
                size="xs", c="dimmed", mb="sm",
            ),
            dmc.Stack(rows, gap="xs"),
        ],
    )


_header = dmc.AppShellHeader(
    dmc.Group(
        [
            dmc.Burger(id="burger", size="sm", hiddenFrom="sm", opened=False),
            dmc.ThemeIcon(DashIconify(icon="tabler:chart-histogram", width=22), size=36, radius="sm", variant="light", color="red"),
            dmc.Title(
                "HÉDONIQUE MYTEK.TN", order=3, tt="uppercase",
                style={"fontSize": "var(--fs-lg)", "letterSpacing": "-0.02em"},
            ),
            dmc.Group(
                [
                    _provenance_badge(),
                    dmc.Text(
                        f"Mise à jour : {last_data_update() or '—'}", size="xs", c="dimmed", visibleFrom="md",
                    ),
                    dmc.Button(
                        "Exporter", id="export-open-btn", size="xs", variant="subtle", color="red",
                        leftSection=DashIconify(icon="tabler:download", width=15),
                    ),
                ],
                ml="auto", gap="sm", wrap="nowrap",
            ),
        ],
        h="100%", px="md", wrap="nowrap",
    )
)

_navbar = dmc.AppShellNavbar(id="navbar-content", children=[], p="md")

_main = dmc.AppShellMain(
    dmc.Container(
        [dash.page_container, app_footer()],
        fluid=True, px="md", py="md",
    )
)

_shell = dmc.AppShell(
    [_header, _navbar, _main],
    header={"height": 60},
    navbar={"width": 260, "breakpoint": "sm", "collapsed": {"mobile": True}},
    padding="md",
    id="appshell",
)

app.layout = dmc.MantineProvider(
    theme=DMC_THEME,
    forceColorScheme="dark",
    children=[dcc.Location(id="url"), _shell, _export_modal(), dcc.Download(id="export-download")],
)


@callback(
    Output("appshell", "navbar"),
    Input("burger", "opened"),
    State("appshell", "navbar"),
)
def _toggle_navbar(opened, navbar):
    navbar["collapsed"] = {"mobile": not opened}
    return navbar


@callback(Output("navbar-content", "children"), Input("url", "pathname"))
def _update_nav(pathname):
    return sidebar_links(pathname)


@callback(Output("export-modal", "opened"), Input("export-open-btn", "n_clicks"), prevent_initial_call=True)
def _open_export_modal(n_clicks):
    return True


@callback(
    Output("export-download", "data"),
    Input({"type": "export-btn", "category": ALL, "kind": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def _export_file(_n_clicks_list):
    """Telechargement direct des artefacts (donnees etiquetees / metriques)
    produits par save_artifacts.py -- lecture seule, aucun recalcul ici."""
    triggered = ctx.triggered_id
    if not triggered or not ctx.triggered[0]["value"]:
        return no_update
    category, kind = triggered["category"], triggered["kind"]
    filename = "pooled_labeled.csv" if kind == "csv" else "metrics.json"
    path = MODELS_DIR / category / filename
    if not path.exists():
        return no_update
    return dcc.send_file(str(path))


if __name__ == "__main__":
    app.run(debug=True)
