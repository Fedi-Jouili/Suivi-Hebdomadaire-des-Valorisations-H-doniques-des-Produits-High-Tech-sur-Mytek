# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/pages/downloads.py
=============================================================================
ROLE :
    Page 6 -- Telechargements. Regroupe, en un seul endroit, les 3 exports
    demandes explicitement par l'utilisateur (2026-07-29) :
      1. Les notebooks academiques, en PDF (pre-generes hors-ligne, cf.
         scripts/generate_notebook_pdfs.py -- jamais convertis a la volee,
         jupyter/nbconvert/playwright ne sont pas installes dans l'image
         de production, cf. requirements-deploy.txt).
      2. Les estimations par cluster x semaine, PAR CATEGORIE (prix reel
         geometrique + estime par les 3 modeles + erreur d'estimation --
         cf. src.models.weekly_report.marque_gamme_model_estimates),
         filtrees a la volee depuis reports/marque_gamme_estimations_
         hebdo.csv (jamais un fichier CSV supplementaire par categorie
         persiste sur disque -- eviterait la proliferation de CSV
         redondants deja identifiee comme un probleme sur ce depot).
      3. Les donnees produit, PAR CATEGORIE (une ligne par produit, toutes
         semaines poolees -- models/<categorie>/pooled_labeled.csv, meme
         fichier que le bouton "Donnees (CSV)" de la modale d'export de
         l'en-tete).

    Tous les liens pointent vers des routes Flask directes (app.py,
    section "PAGE TELECHARGEMENTS") -- de simples <a href> avec l'attribut
    "download", jamais un callback Dash : un telechargement de fichier n'a
    besoin d'aucun etat cote client, un lien direct est plus simple et
    plus robuste qu'un pattern-matching callback pour ce cas d'usage.
=============================================================================
"""

import dash
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from src.dashboard.components import section_header
from src.dashboard.data_loader import weekly_reports_available
from src.models.save_artifacts import MODELS_DIR
from src.models.weekly_report import REPORTS_DIR
from src.utils.config import CATEGORY_LABELS, CATEGORY_ORDER

dash.register_page(__name__, path="/telechargements", name="Téléchargements")

NOTEBOOKS_PDF_DIR = REPORTS_DIR / "notebooks_pdf"

# Titre + description courte (une ligne, jamais un long paragraphe) pour
# chaque notebook -- ordre de lecture recommande du projet (EDA -> Clustering
# -> Comparaison -> Segmentation -> Comparaison ML -> Evolution -> Transitions).
_NOTEBOOKS = [
    ("EDA_produits_technologiques", "Analyse exploratoire",
     "Volumes, prix, marques et caractéristiques techniques, par catégorie."),
    ("Clustering_produits_technologiques", "Clustering technique (N1)",
     "K-Means sur les seules caractéristiques techniques, par catégorie."),
    ("Comparaison_Approches_Clustering", "Comparaison des approches de clustering",
     "N1 (technique pur) vs N2 (marque × gamme) — ARI, silhouette, Kruskal-Wallis."),
    ("Segmentation_Prix_Clustering_produits_technologiques", "Segmentation marque × gamme (N2)",
     "Gammes de prix par marque, puis clustering technique au sein de chaque couple."),
    ("Comparaison_Modeles_ML_Clustering", "Comparaison des modèles",
     "Hedonic OLS / Ridge / Random Forest × aucun/N1/N2, hors-échantillon."),
    ("Evolution_Temporelle_Marche_Mytek", "Évolution temporelle du marché",
     "Indice de prix hédonique dans le temps (effet fixe semaine)."),
    ("Etude_Transitions_Clusters_Marque_Gamme", "Étude des transitions par cluster",
     "Prix réel vs valeur technique implicite, semaine par semaine, par cluster."),
]


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} Ko"
    return f"{num_bytes / (1024 * 1024):.1f} Mo"


def _download_row(icon: str, title: str, description: str, href: str, available: bool):
    return dmc.Group(
        [
            dmc.Group(
                [
                    DashIconify(icon=icon, width=22, color="var(--mantine-color-dimmed)"),
                    dmc.Stack(
                        [
                            dmc.Text(title, fw=600, size="sm"),
                            dmc.Text(description, size="xs", c="dimmed"),
                        ],
                        gap=0,
                    ),
                ],
                gap="sm", wrap="nowrap", style={"flex": 1, "minWidth": 0},
            ),
            dmc.Anchor(
                dmc.Button(
                    "Télécharger" if available else "Indisponible",
                    leftSection=DashIconify(icon="tabler:download", width=16),
                    size="xs", variant="light", disabled=not available,
                ),
                href=href if available else None,
            ),
        ],
        justify="space-between", wrap="nowrap", py="xs",
        style={"borderBottom": "1px solid var(--border)"},
    )


def _notebooks_section():
    rows = []
    for slug, title, description in _NOTEBOOKS:
        pdf_path = NOTEBOOKS_PDF_DIR / f"{slug}.pdf"
        available = pdf_path.exists()
        size_note = f" ({_format_size(pdf_path.stat().st_size)})" if available else ""
        rows.append(_download_row(
            "tabler:file-type-pdf", title, description + size_note,
            f"/download/notebook-pdf/{slug}.pdf", available,
        ))
    if not any((NOTEBOOKS_PDF_DIR / f"{slug}.pdf").exists() for slug, _, _ in _NOTEBOOKS):
        rows.insert(0, dmc.Alert(
            dmc.Text([
                "Aucun PDF généré pour l'instant — exécuter ",
                dmc.Code("python scripts/generate_notebook_pdfs.py", span=True),
                " (hors-ligne, jamais au démarrage du dashboard).",
            ], size="xs"),
            color="yellow", variant="light", icon=DashIconify(icon="tabler:alert-triangle"), mb="sm",
        ))
    return dmc.Card(dmc.Stack(rows, gap=0), p="lg", withBorder=True)


def _per_category_section(icon: str, title_prefix: str, route_prefix: str, available: bool, unavailable_note: str):
    if not available:
        return dmc.Alert(unavailable_note, color="yellow", variant="light",
                          icon=DashIconify(icon="tabler:alert-triangle"))
    rows = [
        _download_row(
            icon, CATEGORY_LABELS[cat], f"Catégorie « {CATEGORY_LABELS[cat]} »",
            f"{route_prefix}/{cat}.csv", True,
        )
        for cat in CATEGORY_ORDER
    ]
    return dmc.Card(dmc.Stack(rows, gap=0), p="lg", withBorder=True)


def layout():
    models_available = MODELS_DIR.exists() and any((MODELS_DIR / cat).exists() for cat in CATEGORY_ORDER)
    reports_available = weekly_reports_available()

    return dmc.Stack(
        [
            section_header(
                "Téléchargements",
                "Notebooks, données produit et estimations par cluster, exportables directement.",
            ),
            dmc.Title("Notebooks (PDF)", order=4, mb="xs"),
            _notebooks_section(),
            dmc.Title("Estimations par cluster — par catégorie", order=4, mt="lg", mb="xs"),
            dmc.Text(
                "Prix réel (moyenne géométrique) vs estimé par les 3 modèles, et erreur d'estimation, "
                "cluster × semaine.",
                size="xs", c="dimmed", mb="xs",
            ),
            _per_category_section(
                "tabler:chart-dots-3", "Estimations", "/download/estimations-cluster",
                reports_available,
                "Rapports indisponibles — exécuter `python -m src.models.weekly_report`.",
            ),
            dmc.Title("Données produit — par catégorie", order=4, mt="lg", mb="xs"),
            dmc.Text(
                "Une ligne par produit, toutes semaines poolées (caractéristiques techniques + segment).",
                size="xs", c="dimmed", mb="xs",
            ),
            _per_category_section(
                "tabler:table", "Produits", "/download/produits",
                models_available,
                "Modèles indisponibles — exécuter `python -m src.models.save_artifacts`.",
            ),
        ],
        maw=900, mx="auto",
        className="page-fade",
    )
