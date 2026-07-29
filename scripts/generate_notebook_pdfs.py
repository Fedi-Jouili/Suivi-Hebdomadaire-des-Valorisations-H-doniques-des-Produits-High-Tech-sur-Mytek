# -*- coding: utf-8 -*-
"""
=============================================================================
Module : scripts/generate_notebook_pdfs.py
=============================================================================
ROLE :
    Convertit chaque notebook de notebooks/ en PDF, pour la page
    Telechargements du dashboard (src/dashboard/pages/downloads.py).

    HORS-LIGNE UNIQUEMENT -- jamais execute au runtime du dashboard/
    conteneur Docker (cf. requirements-deploy.txt, qui exclut deliberement
    jupyter/nbconvert/playwright pour garder l'image de production legere).
    Les PDF sont generes ICI, une fois (ou a chaque mise a jour des
    notebooks), et persistes sous reports/notebooks_pdf/ -- le dashboard ne
    fait ensuite que servir ces fichiers statiques.

    DEUX ETAPES, PAS UNE SEULE (cf. essais infructueux avant ce choix) :
      1. nbconvert --to html (n'a besoin ni de pandoc ni de LaTeX -- les
         cellules markdown sont rendues par le convertisseur `markdown`
         pur Python, contrairement a `--to pdf` qui EXIGE pandoc).
      2. Playwright (API SYNCHRONE, pas nbconvert --to webpdf) ouvre ce
         HTML et imprime en PDF via Chromium. `--to webpdf` a ete essaye
         en premier et echoue sur Windows (NotImplementedError : la boucle
         asyncio de nbconvert ne supporte pas les sous-processus sur cette
         plateforme) -- piloter Playwright nous-memes en API synchrone
         evite ce bug specifique a Windows.

UTILISATION :
    python scripts/generate_notebook_pdfs.py
    python scripts/generate_notebook_pdfs.py --notebook EDA_produits_technologiques.ipynb
=============================================================================
"""

import argparse
import logging
from pathlib import Path

from nbconvert import HTMLExporter
from playwright.sync_api import sync_playwright

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("scripts.generate_notebook_pdfs")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "notebooks_pdf"

_PDF_MARGIN = {"top": "15mm", "bottom": "15mm", "left": "12mm", "right": "12mm"}


def _notebook_to_html(notebook_path: Path) -> str:
    exporter = HTMLExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    body, _resources = exporter.from_filename(str(notebook_path))
    return body


def convert_notebook_to_pdf(notebook_path: Path, output_dir: Path, browser) -> Path:
    html = _notebook_to_html(notebook_path)
    pdf_path = output_dir / f"{notebook_path.stem}.pdf"

    # HTML ecrit sur disque (Playwright charge une URL file://, pas une
    # chaine directement) -- fichier temporaire, supprime apres conversion.
    tmp_html = output_dir / f"_tmp_{notebook_path.stem}.html"
    tmp_html.write_text(html, encoding="utf-8")
    try:
        page = browser.new_page()
        page.goto(tmp_html.resolve().as_uri())
        page.pdf(path=str(pdf_path), format="A4", print_background=True, margin=_PDF_MARGIN)
        page.close()
    finally:
        tmp_html.unlink(missing_ok=True)

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Convertit les notebooks en PDF (reports/notebooks_pdf/).")
    parser.add_argument("--notebook", type=str, default=None,
                        help="Ne convertir qu'un seul notebook (nom de fichier, ex: EDA_produits_technologiques.ipynb).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.notebook:
        notebooks = [NOTEBOOKS_DIR / args.notebook]
        missing = [p for p in notebooks if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Notebook(s) introuvable(s) : {missing}")
    else:
        notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
        if not notebooks:
            raise FileNotFoundError(f"Aucun notebook trouve sous {NOTEBOOKS_DIR}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for notebook_path in notebooks:
            logger.info(f"Conversion : {notebook_path.name}")
            pdf_path = convert_notebook_to_pdf(notebook_path, OUTPUT_DIR, browser)
            size_kb = pdf_path.stat().st_size / 1024
            logger.info(f"  -> {pdf_path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} Ko)")
        browser.close()

    logger.info(f"Termine. PDF ecrits sous {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
