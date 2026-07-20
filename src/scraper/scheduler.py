# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/scraper/scheduler.py
=============================================================================
RÔLE :
    Orchestrateur du pipeline de scraping Mytek.tn, sur un nombre de
    semaines non borné (--week accepte n'importe quel entier >= 1 --
    ajouter une semaine 4, 5... ne demande aucune modification de ce
    fichier, cf. §Historique).

    Fonctionnalités :
    - Exécution manuelle d'une semaine  : --week N (N >= 1)
    - Exécution automatique de plusieurs semaines : --all --end-week N
                                          (délai de 7 jours entre chaque)
    - Reprise sur checkpoint             : reprend la catégorie/le produit où
                                          le processus s'est interrompu
    - Logs horodatés par semaine         : logs/scraping_week_X.log
    - Sorties JSON structurées           : data/raw/week_X/

    Historique : semaines 1-3 collectées avec une version antérieure qui
    limitait --week/--start-from à {1,2,3} (choices= figé dans argparse) --
    limite retirée pour permettre une semaine 4 (et au-delà) sans toucher
    au code à chaque nouvelle semaine, cf. le reste du pipeline
    (pipeline.py, split.py, Evolution_Temporelle_Marche_Mytek.ipynb) qui
    découvre déjà dynamiquement le nombre de semaines disponibles.

UTILISATION :
    python src/scraper/scheduler.py --week 1
    python src/scraper/scheduler.py --week 4 --reset
    python src/scraper/scheduler.py --all --end-week 3
    python src/scraper/scheduler.py --all --start-from 2 --end-week 4
    python src/scraper/scheduler.py --week 1 --debug
=============================================================================
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import signal

from . import CATEGORIES, DATA_RAW_DIR, DELAY_MIN, DELAY_MAX
INTER_CAT_DELAY_MIN = 8.0
INTER_CAT_DELAY_MAX = 15.0
from .spider import MytekSpider
from .parser import ProductParser

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DU SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────

# Répertoire des fichiers de log
LOGS_DIR = Path("logs")

# Fichier de reprise (checkpoint d'avancement global)
CHECKPOINT_FILE = Path("checkpoint.json")

# Sauvegarde partielle tous les N produits (pour limiter les pertes en cas
# d'interruption au milieu d'une catégorie)
PARTIAL_SAVE_INTERVAL = 10

# Délai entre les semaines en mode automatique
WEEK_DELAY_SECONDS = 7 * 24 * 3600  # 7 jours


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def _setup_file_logging(week_number: int) -> Path:
    """
    Ajoute un FileHandler sur le logger racine "scraper" pour écrire
    les logs de la semaine dans logs/scraping_week_X.log.

    Le StreamHandler vers stdout est déjà installé par __init__.py ;
    on n'y touche pas afin de conserver la sortie console.

    Returns : Path du fichier de log créé.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"scraping_week_{week_number}.log"

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_scraper_logger = logging.getLogger("scraper")

    # Éviter d'ajouter plusieurs fois le même handler si la fonction
    # est appelée plusieurs fois dans la même session Python
    already_attached = any(
        isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None)
        == str(log_path.resolve())
        for h in root_scraper_logger.handlers
    )
    if not already_attached:
        root_scraper_logger.addHandler(file_handler)

    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT (REPRISE SUR ERREUR)
# ─────────────────────────────────────────────────────────────────────────────

def _load_checkpoint() -> dict:
    """
    Charge l'état d'avancement global depuis CHECKPOINT_FILE.

    Structure :
    {
      "week_1": {
        "status": "in_progress" | "done" | "partial",
        "started_at": "<ISO>",
        "finished_at": "<ISO>" | null,
        "total_products": <int>,
        "categories": {
          "<cat_key>": {
            "status": "done" | "in_progress" | "error",
            "product_urls": [...],        # URLs crawlées
            "parsed_count": <int>,        # produits parsés jusqu'ici
            "count": <int>,               # produits dans le fichier final
            "output_file": "<path>",      # fichier JSON final
            "partial_file": "<path>",     # fichier JSON partiel (si reprise)
            "crawled_at": "<ISO>",
            "parsed_at": "<ISO>",
            "error": "<msg>"              # si status == "error"
          }
        }
      }
    }
    """
    logger = logging.getLogger("scraper.scheduler")
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                f"Checkpoint illisible ({exc}). Démarrage à zéro."
            )
    return {}


def _save_checkpoint(checkpoint: dict) -> None:
    """
    Sauvegarde atomique du checkpoint :
    écriture dans un fichier temporaire puis renommage,
    pour éviter la corruption en cas d'interruption pendant l'écriture.
    """
    logger = logging.getLogger("scraper.scheduler")
    tmp = CHECKPOINT_FILE.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(checkpoint, fh, ensure_ascii=False, indent=2)
        tmp.replace(CHECKPOINT_FILE)
    except OSError as exc:
        logger.error(f"Impossible de sauvegarder le checkpoint : {exc}")


def reset_checkpoint(week_number: Optional[int] = None) -> None:
    """
    Réinitialise le checkpoint pour une semaine donnée ou entièrement.
    Utile pour forcer un re-scraping complet.

    Args:
        week_number : numéro de la semaine à effacer (None = tout effacer).
    """
    logger = logging.getLogger("scraper.scheduler")
    checkpoint = _load_checkpoint()

    if week_number is not None:
        week_key = f"week_{week_number}"
        if week_key in checkpoint:
            # Supprimer également les fichiers partiels éventuels
            for cat_data in checkpoint[week_key].get("categories", {}).values():
                partial = cat_data.get("partial_file")
                if partial:
                    try:
                        Path(partial).unlink(missing_ok=True)
                    except OSError:
                        pass
            del checkpoint[week_key]
            _save_checkpoint(checkpoint)
            logger.info(f"Checkpoint semaine {week_number} réinitialisé.")
        else:
            logger.info(f"Aucun checkpoint pour la semaine {week_number}.")
    else:
        _save_checkpoint({})
        logger.info("Checkpoint entièrement réinitialisé.")


# ─────────────────────────────────────────────────────────────────────────────
# SAUVEGARDE JSON
# ─────────────────────────────────────────────────────────────────────────────

def _get_output_dir(week_number: int) -> Path:
    """Crée et retourne le répertoire data/raw/week_X/."""
    out_dir = DATA_RAW_DIR / f"week_{week_number}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_json(products: list, filepath: Path) -> bool:
    """
    Écrit la liste de produits dans filepath (format JSON).
    Retourne True si succès, False sinon.
    """
    logger = logging.getLogger("scraper.scheduler")
    try:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(products, fh, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        logger.error(f"Erreur écriture {filepath} : {exc}")
        return False


def _final_filename(output_dir: Path, cat_key: str) -> Path:
    """Génère le nom du fichier JSON final pour une catégorie."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return output_dir / f"produits_{cat_key}_{ts}.json"


def _partial_filename(output_dir: Path, cat_key: str) -> Path:
    """Nom du fichier JSON partiel (en cours de constitution)."""
    return output_dir / f"produits_{cat_key}_partial.json"


# ─────────────────────────────────────────────────────────────────────────────
# CŒUR DU SCRAPING D'UNE CATÉGORIE
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_category(
    cat_key: str,
    cat_info: dict,
    week_number: int,
    output_dir: Path,
    spider: MytekSpider,
    parser: ProductParser,
    cat_checkpoint: dict,
    checkpoint: dict,
    week_key: str,
) -> tuple[list, dict]:
    """
    Gère le crawling + parsing d'une catégorie avec reprise fine.

    La reprise fonctionne à deux niveaux :
      1. Si le crawling était terminé : les URLs sont relues depuis le
         checkpoint (pas de re-crawl du listing).
      2. Si le parsing était en cours : on repart du produit N+1 grâce
         à "parsed_count" et au fichier partiel sauvegardé.

    Args:
        cat_key        : clé de la catégorie (ex. "pc_portables")
        cat_info       : dict de la catégorie (url, label, filename)
        week_number    : semaine en cours (1, 2 ou 3)
        output_dir     : Path du dossier de sortie week_X
        spider         : instance MytekSpider réutilisée
        parser         : instance ProductParser réutilisée
        cat_checkpoint : sous-dict checkpoint pour cette catégorie
        checkpoint     : checkpoint global (pour sauvegarde atomique)
        week_key       : clé "week_X" dans checkpoint

    Returns:
        (all_products, cat_checkpoint) mis à jour.
    """
    logger = logging.getLogger("scraper.scheduler")
    partial_path = _partial_filename(output_dir, cat_key)

    # ── PHASE 1 : CRAWLING DES URLS ──────────────────────────────────────────
    if "product_urls" in cat_checkpoint:
        # Reprise : URLs déjà crawlées
        product_urls = cat_checkpoint["product_urls"]
        logger.info(
            f"  [REPRISE] URLs du listing récupérées du checkpoint "
            f"({len(product_urls)} URLs)."
        )
    else:
        logger.info(f"  Phase 1 — Crawling du listing {cat_info['label']}...")
        product_urls = spider.crawl_category(cat_key, cat_info["url"])

        if not product_urls:
            logger.warning(f"  Aucune URL produit trouvée pour {cat_key}. Skip.")
            cat_checkpoint.update(
                status="done",
                count=0,
                crawled_at=datetime.now().isoformat(),
            )
            _save_checkpoint(checkpoint)
            return [], cat_checkpoint

        cat_checkpoint["product_urls"] = product_urls
        cat_checkpoint["crawled_at"] = datetime.now().isoformat()
        checkpoint[week_key]["categories"][cat_key] = cat_checkpoint
        _save_checkpoint(checkpoint)
        logger.info(f"  {len(product_urls)} URL(s) crawlée(s) et sauvegardées.")

    # ── PHASE 2 : PARSING PRODUIT PAR PRODUIT ────────────────────────────────
    already_parsed = cat_checkpoint.get("parsed_count", 0)

    # Charger les produits déjà parsés depuis le fichier partiel
    all_products: list = []
    if already_parsed > 0 and partial_path.exists():
        try:
            with open(partial_path, "r", encoding="utf-8") as fh:
                all_products = json.load(fh)
            logger.info(
                f"  [REPRISE] {len(all_products)} produit(s) rechargés "
                f"depuis le fichier partiel."
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                f"  Fichier partiel illisible ({exc}). Reparsing depuis 0."
            )
            all_products = []
            already_parsed = 0

    urls_to_parse = product_urls[already_parsed:]
    total_urls = len(product_urls)

    logger.info(
        f"  Phase 2 — Parsing de {len(urls_to_parse)}/{total_urls} produit(s) "
        f"(départ à l'index {already_parsed})..."
    )

    for i, url in enumerate(urls_to_parse, start=already_parsed + 1):
        logger.info(f"  [{i}/{total_urls}] {url}")

        try:
            product = parser.parse_product(url, cat_key, week_number)
            if product:
                all_products.append(product)
        except Exception as exc:
            logger.error(f"  Erreur produit {url} : {exc}")
            # On continue sur le produit suivant

        # Pause entre chaque requête
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        # Sauvegarde partielle périodique
        current_count = i - already_parsed
        if current_count % PARTIAL_SAVE_INTERVAL == 0:
            if _save_json(all_products, partial_path):
                cat_checkpoint["parsed_count"] = i
                cat_checkpoint["partial_file"] = str(partial_path)
                checkpoint[week_key]["categories"][cat_key] = cat_checkpoint
                _save_checkpoint(checkpoint)
                logger.debug(
                    f"  Sauvegarde partielle : {len(all_products)} produit(s) "
                    f"→ {partial_path.name}"
                )

    # ── PHASE 3 : SAUVEGARDE FINALE ──────────────────────────────────────────
    final_path = _final_filename(output_dir, cat_key)
    if _save_json(all_products, final_path):
        logger.info(
            f"  ✔ {len(all_products)} produit(s) sauvegardés → {final_path.name}"
        )
    else:
        logger.error(f"  Sauvegarde finale échouée pour {cat_key}.")

    # Nettoyage du fichier partiel devenu inutile
    try:
        partial_path.unlink(missing_ok=True)
    except OSError:
        pass

    cat_checkpoint.update(
        status="done",
        count=len(all_products),
        parsed_at=datetime.now().isoformat(),
        output_file=str(final_path),
        parsed_count=total_urls,
    )
    cat_checkpoint.pop("partial_file", None)

    return all_products, cat_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPING D'UNE SEMAINE COMPLÈTE
# ─────────────────────────────────────────────────────────────────────────────

def scrape_week(week_number: int, category: Optional[str] = None) -> tuple[int, list[str]]:
    """
    Orchestre le scraping complet d'une semaine.

    Parcourt toutes les catégories de CATEGORIES dans l'ordre,
    en sautant celles déjà terminées selon le checkpoint.
    Si une catégorie échoue, l'erreur est loguée et le scraping
    continue sur la suivante.

    Args:
        week_number : int — numéro de la semaine (1, 2 ou 3)
        category    : si fourni, ne scrape que cette catégorie (clé de
                      CATEGORIES) au lieu de toutes -- utile pour
                      relancer une seule catégorie sans toucher au
                      checkpoint des autres.

    Returns:
        (total_products, failed_categories)
        - total_products    : nombre total de produits collectés
        - failed_categories : liste des clés de catégories en erreur
    """
    logger = logging.getLogger("scraper.scheduler")

    # ── Setup ─────────────────────────────────────────────────────────────────
    log_path = _setup_file_logging(week_number)
    output_dir = _get_output_dir(week_number)

    logger.info("=" * 70)
    logger.info(f"  DÉBUT SCRAPING — SEMAINE {week_number}")
    logger.info(f"  Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Sortie    : {output_dir}")
    logger.info(f"  Log       : {log_path}")
    logger.info("=" * 70)

    # ── Checkpoint ────────────────────────────────────────────────────────────
    checkpoint = _load_checkpoint()
    week_key = f"week_{week_number}"

    if week_key not in checkpoint:
        checkpoint[week_key] = {
            "status": "in_progress",
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "total_products": 0,
            "categories": {},
        }
        _save_checkpoint(checkpoint)

    week_cp = checkpoint[week_key]
    week_cp["status"] = "in_progress"

    # ── Instanciation des objets (une seule fois pour toutes les catégories) ──
    spider = MytekSpider()
    parser = ProductParser()

    total_products = 0
    failed_categories: list[str] = []

    # ── Boucle sur les catégories ─────────────────────────────────────────────
    categories_to_scrape = (
        {category: CATEGORIES[category]} if category else CATEGORIES
    )
    for cat_key, cat_info in categories_to_scrape.items():
        cat_cp = week_cp["categories"].get(cat_key, {})

        # ── Catégorie déjà terminée ──────────────────────────────────────────
        if cat_cp.get("status") == "done":
            done_count = cat_cp.get("count", 0)
            logger.info(
                f"[SKIP] {cat_key} — déjà terminée "
                f"({done_count} produit(s), checkpoint)."
            )
            total_products += done_count
            continue

        # ── Scraping de la catégorie ─────────────────────────────────────────
        logger.info(f"\n{'─' * 60}")
        logger.info(f"Catégorie : {cat_key} ({cat_info['label']})")
        logger.info(f"URL       : {cat_info['url']}")

        try:
            products, cat_cp = _scrape_category(
                cat_key=cat_key,
                cat_info=cat_info,
                week_number=week_number,
                output_dir=output_dir,
                spider=spider,
                parser=parser,
                cat_checkpoint=cat_cp,
                checkpoint=checkpoint,
                week_key=week_key,
            )
            total_products += len(products)
            week_cp["categories"][cat_key] = cat_cp
            _save_checkpoint(checkpoint)
            logger.info(
                f"  Catégorie {cat_key} terminée : {len(products)} produit(s)."
            )

        except KeyboardInterrupt:
            # Propagation pour que main() puisse intercepter et quitter proprement
            week_cp["categories"][cat_key] = cat_cp
            _save_checkpoint(checkpoint)
            raise

        except Exception as exc:
            logger.error(
                f"  ERREUR CATÉGORIE {cat_key} : {exc}",
                exc_info=True,
            )
            failed_categories.append(cat_key)
            cat_cp.update(
                status="error",
                error=str(exc),
                error_at=datetime.now().isoformat(),
            )
            week_cp["categories"][cat_key] = cat_cp
            _save_checkpoint(checkpoint)
            logger.info("  Poursuite sur la catégorie suivante...")

        # Pause inter-catégorie (plus longue pour respecter le serveur)
        if cat_key != list(categories_to_scrape.keys())[-1]:
            inter_delay = random.uniform(INTER_CAT_DELAY_MIN, INTER_CAT_DELAY_MAX)
            logger.debug(f"  Pause inter-catégorie de {inter_delay:.1f}s...")
            time.sleep(inter_delay)

    # ── Bilan final ───────────────────────────────────────────────────────────
    week_cp["status"] = "done" if not failed_categories else "partial"
    week_cp["finished_at"] = datetime.now().isoformat()
    week_cp["total_products"] = total_products
    _save_checkpoint(checkpoint)

    logger.info(f"\n{'=' * 70}")
    logger.info(f"  FIN SCRAPING SEMAINE {week_number}")
    logger.info(f"  Produits collectés  : {total_products}")
    if failed_categories:
        logger.warning(
            f"  Catégories en erreur : {', '.join(failed_categories)}"
        )
    else:
        logger.info("  Toutes les catégories ont été traitées sans erreur.")
    logger.info(f"  Données → {output_dir}")
    logger.info(f"{'=' * 70}\n")

    return total_products, failed_categories


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION MULTI-SEMAINES
# ─────────────────────────────────────────────────────────────────────────────

def schedule_weeks(start_week: int = 1, end_week: int = 3) -> None:
    """
    Lance automatiquement les semaines de scraping (start_week -> end_week,
    bornes incluses) avec un délai de 7 jours entre chaque exécution.

    Le délai est implémenté avec time.sleep() : le processus reste actif.
    Pour un usage en production, préférer un scheduler système (cron,
    systemd-timer) qui appellera `--week N` directement.

    Args:
        start_week : semaine de départ (1 par défaut, utile pour reprendre
                     à une semaine ultérieure sans re-scraper les précédentes).
        end_week   : dernière semaine à scraper (incluse). Anciennement figé
                     à 3 -- désormais un paramètre explicite pour ne pas
                     avoir à modifier ce fichier à chaque semaine ajoutée.
    """
    logger = logging.getLogger("scraper.scheduler")
    logger.info(f"Mode automatique : semaines {start_week} → {end_week}")

    for week in range(start_week, end_week + 1):
        logger.info(f"\n{'#' * 70}")
        logger.info(f"  SEMAINE {week}/{end_week}")
        logger.info(f"{'#' * 70}\n")

        try:
            total, failed = scrape_week(week)
            status = "✔" if not failed else "⚠ partiel"
            logger.info(f"Semaine {week} : {status} — {total} produits.")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.critical(
                f"Erreur fatale semaine {week} : {exc}",
                exc_info=True,
            )

        if week < end_week:
            next_run = datetime.now() + timedelta(seconds=WEEK_DELAY_SECONDS)
            logger.info(
                f"\n  Attente de 7 jours avant la semaine {week + 1}.\n"
                f"  Prochain lancement : {next_run.strftime('%Y-%m-%d à %H:%M:%S')}\n"
                f"  (Ctrl+C pour interrompre)\n"
            )
            try:
                time.sleep(WEEK_DELAY_SECONDS)
            except KeyboardInterrupt:
                logger.warning(
                    "Interruption pendant l'attente inter-semaine. "
                    "Relancer avec --all --start-from "
                    f"{week + 1} --end-week {end_week} pour continuer."
                )
                raise

    logger.info("Toutes les semaines ont été traitées.")


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE CLI
# ─────────────────────────────────────────────────────────────────────────────

def _positive_int(value: str) -> int:
    """Type argparse : entier >= 1 -- remplace l'ancien choices=[1,2,3] figé,
    pour qu'ajouter une semaine 4 (et au-delà) ne demande aucune
    modification de ce fichier."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' n'est pas un entier valide.")
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"le numéro de semaine doit être >= 1 (reçu : {ivalue}).")
    return ivalue


def main() -> None:
    """
    Interface en ligne de commande du scheduler.

    Options disponibles :
      --week N              Scraper une semaine spécifique (N >= 1).
      --all                 Lancer plusieurs semaines automatiquement.
      --start-from N        (avec --all) Semaine de départ. Défaut : 1.
      --end-week N          (avec --all) Dernière semaine (incluse). Défaut : 3.
      --reset               Ignorer le checkpoint existant (repart de zéro).
      --debug               Active le niveau de log DEBUG.
    """
    from . import setup_logging
    setup_logging()

    logger = logging.getLogger("scraper.scheduler")
    logger.info("Scheduler démarré")

    arg_parser = argparse.ArgumentParser(
        prog="scheduler.py",
        description=(
            "Orchestrateur du scraping Mytek.tn — modèle hédonique multi-semaines."
        ),
        epilog=(
            "Exemples :\n"
            "  python src/scraper/scheduler.py --week 1\n"
            "  python src/scraper/scheduler.py --week 4 --reset\n"
            "  python src/scraper/scheduler.py --all --end-week 3\n"
            "  python src/scraper/scheduler.py --all --start-from 2 --end-week 4\n"
            "  python src/scraper/scheduler.py --week 1 --debug"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Arguments mutuellement exclusifs ─────────────────────────────────────
    mode_group = arg_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--week",
        type=_positive_int,
        metavar="N",
        help="Numéro de la semaine à scraper (entier >= 1).",
    )
    mode_group.add_argument(
        "--all",
        action="store_true",
        help=(
            "Lance plusieurs semaines automatiquement (start-from -> end-week) "
            "avec un délai de 7 jours entre chaque."
        ),
    )

    # ── Options supplémentaires ───────────────────────────────────────────────
    arg_parser.add_argument(
        "--start-from",
        type=_positive_int,
        default=1,
        dest="start_from",
        metavar="N",
        help=(
            "(Avec --all) Semaine de départ pour reprendre une exécution "
            "interrompue. Défaut : 1."
        ),
    )
    arg_parser.add_argument(
        "--end-week",
        type=_positive_int,
        default=3,
        dest="end_week",
        metavar="N",
        help="(Avec --all) Dernière semaine à scraper, incluse. Défaut : 3.",
    )
    arg_parser.add_argument(
        "--reset",
        action="store_true",
        help=(
            "Ignore le checkpoint existant et repart de zéro "
            "pour la semaine ciblée (ou toutes si --all)."
        ),
    )
    arg_parser.add_argument(
        "--category",
        type=str,
        choices=list(CATEGORIES.keys()),
        default=None,
        help=(
            "(Avec --week) Ne scrape que cette catégorie au lieu de "
            "toutes, sans toucher au checkpoint des autres catégories."
        ),
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Active le niveau de log DEBUG (très verbeux).",
    )
    def _handle_sigterm(sig, frame):
        logging.getLogger("scraper.scheduler").warning(
            "SIGTERM reçu. Arrêt propre en cours..."
        )
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    args = arg_parser.parse_args()

    # ── Niveau de log ─────────────────────────────────────────────────────────
    if args.debug:
        logging.getLogger("scraper").setLevel(logging.DEBUG)

    logger = logging.getLogger("scraper.scheduler")

    # ── Reset du checkpoint ───────────────────────────────────────────────────
    if args.reset:
        if args.week:
            reset_checkpoint(args.week)
        else:
            # --all --reset : effacer toutes les semaines
            reset_checkpoint(None)

    # ── Lancement ─────────────────────────────────────────────────────────────
    try:
        if args.week:
            logger.info(f"Mode manuel — Semaine {args.week}")
            total, failed = scrape_week(args.week, category=args.category)
            sys.exit(0 if not failed else 1)

        elif args.all:
            logger.info(
                f"Mode automatique — semaines {args.start_from} → {args.end_week}"
            )
            schedule_weeks(start_week=args.start_from, end_week=args.end_week)
            sys.exit(0)

    except KeyboardInterrupt:
        logger.warning(
            "\nInterruption utilisateur (Ctrl+C). "
            "Le checkpoint a été sauvegardé ; vous pouvez reprendre "
            "avec la même commande."
        )
        sys.exit(130)

    except Exception as exc:
        logger.critical(f"Erreur inattendue : {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()