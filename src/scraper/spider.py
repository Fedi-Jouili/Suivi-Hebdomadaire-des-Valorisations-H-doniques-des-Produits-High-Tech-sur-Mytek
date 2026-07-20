# -*- coding: utf-8 -*-
"""
Module : src/scraper/spider.py
ROLE : Robot d'exploration (spider) pour Mytek.tn.
       Parcourt les pages de listing en boucle incrementale, gere la
       pagination Magento (?p=N), et collecte les URLs de chaque fiche
       produit.

NOTE IMPORTANTE :
    Cette version remplace la detection "upfront" du nombre total de
    pages (fragile, dependante de selecteurs CSS qui peuvent ne pas
    correspondre a la structure reelle du site) par une boucle
    incrementale : on continue de demander la page N+1 tant qu'elle
    retourne de nouveaux produits, et on s'arrete des qu'une page est
    vide ou ne contient plus aucun produit inedit. C'est plus lent
    mais beaucoup plus fiable.
"""

import time
import random
import logging
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

from scrapling.fetchers import DynamicFetcher

from . import (
    CATEGORIES, BASE_URL, DELAY_MIN, DELAY_MAX, MAX_RETRIES, USER_AGENT,
)
from .utils import fetch_page

logger = logging.getLogger("scraper.spider")

# Garde-fou anti boucle infinie : aucune categorie Mytek ne depasse
# raisonnablement 200 pages de produits.
MAX_PAGES_PER_CATEGORY = 200


class MytekSpider:
    """
    Spider pour le site Mytek.tn.
    Parcourt les catégories, gère la pagination de façon incrémentale,
    collecte les URLs produits jusqu'à épuisement de la pagination.
    """

    def __init__(self):
        self.fetcher = DynamicFetcher()
        self.collected_links = {}

    def _wait(self):
        """Pause aléatoire entre les requêtes pour respecter le serveur."""
        delay = random.uniform(DELAY_MIN, DELAY_MAX)
        logger.debug(f"  Pause de {delay:.1f}s...")
        time.sleep(delay)

    def _fetch_page(self, url, retries=0):
        """
        Récupère une page via le Fetcher Scrapling avec retry exponentiel.
        Returns: objet Response parsable, ou None si échec.
        """
        return fetch_page(self.fetcher, url, retries)

    def _build_page_url(self, base_url, page_number):
        """Construit l'URL paginée Magento (?p=N)."""
        parsed = urlparse(base_url)
        params = parse_qs(parsed.query)
        params['p'] = [str(page_number)]
        new_query = urlencode(params, doseq=True)
        return urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, new_query, parsed.fragment
        ))

    def get_product_links(self, page):
        """
        Extrait les URLs des produits depuis une page de listing.
        Utilise plusieurs sélecteurs CSS en fallback (Magento 2).
        """
        seen = set()
        links = []

        selectors = [
            "a.product-item-link",
            "li.product-item a.product-item-link",
            "div.product-item-info a.product-item-link",
            "ol.products a.product-item-link",
        ]
        product_elements = []
        for sel in selectors:
            product_elements = page.css(sel)
            if product_elements:
                break

        for element in product_elements:
            href = element.attrib.get("href", "")
            if href:
                absolute_url = urljoin(BASE_URL, href)
                if absolute_url not in seen:
                    seen.add(absolute_url)
                    links.append(absolute_url)

        logger.debug(f"  {len(links)} lien(s) produit(s) sur cette page.")
        return links

    def crawl_category(self, category_key, category_url):
        """
        Parcourt une catégorie en boucle incrémentale jusqu'à épuisement
        de la pagination, sans dépendre d'une détection préalable du
        nombre total de pages.

        Algorithme :
            page = 1
            tant que page <= MAX_PAGES_PER_CATEGORY :
                recuperer la page
                si echec de chargement -> arret
                extraire les liens produits
                si aucun lien -> fin de pagination, arret
                si tous les liens sont deja vus -> fin de pagination, arret
                ajouter les nouveaux liens, page += 1

        Args:
            category_key: identifiant (ex: "pc_portables")
            category_url: URL de la page de listing

        Returns: list[str] — URLs des produits
        """
        logger.info(f"{'='*60}")
        logger.info(f"Crawling : {category_key} — {category_url}")

        all_links = []
        seen = set()
        page_num = 1

        while page_num <= MAX_PAGES_PER_CATEGORY:
            if page_num == 1:
                page_url = category_url
            else:
                self._wait()
                page_url = self._build_page_url(category_url, page_num)

            page = self._fetch_page(page_url)
            if page is None:
                logger.warning(f"  Page {page_num} — échec de chargement, arrêt.")
                break

            links = self.get_product_links(page)

            if not links:
                logger.info(f"  Page {page_num} — vide, fin de la pagination.")
                break

            new_links = [l for l in links if l not in seen]

            if not new_links and page_num > 1:
                # La page renvoie les memes produits que la precedente :
                # Magento a probablement "boucle" sur la derniere page.
                logger.info(
                    f"  Page {page_num} — aucun produit inédit, "
                    f"fin de la pagination."
                )
                break

            for link in new_links:
                seen.add(link)
                all_links.append(link)

            logger.info(
                f"  Page {page_num} — {len(links)} produit(s), "
                f"{len(new_links)} nouveau(x). Total cumulé: {len(all_links)}"
            )

            page_num += 1

        if page_num > MAX_PAGES_PER_CATEGORY:
            logger.warning(
                f"  Limite de {MAX_PAGES_PER_CATEGORY} pages atteinte "
                f"pour {category_key} — arrêt forcé (garde-fou)."
            )

        self.collected_links[category_key] = all_links
        logger.info(f"  Total {category_key} : {len(all_links)} produit(s) unique(s)\n")
        return all_links

    def crawl_all(self):
        """Lance le crawling de toutes les catégories."""
        # Reinitialiser l'etat pour eviter d'accumuler les liens d'un
        # appel precedent si crawl_all() est appele plusieurs fois sur
        # la meme instance.
        self.collected_links = {}

        logger.info("Démarrage du crawling Mytek.tn")
        keys = list(CATEGORIES.keys())
        for i, cat_key in enumerate(keys):
            cat_info = CATEGORIES[cat_key]
            self.crawl_category(cat_key, cat_info["url"])
            if i < len(keys) - 1:
                time.sleep(random.uniform(5, 10))

        total = sum(len(v) for v in self.collected_links.values())
        logger.info(f"RÉSUMÉ CRAWLING — Total : {total} produit(s)")
        for k, v in self.collected_links.items():
            logger.info(f"  {k}: {len(v)}")
        return self.collected_links