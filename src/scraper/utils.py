# -*- coding: utf-8 -*-
import logging
import random
import time
from scrapling.fetchers import DynamicFetcher
DELAY_MIN   = 2.0
DELAY_MAX   = 5.0
MAX_RETRIES = 3
USER_AGENT  = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

logger = logging.getLogger("scraper.utils")

def fetch_page(fetcher: DynamicFetcher, url: str, retries: int = 0):
    """
    Fetches a page with exponential retry on both network errors
    and non-200 HTTP status codes.
    Returns a parsable Response object, or None on definitive failure.
    """
    try:
        logger.debug(f"  GET {url}")
        page = fetcher.fetch(
            url,
            headless=True,
            stealthy_headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml",
            }
        )
        if page.status == 200:
            return page
        logger.warning(f"  Status HTTP {page.status} pour {url}")
        if retries < MAX_RETRIES:
            wait_time = (2 ** retries) + random.uniform(0, 1)
            logger.info(f"  Retry {retries+1}/{MAX_RETRIES} dans {wait_time:.1f}s...")
            time.sleep(wait_time)
            return fetch_page(fetcher, url, retries + 1)
        return None
    except Exception as e:
        logger.error(f"  Erreur requête {url}: {e}")
        if retries < MAX_RETRIES:
            wait_time = (2 ** retries) + random.uniform(0, 1)
            logger.info(f"  Retry {retries+1}/{MAX_RETRIES} dans {wait_time:.1f}s...")
            time.sleep(wait_time)
            return fetch_page(fetcher, url, retries + 1)
        logger.error(f"  Echec définitif pour {url}")
        return None
