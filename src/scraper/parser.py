# -*- coding: utf-8 -*-
"""
Module : src/scraper/parser.py
ROLE : Extraction et structuration des donnees depuis les fiches produits.
       Visite chaque page produit, extrait le nom, prix, marque et les
       caracteristiques techniques (processeur, RAM, stockage, ecran, OS,
       connectivite) via la balise meta og:description (source primaire,
       fiable et toujours presente) avec fallback sur le tableau HTML.
"""
 
import re
import time
import random
import logging
import unicodedata
from datetime import datetime
 
from scrapling.fetchers import DynamicFetcher
 
from . import (
    BASE_URL, DELAY_MIN, DELAY_MAX, MAX_RETRIES, USER_AGENT, KNOWN_BRANDS,
)
from .utils import fetch_page
 
logger = logging.getLogger("scraper.parser")
 
_BRAND_BLACKLIST = {
    "PC", "ORDINATEUR", "TELEPHONE", "SMARTPHONE", "TABLETTE", "ECRAN",
    "TV", "TELEVISEUR", "TELEVISION",
}
 
 
def _css_first(node, selector):
    """
    Remplace l'ancienne methode .css_first() qui n'existe plus dans
    cette version de Scrapling. .css() retourne toujours une liste ;
    on prend le premier element si la liste n'est pas vide.
    """
    if node is None:
        return None
    elements = node.css(selector)
    return elements[0] if elements else None
 
 
def _normalize_text(text):
    """
    Supprime les accents et met en minuscule pour permettre une
    comparaison insensible aux accents (les pages mytek.tn contiennent
    des caracteres accentues comme 'Memoire' vs 'Mémoire').
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c)).lower()
 
 
def _key_matches(search_key, spec_key):
    """Comparaison insensible aux accents et a la casse."""
    return _normalize_text(search_key) in _normalize_text(spec_key)
 
 
class ProductParser:
    """
    Parseur de fiches produits Mytek.tn.
    Extrait les donnees structurees depuis chaque page produit individuelle.
    """
 
    def __init__(self):
        self.fetcher = DynamicFetcher()
 
    def _wait(self):
        """Pause aleatoire entre les requetes."""
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
 
    def _fetch_page(self, url, retries=0):
        """Recupere une page produit avec retry."""
        return fetch_page(self.fetcher, url, retries)
 
    # ─── EXTRACTION DES CHAMPS PRINCIPAUX ──────────────────────────────────
 
    def _extract_name(self, page):
        """Extrait le nom du produit depuis la balise h1."""
        try:
            el = _css_first(page, "h1.page-title span")
            if el and el.text:
                return el.text.strip()
            el = _css_first(page, "h1.page-title")
            if el and el.text:
                return el.text.strip()
            el = _css_first(page, "span[data-ui-id='page-title-wrapper']")
            if el and el.text:
                return el.text.strip()
            el = _css_first(page, "h1")
            if el and el.text and el.text.strip():
                return el.text.strip()
        except Exception as e:
            logger.debug(f"  Erreur extraction nom : {e}")
        return None
 
    def _extract_price(self, page):
        """
        Extrait le prix en TND.
 
        SOURCE PRIMAIRE : balise <meta property="product:price:amount">,
        generee automatiquement par Magento pour le SEO/Open Graph.
        Contrairement aux selecteurs CSS span.price (qui peuvent matcher
        le mauvais element sur la page — produits associes, encarts de
        garantie, etc.), cette balise est garantie unique et exacte.
 
        SOURCE DE SECOURS : selecteurs CSS, utilises seulement si la
        balise meta est absente.
        """
        try:
            meta_el = _css_first(page, "meta[property='product:price:amount']")
            if meta_el:
                content = meta_el.attrib.get("content", "")
                if content:
                    try:
                        return float(content)
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            logger.debug(f"  Erreur extraction prix (meta) : {e}")
 
        try:
            selectors = [
                "span.special-price span.price",
                "span.price-wrapper span.price",
                "div.price-box span.price",
                "span.price",
            ]
            for sel in selectors:
                el = _css_first(page, sel)
                if el and el.text:
                    cleaned = self._clean_price(el.text)
                    if cleaned is not None:
                        return cleaned
        except Exception as e:
            logger.debug(f"  Erreur extraction prix (CSS) : {e}")
        return None
 
    def _clean_price(self, price_text):
        """
        Nettoie un texte de prix et retourne un float.
        Exemples d'entree : "1 299,000 TND", "1299.000", "2,499 TND"
        """
        try:
            cleaned = re.sub(r'[^\d.,]', '', price_text)
            if ',' in cleaned and '.' not in cleaned:
                cleaned = cleaned.replace(',', '.')
            elif ',' in cleaned and '.' in cleaned:
                cleaned = cleaned.replace('.', '').replace(',', '.')
            if not cleaned:
                return None
            return float(cleaned)
        except (ValueError, TypeError):
            return None
 
    def _extract_brand(self, page, product_name):
        """
        Extrait la marque du produit. Strategies en cascade :
        1. Attribut dedie dans les specs (meta + tableau)
        2. Breadcrumb / lien de marque
        3. Marque connue trouvee dans le nom du produit
        4. Premier mot du nom (heuristique, avec garde-fou)
        """
        try:
            specs = self._extract_all_specs(page)
            for key in ["Marque", "Brand", "Fabricant"]:
                for spec_key, spec_val in specs.items():
                    if _key_matches(key, spec_key):
                        return spec_val
 
            brand_el = _css_first(page, "a.brand-link")
            if brand_el and brand_el.text:
                return brand_el.text.strip()
 
            if product_name:
                name_upper = product_name.upper()
                for brand in KNOWN_BRANDS:
                    if brand.upper() in name_upper:
                        return brand
 
                words = product_name.split()
                first_word = words[0] if words else None
                if first_word and first_word.upper() not in _BRAND_BLACKLIST:
                    return first_word
 
        except Exception as e:
            logger.debug(f"  Erreur extraction marque : {e}")
        return None
 
    # ─── EXTRACTION DES SPECIFICATIONS (META + TABLEAU) ────────────────────
 
    def _extract_specs_from_meta(self, page):
        """
        Extrait les specs depuis la balise <meta property="og:description">,
        qui contient systematiquement un resume structure du type :
        "Processeur: ... - Memoire RAM: 8 Go - Disque Dur: 256Go SSD - ..."
 
        C'est la source LA PLUS FIABLE car elle est generee automatiquement
        par Magento pour le SEO et ne depend pas de la structure visuelle
        du tableau de caracteristiques (qui peut varier ou etre absent).
 
        Returns: dict {label: valeur}
        """
        specs = {}
        try:
            meta_el = _css_first(page, "meta[property='og:description']")
            if not meta_el:
                return specs
            content = meta_el.attrib.get("content", "")
            if not content:
                return specs
 
            # Normaliser les espaces insecables (&nbsp; -> espace normal)
            content = content.replace('\xa0', ' ')
 
            # Le format est "Label: Valeur - Label: Valeur - ..."
            # On coupe sur " - " (tiret entoure d'espaces) pour separer
            # chaque paire label/valeur, en evitant de couper les tirets
            # internes (ex: "i3-1305U") qui ne sont pas entoures d'espaces.
            parts = re.split(r'\s+-\s+', content)
            for part in parts:
                if ':' in part:
                    label, _, value = part.partition(':')
                    label = label.strip()
                    value = value.strip()
                    if label and value:
                        specs[label] = value
        except Exception as e:
            logger.debug(f"  Erreur extraction meta specs : {e}")
        return specs
 
    def _extract_specs_table(self, page):
        """
        Extrait la table de caracteristiques techniques (fiche technique),
        en complement de la meta description. Magento stocke parfois les
        specs dans un <table> avec des lignes <tr> contenant <th> et <td>.
 
        Returns: dict {label: valeur}
        """
        specs = {}
        try:
            table_selectors = [
                "table.data.table.additional-attributes",
                "table#product-attribute-specs-table",
                "table.table-additional",
                "div.product-attributes table",
                "div.additional-attributes-wrapper table",
            ]
            table = None
            for sel in table_selectors:
                table = _css_first(page, sel)
                if table:
                    break
 
            if table:
                rows = table.css("tr")
                for row in rows:
                    th = _css_first(row, "th")
                    td = _css_first(row, "td")
                    if th and td and th.text and td.text:
                        label = th.text.strip()
                        value = td.text.strip()
                        if label and value:
                            specs[label] = value
 
            if not specs:
                spec_items = page.css("div.product-attribute")
                for item in spec_items:
                    label_el = _css_first(item, ".label, .attribute-label")
                    value_el = _css_first(item, ".value, .attribute-value")
                    if label_el and value_el and label_el.text and value_el.text:
                        label = label_el.text.strip()
                        value = value_el.text.strip()
                        if label and value:
                            specs[label] = value
 
        except Exception as e:
            logger.debug(f"  Erreur extraction specs table : {e}")
 
        return specs
 
    def _extract_all_specs(self, page):
        """
        Combine les specs de la meta description (source primaire et
        fiable) avec celles du tableau HTML (complement). En cas de
        conflit, la meta description est prioritaire.
 
        Returns: dict {label: valeur}
        """
        specs_table = self._extract_specs_table(page)
        specs_meta = self._extract_specs_from_meta(page)
        # specs_meta en dernier : ses cles ecrasent celles de specs_table
        # en cas de doublon, car c'est la source la plus fiable.
        return {**specs_table, **specs_meta}
 
    # ─── PARSING DES CARACTERISTIQUES TECHNIQUES ───────────────────────────
    # Toutes les comparaisons de cles utilisent _key_matches() qui ignore
    # les accents, pour eviter les ratés du type "Memoire" vs "Mémoire".
 
    def _parse_processor(self, specs, product_name=""):
        """Extrait le modele de processeur depuis les specs ou le nom."""
        proc_keys = [
            "Processeur", "CPU", "Chipset", "Processeur / Chipset",
            "Type de processeur", "Modele de processeur",
        ]
        for key in proc_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    return spec_val
 
        if product_name:
            proc_patterns = [
                r'(Intel\s+Core\s+i\d[^\s,/]*)',
                r'(Intel\s+Core\s+Ultra\s+\d[^\s,/]*)',
                r'(AMD\s+Ryzen\s+\d[^\s,/]*)',
                r'(Snapdragon\s+\d+[^\s,]*)',
                r'(MediaTek\s+\w+[^\s,]*)',
                r'(Exynos\s+\d+)',
                r'(Apple\s+A\d+)',
                r'(Celeron\s+\w+)',
                r'(Pentium\s+\w+)',
            ]
            for pattern in proc_patterns:
                match = re.search(pattern, product_name, re.IGNORECASE)
                if match:
                    return match.group(1)
        return None
 
    def _parse_ram(self, specs, product_name=""):
        """
        Extrait la RAM en Go. Retourne un float ou None.

        Deux passes pour eviter la confusion "Memoire" (RAM) / "Memoire
        Cache" (cache processeur) : le candidat generique "Memoire" est un
        simple substring-match qui matcherait aussi "Memoire Cache" si ce
        dernier apparait avant "Memoire" (RAM) dans le dict specs -- ce qui
        arrive reellement sur certains modeles (ThinkBook/ThinkPad/Chuwi)
        et donnait des RAM aberrantes du type 0.0039 Go (= 4 Mo de cache
        confondus avec la RAM). La passe 1 exige une egalite exacte de cle
        (priorite absolue) ; la passe 2 fait du matching partiel mais
        exclut explicitement toute cle contenant "cache".
        """
        ram_keys = ["RAM", "Memoire RAM", "Memoire vive", "Memoire"]

        for key in ram_keys:
            for spec_key, spec_val in specs.items():
                if _normalize_text(spec_key) == _normalize_text(key):
                    match = re.search(r'(\d+)\s*(Go|GB|Mo|MB)', spec_val, re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                        unit = match.group(2).upper()
                        if unit in ("MO", "MB"):
                            val = val / 1024
                        return val

        for key in ram_keys:
            for spec_key, spec_val in specs.items():
                if "cache" in _normalize_text(spec_key):
                    continue
                if _key_matches(key, spec_key):
                    match = re.search(r'(\d+)\s*(Go|GB|Mo|MB)', spec_val, re.IGNORECASE)
                    if match:
                        val = float(match.group(1))
                        unit = match.group(2).upper()
                        if unit in ("MO", "MB"):
                            val = val / 1024
                        return val

        if product_name:
            match = re.search(r'(\d+)\s*(Go|GB)\s*(RAM|DDR)', product_name, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
 
    def _parse_storage(self, specs, product_name=""):
        """
        Extrait la capacite de stockage (Go) et le type (SSD/HDD).
        Returns: tuple (capacite_go: float, type_stockage: str) ou (None, None)
        """
        storage_keys = [
            "Disque Dur", "Disque dur", "Stockage", "Capacite de stockage",
            "SSD", "HDD", "Disque", "Memoire interne", "Capacite",
            "Stockage interne",
        ]
        for key in storage_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    result = self._parse_storage_value(spec_val)
                    if result[0]:
                        return result
 
        if product_name:
            result = self._parse_storage_value(product_name)
            if result[0]:
                return result
        return None, None
 
    def _parse_storage_value(self, text):
        """Parse une chaine pour en extraire capacite et type de stockage."""
        match = re.search(r'(\d+)\s*To\b', text, re.IGNORECASE)
        if match:
            capacity = float(match.group(1)) * 1024
        else:
            match = re.search(r'(\d+)\s*Go\b', text, re.IGNORECASE)
            if match:
                capacity = float(match.group(1))
            else:
                match = re.search(r'(\d+)\s*GB\b', text, re.IGNORECASE)
                if match:
                    capacity = float(match.group(1))
                else:
                    return None, None
 
        text_upper = text.upper()
        if "SSD" in text_upper:
            storage_type = "SSD"
        elif "HDD" in text_upper:
            storage_type = "HDD"
        elif "EMMC" in text_upper:
            storage_type = "eMMC"
        else:
            storage_type = None
 
        return capacity, storage_type
 
    def _parse_screen_size(self, specs, product_name=""):
        """Extrait la taille d'ecran en pouces. Retourne un float ou None."""
        screen_keys = [
            "Taille de l'ecran", "Taille ecran", "Ecran", "Diagonale",
            "Taille d'ecran", "Affichage",
        ]
        for key in screen_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    match = re.search(r'([\d.,]+)\s*["\u201Dpouces\']', spec_val, re.IGNORECASE)
                    if match:
                        return float(match.group(1).replace(',', '.'))
                    match = re.search(r'([\d.,]+)', spec_val)
                    if match:
                        val = float(match.group(1).replace(',', '.'))
                        if 1 <= val <= 110:  # couvre laptops (10-17") et TVs (24-98"+)
                            return val
 
        if product_name:
            match = re.search(r'([\d.,]+)\s*["\u201Dpouces\']', product_name, re.IGNORECASE)
            if match:
                return float(match.group(1).replace(',', '.'))
        return None
 
    def _parse_os(self, specs):
        """Extrait le systeme d'exploitation."""
        os_keys = ["Systeme d'exploitation", "Systeme", "OS"]
        for key in os_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    return spec_val
        return None
 
    def _parse_connectivity(self, specs):
        """Extrait les informations de connectivite (4G/5G/Wi-Fi)."""
        conn_keys = [
            "Connectivite", "Reseau", "Wi-Fi", "WiFi", "4G", "5G",
            "Connexion", "Sans fil", "Wireless",
        ]
        connectivity = []
        for spec_key, spec_val in specs.items():
            for key in conn_keys:
                if _key_matches(key, spec_key):
                    connectivity.append(f"{spec_key}: {spec_val}")
                    break
 
        all_text = " ".join(specs.values())
        if "5G" in all_text and "5G" not in str(connectivity):
            connectivity.append("5G")
        if "4G" in all_text and "4G" not in str(connectivity):
            connectivity.append("4G")
        if re.search(r'Wi-?Fi', all_text, re.IGNORECASE) and "Wi-Fi" not in str(connectivity):
            connectivity.append("Wi-Fi")
 
        return "; ".join(connectivity) if connectivity else None
 
    # ─── PARSING SPÉCIFIQUE TÉLÉVISEURS ────────────────────────────────────

    def _parse_resolution(self, specs, product_name=""):
        """Extrait la résolution d'affichage : '4K', 'Full HD', 'HD Ready', '8K'."""
        res_keys = [
            "Resolution", "Résolution", "Définition", "Definition",
            "Format image", "Format d'image",
        ]
        text = ""
        for key in res_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    text = spec_val
                    break
            if text:
                break
        if not text:
            text = product_name

        t = text.upper()
        if "8K" in t or "7680" in t:
            return "8K"
        if "4K" in t or "UHD" in t or "3840" in t:
            return "4K"
        if "FULL HD" in t or "FHD" in t or "1920" in t or "1080" in t:
            return "Full HD"
        if "HD READY" in t or "HD" in t or "1280" in t or "720" in t:
            return "HD Ready"
        return None

    def _parse_display_tech(self, specs, product_name=""):
        """Extrait la technologie de dalle : OLED, QLED, QNED, NanoCell, Mini-LED, LED."""
        tech_keys = [
            "Technologie", "Type d'ecran", "Type ecran", "Dalle",
            "Technologie d'affichage", "Technologie ecran",
        ]
        text = ""
        for key in tech_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    text = spec_val
                    break
            if text:
                break
        if not text:
            text = product_name

        t = text.upper()
        if "OLED" in t:
            return "OLED"
        if "QNED" in t:
            return "QNED"
        if "QLED" in t:
            return "QLED"
        if "NANOCELL" in t or "NANO CELL" in t:
            return "NanoCell"
        if "MINI-LED" in t or "MINI LED" in t or "MINILED" in t:
            return "Mini-LED"
        if "LED" in t:
            return "LED"
        return None

    def _parse_hdr(self, specs, product_name=""):
        """
        Retourne True si le téléviseur supporte HDR (toute variante).

        Priorite a une cle de specs explicite (typiquement "Compatible
        HDR" avec valeur "Oui"/"Non") : le bug initial ne cherchait
        "HDR" que dans les VALEURS de specs.values(), jamais dans les
        CLES -- or c'est presque toujours dans le nom de la cle elle-meme
        ("Compatible HDR": "Oui") que "HDR" apparait, la valeur seule
        ("Oui") ne le mentionne jamais. Resultat reel observe : 106/112
        televiseurs faussement detectes hdr=False alors que 106/106
        mentionnaient bien HDR quelque part dans specs_brutes.
        """
        hdr_keys = ["Compatible HDR", "HDR"]
        for key in hdr_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    v = str(spec_val).upper()
                    if "OUI" in v or "YES" in v:
                        return True
                    if "NON" in v or "NO" in v:
                        return False

        all_text = " ".join(f"{k} {v}" for k, v in specs.items()) + " " + product_name
        return bool(re.search(r'\bHDR\b', all_text, re.IGNORECASE))

    def _parse_smart_tv(self, specs, product_name=""):
        """Retourne True si Smart TV détecté (mention OS ou 'Smart' dans les specs)."""
        smart_keys = ["Smart TV", "Smart", "Fonctionnalite Smart", "Type de TV"]
        for key in smart_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    v = spec_val.upper()
                    if "SMART" in v or "OUI" in v:
                        return True
                    if "NON" in v:
                        return False
        all_text = " ".join(specs.values()) + " " + product_name
        return bool(
            re.search(r'smart\s*tv', all_text, re.IGNORECASE)
            or re.search(r'android\s*tv|webos|tizen|fire\s*tv|roku', all_text, re.IGNORECASE)
        )

    def _parse_refresh_rate(self, specs, product_name=""):
        """
        Extrait le taux de rafraîchissement en Hz (entier) ou None.

        Repli supplementaire : sur certaines fiches, le texte "Taux de
        rafraichissement : 60 Hz" se retrouve fusionne dans la VALEUR
        d'une cle sans rapport (ex. "Luminosite (cd/m2)"), probablement
        a cause de lignes de tableau mal separees cote source -- un
        artefact de mise en page, pas une absence reelle de donnee. On
        cherche donc, en dernier recours, le mot "rafraichissement" suivi
        d'un nombre de Hz n'importe ou dans les valeurs, plutot que de se
        limiter aux cles candidates ci-dessus. Ce repli est volontairement
        specifique au mot "rafraichissement" (et non un simple "Hz") pour
        eviter les faux positifs frequents avec la frequence secteur de
        l'alimentation ("50/60 Hz") ou la frequence du processeur ("1.5GHz").
        """
        hz_keys = [
            "Taux de rafraichissement", "Frequence", "Frequency",
            "Taux rafraichissement", "Hz",
        ]
        for key in hz_keys:
            for spec_key, spec_val in specs.items():
                if _key_matches(key, spec_key):
                    m = re.search(r'(\d+)\s*Hz', spec_val, re.IGNORECASE)
                    if m:
                        return int(m.group(1))

        for spec_val in specs.values():
            if _key_matches("rafraichissement", str(spec_val)):
                m = re.search(r'rafraichissement[^\d]*(\d+)\s*hz', _normalize_text(str(spec_val)))
                if m:
                    return int(m.group(1))

        m = re.search(r'(\d+)\s*Hz', product_name, re.IGNORECASE)
        if m:
            return int(m.group(1))
        return None

    # ─── METHODE PRINCIPALE ─────────────────────────────────────────────────
 
    def parse_product(self, url, category, week_number):
        """
        Extrait toutes les donnees d'une page produit.
 
        Args:
            url: URL de la fiche produit
            category: categorie du produit (ex: "pc_portables")
            week_number: numero de la semaine (1, 2 ou 3)
 
        Returns: dict avec toutes les donnees extraites, ou None si echec
        """
        page = self._fetch_page(url)
        if page is None:
            logger.warning(f"  Echec chargement produit : {url}")
            return None
 
        try:
            name = self._extract_name(page)
            price = self._extract_price(page)
            brand = self._extract_brand(page, name)
 
            specs = self._extract_all_specs(page)
 
            processor = self._parse_processor(specs, name or "")
            ram = self._parse_ram(specs, name or "")
            storage_capacity, storage_type = self._parse_storage(specs, name or "")
            screen_size = self._parse_screen_size(specs, name or "")
            os = self._parse_os(specs)
            connectivity = self._parse_connectivity(specs)
 
            product_data = {
                "url": url,
                "nom": name,
                "marque": brand,
                "prix_tnd": price,
                "processeur": processor,
                "ram_go": ram,
                "stockage_go": storage_capacity,
                "type_stockage": storage_type,
                "taille_ecran": screen_size,
                "os": os,
                "connectivite": connectivity,
                "categorie": category,
                "semaine": week_number,
                "date_collecte": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "specs_brutes": specs if specs else None,
            }

            if category == "televiseurs":
                product_data.update({
                    "resolution_affichage": self._parse_resolution(specs, name or ""),
                    "technologie_dalle": self._parse_display_tech(specs, name or ""),
                    "hdr": self._parse_hdr(specs, name or ""),
                    "smart_tv": self._parse_smart_tv(specs, name or ""),
                    "taux_rafraichissement": self._parse_refresh_rate(specs, name or ""),
                })
 
            fields_filled = sum(1 for v in product_data.values() if v is not None)
            total_fields = len(product_data)
            logger.info(
                f"  OK {name[:50] if name else 'Sans nom'} "
                f"| {price} TND | {fields_filled}/{total_fields} champs"
            )
 
            return product_data
 
        except Exception as e:
            logger.error(f"  Erreur parsing {url}: {e}")
            return None
 
    def parse_products(self, urls, category, week_number):
        """
        Parse une liste de produits avec des pauses entre chaque requete.
 
        Args:
            urls: liste d'URLs de produits
            category: categorie
            week_number: numero de semaine
 
        Returns: list[dict] — donnees de tous les produits parses
        """
        products = []
        total = len(urls)
        logger.info(f"Parsing de {total} produit(s) pour {category}...")
 
        for i, url in enumerate(urls, 1):
            logger.info(f"  [{i}/{total}] {url}")
            product = self.parse_product(url, category, week_number)
            if product:
                products.append(product)
            if i < total:
                self._wait()
 
        success = len(products)
        logger.info(f"  Resultat {category}: {success}/{total} produits parses.\n")
        return products