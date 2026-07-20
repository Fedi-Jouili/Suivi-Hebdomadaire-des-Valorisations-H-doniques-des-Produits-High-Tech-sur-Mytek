# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/tests/test_scraper/test_parser.py
=============================================================================
ROLE :
    Tests de ProductParser.parse_product() (src/scraper/parser.py) avec une
    page MOCKEE -- aucun appel reseau reel.

    Le parser n'utilise PAS BeautifulSoup/requests : il recoit un objet
    scrapling-like (methode .css(selecteur) -> liste d'elements, chaque
    element ayant .text et .attrib.get(cle)). FakeElement/FakePage
    ci-dessous reproduisent exactement cette interface, pour exercer le
    VRAI code de parsing (regex, priorites de selecteurs) plutot que de le
    contourner avec un mock generique.
=============================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from src.scraper.parser import ProductParser


class FakeElement:
    """Reproduit un element scrapling : .text et .attrib.get(cle)."""

    def __init__(self, text: str | None = None, attrib: dict | None = None):
        self.text = text
        self.attrib = attrib or {}


class FakePage:
    """Reproduit une page scrapling : .css(selecteur) -> liste de FakeElement."""

    def __init__(self, css_map: dict):
        self._css_map = css_map

    def css(self, selector: str):
        return self._css_map.get(selector, [])


@pytest.fixture
def parser():
    return ProductParser()


@pytest.fixture
def complete_page():
    """
    Page mockee avec le strict minimum pour que le VRAI code de
    parser.py extraie correctement nom/prix/marque/processeur --
    via meta[og:description], la source la plus fiable (cf. docstring de
    _extract_specs_from_meta dans parser.py).
    """
    return FakePage({
        "h1.page-title span": [FakeElement(text="PC Portable ASUS VivoBook 15 i5 8Go 512Go SSD")],
        "meta[property='product:price:amount']": [FakeElement(attrib={"content": "1500.0"})],
        "meta[property='og:description']": [FakeElement(attrib={
            "content": "Processeur: Intel Core i5-1235U - Marque: ASUS - Memoire: 8 Go - Disque Dur: 512Go SSD"
        })],
    })


@pytest.fixture
def minimal_page():
    """Page mockee avec SEULEMENT un nom -- tout le reste (prix, marque,
    processeur, connectivite...) est absent, pour verifier que
    parse_product() degrade proprement (retourne None sur les champs
    manquants) plutot que de planter."""
    return FakePage({
        "h1.page-title span": [FakeElement(text="Produit Sans Details")],
    })


class TestParseProductChampsPresents:
    """parse_product() extrait correctement nom/marque/prix/processeur
    quand la page contient une meta og:description exploitable."""

    def test_extrait_le_nom(self, parser, complete_page, mocker):
        mocker.patch.object(ProductParser, "_fetch_page", return_value=complete_page)
        produit = parser.parse_product("https://exemple.test/produit.html", "pc_portables", 1)
        assert produit is not None
        assert produit["nom"] == "PC Portable ASUS VivoBook 15 i5 8Go 512Go SSD"

    def test_extrait_le_prix(self, parser, complete_page, mocker):
        mocker.patch.object(ProductParser, "_fetch_page", return_value=complete_page)
        produit = parser.parse_product("https://exemple.test/produit.html", "pc_portables", 1)
        assert produit["prix_tnd"] == 1500.0

    def test_extrait_la_marque(self, parser, complete_page, mocker):
        mocker.patch.object(ProductParser, "_fetch_page", return_value=complete_page)
        produit = parser.parse_product("https://exemple.test/produit.html", "pc_portables", 1)
        assert produit["marque"] == "ASUS"

    def test_extrait_le_processeur(self, parser, complete_page, mocker):
        mocker.patch.object(ProductParser, "_fetch_page", return_value=complete_page)
        produit = parser.parse_product("https://exemple.test/produit.html", "pc_portables", 1)
        assert "i5-1235U" in produit["processeur"]


class TestParseProductChampsManquants:
    """Des champs absents de la page (connectivite, prix...) doivent
    donner None -- jamais une exception."""

    def test_champs_manquants_retournent_none_sans_planter(self, parser, minimal_page, mocker):
        mocker.patch.object(ProductParser, "_fetch_page", return_value=minimal_page)
        produit = parser.parse_product("https://exemple.test/produit-minimal.html", "pc_portables", 1)

        assert produit is not None  # ne doit pas planter
        assert produit["nom"] == "Produit Sans Details"
        assert produit["connectivite"] is None
        assert produit["prix_tnd"] is None

    def test_page_introuvable_retourne_none(self, parser, mocker):
        """Si le fetch echoue completement (page=None), parse_product()
        doit retourner None plutot que de lever une exception."""
        mocker.patch.object(ProductParser, "_fetch_page", return_value=None)
        produit = parser.parse_product("https://exemple.test/404.html", "pc_portables", 1)
        assert produit is None
