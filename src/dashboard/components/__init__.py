# -*- coding: utf-8 -*-
"""Composants reutilisables du dashboard (src/dashboard/components/)."""

from .kpi_card import kpi_row, kpi_card
from .section_header import section_header
from .provenance_strip import provenance_strip
from .category_selector import category_selector
from .nav import sidebar_links

__all__ = [
    "kpi_row", "kpi_card", "section_header", "provenance_strip",
    "category_selector", "sidebar_links",
]
