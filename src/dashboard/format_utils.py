# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/dashboard/format_utils.py
=============================================================================
ROLE :
    Formatage numerique FRANCAIS coherent partout dans le dashboard :
    separateur de milliers = espace insecable, decimale = virgule, "TND"
    pour les montants. Une seule implementation, jamais un f-string
    ad-hoc redondant par page.
=============================================================================
"""

_NBSP = " "  # espace fine insecable -- separateur de milliers francais


def fmt_number(value, decimals: int = 0) -> str:
    """1234567.5 -> '1 234 568' (decimals=0) ou '1 234 567,5' (decimals=1)."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    formatted = f"{value:,.{decimals}f}"
    # f-string donne "1,234,567.5" (convention US) -> on bascule vers la
    # convention francaise (espace milliers, virgule decimale) par substitution.
    formatted = formatted.replace(",", " ").replace(".", ",").replace(" ", _NBSP)
    return formatted


def fmt_price(value, decimals: int = 0) -> str:
    """1234567.5 -> '1 234 568 TND'."""
    if value is None:
        return "—"
    return f"{fmt_number(value, decimals)} TND"


def fmt_pct(value, decimals: int = 1) -> str:
    """0.1234 -> '12,3 %' -- attend une fraction (pas deja *100)."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{fmt_number(value * 100, decimals)} %"


def fmt_pct_effect(value, decimals: int = 1) -> str:
    """12.34 -> '+12,3 %' -- attend une valeur DEJA en %, ex. pct_effect
    des modeles hedoniques (signe explicite, jamais ambigu sur le sens)."""
    if value is None:
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "+" if value >= 0 else ""
    return f"{sign}{fmt_number(value, decimals)} %"
