# -*- coding: utf-8 -*-
"""Libelle humains des clusters N1, derives du profil technique.

Source unique pour le dashboard et les rapports hebdomadaires : les noms
N1 sont stables, category-specific et ne depandent pas des donnees N2.
"""

N1_CLUSTER_NAMES = {
    "pc_bureau": {
        0: "Performant milieu de gamme",
        1: "Standard bureautique",
        2: "Compact connecté",
        3: "Haute performance",
        4: "Station d'entrée de gamme",
    },
    "pc_portables": {
        0: "Polyvalent grand public",
        1: "Productivité Core i5",
        2: "Haute performance Core i7",
        3: "Entrée de gamme Core i3",
    },
    "smartphones": {
        0: "Premium 5G",
        1: "Milieu de gamme 4G",
        2: "Entrée de gamme basique",
        3: "Ultra-premium 5G",
    },
    "telephones_portables": {
        0: "Basique compact",
        1: "Standard multimédia",
        2: "Intermédiaire renforcé",
        3: "Feature phone avancé",
        4: "Connecté haut de gamme",
    },
    "televiseurs": {
        0: "Smart TV 4K HDR",
        1: "TV Full HD standard",
    },
}


def n1_cluster_name(category: str, cluster_int: int) -> str:
    """Retourne un libelle humain pour un cluster N1, avec repli explicite."""
    try:
        return N1_CLUSTER_NAMES[category][int(cluster_int)]
    except (KeyError, TypeError, ValueError):
        return f"Cluster {cluster_int}"