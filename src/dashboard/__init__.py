# -*- coding: utf-8 -*-
"""
=============================================================================
Package : src/dashboard
=============================================================================
ROLE :
    Dashboard Dash (Dash Mantine Components + dash-ag-grid + Plotly) pour
    visualiser et servir l'analyse hedonique du projet -- STRICTEMENT
    LECTURE SEULE sur data/processed/ et models/ (jamais de reentrainement
    ni de mutation de donnees source, cf. data_loader.py).

    - theme.py            : theme DMC + template Plotly, source unique.
    - format_utils.py      : formatage numerique francais coherent.
    - data_loader.py       : acces cache aux donnees/artefacts.
    - prediction_utils.py  : logique de la page Prediction (encodage,
                              prediction, assignation de segment N1/N2).
    - components/           : elements d'UI reutilisables.
    - pages/                 : les 5 pages (accueil, descriptif, modeles,
                              prediction, a propos), framework dash.pages.
    - app.py                 : point d'entree (python -m src.dashboard.app).

PREREQUIS :
    Les artefacts consommes par ce dashboard sont produits par
    `python -m src.models.save_artifacts` (src/models/save_artifacts.py) --
    a executer avant le premier lancement du dashboard, et a chaque fois
    que de nouvelles semaines de donnees sont disponibles.
=============================================================================
"""
