# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/models/rf_model.py
=============================================================================
ROLE :
    Modele hedonique non-lineaire : log(prix) ~ RandomForestRegressor,
    avec selection d'hyperparametres par validation croisee (GridSearchCV).
    Complement non-lineaire au RidgeModel (log-lineaire) -- capture les
    interactions et effets de seuil qu'une regression lineaire ne peut
    representer qu'via des termes d'interaction explicites.

    Aucun Pipeline de mise a l'echelle n'est necessaire ici : les arbres de
    decision (et donc les forets aleatoires) sont invariants a toute
    transformation monotone des variables (standardiser RAM en Go ne
    change pas les seuils de split que l'arbre apprend), contrairement a
    Ridge qui exige une echelle commune entre variables.

UTILISATION :
    from src.models.rf_model import RandomForestModel
    model = RandomForestModel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    importances = model.get_importances(feature_names=list(X_train.columns))
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.exceptions import NotFittedError


class RandomForestModel:
    """
    RandomForestRegressor hedonique avec selection d'hyperparametres par
    validation croisee (GridSearchCV) et importances de variables.
    """

    def __init__(self, param_grid: dict | None = None):
        """
        Args:
            param_grid: grille d'hyperparametres testee par GridSearchCV.
                Par defaut :
                    n_estimators   : [100, 200, 400]
                    max_depth      : [10, 20, None]  (None = pas de limite)
                    min_samples_split : [2, 5]
                n_estimators elargi le 2026-07-28 (etait [100, 200]) :
                plafonnait exactement a 200 pour 3 categories sur 5 une
                fois la CV correctement groupee par produit (cf. fit) --
                verifie empiriquement qu'une grille [100,200,400,600] pousse
                le choix jusqu'a 600 (le plafond teste) pour les deux
                categories les plus concernees, mais le gain de R² au-dela
                de 400 est marginal (plus d'arbres n'aggrave jamais la
                performance, seulement le temps de calcul) -- 400 retenu
                comme compromis plutot que de repousser indefiniment la
                limite a chaque elargissement.
        """
        self.param_grid = dict(param_grid) if param_grid is not None else {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        self.best_estimator_ = None
        self.grid_search_ = None

    def fit(self, X, y, groups=None):
        """
        Ajuste un RandomForestRegressor(random_state=42) avec selection
        d'hyperparametres par validation croisee (5 plis, scoring R² par
        defaut, n_jobs=-1 pour paralleliser sur tous les coeurs).

        groups : identifiant de regroupement (typiquement l'url du produit)
            -- si fourni, utilise GroupKFold au lieu d'un KFold ordinaire,
            meme raison et meme convention que RidgeModel.fit (cf. sa
            docstring) : X peut contenir plusieurs lignes quasi-identiques
            du meme produit (une par semaine), un KFold ordinaire risquerait
            de "valider" un pli sur une ligne quasi-dupliquee d'une ligne
            d'entrainement, biaisant la selection vers des arbres plus
            complexes (max_depth/n_estimators plus eleves) que ce que la
            vraie capacite de generalisation justifie. None (par defaut) :
            KFold ordinaire, comportement inchange.
        """
        rf = RandomForestRegressor(random_state=42)
        cv = GroupKFold(n_splits=5) if groups is not None else 5
        grid_search = GridSearchCV(rf, self.param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X, y, groups=groups)

        self.grid_search_ = grid_search
        self.best_estimator_ = grid_search.best_estimator_
        return self

    def _check_fitted(self):
        if self.best_estimator_ is None:
            raise NotFittedError(
                "RandomForestModel n'est pas encore ajuste -- appeler fit(X, y) d'abord."
            )

    def predict(self, X):
        """Predit log(prix) pour de nouvelles observations."""
        self._check_fitted()
        return self.best_estimator_.predict(X)

    def get_importances(self, feature_names: list) -> pd.DataFrame:
        """
        Retourne les importances de variables (feature_importances_,
        basees sur la reduction moyenne d'impurete -- MDI) triees par
        importance decroissante.

        Args:
            feature_names: noms des colonnes de X, dans le meme ordre
                que celui utilise lors de fit() (pas d'expansion
                polynomiale ici, contrairement a RidgeModel -- une seule
                valeur d'importance par colonne d'origine).

        Returns: DataFrame ["feature", "importance"].
        """
        self._check_fitted()
        importances = self.best_estimator_.feature_importances_
        df = pd.DataFrame({"feature": feature_names, "importance": importances})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_permutation_importance(
        self, X_test, y_test, feature_names: list, n_repeats: int = 10, random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Importance par permutation (Breiman, 2001 ; Fisher, Rudin & Dominici,
        2019), calculee sur le jeu de TEST -- contrairement a MDI
        (get_importances), qui est calculee sur le train et biaisee en
        faveur des variables continues/a forte cardinalite (Strobl et al.,
        2007, cf. importance_note dans save_artifacts.py). Le principe :
        permuter aleatoirement UNE colonne a la fois (en gardant tout le
        reste identique) et mesurer de combien le R² hors-echantillon se
        degrade -- une variable dont la permutation ne change presque rien
        au R² n'apporte presque rien au modele, quelle que soit sa
        cardinalite. n_repeats permutations independantes par variable
        donnent aussi un ecart-type (importance_std), absent de MDI.

        Returns: DataFrame ["feature", "importance_mean", "importance_std"],
            trie par importance_mean decroissante.
        """
        self._check_fitted()
        result = permutation_importance(
            self.best_estimator_, X_test, y_test,
            n_repeats=n_repeats, random_state=random_state, scoring="r2",
        )
        df = pd.DataFrame({
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # ── Exemple jouet : meme jeu de donnees simule que ridge_model.py,
    # pour comparaison directe des deux approches. ───────────────────────
    rng = np.random.default_rng(42)
    n = 200
    ram_go = rng.choice([4, 8, 16, 32], size=n).astype(float)
    stockage_go = rng.choice([128, 256, 512, 1024], size=n).astype(float)
    taille_ecran = rng.uniform(10, 17, size=n)

    log_prix = (
        6.0 + 0.03 * ram_go + 0.0008 * stockage_go + 0.05 * taille_ecran
        + 0.00002 * ram_go * stockage_go + rng.normal(0, 0.05, size=n)
    )

    X_toy = pd.DataFrame({"ram_go": ram_go, "stockage_go": stockage_go, "taille_ecran": taille_ecran})
    y_toy = pd.Series(log_prix, name="log_prix")

    model = RandomForestModel()
    model.fit(X_toy, y_toy)

    print("Meilleurs hyperparametres :", model.grid_search_.best_params_)
    print("\nImportances des variables :")
    print(model.get_importances(feature_names=list(X_toy.columns)))
    print("\nPredictions (5 premieres lignes) :", model.predict(X_toy.head()))
