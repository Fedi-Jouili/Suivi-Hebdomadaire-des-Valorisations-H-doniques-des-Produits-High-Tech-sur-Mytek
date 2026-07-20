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
from sklearn.model_selection import GridSearchCV
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
                    n_estimators   : [100, 200]
                    max_depth      : [10, 20, None]  (None = pas de limite)
                    min_samples_split : [2, 5]
        """
        self.param_grid = dict(param_grid) if param_grid is not None else {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        }
        self.best_estimator_ = None
        self.grid_search_ = None

    def fit(self, X, y):
        """
        Ajuste un RandomForestRegressor(random_state=42) avec selection
        d'hyperparametres par validation croisee (cv=5, scoring R² par
        defaut, n_jobs=-1 pour paralleliser sur tous les coeurs).
        """
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, self.param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X, y)

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
