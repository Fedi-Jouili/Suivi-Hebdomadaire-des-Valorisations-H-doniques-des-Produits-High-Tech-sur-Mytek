# -*- coding: utf-8 -*-
"""
=============================================================================
Module : src/models/ridge_model.py
=============================================================================
ROLE :
    Modele hedonique semi-log (Rosen, 1974) via regression Ridge :
    log(prix) ~ Pipeline(PolynomialFeatures -> StandardScaler -> Ridge).

    Le Pipeline sklearn garantit qu'aucune fuite de donnees ne se produit
    lors de la validation croisee : PolynomialFeatures et StandardScaler
    sont refits sur chaque fold d'entrainement par GridSearchCV, jamais sur
    l'ensemble des donnees avant le split.

ATTENTION -- INTERACTIONS SUR VARIABLES ONE-HOT :
    interaction_only=True evite les termes au carre (ram_go^2), mais si X
    contient aussi des indicatrices one-hot (marque_*, categorie_*, os_*),
    PolynomialFeatures generera des interactions ENTRE ces indicatrices.
    Deux indicatrices issues du MEME groupe one-hot (ex: marque_ASUS et
    marque_DELL) sont mutuellement exclusives -- leur produit est TOUJOURS
    nul, une colonne totalement degeneree qui n'apporte aucune information
    mais gonfle inutilement la dimension et la colinearite du probleme.
    Pour un usage optimal, ne passer a RidgeModel.fit() que les variables
    numeriques continues (ram_go, stockage_go, cpu_serie, taille_ecran...),
    pas les colonnes one-hot -- ou accepter ce bruit en connaissance de
    cause si l'on passe la matrice X complete deja encodee.

    CE QUI EST REELLEMENT DEPLOYE (save_artifacts.py) : `RidgeModel(degree=1)`,
    precisement pour eviter le probleme ci-dessus (X_train contient les
    colonnes one-hot). PolynomialFeatures(degree=1, interaction_only=True)
    ne genere AUCUN terme d'interaction (verifie : get_feature_names_out()
    retourne les colonnes d'origine inchangees) -- le Ridge de production
    est donc un modele semi-log LINEAIRE, sans interactions, malgre le
    "avec Pipeline(PolynomialFeatures...)" du paragraphe ROLE ci-dessus qui
    decrit la CAPACITE generale de la classe, pas ce qui est effectivement
    utilise. degree=2 (interactions) reste utilisable directement (cf.
    UTILISATION) mais n'a jamais ete compare empiriquement a degree=1 sur
    les donnees reelles du projet (audit methodologique, 2026-07-28).

UTILISATION :
    from src.models.ridge_model import RidgeModel
    model = RidgeModel(degree=2, alphas=[0.01, 0.1, 1, 10, 100])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coefs = model.get_coefficients(feature_names=list(X_train.columns))
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.exceptions import NotFittedError


class RidgeModel:
    """
    Regression Ridge hedonique avec selection d'alpha par validation
    croisee (GridSearchCV) et interpretation des coefficients en
    semi-elasticites (cf. get_coefficients).
    """

    def __init__(self, degree: int = 2, alphas: list | None = None):
        """
        Args:
            degree: degre polynomial maximal des interactions.
            alphas: grille de regularisation Ridge testee par GridSearchCV.
                Par defaut [0.001, 0.01, 0.1, 1, 10, 100, 500] (grille
                log-echelonnee elargie le 2026-07-28 -- l'ancienne grille
                [0.01, 0.1, 1, 10, 100] plafonnait exactement a sa borne
                pour au moins une categorie (pc_portables, alpha=100) une
                fois la CV correctement groupee par produit (cf. fit) ;
                verifie empiriquement qu'une grille plus large [0.001..5000]
                ne deplace PAS ce choix (100 reste l'optimum, pas juste la
                limite testee) mais en deplace un autre (telephones_
                portables, alpha=0.01 -> 0.001) -- une grille qui plafonne
                exactement a son propre optimum est un signal a verifier,
                jamais a ignorer.

                LIMITE RESIDUELLE CONNUE (telephones_portables) : cette
                categorie (la plus petite, n_train~233) choisit alpha=0.001,
                la borne BASSE de la grille ci-dessus -- ET CONTINUE DE
                DESCENDRE des que la borne basse est abaissee (verifie
                jusqu'a 0.0001, qui est aussi choisi) : ce n'est donc pas un
                optimum interieur cache juste sous 0.001, mais une
                preference asymptotique pour AUCUNE regularisation (la CV
                n'y trouve, sur cette petite categorie, aucun benefice au
                shrinkage Ridge -- un Ridge quasi-nul se comporte comme un
                OLS ordinaire). Non corrige : descendre encore la borne ne
                ferait que deplacer le meme phenomene plus bas, jamais le
                resoudre. A lire comme un signal legitime (le Ridge n'aide
                pas sur cette categorie, contrairement aux autres) plutot
                qu'un defaut de grille.
        """
        self.degree = degree
        self.alphas = list(alphas) if alphas is not None else [0.001, 0.01, 0.1, 1, 10, 100, 500]
        self.best_estimator_ = None
        self.grid_search_ = None

    def fit(self, X, y, groups=None):
        """
        Ajuste le Pipeline complet (Poly -> Scaler -> Ridge) avec
        selection d'alpha par validation croisee (5 plis, scoring R² par
        defaut). Le Pipeline empeche toute fuite de donnees : Poly et
        Scaler sont reajustes sur chaque fold d'entrainement, jamais sur
        l'ensemble du jeu de donnees.

        groups : identifiant de regroupement (typiquement l'url du produit)
            -- si fourni, utilise GroupKFold au lieu d'un KFold ordinaire.
            X peut contenir plusieurs lignes quasi-identiques pour un meme
            produit (une par semaine ou il a ete vu, cf. save_artifacts.py::
            load_pooled_category) ; un KFold ordinaire peut alors placer
            deux lignes du MEME produit de part et d'autre d'un pli
            (entrainement/validation), une fuite qui biaise la selection
            d'alpha vers moins de regularisation (le modele "valide" en
            partie sur une ligne quasi-identique a une ligne d'entrainement).
            None (par defaut) : KFold ordinaire, comportement inchange.
        """
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=self.degree, include_bias=False, interaction_only=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(fit_intercept=True)),
        ])

        param_grid = {"ridge__alpha": self.alphas}
        cv = GroupKFold(n_splits=5) if groups is not None else 5
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1)
        grid_search.fit(X, y, groups=groups)

        self.grid_search_ = grid_search
        self.best_estimator_ = grid_search.best_estimator_
        return self

    def _check_fitted(self):
        if self.best_estimator_ is None:
            raise NotFittedError(
                "RidgeModel n'est pas encore ajuste -- appeler fit(X, y) d'abord."
            )

    def predict(self, X):
        """Predit log(prix) pour de nouvelles observations."""
        self._check_fitted()
        return self.best_estimator_.predict(X)

    def get_best_alpha(self) -> float:
        """Retourne l'alpha retenu par la validation croisee (GridSearchCV)."""
        self._check_fitted()
        return self.best_estimator_.named_steps["ridge"].alpha

    def get_coefficients(self, feature_names: list) -> pd.DataFrame:
        """
        Retourne les coefficients Ridge RE-EXPRIMES SUR L'ECHELLE
        D'ORIGINE des variables (pas sur l'echelle standardisee interne
        au Pipeline), pour rester interpretables.

        Pourquoi diviser par scaler.scale_ :
            Le Ridge est ajuste sur x_std = (x_poly - mean_) / scale_, donc
            log(prix) = beta0 + sum(coef_j * x_std_j)
                      = cte + sum[(coef_j / scale_j) * x_poly_j]
            Le coefficient sur l'echelle ORIGINALE de x_poly_j est donc
            coef_j / scale_j -- c'est exactement (et seulement) cette
            division qui annule la standardisation, sans quoi le
            coefficient dependrait de l'ecart-type de la variable plutot
            que de son unite reelle (Go de RAM, pouces d'ecran...).

        Interpretation economique (y = log(prix)) : chaque coefficient
        est une SEMI-ELASTICITE -- une augmentation d'une unite de la
        variable est associee a une variation d'environ
        (coefficient * 100)% du prix, toutes choses egales par ailleurs
        (Halvorsen & Palmquist, 1980, pour les variables continues ;
        cf. src/notebooks/Clustering_produits_technologiques.ipynb §2.5
        pour la meme logique appliquee aux variables indicatrices, ou la
        formule exacte (exp(coef)-1)*100 est preferable).

        Args:
            feature_names: noms des colonnes ORIGINALES de X (avant
                l'expansion polynomiale), dans le meme ordre que celui
                utilise lors de fit(). Sert a nommer les termes
                d'interaction generes par PolynomialFeatures (ex:
                "ram_go stockage_go" pour l'interaction RAM x stockage --
                cf. poly.get_feature_names_out()).

        Returns: DataFrame ["feature", "coefficient"], trie par valeur
            absolue du coefficient decroissante.
        """
        self._check_fitted()
        poly = self.best_estimator_.named_steps["poly"]
        scaler = self.best_estimator_.named_steps["scaler"]
        ridge = self.best_estimator_.named_steps["ridge"]

        poly_feature_names = poly.get_feature_names_out(feature_names)
        coefficients_echelle_originale = ridge.coef_ / scaler.scale_

        df = pd.DataFrame({
            "feature": poly_feature_names,
            "coefficient": coefficients_echelle_originale,
        })
        return df.sort_values("coefficient", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # ── Exemple jouet : verifie que le pipeline s'ajuste et s'interprete
    # correctement, sans dependre des vraies donnees du projet. ──────────
    rng = np.random.default_rng(42)
    n = 200
    ram_go = rng.choice([4, 8, 16, 32], size=n).astype(float)
    stockage_go = rng.choice([128, 256, 512, 1024], size=n).astype(float)
    taille_ecran = rng.uniform(10, 17, size=n)

    # Prix simule : log-lineaire en RAM/stockage + interaction + bruit
    log_prix = (
        6.0 + 0.03 * ram_go + 0.0008 * stockage_go + 0.05 * taille_ecran
        + 0.00002 * ram_go * stockage_go + rng.normal(0, 0.05, size=n)
    )

    X_toy = pd.DataFrame({"ram_go": ram_go, "stockage_go": stockage_go, "taille_ecran": taille_ecran})
    y_toy = pd.Series(log_prix, name="log_prix")

    model = RidgeModel(degree=2, alphas=[0.01, 0.1, 1, 10, 100])
    model.fit(X_toy, y_toy)

    print("Meilleur alpha :", model.get_best_alpha())
    print("\nCoefficients (echelle originale, semi-elasticites) :")
    print(model.get_coefficients(feature_names=list(X_toy.columns)))
    print("\nPredictions (5 premieres lignes) :", model.predict(X_toy.head()))
