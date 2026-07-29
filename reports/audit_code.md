# Audit de logique et de correction — pipeline hédonique Mytek.tn

> **NOTE DE STATUT (2026-07-29)** — Ce document est un **instantané figé** de l'audit du 2026-07-21 et des
> corrections qui ont suivi immédiatement. Il ne reflète PAS l'état actuel du code : des travaux de rigueur
> méthodologique substantiels ont eu lieu depuis (erreurs-types groupées par produit sur l'OLS de production,
> stabilité bootstrap du clustering N2, justification du choix de k, sélection de features calculée une seule
> fois sur les données poolées, recalibration des bornes de plausibilité, importance par permutation en plus de
> MDI...) — tous documentés dans `RAPPORT_SYNTHESE.md` (section méthodologie), pas ici. Conserver ce fichier
> comme trace historique de la découverte du bug de circularité (§3.1, le plus important du projet), mais ne
> jamais le lire comme une liste de problèmes encore ouverts sans vérifier `RAPPORT_SYNTHESE.md` d'abord.

Date : 2026-07-21
Périmètre : tous les fichiers `.py` de `src/` (scraper, preprocessing, models, utils) et les 6 notebooks de
`notebooks/` (EDA, Clustering, Comparaison des approches de clustering, Segmentation, Comparaison des modèles ML,
Évolution temporelle), lus intégralement. Les tests (`tests/`) ne sont pas dans le périmètre demandé.

Vérification de l'ordre d'exécution : les 6 notebooks ont des `execution_count` strictement séquentiels (1..N,
sans trou ni réordonnancement) — aucun notebook n'a été exécuté hors ordre. Aucune preuve directe de sortie
« périmée » n'a été trouvée (voir §5, une réserve structurelle subsiste néanmoins).

Légende : **BUG** = résultat faux/incorrect. **RISK** = fuite de données / choix méthodologique discutable qui
biaise un résultat sans le rendre trivialement faux. **STYLE** = hygiène, sans impact sur les résultats.

**Mise à jour 2026-07-21 (post-audit)** : le finding §3.1 (le plus critique) a été **corrigé** dans
`src/models/hedonic_model.py` — `cluster_id` est désormais dans `FORBIDDEN_REGRESSORS` au même titre que
`gamme_prix`, et `fit_strategy_a`/`fit_strategy_b`/`fit_strategy_c_pooled_time`/`compare_strategies` utilisent
`marque` comme effet fixe à la place. Tests mis à jour et passants (111/111), `demo_hedonic.py`/
`demo_hedonic_pooled_time.py` ré-exécutés avec succès — nouveaux adj-R² honnêtes, nettement plus bas que ceux
cités ci-dessous (ex. pc_bureau : 0.90 → 0.72 ; pc_portables : 0.85 → 0.60 ; télé_portables : 0.91 → 0.81 ;
téléviseurs quasi inchangé : 0.98 → 0.97, cf. détail dans la conversation). **Reste non corrigé** (hors périmètre
de ce correctif, scopé à `src/models/hedonic_model.py` uniquement) : la logique dupliquée `_group_n2`/
`attach_cluster_labels` dans `Comparaison_Modeles_ML_Clustering.ipynb` (qui ne passe pas par `hedonic_model.py`)
et les sorties déjà enregistrées d'`Evolution_Temporelle_Marche_Mytek.ipynb` (qui importe et utilise directement
les fonctions corrigées — ses sorties sauvegardées reflètent donc encore l'ancien comportement tant que le
notebook n'est pas ré-exécuté). Le reste de ce document est laissé tel quel comme trace de l'audit original.

---

## 0. Constat central (à lire en premier)

Le finding le plus important de cet audit (§3.1) traverse plusieurs fichiers : `cluster_id`
(`src/models/hedonic_model.py`), réutilisé tel quel dans `Evolution_Temporelle_Marche_Mytek.ipynb`, et sa variante
`group_n2` dans `Comparaison_Modeles_ML_Clustering.ipynb`, encodent tous les trois la **gamme de prix**
(`gamme_prix`, un quantile de `prix_tnd`) comme partie du nom de la modalité catégorielle utilisée comme
régresseur du modèle hédonique — alors même que le garde-fou de circularité du projet (`_check_no_circularity`,
`hedonic_model.py:160-171`) a été conçu explicitement pour interdire `gamme_prix` comme régresseur. Le garde-fou
ne vérifie que les **noms de colonnes**, pas le contenu sémantique des valeurs d'une colonne dérivée — il ne
détecte donc jamais ce cas. Conséquence : une partie substantielle des R²/adj-R² très élevés rapportés dans le
projet (souvent >0.90, jusqu'à 0.98) reflète le fait que le modèle « explique » le prix en partie avec une version
discrétisée du prix lui-même, pas uniquement avec les caractéristiques hédoniques. Détail complet en §3.1.

---

## 1. Fuites de données (data leakage)

### 1.1 — RISK — Sélection de features (Spearman/Kruskal-Wallis) calculée AVANT le split train/test

**Fichiers** : `src/preprocessing/pipeline.py:122-193` (`process_category`, appelle
`select_features_for_category` à la ligne 155-157) ; `src/preprocessing/select_features.py:150-186`
(`select_features_for_category`) ; `src/preprocessing/split.py` (le split train/test n'intervient qu'après).

**Scénario concret** : `build_processed_datasets` (`pipeline.py:196-243`) nettoie, impute et **sélectionne les
colonnes** (test de Spearman/Kruskal-Wallis contre `prix_tnd`, `select_features.py:67-147`) sur **toute** la
catégorie (ex. les 217 pc_bureau de S1) et écrit `<catégorie>_clean.csv`. Ce n'est que dans un second temps,
`src/preprocessing/split.py:151-180` lit ce CSV **déjà réduit aux colonnes retenues** et fait
`train_test_split` 80/20. Autrement dit, la décision « quelles colonnes entrent dans le modèle » a été prise en
utilisant les valeurs de `prix_tnd` de lignes qui deviendront ensuite le jeu de test.

**Impact réel probablement faible** : les effets mesurés (`|Spearman| >= 0.1`, `p < 0.05`) sont pour la plupart
très marqués (ex. `ram_go` Spearman=0.815 pour smartphones, `marque` Kruskal p<0.0001, cf. logs pipeline) —
la probabilité qu'un sous-échantillonnage à 80% inverse la décision « garder/écarter » est faible pour la
majorité des colonnes. Mais c'est une violation de principe qui rend les R² hors-échantillon rapportés
légèrement optimistes, et le risque grandit pour les catégories à petit effectif (telephones_portables, n=58-74).

### 1.2 — RISK — Imputation KNN calculée sur category entière (train+test) avant le split

**Fichier** : `src/preprocessing/impute.py:55-127` (`impute_numeric_cascade`), appelé par
`pipeline.py:135-149` **avant** que `split.py` ne sépare train/test.

Le docstring d'`impute_numeric_cascade` (lignes 67-72) documente soigneusement l'exclusion de `prix_tnd` des
`neighbor_columns` pour éviter une fuite de la **cible** — bonne pratique, bien pensée. Mais rien n'empêche
qu'une valeur manquante d'une ligne qui finira dans le **test** soit imputée par KNN en utilisant des voisins qui
finiront dans le **train** (et réciproquement) : l'imputation est une fonction de tout l'échantillon, pas du
train seul. C'est une fuite de covariables (pas de la cible), plus légère que 1.1, mais réelle — et
quantitativement non négligeable pour les petites catégories : les logs de la dernière exécution (semaine 4)
montrent p. ex. `telephones_portables` — `ram_go` : 31/74 valeurs imputées, `stockage_go` : 60/74 imputées.

**Comparer avec la bonne pratique déjà appliquée ailleurs dans le projet** : dans
`Comparaison_Modeles_ML_Clustering.ipynb` (cellule 6, fonction `detect_outliers_iqr`), les bornes de Tukey sont
explicitement calculées sur le train seul avant d'être appliquées au test — la discipline existe dans le projet,
mais n'a pas été appliquée à l'étape d'imputation du pipeline principal.

### 1.3 — BUG (résultat faux) — Circularité : `cluster_id` encode `gamme_prix` (dérivée du prix) et sert de régresseur

Voir le détail complet en **§3.1** (classé dans la section « correction du modèle hédonique » pour cohérence
avec le reste de l'analyse du modèle, mais c'est fondamentalement une fuite de la cible dans les régresseurs).

### 1.4 — Split train/test et structure temporelle : pas de problème identifié

`src/preprocessing/split.py:183-220` (`build_week_splits`) fait un split 80/20 **indépendamment pour chaque
semaine** (`discover_weeks` puis boucle), jamais un split qui mélangerait des observations de semaines
différentes. Le panel multi-semaines (`fit_strategy_c_pooled_time`, `hedonic_model.py:681-766`) n'utilise lui-même
aucun split train/test — c'est un ajustement en pleine information avec effet fixe « semaine », ce qui est la
pratique standard pour estimer un indice de prix hédonique (Triplett 2004), pas un exercice prédictif nécessitant
un holdout. Aucune fuite passé→futur identifiée sur ce point précis.

---

## 2. Extraction des spécifications (parsing)

### 2.1 — BUG — Unité de stockage anglaise « TB » jamais reconnue (Go/GB/To gérés, TB non)

**Fichiers** :
- `src/scraper/parser.py:401-427` (`_parse_storage_value`)
- `src/preprocessing/clean.py:473-501` (`parse_storage_from_text`, logique dupliquée à l'identique)

```python
match = re.search(r'(\d+)\s*To\b', text, re.IGNORECASE)   # français
...
match = re.search(r'(\d+)\s*Go\b', text, re.IGNORECASE)
...
match = re.search(r'(\d+)\s*GB\b', text, re.IGNORECASE)
...
else:
    return None, None
```

Le motif ne teste jamais `TB` (téraoctet, anglais) — seulement `To` (français). Une fiche produit dont le texte
contiendrait « 1TB » ou « 2 TB SSD » (variante orthographique plausible sur un site qui mélange contenu FR/EN,
ou sur une future catégorie) ne matche **aucun** des trois motifs et retourne silencieusement `(None, None)` :
ni `stockage_go`, ni `type_stockage` ne sont renseignés pour cette ligne, sans aucun flag ni log — viole le
principe « aucun filtrage silencieux » du README. Impact non quantifié ici (nécessiterait de re-scanner
`specs_brutes` de toutes les semaines pour compter les occurrences de « TB » non capturées) — à faire avant de
corriger, pour savoir si le nombre de lignes concerné justifie une correction rétroactive comme celle déjà faite
pour le bug RAM/cache (`clean.py`, `KNOWN_DATA_CORRECTIONS`).

### 2.2 — RISK — RAM/stockage : pas de support des valeurs décimales (contrairement à la taille d'écran)

**Fichiers** : `parser.py:332-376` (`_parse_ram`, regex `(\d+)\s*(Go|GB|Mo|MB)`), `parser.py:401-427`
(`_parse_storage_value`, regex `(\d+)\s*To\b` / `Go\b` / `GB\b`) — tous utilisent `\d+` (entier), jamais
`[\d.,]+`. À comparer avec `_parse_screen_size` (`parser.py:429-451`) qui utilise `[\d.,]+` et gère
explicitement virgule/point décimal. Un stockage annoncé « 1,5 To » (1500 Go) ou « 0,5 To » échouerait
silencieusement (retombe sur le motif Go/GB, ne matche rien, `(None, None)`). Risque faible en pratique — les
capacités RAM/stockage du marché sont presque toujours des puissances de 2 entières (8/16/32/…, 256/512/1024…)
— mais c'est une asymétrie de robustesse entre `parser.py`/`clean.py` non documentée.

### 2.3 — RISK — Ambiguïté virgule décimale vs séparateur de milliers dans `_clean_price`

**Fichier** : `src/scraper/parser.py:142-157` (`_clean_price`).

```python
cleaned = re.sub(r'[^\d.,]', '', price_text)
if ',' in cleaned and '.' not in cleaned:
    cleaned = cleaned.replace(',', '.')
```

Le docstring donne lui-même trois exemples d'entrée (« 1 299,000 TND », « 1299.000 », « 2,499 TND ») sans jamais
préciser la sortie attendue pour le troisième — et le code, appliqué à `"2,499 TND"`, retourne `2.499` (traite la
virgule comme décimale), ce qui serait absurde si `"2,499"` désignait en réalité 2499 TND (formatage anglais,
virgule = séparateur de milliers). Ce n'est heureusement qu'un **chemin de repli** : la source primaire du prix
(`_extract_price`, lignes 100-140) lit `meta[property='product:price:amount']` et fait `float(content)`
directement, sans jamais passer par `_clean_price` — cette fonction n'est utilisée que si la balise meta est
absente. Impact probablement faible (source primaire fiable, secours rarement sollicité) mais non quantifié
(nombre de lignes où le secours CSS a effectivement été utilisé — non tracé dans les données).

### 2.4 — STYLE — Bornes de valeurs aberrantes : correctement appliquées PAR CATÉGORIE, pas globalement

Vérifié explicitement : `clean.py:107-140` (`VALIDITY_BOUNDS`), `bounds.py:44-91` (`compute_bounds_from_data`,
paramètre `by="categorie"`, groupby par catégorie) — **aucun problème ici**, contrairement à l'hypothèse à
vérifier listée dans la consigne. C'est un point positif du projet, mentionné pour mémoire.

### 2.5 — STYLE — Colonnes structurellement exclues (`ALWAYS_DROP`) absentes du rapport de sélection

**Fichier** : `src/preprocessing/select_features.py:44-52` (`ALWAYS_DROP`), `select_features_for_category:170-174`.

Les colonnes de `ALWAYS_DROP` (gtin, dedup_key, specs_brutes, processeur, os, flags `*_suspect`…) sont retirées
des `candidate_columns` **avant** l'appel à `compute_effect_sizes` — elles n'apparaissent donc jamais dans le
DataFrame `report` loggé (« Rapport de sélection des colonnes »). Le principe « toute colonne exclue est
reportée avec sa raison » (README) est respecté pour les colonnes écartées par test statistique, mais pas pour
celles écartées structurellement — leur exclusion est documentée dans le code (docstring d'`ALWAYS_DROP`) mais
absente du log d'exécution que lirait un utilisateur du pipeline. Écart mineur, cohérence à améliorer plutôt que
bug.

### 2.6 — STYLE — Erreurs d'extraction du scraper journalisées uniquement en `DEBUG`, invisibles par défaut

**Fichier** : `src/scraper/parser.py` — tous les blocs `except Exception as e: logger.debug(...)` (ex. lignes 96,
122, 138, 189, 231, 280) ; `src/scraper/__init__.py:99-123` (`setup_logging`, niveau par défaut `INFO`), appelé
uniquement dans `scheduler.py:main()` — jamais lors d'un simple `import src.scraper`.

Si un sélecteur CSS casse pour une famille entière de fiches produit (changement de mise en page côté Mytek.tn),
l'échec d'extraction correspondant est avalé silencieusement à un niveau de log qui n'est, par défaut, jamais
affiché — seul `--debug` explicite le révèle. Aucun compteur agrégé (« N échecs d'extraction du nom sur cette
catégorie ») n'est produit en fin de run. Pas un bug de résultat, mais une réserve vis-à-vis du principe « aucun
filtrage silencieux ».

---

## 3. Correction de la décomposition hédonique (modèles)

### 3.1 — BUG (critique) — `cluster_id` / `group_n2` encodent `gamme_prix` (dérivée du prix) et contournent le garde-fou de circularité

**Fichiers concernés** :
- `src/models/hedonic_model.py:201-244` (`compute_price_tiers` — assigne `gamme_prix` via `pd.qcut(prix_tnd, …)`
  **par marque**)
- `src/models/hedonic_model.py:333-402` (`compute_cluster_labels`, ligne 396 :
  `cluster_ids.loc[df_unit.index] = [f"{brand}::{gamme}::c{lbl}" ...]` — `gamme` est ici directement la valeur de
  `gamme_prix`)
- `src/models/hedonic_model.py:150-171` (`FORBIDDEN_REGRESSORS = {"prix_tnd", "gamme_prix", "log_prix_tnd"}` et
  `_check_no_circularity`, qui ne compare que des **noms** de colonnes à cette liste)
- `src/models/hedonic_model.py:586-606, 621-661, 681-766, 838-877` (`fit_strategy_a`, `fit_strategy_b`,
  `fit_strategy_c_pooled_time`, `check_tier_monotonicity` — tous ajoutent `cluster_id` comme catégorielle du
  modèle)
- `notebooks/Comparaison_Modeles_ML_Clustering.ipynb`, cellule 6, fonction `_group_n2` (dump interne, ~L2680-2691) :
  `f"{marque} | {gamme} | c{cluster_segmentation}"`, où `gamme` vient de `outputs/labels/segmentation_<cat>.csv`
- `notebooks/Segmentation_Prix_Clustering_produits_technologiques.ipynb`, §2.4 (« `gamme_prix` est exclue au même
  titre que `prix_tnd` ») — l'exclusion n'est faite que pour l'**entrée** du clustering (bonne pratique, bien
  documentée), pas pour la **réutilisation** de l'étiquette de sortie comme régresseur.

**Nature du problème.** `gamme_prix` (Économique / Milieu de gamme / Premium) est un quantile de `prix_tnd`
calculé **par marque** (`qcut`). Le projet a un garde-fou explicite, documenté et testé
(`hedonic_model.py:155-171`, avec un test dans le bloc `__main__` du même fichier) pour empêcher `gamme_prix`
d'atteindre la matrice de design — parce qu'un régresseur dérivé du prix « expliquerait le prix par lui-même ».
Mais `cluster_id` (et `group_n2` dans le notebook de comparaison) est une chaîne composite
`"marque::gamme::sous-cluster"` : elle contient littéralement `gamme_prix` comme sous-chaîne. Le garde-fou
`_check_no_circularity` ne vérifie que si un **nom de colonne** appartient à `FORBIDDEN_REGRESSORS` — il ne peut
pas détecter qu'une colonne nommée `cluster_id` encode, dans ses valeurs, l'information qu'il est censé bannir.

**Preuve empirique à l'appui.** Dans `Comparaison_Modeles_ML_Clustering.ipynb` (cellule 8), le passage de la
condition « sans cluster » à « N2 » fait bondir le R² hors-échantillon de façon spectaculaire et systématique :

| Catégorie | OLS sans cluster | OLS + N2 |
|---|---|---|
| PC de Bureau | 0.470 | **0.905** |
| PC Portables | 0.562 | **0.845** |
| Téléphones Portables | 0.444 | **0.860** |
| Téléviseurs | 0.914 | 0.898 (ici N2 n'apporte rien : couverture 0%, cf. §2 ci-dessous) |

Le saut est bien plus marqué pour N2 (qui embarque `gamme_prix`) que pour N1 (`cluster_direct`, construit
**uniquement** sur des caractéristiques techniques, sans prix ni marque — cf.
`Clustering_produits_technologiques.ipynb`), où le gain est marginal voire négatif (PC Portables : OLS
0.562→0.301 avec N1, cf. cellule 8 du même notebook). C'est exactement le signal attendu si une partie du gain
de R² sous N2 est un artefact de circularité plutôt qu'un véritable pouvoir explicatif technique. Cohérent
également avec les adj-R² extrêmement élevés (0.90 à 0.98) rapportés dans `Evolution_Temporelle_Marche_Mytek.ipynb`
(`fit_strategy_a` par semaine, ex. téléviseurs S1-S3 : adj-R² = 0.978/0.979/0.979 avec seulement 2 variables
continues) — un ajustement aussi élevé pour un modèle log-linéaire à 2-3 prédicteurs continus est peu plausible
sans qu'une part substantielle du signal vienne des effets fixes `cluster_id`.

**Conséquence.** Toute conclusion du projet qui s'appuie sur les coefficients ou le pouvoir explicatif d'un
modèle incluant `cluster_id` (§7 `fit_strategy_a`/`fit_strategy_b`, §7bis `fit_strategy_c_pooled_time`, la
fonction `check_tier_monotonicity` dont la vocation même — vérifier que « Premium » prédit un prix plus élevé —
devient quasi tautologique si `cluster_id` sait déjà dans quel tercile de prix se trouve le produit) est à relire
avec cette réserve. Ce n'est PAS vrai des coefficients `ram_go`/`stockage_go`/`taille_ecran` en tant que tels
(l'estimation reste un effet **partiel**, conditionnel aux effets fixes — statistiquement valide en soi), mais le
R²/adj-R²/AIC-BIC rapportés comme mesure de qualité du modèle, et toute comparaison de stratégies fondée dessus
(§8 `compare_strategies`, le test de Chow), sont optimistes dans une mesure non quantifiée. `check_tier_monotonicity`
en particulier perd une grande partie de sa valeur diagnostique : elle teste si le prix prédit croît avec la
gamme, alors que le modèle qui prédit sait déjà (via `cluster_id`) dans quelle gamme est le produit.

**Ce qui n'est PAS affecté** : le clustering N1 lui-même (construit sans prix ni marque, `Clustering_produits_
technologiques.ipynb`) et son usage comme validation *a posteriori* (comparer le prix entre clusters *après*
les avoir formés sans le prix) restent méthodologiquement sains — c'est explicitement la logique du notebook
(§0, « le clustering se fait uniquement sur les caractéristiques techniques »). Le problème est spécifique à
`gamme_prix`/`cluster_id`/`group_n2`, pas au clustering en général.

### 3.2 — RISK — Importance Random Forest = impureté (MDI), jamais permutation, sans mise en garde

**Fichier** : `src/models/rf_model.py:83-100` (`get_importances`).

```python
importances = self.best_estimator_.feature_importances_  # MDI
```

Aucune alternative par permutation (`sklearn.inspection.permutation_importance`) n'est implémentée dans le
projet, et le docstring ne mentionne pas le biais connu du MDI en faveur des variables continues/à forte
cardinalité par rapport aux variables catégorielles à faible cardinalité (Strobl et al., 2007) — pertinent ici
car le jeu de features mélange des continues (`ram_go`, `stockage_go`) et des indicatrices one-hot (`has_5g`,
`os_platform_*`). Dans `Comparaison_Modeles_ML_Clustering.ipynb` (§6), l'importance de `stockage_go` domine
très largement (0.44-0.59) devant les indicatrices catégorielles — cohérent avec le biais MDI documenté dans la
littérature, sans qu'on puisse trancher ici quelle part vient d'un effet réel vs du biais de la métrique.

### 3.3 — RISK (documenté, faible sévérité) — Rétro-transformation log→TND sans correction de biais (Duan/Miller)

**Fichier** : `notebooks/Comparaison_Modeles_ML_Clustering.ipynb`, cellule 6, fonction `evaluate_predictions`
(dump ~L2694-2712).

```python
price_pred = np.exp(y_pred_log)   # pas de correction de smearing
rmse_tnd = np.sqrt(mean_squared_error(price_true, price_pred))
```

Le docstring reconnaît explicitement l'absence de correction de biais de retransformation et argumente que la
comparaison **relative** entre modèles reste valide sans elle — argument défendable pour un classement de
modèles. Reste que si `rmse_tnd_test`/`mae_tnd_test` sont un jour cités comme une estimation absolue de
l'erreur en dinars (ex. dans un rapport final ou une présentation), le lecteur devrait être averti que ces
valeurs sous-estiment systématiquement l'erreur/le prix moyen prédit — la mention actuelle n'existe que dans un
docstring interne au notebook, pas dans les cellules d'interprétation visibles (§5, §6, §8).

### 3.4 — Vérifié conforme — Interprétation des coefficients log-linéaires (Halvorsen & Palmquist, 1980)

**Fichiers** : `hedonic_model.py:514-549` (`HedonicOLS.get_coefficients`), `ridge_model.py:104-152`
(`RidgeModel.get_coefficients`).

Contrairement à l'hypothèse à vérifier listée dans la consigne, le projet applique correctement la distinction
entre variable continue (`pct_effect = coefficient * 100`, approximation valide près de 0) et variable
catégorielle/indicatrice (`pct_effect = (exp(coefficient) - 1) * 100`, formule exacte) — vérifié dans
`hedonic_model.py:536-542` avec un test explicite `if name in self.continuous_cols_`. Aucun endroit du code ou
des notebooks n'a été trouvé où un coefficient log-space serait présenté comme un montant en dinars. Point
positif, mentionné pour mémoire — sous réserve de la remarque en §3.1 sur ce que ces coefficients signifient
réellement une fois `cluster_id` dans le modèle.

### 3.5 — Vérifié conforme — Standardisation Ridge, sélection d'alpha par CV

**Fichier** : `src/models/ridge_model.py:66-86` (`RidgeModel.fit`).

`Pipeline([PolynomialFeatures, StandardScaler, Ridge])` + `GridSearchCV(cv=5)` — le `Pipeline` garantit que
`StandardScaler`/`PolynomialFeatures` sont réajustés à chaque fold **d'entraînement**, jamais sur l'ensemble
(cf. commentaire explicite du module, ligne 10-13). La CV est bien interne au train (le notebook de comparaison
appelle `RidgeModel().fit(X_train, y_train)` puis évalue sur `X_test` séparément — `Comparaison_Modeles_ML_
Clustering.ipynb`, cellule 6, `fit_and_evaluate`). Aucun problème trouvé sur ce point précis.

### 3.6 — Vérifié conforme — Métriques en log-espace ET en espace prix, toutes deux explicitement nommées

`evaluate_predictions` (cf. §3.3) calcule et retourne séparément `r2_test`/`rmse_log_test`/`mae_log_test` (échelle
log) et `rmse_tnd_test`/`mae_tnd_test` (échelle prix, via `np.exp`), avec des noms de clé sans ambiguïté — pas de
confusion entre les deux échelles trouvée dans les cellules d'affichage.

---

## 4. Cohérence code / README

| Affirmation du README | Vérifiée dans le code | Verdict |
|---|---|---|
| « Chaque catégorie est analysée séparément » | `pipeline.py` boucle `for category in df["categorie"].unique()` ; tous les notebooks segmentent par catégorie | **Conforme** |
| « Analyse par catégorie (jamais groupée) » | Idem, + `fit_strategy_c_pooled_time` groupe des **semaines**, pas des catégories (pooling temporel explicite et distinct, documenté comme tel) | **Conforme** (le pooling multi-semaines n'est pas ce que le README vise) |
| « segmentation par marque avant clustering » | `Segmentation_Prix_Clustering...ipynb` §2.1-2.4 : marque → gamme → clustering, dans cet ordre | **Conforme** |
| « Aucun filtrage silencieux : toute ligne ou colonne exclue est reportée » | Vrai pour le filtrage statistique (`select_features.py`, rapport complet) et pour `clean_products` (dédoublonnage/bornes toutes loggées) ; **pas vrai** pour : l'unité « TB » non reconnue (§2.1, silencieuse), les colonnes `ALWAYS_DROP` absentes du rapport (§2.5, mineur), les exceptions de parsing en DEBUG (§2.6) | **Partiellement conforme** — voir §2.1/2.5/2.6 |

Aucune autre divergence README/code identifiée.

---

## 5. Réserve structurelle sur la fraîcheur des notebooks

Les 6 notebooks pointent tous explicitement et en dur vers `data/processed/week_1/` (jamais vers le panel
multi-semaines à la racine de `data/processed/`), avec des commentaires qui expliquent pourquoi (éviter de
compter deux fois un même produit observé sur plusieurs semaines). C'est délibéré et documenté. Mais aucun
mécanisme automatisé (test, hook CI, notebook re-exécuté en CI) ne garantit que les *sorties déjà enregistrées*
dans les fichiers `.ipynb` reflètent l'état courant de `data/processed/week_1/*.csv` — si ce dernier est
régénéré (nouvelle collecte, correction dans `clean.py`, recalibration des bornes), les notebooks doivent être
ré-exécutés **manuellement** pour rester synchrones, sans qu'un désaccord soit détecté automatiquement. Une
vérification croisée ponctuelle (comparaison des silhouettes/k retenus pour smartphones entre
`Clustering_produits_technologiques.ipynb` et le correctif documenté dans `Comparaison_Modeles_ML_Clustering.
ipynb` §1) montre une cohérence correcte au moment de cet audit — mais c'est une vérification manuelle, pas une
garantie structurelle. RISK d'hygiène, pas un bug constaté.

---

## 6. Liste priorisée

### Changent les résultats (must-fix)

1. **§3.1 (BUG, critique)** — Circularité `cluster_id`/`group_n2` via `gamme_prix`. Affecte le R²/AIC/BIC de
   toutes les stratégies de régression hédonique utilisant `cluster_id` (A, B, C, comparaison de stratégies,
   test de monotonie des gammes) et les conclusions de `Comparaison_Modeles_ML_Clustering.ipynb` sur l'intérêt
   prédictif du clustering N2. Recommandation : soit remplacer `cluster_id` par un identifiant qui ne fait QUE
   grouper les observations (fixed effect anonyme, ex. un entier arbitraire par groupe marque×gamme×sous-cluster,
   sans que la marque ni la gamme ne soient elles-mêmes ajoutées séparément comme catégorielles) — mais le
   problème resterait le même tant que la partition elle-même dépend de `gamme_prix` ; soit, plus proprement,
   distinguer analytiquement l'effet de la **marque seule** (fixed effect légitime, ne dépend pas du prix
   individuel) de celui de la **gamme** (dérivée du prix, à exclure du régresseur ou à traiter comme la variable
   dont on veut justement mesurer l'écart résiduel, jamais comme un contrôle).
2. **§2.1 (BUG)** — Unité « TB » jamais reconnue dans le parsing du stockage (`parser.py` et `clean.py`,
   logique dupliquée). Corriger le motif regex dans les deux fichiers, puis quantifier combien de lignes
   historiques sont concernées avant de décider s'il faut une correction rétroactive de type
   `KNOWN_DATA_CORRECTIONS`.
3. **§1.1 et §1.2 (RISK)** — Sélection de features et imputation calculées avant le split train/test. Impact
   probablement faible vu la marge des effets mesurés, mais à corriger si le projet doit un jour produire des
   métriques hors-échantillon présentées comme rigoureusement propres (ex. publication, revue externe) : faire le
   split AVANT `select_features_for_category`/`impute_numeric_cascade`, ou au moins documenter explicitement
   cette limite à côté des R² rapportés.

### Hygiène (ne changent pas les résultats actuels)

4. **§3.2 (RISK)** — Ajouter une mise en garde sur le biais MDI de `RandomForestModel.get_importances`, ou
   ajouter une option d'importance par permutation.
5. **§2.2 (RISK)** — Support des valeurs décimales pour RAM/stockage, par cohérence avec `_parse_screen_size`.
6. **§2.3 (RISK)** — Clarifier/tester le comportement de `_clean_price` sur l'ambiguïté virgule décimale vs
   milliers (chemin de repli, faible exposition réelle).
7. **§3.3 (RISK, déjà documenté)** — Rendre visible dans les cellules d'interprétation (pas seulement le
   docstring) que `rmse_tnd`/`mae_tnd` ne sont pas corrigés du biais de retransformation.
8. **§2.5, §2.6 (STYLE)** — Inclure les colonnes `ALWAYS_DROP` dans le rapport de sélection ; remonter les
   erreurs de parsing du scraper au moins en `WARNING` avec un compteur agrégé par catégorie.
9. **§5 (STYLE)** — Envisager un test/notebook de fumée qui vérifie que les métriques clés d'un notebook
   (silhouette, k retenu, R²) correspondent à une ré-exécution rapide sur les données courantes, pour détecter
   automatiquement une dérive future.

---

## Points positifs constatés (pour contexte, non exhaustif)

- Bornes de valeurs aberrantes calibrées et appliquées **par catégorie**, jamais globalement (§2.4).
- Interprétation des coefficients log-linéaires correcte (formule exacte pour les catégorielles) (§3.4).
- Pipeline Ridge sans fuite de standardisation/sélection d'alpha (§3.5).
- Détection des outliers (Tukey) dans `Comparaison_Modeles_ML_Clustering.ipynb` calculée sur train seul,
  appliquée à train+test — bonne pratique correctement appliquée à cet endroit précis (contraste avec §1.2).
- Traçabilité et transparence globalement fortes : bugs déjà trouvés et documentés dans l'historique du projet
  lui-même (RAM/cache, HDR, Wi-Fi, taille d'écran aberrante) avec correction à la source ET rapport chiffré de
  l'impact — méthodologie exemplaire, à reproduire pour les points 1 et 2 de la liste priorisée ci-dessus.
