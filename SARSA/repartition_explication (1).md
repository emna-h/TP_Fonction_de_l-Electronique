# Explication ligne par ligne de `repartition.py`

**Rôle du script (étape 2/4)** : séparer les 1000 configurations en trois groupes : **training** (810), **validation** (90) et **test** (100).

## L'idée en quelques phrases

Pour savoir si l'agent SARSA a vraiment appris quelque chose, il faut le juger sur des configurations qu'il n'a **jamais rencontrées** pendant l'entraînement. Comme chaque colonne de `Minput`/`Moutput` est un scénario indépendant, on sépare des **colonnes entières**.

- **training** : les configurations sur lesquelles l'agent s'entraîne ;
- **validation** : sert à régler les hyperparamètres et à suivre la convergence (jamais à l'entraînement) ;
- **test** : jugement final. On ne s'en sert **qu'à la fin**.

Les 100 configurations de test sont **les mêmes que celles du FFNN et du cGAN** (même graine 46, même permutation), pour pouvoir comparer les trois approches sur le même ensemble.

## Vocabulaire

| Terme | Signification simple |
|---|---|
| Graine (seed) | Nombre qui fixe le « hasard » : avec la même graine, on obtient toujours le même mélange. |
| Permutation | Un mélange aléatoire de la liste des indices 0, 1, ..., 999. |
| Indice | Le numéro d'une colonne (0 = première configuration). |
| Fuite (leakage) | Quand une même information se retrouve à la fois dans l'entraînement et dans le test : le test devient trop facile et ne mesure plus rien. |
| `.npy` / `.mat` | Formats de fichiers : NumPy (Python) et MATLAB. |

---

## 1. Imports

```python
import numpy as np
import scipy.io
import os
```
- `numpy` : calcul sur des tableaux (nos matrices) ;
- `scipy.io` : permet d'écrire des fichiers `.mat` pour MATLAB ;
- `os` : manipuler des chemins de fichiers.

## 2. Bannière de démarcation

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")

clear_console()
```
- `def clear_console():` définit une petite fonction ;
- `print(...)` affiche une ligne de 80 signes « = », le texte « NOUVELLE EXÉCUTION », puis une autre ligne : cela sépare visuellement deux exécutions dans la console ;
- `"\n"` signifie « retour à la ligne », `"=" * 80` répète le signe 80 fois ;
- la ligne `clear_console()` appelle la fonction : sans cet appel, rien ne s'afficherait.

## 3. Graine aléatoire

```python
np.random.seed(46)
```
Fixe la graine à **46**, comme tous les autres scripts du projet. Conséquence : la permutation de la ligne 36 est toujours la même, donc le test est toujours le même ensemble de 100 configurations (et identique à celui du FFNN et du cGAN).

## 4. Chargement des données

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput = np.load(os.path.join(script_dir, 'Minput.npy'))
Moutput = np.load(os.path.join(script_dir, 'Moutput.npy'))
```
- `__file__` est le chemin du script ; `os.path.abspath` le rend complet ; `os.path.dirname` garde seulement le **dossier**. Ainsi le script lit et écrit toujours dans son propre dossier, où que tu le lances ;
- `np.load` lit `Minput.npy` (4 lignes × 1000 colonnes : gain, SSL, HPBW, θ0) et `Moutput.npy` (5 lignes × 1000 colonnes : éléments par anneau), créés par `dataset.py` ;
- `os.path.join` colle proprement le dossier et le nom de fichier.

## 5. Mélange des indices

```python
total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)
```
- `Minput.shape` est `(4, 1000)` ; `[1]` prend le nombre de colonnes : `total_cols = 1000` ;
- `np.random.permutation(1000)` renvoie les nombres 0 à 999 dans un ordre mélangé (graine 46). Les **premières** valeurs serviront au test, puis à la validation, puis le reste à l'entraînement.

## 6. Découpage

```python
nb_test = round(0.10 * total_cols)                  # 100
nb_val = round(0.10 * (total_cols - nb_test))       # 90
```
- `nb_test = round(0.10 * 1000) = 100` : 10 % des configurations pour le test ;
- `nb_val = round(0.10 * 900) = 90` : 10 % des 900 restantes pour la validation.

```python
test_indices = rand_indices[:nb_test]
val_indices = rand_indices[nb_test:nb_test + nb_val]
train_indices = rand_indices[nb_test + nb_val:]
```
Les crochets `[a:b]` découpent la liste mélangée :
- `rand_indices[:100]` : les 100 premiers indices → **test** (le même découpage que dans le script du FFNN/cGAN) ;
- `rand_indices[100:190]` : les 90 suivants → **validation** ;
- `rand_indices[190:]` : tout le reste (810) → **training**.

## 7. Contrôle de fuite entre ensembles

```python
assert len(set(test_indices) & set(val_indices)) == 0
assert len(set(test_indices) & set(train_indices)) == 0
assert len(set(val_indices) & set(train_indices)) == 0
assert len(test_indices) + len(val_indices) + len(train_indices) == total_cols
```
`assert condition` arrête le script avec une erreur si la condition est fausse. On vérifie ici :
- `set(...) & set(...)` est l'**intersection** de deux ensembles : elle doit être vide (taille 0) pour test/validation, test/training et validation/training : aucune configuration n'est commune à deux groupes ;
- la somme des trois tailles doit faire 1000 : rien n'a été oublié.

## 8. Fuite approchée : architectures en double

Le dataset contient 1000 scénarios mais seulement **998 architectures différentes** : deux architectures sont tirées deux fois (avec des θ0 différents, donc ce sont bien deux scénarios différents). Si l'une des deux copies était au test et l'autre à l'entraînement, il y aurait une petite fuite. Les lignes suivantes le vérifient.

```python
nom_ens = {}
for nom, idx in [('training', train_indices), ('validation', val_indices), ('test', test_indices)]:
    for i in idx:
        nom_ens[i] = nom
```
- `nom_ens = {}` crée un **dictionnaire** (une table « clé → valeur ») ;
- la boucle `for nom, idx in [...]` parcourt les trois groupes avec leur nom ;
- la boucle intérieure `for i in idx` enregistre, pour chaque indice `i`, le nom de son groupe. À la fin : `nom_ens[366] = 'training'`, par exemple.

```python
archi = {}
for i in range(total_cols):
    archi.setdefault(tuple(int(x) for x in Moutput[:, i]), []).append(i)
```
- `archi = {}` : dictionnaire qui associera chaque architecture à la liste des colonnes qui la contiennent ;
- `Moutput[:, i]` est la colonne `i` (5 nombres) ; `tuple(int(x) for x in ...)` la transforme en n-uplet d'entiers, par exemple `(0, 9, 6, 6, 10)`, utilisable comme clé de dictionnaire ;
- `setdefault(clé, [])` crée une liste vide si la clé est nouvelle, puis `.append(i)` y ajoute la colonne.

```python
doublons = {a: ids for a, ids in archi.items() if len(ids) > 1}
```
Ne garde que les architectures qui apparaissent **plus d'une fois** (`len(ids) > 1`). Le résultat est une compréhension de dictionnaire : `{clé: valeur for ... if condition}`.

```python
print(f"Architectures apparaissant plusieurs fois : {len(doublons)}")
for a, ids in doublons.items():
    print(f"  architecture {a} : colonnes {ids} -> ensembles {[nom_ens[i] for i in ids]}, "
          f"theta0 = {[round(float(Minput[3, i]), 2) for i in ids]}")
    if len({nom_ens[i] for i in ids}) > 1:
        print("  ATTENTION : fuite faible (même architecture, theta0 différent, ensembles différents).")
```
- affiche le nombre d'architectures en double ;
- pour chacune, affiche les colonnes concernées, leurs groupes et leurs θ0 (arrondis à 2 décimales avec `round`) ;
- `{nom_ens[i] for i in ids}` est un ensemble des groupes concernés : s'il contient plus d'un nom, les copies sont dans des groupes différents et le script affiche un **avertissement**. Dans nos données, les deux doublons tombent tous les deux dans le training : aucune fuite.

## 9. Construction des six matrices

```python
matrices = {
    'Minput_training': Minput[:, train_indices], 'Moutput_training': Moutput[:, train_indices],
    'Minput_validation': Minput[:, val_indices], 'Moutput_validation': Moutput[:, val_indices],
    'Minput_test': Minput[:, test_indices], 'Moutput_test': Moutput[:, test_indices],
}
```
Dictionnaire qui associe un nom de fichier à la sous-matrice correspondante. `Minput[:, train_indices]` signifie « toutes les lignes, uniquement les colonnes dont l'indice est dans `train_indices` ». On obtient ainsi `Minput_training` (4×810), `Moutput_training` (5×810), `Minput_validation` (4×90), etc.

```python
for nom, M in matrices.items():
    print(f"\nMatrice {nom} : shape {M.shape}")
    print(np.round(M[:, :3], 4) if nom.startswith('Minput') else M[:, :3].astype(int))
```
Pour chaque matrice, affiche sa taille (`shape`) et ses 3 premières colonnes. Pour les `Minput`, on arrondit à 4 décimales (`np.round`) ; pour les `Moutput`, on convertit en entiers (`.astype(int)`) car ce sont des nombres d'éléments.

```python
print(f"\nTest : {nb_test} configurations, validation : {nb_val}, training : {len(train_indices)}")
```
Récapitulatif des trois tailles : 100 / 90 / 810.

## 10. Enregistrement

```python
matrices['indices_split'] = np.array([np.concatenate([train_indices, val_indices, test_indices]),
                                      np.concatenate([np.zeros(len(train_indices)), np.ones(len(val_indices)),
                                                      2 * np.ones(len(test_indices))])])
```
Ajoute une septième matrice, `indices_split`, à deux lignes :
- ligne 1 : l'**indice d'origine** de chaque configuration (utile pour retrouver une configuration dans `Minput.npy`) ;
- ligne 2 : le groupe (`0` training, `1` validation, `2` test), construit avec `np.zeros`, `np.ones` et `2 * np.ones` ;
- `np.concatenate` colle les trois morceaux bout à bout.

```python
for nom, M in matrices.items():
    scipy.io.savemat(os.path.join(script_dir, nom + '.mat'), {nom: M})   # .mat
    np.save(os.path.join(script_dir, nom + '.npy'), M)                   # .npy
```
Pour chaque matrice du dictionnaire :
- `scipy.io.savemat(chemin, {nom: M})` écrit un fichier `.mat` (MATLAB) où la matrice porte le même nom ;
- `np.save(chemin, M)` écrit un fichier `.npy` (Python).
Les deux formats sont écrits dans le dossier du script, comme tu l'as demandé.

```python
print("\n✅ Matrices training / validation / test enregistrées en .mat et .npy dans le dossier du script.")
print("   indices_split : ligne 1 = indice d'origine, ligne 2 = ensemble (0 training, 1 validation, 2 test).")
```
Messages de fin : confirmation et rappel de la signification des deux lignes de `indices_split`.

---

## Résumé du flux

```
Minput.npy, Moutput.npy  (1000 colonnes)
        │ permutation (graine 46)
        ▼
 100 test  |  90 validation  |  810 training
        │
        ▼
 6 matrices + indices_split  (.npy et .mat)
```
