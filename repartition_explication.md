# Explication ligne par ligne de `repartition_training_test.py`

**Rôle du script (étape 2/4)** : couper le dataset en deux parties.

- **Training (90 %)** : les exemples sur lesquels le réseau apprend.
- **Test (10 %)** : des exemples **jamais vus** pendant l'apprentissage, gardés pour mesurer honnêtement la qualité du modèle.

Pourquoi séparer ? Un élève qui apprend par cœur les corrigés réussit les exercices qu'il connaît, mais cela ne prouve pas qu'il a compris. Le jeu de test joue le rôle de l'examen avec des questions nouvelles.

---

## 1. Imports

```python
import numpy as np
```
NumPy : calcul sur des tableaux.

```python
import scipy.io
```
Écriture de fichiers `.mat`.

```python
import os
```
Gestion des chemins de fichiers.

---

## 2. Bannière et graine

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
clear_console()
```
Définit puis appelle la fonction qui affiche la grande barre « NOUVELLE EXÉCUTION » (voir `dataset.py`).

```python
np.random.seed(46)
```
Fixe la graine aléatoire. Le « mélange » sera donc **le même** à chaque exécution : la répartition training/test est reproductible, et identique à celle de ton FFNN.

---

## 3. Chargement des matrices

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
```
Dossier qui contient le script (les fichiers sont lus et écrits là).

```python
Minput = np.load(os.path.join(script_dir, 'Minput.npy'))
Moutput = np.load(os.path.join(script_dir, 'Moutput.npy'))
```
Recharge les matrices créées par `dataset.py` : `Minput` (4 × 1000) et `Moutput` (5 × 1000). `np.load` lit un fichier `.npy`. **Il faut donc avoir lancé `dataset.py` avant.**

---

## 4. Mélange et découpage

```python
total_cols = Minput.shape[1]
```
`shape` est le tuple `(lignes, colonnes)`. `shape[1]` est le nombre de **colonnes**, c'est-à-dire d'exemples (1000).

```python
rand_indices = np.random.permutation(total_cols)
```
Crée une **permutation aléatoire** de 0…999 (les numéros d'exemples dans le désordre). Le mélange évite que le test ne contienne que les derniers exemples générés.

```python
nb_test = round(0.10 * total_cols)
```
10 % du total : 100 exemples pour le test.

```python
test_indices = rand_indices[:nb_test]
```
Les 100 **premiers** indices du mélange → exemples de test.

```python
train_indices = rand_indices[nb_test:]
```
Tous les **suivants** (900) → exemples de training. Un exemple ne peut donc jamais être dans les deux groupes.

---

## 5. Sélection des colonnes

```python
Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]
```
`Minput[:, indices]` signifie « toutes les lignes, mais seulement les colonnes listées ». On applique **exactement les mêmes indices** à `Minput` et `Moutput`, pour qu'une architecture reste associée à ses propres performances. Résultat :

| Matrice | Forme |
|---|---|
| `Minput_training` | 4 × 900 |
| `Moutput_training` | 5 × 900 |
| `Minput_test` | 4 × 100 |
| `Moutput_test` | 5 × 100 |

---

## 6. Affichage de contrôle

```python
for nom, M in [('Minput_training', Minput_training), ('Moutput_training', Moutput_training),
               ('Minput_test', Minput_test), ('Moutput_test', Moutput_test)]:
```
Boucle sur une liste de 4 paires (nom, matrice). À chaque tour, `nom` est le texte et `M` la matrice. Cela évite de recopier 4 fois les mêmes lignes d'affichage.

```python
    print(f"\nCaractéristiques de la matrice {nom} :")
    print(f"- Shape (lignes, colonnes) : {M.shape}")
    print("→ Premières colonnes (extraits) :")
```
Affiche le nom et la forme de la matrice.

```python
    print(np.round(M[:, :5], 4) if nom.startswith('Minput') else M[:, :5].astype(int))
```
Affiche les 5 premières colonnes. Si le nom commence par `Minput` (valeurs réelles), on arrondit à 4 décimales ; sinon (nombres d'éléments) on convertit en entiers. C'est une écriture condensée `A if condition else B`.

---

## 7. Sauvegarde en `.mat` et `.npy`

```python
matrices = {'Minput_training': Minput_training, 'Moutput_training': Moutput_training,
            'Minput_test': Minput_test, 'Moutput_test': Moutput_test}
```
Un **dictionnaire** : à chaque nom (clé) on associe la matrice (valeur).

```python
for nom, M in matrices.items():
```
`.items()` donne les paires (nom, matrice) du dictionnaire, une par tour.

```python
    scipy.io.savemat(os.path.join(script_dir, nom + '.mat'), {nom: M})
```
Écrit `<nom>.mat` ; la variable s'appellera `<nom>` dans MATLAB.

```python
    np.save(os.path.join(script_dir, nom + '.npy'), M)
```
Écrit `<nom>.npy`, lu par les scripts d'entraînement et de test.

```python
print("\n✅ Minput/Moutput _training et _test enregistrées en .mat et .npy dans le dossier du script.")
```
Message de confirmation.

---

## Résultat

8 fichiers : `Minput_training`, `Moutput_training`, `Minput_test`, `Moutput_test`, chacun en `.mat` et `.npy`.

Point important : **le test ne doit servir qu'à la fin**. Les statistiques de normalisation seront calculées uniquement sur le training (script suivant), pour que rien du test ne « fuite » dans l'apprentissage.
