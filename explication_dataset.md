# Explication ligne par ligne de `dataset.py`

**Rôle du script (étape 1/4)** : fabriquer le *dataset*, c'est-à-dire une grande table d'exemples. Chaque exemple associe :

- une **architecture d'antenne** (combien d'éléments sur chacun des 5 anneaux) ;
- les **performances** de cette architecture (gain, lobe secondaire, largeur de faisceau, direction visée).

Le cGAN apprendra plus tard à faire le chemin inverse : à partir des performances voulues, proposer une architecture.

## Vocabulaire à connaître avant de lire

| Terme | Signification simple |
|---|---|
| Réseau circulaire multi-anneaux | Des antennes placées sur des cercles concentriques (ici 5 cercles). |
| `theta0` | La direction (en degrés) dans laquelle on veut pointer le faisceau. |
| Diagramme de rayonnement | La courbe qui dit, pour chaque direction, quelle puissance l'antenne émet. |
| Gmax | Gain du lobe principal : la puissance maximale, en dB. |
| SSL (*Side Lobe Level*) | Niveau du plus grand lobe secondaire (parasite), en dB, relatif au lobe principal. Plus il est négatif, mieux c'est. |
| HPBW | Largeur du lobe principal à −3 dB (en degrés). |
| Facteur de réseau (AF) | La somme des ondes émises par tous les éléments. C'est avec lui qu'on calcule le diagramme. |

---

## 1. En-tête et imports

```python
import numpy as np
```
Charge **NumPy**, la bibliothèque de calcul sur des tableaux de nombres. `as np` permet d'écrire `np.` au lieu de `numpy.`.

```python
from scipy.signal import find_peaks
```
Charge de **SciPy** la fonction `find_peaks`, qui repère les « sommets » d'une courbe. On s'en servira pour trouver les lobes du diagramme.

```python
import scipy.io
```
Charge le module de SciPy qui lit et écrit des fichiers **`.mat`** (le format de MATLAB).

```python
import os
```
Charge le module `os`, qui sert à manipuler les chemins de fichiers et dossiers.

---

## 2. Bannière de démarrage

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
```
- `def clear_console():` définit une **fonction** (un bloc de code réutilisable) nommée `clear_console`.
- `print(...)` affiche du texte. `"\n"` est un saut de ligne ; `"=" * 80` répète le signe `=` 80 fois ; `" " * 30` met 30 espaces pour centrer le titre.
- Résultat : une grande barre « NOUVELLE EXÉCUTION » qui sépare visuellement chaque lancement dans la console.

```python
clear_console()
```
**Appelle** la fonction : sans cette ligne, la fonction serait définie mais jamais utilisée.

---

## 3. Paramètres modifiables

```python
nb_samples = 1000
```
Nombre d'exemples à générer. Plus il y en a, mieux le réseau apprend, mais plus le script est long.

```python
max_rings = 5
```
Nombre d'anneaux de l'antenne.

```python
max_elements = 10
```
Nombre maximal d'éléments par anneau (donc chaque anneau en contient entre 0 et 10).

```python
theta0_max_deg = 180
```
La direction visée `theta0` sera tirée entre 0° et 180°.

```python
carrierFreq = 2.45e9
```
Fréquence de fonctionnement : 2,45 GHz (`2.45e9` = 2,45 × 10⁹ Hz).

```python
c = 3e8
```
Vitesse de la lumière : 300 000 000 m/s.

```python
lambda_ = c / carrierFreq
```
**Longueur d'onde** λ = c / f (environ 12,2 cm). Le nom a un `_` final car `lambda` est un mot réservé de Python.

```python
k = 2 * np.pi / lambda_
```
**Nombre d'onde** k = 2π/λ. Il convertit une distance en un déphasage (un angle) pour l'onde.

```python
r0 = 0.2 * lambda_
```
Rayon du **premier anneau** : 0,2 λ.

```python
delta_r = 0.5 * lambda_
```
Écart entre deux anneaux consécutifs : 0,5 λ. Les rayons valent donc 0,2λ ; 0,7λ ; 1,2λ ; 1,7λ ; 2,2λ.

```python
np.random.seed(46)
```
Fixe la **graine** du générateur aléatoire. Avec la même graine, on obtient exactement les mêmes nombres « aléatoires » à chaque exécution, donc un dataset reproductible (et identique à celui de ton FFNN).

---

## 4. Initialisation des matrices

```python
Moutput = np.zeros((max_rings, nb_samples))
```
Crée un tableau de **5 lignes × 1000 colonnes** rempli de zéros. Chaque **colonne** sera un exemple ; les 5 lignes seront le nombre d'éléments des 5 anneaux.

```python
Minput = np.zeros((4, nb_samples))
```
Tableau de **4 lignes × 1000 colonnes**. Les 4 lignes : `[Gmax ; SSL ; HPBW ; theta0]`.

> Convention des noms : « input » et « output » sont vus du côté du **FFNN** (entrée = performances, sortie = architecture). Pour le cGAN, `Minput` joue le rôle de la *condition* et `Moutput` celui de la *donnée à générer*.

---

## 5. Boucle principale : un tour = un exemple

```python
for sample_idx in range(nb_samples):
```
Répète le bloc indenté pour `sample_idx` = 0, 1, 2, …, 999.

### 5.1 Tirage de l'architecture

```python
    while True:
```
Boucle **infinie** : on la quittera avec `break` quand une architecture valide est trouvée.

```python
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
```
Tire 5 entiers aléatoires entre 0 et 10 inclus (le `+ 1` car la borne haute de `randint` est exclue). C'est le nombre d'éléments de chaque anneau.

```python
        if np.sum(elements_per_ring > 0) > 0:
```
`elements_per_ring > 0` donne 5 booléens (vrai si l'anneau a au moins un élément). `np.sum` compte les vrais. On teste qu'**au moins un anneau est actif**.

```python
            break
```
Si oui, on sort de la boucle. Sinon on retire une autre architecture (une antenne sans aucun élément n'aurait pas de diagramme).

```python
    Moutput[:, sample_idx] = elements_per_ring
```
Range ces 5 nombres dans la colonne `sample_idx` de `Moutput`. `:` veut dire « toutes les lignes ».

### 5.2 Direction visée

```python
    theta0deg = np.random.uniform(0, theta0_max_deg)
```
Tire un nombre réel entre 0° et 180°.

```python
    theta0 = np.deg2rad(theta0deg)
```
Convertit en **radians** (les fonctions `sin` et `cos` de NumPy attendent des radians).

```python
    phi0 = 0
```
Angle d'azimut de pointage fixé à 0 (on travaille dans un seul plan).

### 5.3 Rayons et axe des angles

```python
    radii = r0 + delta_r * np.arange(max_rings)
```
`np.arange(5)` = [0,1,2,3,4]. Donc `radii` = r0 + 0,5λ × [0..4] : les 5 rayons des anneaux.

```python
    theta = np.linspace(0, 2 * np.pi, 1000)
```
1000 angles régulièrement répartis de 0 à 360° (en radians) : ce sont les directions où l'on calcule le diagramme.

```python
    phi = 0
```
Plan de coupe (azimut 0).

```python
    AF_az = np.zeros_like(theta, dtype=complex)
```
Tableau de 1000 **nombres complexes** nuls (même taille que `theta`). On y accumulera la somme des ondes. Un nombre complexe encode à la fois l'amplitude et la phase d'une onde.

### 5.4 Somme des ondes de tous les éléments

```python
    for ring in range(max_rings):
```
Parcourt les 5 anneaux.

```python
        a = radii[ring]
```
Rayon de l'anneau courant.

```python
        N = elements_per_ring[ring]
```
Nombre d'éléments de cet anneau.

```python
        if N == 0:
            continue
```
Si l'anneau est vide, `continue` saute directement à l'anneau suivant.

```python
        phi_n = 2 * np.pi * np.arange(N) / N
```
Positions angulaires des N éléments, **régulièrement espacés** sur le cercle (0, 2π/N, 2×2π/N, …).

```python
        for n in range(N):
```
Parcourt chaque élément de l'anneau.

```python
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
```
Le déphasage de l'élément n, pour chacune des 1000 directions :
- `k * a * sin(theta) * cos(phi - phi_n)` : retard de propagation de l'onde de cet élément dans la direction `theta` ;
- `- k * a * sin(theta0) * cos(phi0 - phi_n)` : **correction de pointage**, qui aligne toutes les ondes en phase dans la direction `theta0`. C'est ce qui oriente le faisceau.

```python
            AF_az += np.exp(1j * phase)
```
`np.exp(1j * phase)` est l'onde de l'élément (de module 1 et de phase `phase`). `+=` l'ajoute à la somme. Après toutes les boucles, `AF_az` est le **facteur de réseau**.

### 5.5 Du facteur de réseau au diagramme en dB

```python
    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
```
- `np.abs` : module (amplitude) du complexe ;
- on divise par le maximum, donc la valeur max devient 1 (**normalisation**) ;
- `np.finfo(float).eps` est un tout petit nombre (≈ 2·10⁻¹⁶) ajouté pour éviter une division par zéro.

```python
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
```
Conversion en **décibels** : 20·log10(amplitude). Le maximum vaut 0 dB. Le `eps` évite `log10(0)`.

```python
    AF_dB_az[AF_dB_az < -40] = -40
```
Tout ce qui est en dessous de −40 dB est ramené à −40 dB (on coupe le « plancher » du diagramme).

```python
    theta_deg = np.rad2deg(theta)
```
Angles reconvertis en degrés, pour les calculs de largeur et pour les tracés.

### 5.6 Gain maximal non normalisé

```python
    AF_abs_az = np.abs(AF_az)
```
Module de la somme, **sans** normalisation.

```python
    maxVal = np.max(AF_abs_az)
```
Valeur maximale : c'est le nombre total d'éléments si toutes les ondes sont en phase.

```python
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
```
Gain du lobe principal en dB : **Gmax**. Exemple : 20 éléments en phase donnent 20·log10(20) ≈ 26 dB.

### 5.7 Calcul du HPBW (largeur à −3 dB)

```python
    maxVal_dB = np.max(AF_dB_az)
```
Maximum du diagramme normalisé (0 dB).

```python
    maxIdx = np.argmax(AF_dB_az)
```
**Position** (indice) de ce maximum dans le tableau.

```python
    halfPower = maxVal_dB - 3
```
Le niveau « demi-puissance » : 3 dB sous le maximum.

```python
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
```
Colle le diagramme 3 fois bout à bout. Le diagramme est **circulaire** (360° rejoint 0°) : si le lobe est près de 0° ou de 360°, il serait coupé ; avec 3 copies, on peut chercher des deux côtés sans être bloqué par le bord.

```python
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
```
Les angles associés aux 3 copies : −360…0, 0…360, 360…720.

```python
    maxIdx_ext = maxIdx + len(theta_deg)
```
Position du maximum dans la copie du **milieu**.

```python
    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None
```
Côté **gauche** du maximum : on cherche les points sous le seuil −3 dB (`np.where`), on garde le **dernier** (`[-1]`, donc le plus proche du pic). `np.any(...)` vérifie qu'il en existe un ; sinon on met `None` (introuvable).

```python
    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
```
Côté **droit** : tous les points sous le seuil après le maximum (indices relatifs au maximum).

```python
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None
```
On prend le **premier** (`[0]`, le plus proche du pic) et on ajoute `maxIdx_ext` pour revenir à un indice absolu ; `None` s'il n'y en a pas.

```python
    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]
```
Si un des deux bords est introuvable (lobe très large), on fixe HPBW = 180° par convention. Sinon HPBW = angle de droite − angle de gauche.

### 5.8 Gain et SSL

```python
    responseLin = AF_norm_az
```
On travaille sur le diagramme normalisé **linéaire** (pas en dB) pour chercher les pics.

```python
    peaks, _ = find_peaks(responseLin, distance=5)
```
`find_peaks` renvoie les indices des sommets. `distance=5` impose au moins 5 points d'écart entre deux sommets (évite de compter du bruit). Le `_` ignore la deuxième valeur renvoyée (des propriétés dont on n'a pas besoin).

```python
    pk = responseLin[peaks]
```
La hauteur de chaque sommet.

```python
    if len(pk) == 0:
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 0
```
Cas particulier : aucun sommet trouvé. On met le gain max et un SSL de 0.

```python
    else:
        sorted_idx = np.argsort(pk)[::-1]
```
`np.argsort` donne l'ordre qui trierait les hauteurs en croissant ; `[::-1]` inverse : du **plus haut au plus bas**.

```python
        sorted_pk = pk[sorted_idx]
```
Les hauteurs triées.

```python
        sorted_angles = theta_deg[peaks][sorted_idx]
```
Les angles correspondants, dans le même ordre (calculés ici mais non utilisés dans ce script ; ils servent dans le code de tracé fourni).

```python
        threshold_dB = 1
```
Tolérance de 1 dB.

```python
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]
```
On considère comme **lobe principal** tout sommet situé à moins de 1 dB du plus haut. (Il peut y en avoir plusieurs s'ils sont presque égaux.)

```python
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)
```
Les sommets **restants** (tous les indices sauf ceux du lobe principal) : ce sont les lobes secondaires.

```python
        main_lobe_gain = maxVal_non_norm
```
Gain du lobe principal.

```python
        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0
```
Le SSL est le **plus haut lobe secondaire** (`side_lobe_idx[0]`, car trié), converti en dB. S'il n'y en a aucun, SSL = 0.

### 5.9 Stockage

```python
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg
```
Remplit la colonne `sample_idx` de `Minput` avec les 4 valeurs (ligne 0 : Gmax, ligne 1 : SSL, ligne 2 : HPBW, ligne 3 : theta0).

---

## 6. Affichage de contrôle

```python
print("\nCaractéristiques de la matrice Minput :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
```
Affiche un titre puis la **forme** de la matrice. `f"..."` est une chaîne où `{...}` est remplacé par la valeur. Attendu : `(4, 1000)`.

```python
print("→ Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))
```
Affiche les 5 premières colonnes (`:5`), arrondies à 4 décimales.

```python
print("\nCaractéristiques de la matrice Moutput :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))
```
Même chose pour `Moutput` (forme attendue `(5, 1000)`), avec `.astype(int)` pour afficher des entiers (3 plutôt que 3.0).

---

## 7. Sauvegarde

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
```
- `__file__` : chemin du script en cours ;
- `os.path.abspath` : chemin complet ;
- `os.path.dirname` : le **dossier** qui le contient.

Les fichiers seront donc écrits à côté du script, où que tu le lances.

```python
scipy.io.savemat(os.path.join(script_dir, 'Minput.mat'), {'Minput': Minput})
scipy.io.savemat(os.path.join(script_dir, 'Moutput.mat'), {'Moutput': Moutput})
```
- `os.path.join` assemble dossier + nom de fichier correctement ;
- `savemat` écrit un fichier `.mat` ; le dictionnaire `{'Minput': Minput}` donne à la variable le nom `Minput` quand on l'ouvre dans MATLAB.

```python
np.save(os.path.join(script_dir, 'Minput.npy'), Minput)
np.save(os.path.join(script_dir, 'Moutput.npy'), Moutput)
```
Enregistre les mêmes matrices au format **`.npy`** (NumPy), que les scripts suivants rechargent.

```python
print("✅ Minput.mat/.npy et Moutput.mat/.npy ont été enregistrés dans le dossier du script.")
```
Message de confirmation.

---

## Résultat

Quatre fichiers : `Minput.mat`, `Minput.npy` (4 × 1000) et `Moutput.mat`, `Moutput.npy` (5 × 1000). **Une colonne = un exemple.**
