# Explication ligne par ligne de `dataset.py`

**Rôle du script (étape 1/4)** : produire la **banque de scénarios** du projet SARSA : 1000 configurations de réseau d'antennes circulaire, avec leurs performances de référence.

## L'idée en quelques phrases

Dans le FFNN et le cGAN, ce script fabrique des paires (performances → architecture) servant d'exemples à imiter. **Ici, la même matrice sert à autre chose** : chaque colonne est un **scénario**, c'est-à-dire un réseau complet qu'on voudra *alléger* en éteignant des antennes. Il n'y a pas de « bonne réponse » à prédire : les performances de référence servent de **contraintes** (on veut les garder quasiment intactes).

Le script est **volontairement identique** à celui du FFNN/cGAN (même physique, même graine 46) : tu obtiens exactement les mêmes matrices. On y a seulement ajouté une étape de **validation** (section 12).

## Les deux matrices produites

| Matrice | Taille | Contenu d'une colonne |
|---|---|---|
| `Minput` | 4 × 1000 | Gmax (dB), SSL (dB), HPBW (°), θ0 (°) : performances de référence du réseau complet |
| `Moutput` | 5 × 1000 | nombre d'éléments (0 à 10) des anneaux 1 à 5 |

## Vocabulaire

| Terme | Signification simple |
|---|---|
| Réseau d'antennes | Plusieurs petites antennes qui travaillent ensemble pour former un faisceau. |
| Anneau (ring) | Cercle sur lequel les éléments sont répartis régulièrement. On en a 5, de rayons croissants. |
| θ0 | Direction vers laquelle on veut pointer le faisceau. |
| Facteur de réseau (AF) | Somme des ondes émises par tous les éléments, en fonction de la direction θ. Son module donne la forme du diagramme de rayonnement. |
| Gmax | Valeur maximale du diagramme (le gain du lobe principal), en dB. |
| SSL (Side Lobe Level) | Niveau du plus haut lobe secondaire, en dB (plus il est bas, mieux c'est). |
| HPBW | Largeur du faisceau à −3 dB (« demi-puissance »), en degrés. |
| dB | Échelle logarithmique : 20·log10 d'un rapport d'amplitudes. |

---

## 1. Imports

```python
import numpy as np
from scipy.signal import find_peaks
import scipy.io
import os
```
- `numpy` : tableaux et calcul numérique ;
- `find_peaks` (de `scipy.signal`) : trouve les **pics** d'une courbe, utile pour repérer les lobes ;
- `scipy.io` : écriture des fichiers `.mat` ;
- `os` : chemins de fichiers.

## 2. Bannière

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")

# Appel immédiat au début du script
clear_console()
```
Même petite fonction que dans les autres scripts : elle affiche une ligne de séparation « NOUVELLE EXÉCUTION » pour distinguer deux lancements successifs dans la console. La ligne `clear_console()` l'appelle immédiatement.

## 3. Paramètres du problème

```python
nb_samples = 1000          # (le cGAN profite d'un dataset plus grand : ex. 20000 ou 50000)
max_rings = 5
max_elements = 10
theta0_max_deg = 180
```
- `nb_samples = 1000` : nombre de scénarios (colonnes) ;
- `max_rings = 5` : nombre d'anneaux ;
- `max_elements = 10` : nombre maximal d'éléments par anneau (de 0 à 10) ;
- `theta0_max_deg = 180` : θ0 sera tiré entre 0° et 180°.

```python
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
```
Constantes physiques :
- `carrierFreq = 2.45e9` : fréquence de 2,45 GHz (notation scientifique : 2,45 × 10⁹) ;
- `c = 3e8` : vitesse de la lumière en m/s ;
- `lambda_ = c / carrierFreq` : **longueur d'onde** (≈ 12,2 cm). On écrit `lambda_` avec un tiret bas car `lambda` est un mot réservé de Python ;
- `k = 2 * np.pi / lambda_` : **nombre d'onde**, utilisé dans la phase de chaque onde ;
- `r0 = 0.2 * lambda_` : rayon du premier anneau (0,2 λ) ;
- `delta_r = 0.5 * lambda_` : écart entre deux anneaux successifs (0,5 λ). Le rayon de l'anneau m vaut donc 0,2λ + 0,5λ × (m−1).

```python
np.random.seed(46)
```
Fixe la graine à **46** : tous les tirages aléatoires qui suivent sont reproductibles, et identiques à ceux du FFNN et du cGAN.

## 4. Préparation des matrices vides

```python
Moutput = np.zeros((max_rings, nb_samples))  # Architecture
Minput = np.zeros((4, nb_samples))           # [MainLobe, SSL, HPBW, theta0]
```
- `np.zeros((5, 1000))` crée un tableau 5×1000 rempli de zéros, qu'on remplira colonne par colonne : `Moutput` (architectures) ;
- idem pour `Minput` avec 4 lignes : gain, SSL, HPBW, θ0.

## 5. Boucle principale : un scénario par tour

```python
for sample_idx in range(nb_samples):
```
`sample_idx` prend les valeurs 0, 1, ..., 999 : un tour de boucle fabrique un scénario complet.

```python
    # --- Architecture complètement aléatoire ---
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:  # S'assurer qu'au moins un ring est actif
            break
    Moutput[:, sample_idx] = elements_per_ring
```
**Architecture aléatoire.**
- `while True:` répète jusqu'à ce qu'on sorte avec `break` ;
- `np.random.randint(0, 11, size=5)` tire 5 entiers entre 0 et 10 : le nombre d'éléments de chaque anneau ;
- `np.sum(elements_per_ring > 0) > 0` : on compte les anneaux non vides et on exige qu'il y en ait au moins un (sinon, pas d'antenne du tout). Si la condition est vraie, `break` termine le `while` ;
- `Moutput[:, sample_idx] = ...` range ces 5 nombres dans la colonne du scénario.

```python
    theta0deg = np.random.uniform(0, theta0_max_deg)
    theta0 = np.deg2rad(theta0deg)
    phi0 = 0
```
**Direction de pointage.**
- `np.random.uniform(0, 180)` tire un angle θ0 en degrés ;
- `np.deg2rad` le convertit en radians (les fonctions sin/cos de NumPy travaillent en radians) ;
- `phi0 = 0` : le plan de calcul est le plan φ = 0.

```python
    radii = r0 + delta_r * np.arange(max_rings)
```
Rayons des 5 anneaux : `np.arange(5)` donne `[0,1,2,3,4]`, donc `radii = r0 + delta_r × [0,1,2,3,4]`.

```python
    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = 0
    AF_az = np.zeros_like(theta, dtype=complex)
```
**Diagramme de rayonnement.**
- `theta` : 1000 angles régulièrement répartis de 0 à 2π (la grille de calcul, pas de 0,36°) ;
- `phi = 0` : plan d'observation ;
- `AF_az` : tableau de 1000 **nombres complexes** initialisés à 0 ; on y ajoutera la contribution de chaque antenne.

```python
    for ring in range(max_rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
```
Boucle sur les anneaux :
- `a = radii[ring]` : rayon de l'anneau ;
- `N = elements_per_ring[ring]` : nombre d'éléments de cet anneau ;
- `if N == 0: continue` : anneau vide, on passe au suivant ;
- `phi_n` : angles des N éléments, **régulièrement espacés** autour du cercle (`2π × n / N`, pour n = 0..N−1).

```python
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)
```
Boucle sur les éléments de l'anneau :
- `phase` : phase de l'onde émise par l'élément n, dans chaque direction θ. Le premier terme `sin(theta) cos(phi - phi_n)` dépend de la direction observée ; le second `sin(theta0) cos(phi0 - phi_n)` est la **correction qui oriente le faisceau vers θ0** (alignement des phases dans la direction θ0) ;
- `np.exp(1j * phase)` : l'onde complexe d'amplitude 1 ;
- `AF_az += ...` : on additionne les ondes de tous les éléments. Le résultat final est le **facteur de réseau**.

```python
    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    theta_deg = np.rad2deg(theta)
```
Normalisation et passage en dB :
- `np.abs(AF_az)` : module de l'onde totale ;
- division par le maximum (`+ eps` pour éviter une division par zéro ; `np.finfo(float).eps` est le plus petit écart représentable) : le pic vaut 1 ;
- `20 * np.log10(...)` : conversion en dB (le pic vaut 0 dB) ;
- `AF_dB_az[AF_dB_az < -40] = -40` : on écrase tout ce qui est sous −40 dB (plancher) ;
- `theta_deg` : les angles convertis en degrés.

```python
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)
```
**Gain non normalisé** : valeur maximale du module de l'onde totale, convertie en dB. Si tous les N éléments sont en phase, le maximum vaut N, donc **Gmax = 20·log10(N)** (loi du gain, très importante pour le projet : le gain ne dépend que du nombre d'éléments actifs).

## 6. Calcul du HPBW (largeur à −3 dB)

```python
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    maxIdx_ext = maxIdx + len(theta_deg)
```
- `maxVal_dB` : maximum du diagramme en dB (0) ; `maxIdx` : sa position ;
- `halfPower = maxVal_dB - 3` : le niveau « −3 dB », c'est-à-dire la demi-puissance ;
- on **répète le diagramme trois fois** (`np.concatenate`, avec les angles décalés de −360° et +360°) pour que le faisceau qui déborde de 0° ou de 360° soit mesuré correctement ;
- `maxIdx_ext` : position du maximum dans le tableau étendu (on ajoute la longueur d'un tour).

```python
    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None

    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None

    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]
```
- `leftIdx_ext` : à gauche du maximum, on cherche le **dernier** point situé sous −3 dB ; `None` s'il n'y en a pas ;
- `rightIdx_ext` : à droite, le **premier** point sous −3 dB ;
- si l'un des deux manque, on met 180 par défaut (valeur de repli) ; sinon, **HPBW = angle de droite − angle de gauche**. Dans nos données, la valeur de repli n'apparaît jamais.

## 7. Gain du lobe principal et SSL

```python
    responseLin = AF_norm_az
    peaks, _ = find_peaks(responseLin, distance=5)

    pk = responseLin[peaks]
    if len(pk) == 0:
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 0
```
- `find_peaks(..., distance=5)` repère les pics du diagramme (deux pics doivent être distants d'au moins 5 points de grille) ;
- `pk` : les hauteurs de ces pics ;
- s'il n'y a aucun pic, on met SSL = 0 par défaut.

```python
    else:
        sorted_idx = np.argsort(pk)[::-1]
        sorted_pk = pk[sorted_idx]
        sorted_angles = theta_deg[peaks][sorted_idx]

        threshold_dB = 1
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)
```
- `np.argsort(pk)[::-1]` : ordre décroissant des pics ; `sorted_pk` : pics triés du plus haut au plus bas (`sorted_angles` garde leurs angles) ;
- `threshold_dB = 1` : sont considérés comme **lobes principaux** tous les pics situés à moins de 1 dB du plus haut ;
- `main_lobes_idx` : indices de ces pics principaux ;
- `side_lobe_idx` : tous les autres = **lobes secondaires** (`np.setdiff1d` calcule « tous sauf »).

```python
        main_lobe_gain = maxVal_non_norm

        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0
```
- `main_lobe_gain` : le gain non normalisé calculé plus haut ;
- `true_SSL_gain` : le niveau (en dB) du **plus haut lobe secondaire**, ou 0 s'il n'y en a pas.

## 8. Rangement dans les matrices

```python
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg
```
On écrit les quatre performances de référence dans la colonne du scénario : ligne 0 = gain, ligne 1 = SSL, ligne 2 = HPBW, ligne 3 = θ0 en degrés. Fin de la boucle : on passe au scénario suivant.

## 9. Affichage de contrôle

```python
print("\nCaractéristiques de la matrice Minput :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))
```
Affiche la forme (`shape`) et les 5 premières colonnes de `Minput` (arrondies à 4 décimales) puis de `Moutput` (converties en entiers), pour vérifier à l'œil que tout est cohérent.

## 10. Validation de la banque (nouveauté pour SARSA)

```python
eps_G_defaut = 1.0   # dB, tolérance de gain par défaut (sert seulement à l'affichage)
assert Minput.shape == (4, nb_samples) and Moutput.shape == (max_rings, nb_samples)
assert np.all(Moutput == np.round(Moutput)) and Moutput.min() >= 0 and Moutput.max() <= max_elements
```
- `eps_G_defaut = 1.0` : tolérance de gain par défaut en dB (utilisée seulement pour l'affichage ci-dessous) ;
- première `assert` : les matrices ont bien les tailles attendues ;
- deuxième `assert` : tous les nombres d'éléments sont des **entiers** entre 0 et 10.

```python
N_ref_all = Moutput.sum(axis=0)
assert N_ref_all.min() >= 1, "une configuration sans aucun élément"
assert Minput[3].min() >= 0 and Minput[3].max() <= theta0_max_deg
```
- `N_ref_all` : nombre total d'éléments de chaque scénario (somme de la colonne de `Moutput`, `axis=0`) ;
- on vérifie qu'aucun scénario n'est vide ;
- on vérifie que θ0 est bien dans [0°, 180°].

```python
# Loi du gain : Gmax = 20 log10(N_ref) (amplitudes unitaires, phases de pointage)
ecart_loi_gain = np.max(np.abs(Minput[0] - 20 * np.log10(N_ref_all)))
r_max_all = np.floor(N_ref_all * (1 - 10 ** (-eps_G_defaut / 20)) + 1e-9).astype(int)
```
- `ecart_loi_gain` : plus grand écart entre le gain `Minput[0]` et la loi théorique `20·log10(N_ref)`. Il est très petit (de l'ordre de 10⁻³ dB, dû à la grille de 1000 angles qui ne tombe pas exactement sur le pic) ;
- `r_max_all` : pour chaque scénario, **nombre maximal d'antennes qu'on peut retirer** sans perdre plus de 1 dB de gain. Formule : `floor(N_ref × (1 − 10^(−ε_G/20)))`, où `floor` arrondit à l'entier inférieur ; le `+ 1e-9` évite une erreur d'arrondi numérique ; `.astype(int)` convertit en entiers.

```python
print("\nValidation de la banque de scénarios :")
print(f"- N_ref : min {int(N_ref_all.min())}, médiane {np.median(N_ref_all):.0f}, max {int(N_ref_all.max())}")
print(f"- Écart maximal à la loi du gain 20*log10(N_ref) : {ecart_loi_gain:.2e} dB")
print(f"- Valeurs de repli (HPBW = 180 ou SSL = 0) : {int(np.sum((Minput[2] == 180) | (Minput[1] == 0)))}")
print(f"- Architectures distinctes : {np.unique(Moutput.T, axis=0).shape[0]} / {nb_samples}")
print(f"- r_max (nombre maximal de suppressions) pour eps_G = {eps_G_defaut} dB : "
      f"moyenne {r_max_all.mean():.2f}, max {r_max_all.max()}, "
      f"part de scénarios avec r_max = 0 : {100 * np.mean(r_max_all == 0):.1f} %")
```
Affiche un rapport : plage de N_ref, écart à la loi du gain, nombre de valeurs de repli, nombre d'architectures distinctes (`np.unique(..., axis=0)` supprime les colonnes en double : on trouve 998 sur 1000), et statistiques de `r_max` (moyenne ≈ 2,22 ; maximum 5 ; 1,9 % des scénarios n'ont aucune suppression possible).

## 11. Enregistrement

```python
script_dir = os.path.dirname(os.path.abspath(__file__))

scipy.io.savemat(os.path.join(script_dir, 'Minput.mat'), {'Minput': Minput})
scipy.io.savemat(os.path.join(script_dir, 'Moutput.mat'), {'Moutput': Moutput})

np.save(os.path.join(script_dir, 'Minput.npy'), Minput)
np.save(os.path.join(script_dir, 'Moutput.npy'), Moutput)
```
- `script_dir` : dossier du script (voir `repartition_explication.md`) ;
- `scipy.io.savemat` écrit `Minput.mat` et `Moutput.mat` (pour MATLAB) ;
- `np.save` écrit `Minput.npy` et `Moutput.npy` (pour Python). Les scripts suivants lisent les `.npy`.

```python
print("✅ Minput.mat/.npy et Moutput.mat/.npy ont été enregistrés dans le dossier du script.")
```
Message de confirmation.

---

## Résumé du flux

```
5 anneaux, 0 à 10 éléments chacun, θ0 aléatoire   (graine 46)
        │ pour chaque scénario : somme des ondes → facteur de réseau
        ▼
Gmax, SSL, HPBW, θ0   →  Minput (4×1000)
éléments par anneau   →  Moutput (5×1000)
        │ validation (tailles, entiers, loi du gain, r_max)
        ▼
Minput / Moutput en .npy et .mat
```
