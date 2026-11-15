# Explication ligne par ligne de `training.py`

**Rôle du script (étape 3/4)** : entraîner l'agent SARSA à éteindre le plus d'antennes possible sans dégrader le diagramme au-delà des tolérances.

C'est le script le plus long. Il contient **trois blocs** :
1. le **modèle physique** (calculer gain, SSL, HPBW d'un réseau dont certains éléments sont éteints) ;
2. l'**environnement** (les règles du jeu : états, actions, récompenses) ;
3. l'**agent SARSA** et la boucle d'entraînement.

Rien n'est caché dans une bibliothèque : la mise à jour SARSA est écrite à la main (section 9).

## L'idée en quelques phrases

On part d'un réseau complet. À chaque pas, l'agent choisit **un anneau où éteindre un élément**, ou **s'arrêter**. Après chaque extinction, on recalcule le diagramme :
- si les performances restent dans les tolérances, l'agent reçoit une petite **récompense** (+1/r_max) et peut continuer ;
- si une tolérance est dépassée, il reçoit une **pénalité** (−2), l'épisode s'arrête, et la solution retenue est la dernière configuration valide ;
- s'il choisit de s'arrêter, la récompense est 0.

L'agent apprend ainsi, par essais et erreurs, **quand continuer et quand s'arrêter**. Comme les performances dépendent de la configuration, l'agent ne voit pas le réseau en détail mais une **observation compacte** de 54 valeurs possibles ; il apprend une **seule table** partagée par toutes les configurations.

## Vocabulaire du renforcement

| Terme | Signification simple |
|---|---|
| Agent | Le programme qui apprend à décider. |
| Environnement | Le monde simulé dans lequel l'agent agit (ici : le réseau d'antennes et son calcul de diagramme). |
| Observation `o` | Ce que l'agent voit à un instant donné. |
| Action `a` | Ce que l'agent décide : amincir l'anneau 1 à 5, ou STOP. |
| Récompense `r` | Un nombre qui dit si l'action était bonne. |
| Épisode | Une partie complète, du réseau entier jusqu'à l'arrêt. |
| Table Q | Tableau `Q[o, a]` : « combien de récompense future puis-je espérer si, dans l'observation `o`, je fais l'action `a` ? » |
| SARSA | Algorithme qui met à jour Q à partir de **S**tate, **A**ction, **R**eward, **S**tate suivant, **A**ction suivante réellement choisie. |
| α (alpha) | Vitesse d'apprentissage (0,1 : on corrige de 10 % de l'erreur à chaque fois). |
| γ (gamma) | Importance du futur (1 : on compte tout le futur de l'épisode). |
| ε (epsilon) | Probabilité d'**explorer** (action au hasard) au lieu d'**exploiter** (meilleure action connue). |
| Politique gloutonne | Toujours choisir l'action de plus grande valeur Q (ε = 0). |

## L'équation de SARSA (écrite à la ligne 253)

`Q(o,a) ← Q(o,a) + α [ r + γ · Q(o',a') − Q(o,a) ]`

- `Q(o,a)` : ce qu'on croyait de l'action `a` dans l'observation `o` ;
- `r + γ·Q(o',a')` : ce qu'on vient de **constater** (récompense obtenue, plus la valeur de l'action suivante qu'on va réellement faire) ;
- la différence entre les deux est l'**erreur de prédiction** ; on corrige Q d'une fraction α de cette erreur.

---

## 1. Imports

```python
import argparse
import itertools
import os
import time

import numpy as np
import scipy.io
from scipy.signal import find_peaks
```
- `argparse` : lire des options données en ligne de commande (par exemple `--eps_G 2`) ;
- `itertools` : outils de boucles (importé mais non utilisé dans ce fichier ; on peut le supprimer sans effet) ;
- `os` : chemins de fichiers ; `time` : chronomètre ;
- `numpy` : tableaux et calcul ; `scipy.io` : fichiers `.mat` ;
- `find_peaks` : détection des pics du diagramme (pour le SSL).

## 2. Constantes physiques

```python
C0 = 3e8
F0 = 2.45e9
LAMBDA = C0 / F0
K_WAVE = 2 * np.pi / LAMBDA
R0 = 0.2 * LAMBDA            # rayon de l'anneau 1
DR = 0.5 * LAMBDA            # pas entre anneaux
THETA = np.linspace(0, 2 * np.pi, 1000)
THETA_DEG = np.rad2deg(THETA)
EPS_MACH = np.finfo(float).eps
```
Ce sont les mêmes que dans `dataset.py` (voir `dataset_explication.md`) :
- `C0`, `F0` : vitesse de la lumière et fréquence 2,45 GHz ; `LAMBDA` : longueur d'onde ; `K_WAVE` : nombre d'onde ;
- `R0` : rayon du premier anneau ; `DR` : écart entre anneaux ;
- `THETA` : grille de 1000 angles de 0 à 2π ; `THETA_DEG` : la même en degrés ;
- `EPS_MACH` : plus petit nombre positif distinguable de zéro pour des réels (sert à éviter `log(0)` et les divisions par 0).
Les noms sont en MAJUSCULES : c'est la convention Python pour les constantes.

## 3. Constantes de l'agent

```python
N_RINGS = 5
N_ACTIONS = N_RINGS + 1      # actions 0..4 : amincir l'anneau 1..5 ; action 5 : STOP
STOP = N_RINGS
```
- `N_RINGS = 5` : nombre d'anneaux ;
- `N_ACTIONS = 6` : actions numérotées 0 à 4 (« éteindre un élément de l'anneau 1 à 5 ») et 5 (« STOP ») ;
- `STOP = 5` : le numéro de l'action STOP.

```python
L_MAX = 5                    # l = min(5, r_max - r) : budget de suppressions restant (0..5)
N_OBS = (L_MAX + 1) * 3 * 3  # 54 observations (l, q_S, q_H)
Q_THRESH = (0.5, 0.8)        # seuils de discrétisation des écarts normalisés u
W_ERR = (0.33, 0.33, 0.34)   # pondérations de l'erreur globale (comme test_model_FFNN / cGAN)
```
- `L_MAX = 5` : on suit le « budget de suppressions restant » `l`, plafonné à 5 ;
- `N_OBS = 6 × 3 × 3 = 54` : nombre d'observations possibles. Une observation est le triplet (`l` de 0 à 5, `q_S` de 0 à 2, `q_H` de 0 à 2) ;
- `Q_THRESH = (0.5, 0.8)` : seuils qui transforment l'écart normalisé `u` en trois niveaux : < 0,5 (marge confortable), entre 0,5 et 0,8 (marge qui s'amenuise), ≥ 0,8 (proche de la limite) ;
- `W_ERR` : pondérations 0,33 / 0,33 / 0,34 de l'erreur globale, les mêmes que dans les tests du FFNN et du cGAN.

## 4. Bannière

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
```
Même fonction de séparation « NOUVELLE EXÉCUTION » que dans les autres scripts.

## 5. Modèle physique

### 5.1 `layout` : où sont les éléments ?

```python
def layout(arch):
    """Position de TOUS les éléments de référence : rayon, angle dans l'anneau, numéro d'anneau."""
    a, phi, ring = [], [], []
    for m, N in enumerate(arch):
        N = int(N)
        for n in range(N):
            a.append(R0 + DR * m)
            phi.append(2 * np.pi * n / N)
            ring.append(m)
    return np.array(a), np.array(phi), np.array(ring)
```
- `arch` : les 5 nombres d'éléments par anneau, par exemple `[2, 9, 4, 5, 5]` ;
- `a, phi, ring = [], [], []` : trois listes vides qui contiendront, **pour chaque élément**, son rayon, son angle sur l'anneau, et son numéro d'anneau ;
- `for m, N in enumerate(arch)` : `m` = numéro d'anneau (0 à 4), `N` = nombre d'éléments de cet anneau ;
- `for n in range(N)` : on parcourt les éléments de l'anneau ;
- `a.append(R0 + DR * m)` : rayon de l'anneau ; `phi.append(2π n / N)` : éléments régulièrement espacés ; `ring.append(m)` ;
- `return np.array(...)` : on renvoie les trois listes converties en tableaux NumPy. Les éléments sont numérotés dans l'ordre : tous ceux de l'anneau 1, puis ceux de l'anneau 2, etc. Cette numérotation est utilisée partout ensuite (élément « 6 » = 7e élément).

### 5.2 `contributions` : l'onde de chaque élément

```python
def contributions(a, phi, theta0_deg):
    """Matrice E (1000 x N) : contribution complexe de chaque élément au facteur de réseau.
    AF(theta) = somme des colonnes de E pour les éléments actifs."""
    th0 = np.deg2rad(theta0_deg)
    phase = K_WAVE * a[None, :] * (np.sin(THETA)[:, None] * np.cos(0 - phi)[None, :]
                                   - np.sin(th0) * np.cos(0 - phi)[None, :])
    return np.exp(1j * phase)
```
Au lieu de recalculer tout le diagramme à chaque suppression, on calcule **une fois** la contribution de chaque élément, puis on additionne seulement les éléments actifs. C'est ce qui rend l'entraînement rapide.
- `th0` : θ0 en radians ;
- `phase` : tableau de taille 1000 × N (1000 angles, N éléments) grâce au **broadcasting** NumPy : `a[None, :]` transforme `a` en ligne, `np.sin(THETA)[:, None]` transforme les angles en colonne, et leurs produits donnent toutes les combinaisons d'un coup. La formule est la même que dans `dataset.py` ;
- `np.exp(1j * phase)` : l'onde complexe de chaque élément. Le facteur de réseau d'un réseau complet est la somme des colonnes ; en éteindre un revient à **retirer sa colonne**.

### 5.3 `metrics_from_AF` : gain, SSL, HPBW

```python
def metrics_from_AF(AF):
    """Gmax (non normalisé), SSL, HPBW et direction du maximum : même calcul que le dataset.py."""
    mag = np.abs(AF)
    mx = mag.max()
    n = mag / (mx + EPS_MACH)
    dB = 20 * np.log10(n + EPS_MACH)
    dB[dB < -40] = -40
    gain = 20 * np.log10(mx + EPS_MACH)
```
Cette fonction reprend **exactement** le calcul de `dataset.py` (sections 5 à 7 de `dataset_explication.md`) :
- `mag` : module de l'onde totale ; `mx` : son maximum ; `n` : diagramme normalisé (pic = 1) ;
- `dB` : le diagramme en dB, avec plancher à −40 dB ;
- `gain` : le gain non normalisé `20·log10(mx)`.

```python
    # HPBW : croisement à -3 dB, sur un diagramme répété 3 fois
    half = dB.max() - 3
    ext = np.concatenate((dB, dB, dB))
    th_ext = np.concatenate((THETA_DEG - 360, THETA_DEG, THETA_DEG + 360))
    imax = np.argmax(dB) + len(THETA_DEG)
    left = np.where(ext[:imax] <= half)[0]
    right = np.where(ext[imax:] <= half)[0]
    hpbw = 180.0 if (len(left) == 0 or len(right) == 0) else th_ext[right[0] + imax] - th_ext[left[-1]]
```
**HPBW** :
- `half` : niveau −3 dB ; on répète le diagramme trois fois (`ext`, avec les angles `th_ext` décalés de ∓360°) pour gérer les faisceaux près de 0° ou 360° ;
- `imax` : position du maximum dans le tableau étendu ;
- `left` / `right` : positions sous −3 dB à gauche et à droite du maximum ;
- `hpbw` : 180 par défaut si un côté manque, sinon angle de droite − angle de gauche.

```python
    # SSL : plus haut pic hors de 1 dB du sommet
    pk, _ = find_peaks(n, distance=5)
    pv = n[pk]
    if len(pv) == 0:
        ssl = 0.0
    else:
        sp = np.sort(pv)[::-1]
        main = np.where(20 * np.log10(sp + EPS_MACH) >= 20 * np.log10(sp[0] + EPS_MACH) - 1)[0]
        side = np.setdiff1d(np.arange(len(sp)), main)
        ssl = 20 * np.log10(sp[side[0]] + EPS_MACH) if len(side) else 0.0
    return gain, ssl, hpbw, THETA_DEG[int(np.argmax(dB))]
```
**SSL** :
- `find_peaks` repère les pics ; `pv` : leurs hauteurs ; s'il n'y en a pas, `ssl = 0` ;
- sinon `sp` : pics triés du plus haut au plus bas ; `main` : ceux à moins de 1 dB du plus haut (lobes principaux) ; `side` : tous les autres ; `ssl` : niveau en dB du plus haut lobe secondaire ;
- la fonction renvoie le quadruplet (gain, SSL, HPBW, **direction du maximum** en degrés). La direction du maximum sert à vérifier que le faisceau pointe toujours vers θ0.

### 5.4 `delta_theta`

```python
def delta_theta(peak_deg, theta0_deg):
    """Écart de direction : le diagramme a deux lobes principaux symétriques, theta0 et 180-theta0."""
    return min(abs(peak_deg - theta0_deg), abs(peak_deg - (180.0 - theta0_deg)))
```
Le diagramme est symétrique : il a **deux** lobes principaux, à θ0 et à 180° − θ0. Le maximum trouvé peut être l'un ou l'autre ; l'écart de direction est donc la plus petite des deux distances (`min`), en valeur absolue (`abs`).

## 6. Environnement : la classe `Scenario`

Une **classe** regroupe des données et des fonctions qui vont ensemble. `Scenario` représente **un** scénario : une configuration de référence, ses performances de référence, ses tolérances et ses résultats déjà calculés.

```python
class Scenario:
    """Une configuration de référence (une colonne de Minput / Moutput) et ses contraintes."""

    def __init__(self, arch, theta0_deg, tol):
        self.arch = np.array(arch).astype(int)
        self.theta0 = float(theta0_deg)
        self.a, self.phi, self.ring = layout(self.arch)
        self.n_ref = len(self.a)
```
- `class Scenario:` définit la classe ;
- `__init__` est le **constructeur**, appelé à la création d'un scénario ; `self` désigne l'objet lui-même ;
- `arch` : les 5 nombres d'éléments ; `theta0_deg` : direction ; `tol` : dictionnaire des tolérances ;
- on stocke l'architecture, θ0, et on appelle `layout` pour obtenir les positions ; `n_ref` : nombre total d'éléments (longueur de `a`).

```python
        self.tol = tol                                    # dict eps_G, eps_S, eps_H_rel, eps_theta
        self._E = None                                    # calculée au premier besoin
        self.cache = {}                                   # {tuple des éléments retirés : métriques}
        self.ref = self.metrics(())                       # performances de référence (réseau complet)
        self.eps_H = tol['eps_H_rel'] * self.ref[2]
        self.r_max = int(np.floor(self.n_ref * (1 - 10 ** (-tol['eps_G'] / 20)) + 1e-9))
```
- `self.tol` : tolérances ;
- `self._E = None` : la matrice des contributions n'est calculée qu'au premier besoin (économie de mémoire) ;
- `self.cache = {}` : **mémoire des résultats** déjà calculés : la clé est le n-uplet des éléments éteints, la valeur les métriques. Si l'agent retombe sur la même configuration, on ne recalcule pas ;
- `self.ref` : performances de **référence** = métriques du réseau complet (aucun élément retiré, n-uplet vide `()`) ;
- `self.eps_H` : tolérance de HPBW en degrés = pourcentage (10 %) × HPBW de référence ;
- `self.r_max` : nombre maximal de suppressions que permet la tolérance de gain (même formule que dans `dataset.py`).

```python
    @property
    def E(self):
        if self._E is None:
            self._E = contributions(self.a, self.phi, self.theta0)
        return self._E
```
`@property` permet d'écrire `sc.E` comme un attribut alors que c'est un calcul : si `_E` est vide, on le calcule avec `contributions` (une seule fois), puis on le renvoie.

```python
    def metrics(self, removed):
        """(G, SSL, HPBW, pic) du réseau privé des éléments 'removed' (n-uplet trié)."""
        if removed not in self.cache:
            AF = self.E.sum(axis=1) if not removed else self.E.sum(axis=1) - self.E[:, list(removed)].sum(axis=1)
            self.cache[removed] = metrics_from_AF(AF)
        return self.cache[removed]
```
- `metrics(removed)` renvoie (gain, SSL, HPBW, pic) du réseau privé des éléments de la liste `removed` ;
- `if removed not in self.cache` : seulement si ce cas n'a pas déjà été calculé ;
- `AF` : somme des contributions de tous les éléments, **moins** celles des éléments retirés (`self.E[:, list(removed)]` sélectionne leurs colonnes) ;
- on stocke le résultat dans le cache et on le renvoie.

```python
    def deviations(self, removed):
        """Écarts normalisés (u_G, u_S, u_H, u_theta) ; admissible <=> max <= 1."""
        g, s, h, pk = self.metrics(removed)
        rg, rs, rh, _ = self.ref
        return (abs(g - rg) / self.tol['eps_G'], abs(s - rs) / self.tol['eps_S'],
                abs(h - rh) / self.eps_H, delta_theta(pk, self.theta0) / self.tol['eps_theta'])
```
- `deviations` calcule les **écarts normalisés** `u` : chaque écart absolu à la référence divisé par sa tolérance :
  - `u_G = |ΔG| / ε_G`, `u_S = |ΔSSL| / ε_S`, `u_H = |ΔHPBW| / ε_H`, `u_θ = Δθ / ε_θ` ;
- une valeur `u = 0,5` veut dire « la moitié de la marge est consommée » ; `u > 1` veut dire « tolérance dépassée » ;
- `rg, rs, rh, _ = self.ref` : le `_` ignore la 4e valeur (pic de référence).

```python
    def u_max(self, removed):
        return max(self.deviations(removed))
```
`u_max` : le plus grand des quatre `u`. Une configuration est **admissible** si et seulement si `u_max ≤ 1`.

## 7. Environnement : la classe `AntennaEnv`

Un **épisode** de SARSA = une partie sur un scénario. `AntennaEnv` gère l'état de la partie (quels éléments sont éteints) et applique les actions.

```python
class AntennaEnv:
    """Environnement SARSA : un épisode = suite de suppressions d'éléments sur un scénario."""

    def __init__(self, scenario, penalty=2.0):
        self.sc = scenario
        self.penalty = penalty
        self.reset()
```
- le constructeur reçoit un scénario et la valeur de la pénalité (2 par défaut) ;
- il appelle `reset()` pour démarrer une partie.

```python
    def reset(self):
        self.removed = []                                 # éléments retirés (dans l'ordre)
        self.active = np.ones(self.sc.n_ref, dtype=bool)
        self.last_admissible = tuple()
        return self.observation()
```
`reset` remet le jeu au début :
- `removed` : liste des éléments éteints (vide) ;
- `active` : tableau de booléens, tous `True` (tous les éléments allumés) ;
- `last_admissible` : dernière configuration admissible (vide au départ) : c'est la **solution** retenue à la fin ;
- il renvoie la première observation.

```python
    # --- observation compacte o = (l, q_S, q_H) -> indice 0..53 ---
    def observation(self):
        sc = self.sc
        u = sc.deviations(tuple(sorted(self.removed)))
        l = min(L_MAX, max(0, sc.r_max - len(self.removed)))
        qs = int(u[1] >= Q_THRESH[0]) + int(u[1] >= Q_THRESH[1])
        qh = int(u[2] >= Q_THRESH[0]) + int(u[2] >= Q_THRESH[1])
        return l * 9 + qs * 3 + qh
```
`observation` convertit la situation en **un numéro de 0 à 53** :
- `u` : les écarts normalisés de la configuration actuelle ;
- `l = min(5, max(0, r_max − nombre d'éléments retirés))` : combien de suppressions il reste avant la limite de gain, plafonné à 5 ;
- `qs`, `qh` : niveaux 0, 1 ou 2 pour le SSL et le HPBW : on additionne deux booléens (`int(u ≥ 0,5) + int(u ≥ 0,8)`) ;
- `l * 9 + qs * 3 + qh` : numéro unique de l'observation (comme un nombre écrit en « base mixte »).
L'agent **ne voit ni les positions, ni le nombre exact d'éléments** : c'est ce qui permet une seule table Q pour toutes les configurations.

```python
    def valid_actions(self):
        """STOP toujours valide ; amincir un anneau est valide s'il lui reste un élément actif
        et s'il reste au moins 2 éléments actifs au total (les suppressions non admissibles
        ne sont PAS masquées : l'agent doit apprendre à les éviter)."""
        acts = []
        if self.active.sum() > 1:
            for m in range(N_RINGS):
                if np.any(self.active & (self.sc.ring == m)):
                    acts.append(m)
        acts.append(STOP)
        return acts
```
`valid_actions` donne la liste des actions autorisées :
- on part d'une liste vide ;
- si plus d'un élément est actif (`self.active.sum() > 1`), on ajoute chaque anneau `m` qui contient encore un élément actif (`np.any(...)` ; `&` est le « et » élément par élément) ;
- `STOP` est toujours possible ;
- Les suppressions qui violent les contraintes ne sont **pas** interdites : c'est l'agent qui doit apprendre à les éviter.

```python
    def element_to_remove(self, ring):
        """Règle fixe dans l'anneau : on retire l'élément actif dont la somme des deux écarts
        angulaires aux voisins actifs est la plus petite (égalité -> plus petit indice)."""
        idx = np.where(self.active & (self.sc.ring == ring))[0]
        if len(idx) == 1:
            return int(idx[0])
        ang = self.sc.phi[idx]                            # angles croissants (construits ainsi)
        nxt = np.roll(ang, -1)
        prv = np.roll(ang, 1)
        gap = ((nxt - ang) % (2 * np.pi)) + ((ang - prv) % (2 * np.pi))
        return int(idx[int(np.argmin(gap))])              # argmin renvoie le premier minimum
```
`element_to_remove(ring)` décide **quel élément** éteindre dans l'anneau choisi, par une règle fixe : on retire celui dont les deux voisins actifs sont les plus proches (somme des deux écarts angulaires minimale), pour garder une répartition aussi régulière que possible.
- `idx` : indices des éléments actifs de cet anneau ;
- si un seul reste, c'est lui ;
- `ang` : leurs angles ; `nxt` / `prv` : angle du voisin suivant et précédent (`np.roll` décale le tableau circulairement) ;
- `gap` : somme des écarts aux deux voisins (le `% 2π` ramène à un angle positif sur le cercle) ;
- `np.argmin(gap)` : position du plus petit écart ; en cas d'égalité, c'est la première, donc le plus petit indice.

```python
    def step(self, action):
        """Retourne (récompense, terminé, info). Récompenses : STOP 0 ; suppression admissible
        +1/r_max ; violation -penalty (retour à la dernière configuration admissible)."""
        if action == STOP:
            return 0.0, True, 'stop'
        e = self.element_to_remove(action)
        self.active[e] = False
        self.removed.append(e)
        u = self.sc.u_max(tuple(sorted(self.removed)))
        if u > 1.0:
            self.active[e] = True                         # annulation : la solution reste la config précédente
            self.removed.pop()
            return -self.penalty, True, 'violation'
        self.last_admissible = tuple(sorted(self.removed))
        if len(self.valid_actions()) == 1:                # plus aucune action de suppression possible
            return 1.0 / max(1, self.sc.r_max), True, 'plus_daction'
        return 1.0 / max(1, self.sc.r_max), False, 'ok'
```
`step(action)` applique l'action et renvoie (récompense, épisode terminé ?, raison) :
- **STOP** : récompense 0, fin de l'épisode ;
- sinon on choisit l'élément `e` à éteindre (`element_to_remove`), on le marque éteint (`active[e] = False`) et on l'ajoute à `removed` ;
- `u = u_max(...)` : on calcule le pire écart normalisé de la nouvelle configuration ;
- **violation** (`u > 1`) : on **annule** l'extinction (`active[e] = True`, `removed.pop()`), récompense **−pénalité**, fin d'épisode. La solution reste la configuration précédente ;
- sinon la configuration est admissible : on la mémorise dans `last_admissible` ;
- si plus aucune suppression n'est possible, fin d'épisode avec récompense positive ; sinon récompense **+1/r_max** et l'épisode continue. `max(1, r_max)` évite une division par zéro.

## 8. L'agent : choix des actions

```python
def choose_action(Q, o, valid, eps, rng, tie='random'):
    """Politique epsilon-glouton restreinte aux actions valides."""
    if eps > 0 and rng.random() < eps:
        return int(valid[rng.integers(len(valid))])
    q = np.array([Q[o, a] for a in valid])
    best = np.flatnonzero(q == q.max())
    return int(valid[best[rng.integers(len(best))]]) if tie == 'random' else int(valid[best[0]])
```
`choose_action` applique la politique **ε-gloutonne**, restreinte aux actions valides :
- avec probabilité `eps` (`rng.random() < eps`), on **explore** : action valide tirée au hasard (`rng.integers(n)` donne un entier entre 0 et n−1) ;
- sinon on **exploite** : `q` = valeurs Q des actions valides ; `best` = positions de la ou des meilleures valeurs ;
- en cas d'égalité, `tie='random'` choisit au hasard parmi les meilleures (utile pendant l'entraînement, car toute la table vaut 0 au départ) ; `tie='first'` prend toujours la première (évaluation déterministe).

## 9. L'agent : un épisode SARSA

```python
def run_episode(env, Q, visits, alpha, gamma, eps, rng, learn=True, tie='random'):
    """Un épisode SARSA complet. Si learn=False : aucune mise à jour (évaluation)."""
    o = env.reset()
    a = choose_action(Q, o, env.valid_actions(), eps, rng, tie)
    ret, n_steps, end = 0.0, 0, ''
```
- `learn=True` : on met à jour Q ; `learn=False` : on ne fait que jouer (évaluation) ;
- `o = env.reset()` : observation de départ ;
- `a` : première action choisie ;
- `ret` : retour (somme des récompenses), `n_steps` : nombre de pas, `end` : raison de la fin.

```python
    while True:
        r, done, end = env.step(a)
        ret += r
        n_steps += 1
        if done:
            target = r                                    # état terminal : Q(s', .) = 0
        else:
            o2 = env.observation()
            a2 = choose_action(Q, o2, env.valid_actions(), eps, rng, tie)
            target = r + gamma * Q[o2, a2]
```
La boucle `while True` joue un pas à la fois :
- `env.step(a)` applique l'action : on obtient la récompense `r`, l'indicateur `done` et la raison ;
- `ret += r` cumule le retour ; `n_steps += 1` compte les pas ;
- si l'épisode est fini, la **cible** de la mise à jour est juste `r` (pas de futur) ;
- sinon : `o2` = nouvelle observation ; `a2` = action qu'on va **réellement faire ensuite** (c'est le « A » final de SARSA) ; la cible vaut `r + γ · Q[o2, a2]`.

```python
        if learn:
            Q[o, a] += alpha * (target - Q[o, a])         # mise à jour SARSA
            visits[o, a] += 1
```
**Mise à jour SARSA** (seulement si `learn`) :
- `Q[o, a] += alpha * (target − Q[o, a])` : on rapproche la valeur Q de la cible, d'une fraction α de l'erreur ;
- `visits[o, a] += 1` compte combien de fois cette case a été mise à jour (pour savoir quelles cases sont explorées).

```python
        if done:
            break
        o, a = o2, a2
        assert n_steps <= env.sc.r_max + 1, "épisode trop long : la loi du gain devrait l'interdire"
    n_removed = len(env.last_admissible)
    return dict(ret=ret, r=n_removed, end=end, config=env.last_admissible, steps=n_steps)
```
- si l'épisode est terminé, `break` sort de la boucle ;
- sinon on passe au pas suivant : l'observation et l'action suivantes deviennent l'observation et l'action courantes ;
- `assert` : filet de sécurité (la loi du gain interdit qu'un épisode dépasse `r_max + 1` actions) ;
- à la fin, on renvoie un dictionnaire : retour, nombre d'éléments éteints (`r`, la taille de la dernière configuration admissible), raison de fin, configuration finale et nombre de pas.

## 10. Fonctions utilitaires

```python
def error_global(sc, removed, norm):
    """Erreur globale pondérée (%) comme dans test_model_FFNN / cGAN (départage des égalités)."""
    g, s, h, _ = sc.metrics(tuple(sorted(removed)))
    rg, rs, rh, _ = sc.ref
    return (W_ERR[0] * 100 * abs(g - rg) / norm[0] + W_ERR[1] * 100 * abs(s - rs) / norm[1]
            + W_ERR[2] * 100 * abs(h - rh) / norm[2])
```
`error_global` : l'erreur globale pondérée de ton FFNN/cGAN, appliquée à une configuration allégée : erreurs relatives en % sur gain, SSL, HPBW (normalisées par `norm`, les valeurs maximales du jeu de données), pondérées par 0,33 / 0,33 / 0,34. Elle sert à **départager** des solutions de même nombre d'antennes (elle n'entre pas dans la récompense).

```python
def build_scenarios(Minput, Moutput, tol):
    return [Scenario(Moutput[:, j], Minput[3, j], tol) for j in range(Minput.shape[1])]
```
`build_scenarios` crée un objet `Scenario` pour chaque colonne : `Moutput[:, j]` (architecture), `Minput[3, j]` (θ0). C'est une compréhension de liste.

```python
def evaluate_greedy(scenarios, Q, penalty):
    """Politique gloutonne (eps = 0), SANS mise à jour, sur une liste de scénarios."""
    rng = np.random.default_rng(0)
    out = []
    for sc in scenarios:
        res = run_episode(AntennaEnv(sc, penalty), Q, None, 0.0, 1.0, 0.0, rng, learn=False, tie='first')
        out.append(res)
    return out
```
`evaluate_greedy` joue la **politique gloutonne** (ε = 0) sur une liste de scénarios, **sans mise à jour** de Q :
- un générateur aléatoire à graine 0 (inutile ici car pas d'exploration, mais requis par la fonction) ;
- pour chaque scénario : on joue un épisode avec `learn=False`, `eps=0` et `tie='first'` (choix déterministe) ;
- on renvoie la liste des résultats.

## 11. Programme principal

La fonction `main()` est exécutée seulement quand on lance `python training.py` (voir la dernière ligne). `test.py` peut ainsi **importer** les fonctions ci-dessus sans relancer l'entraînement.

```python
def main():
    clear_console()
```
Définit `main` et affiche la bannière.

```python
    p = argparse.ArgumentParser()
    p.add_argument('--eps_G', type=float, default=1.0, help="tolérance de gain (dB)")
    p.add_argument('--eps_S', type=float, default=2.0, help="tolérance de SSL (dB)")
    p.add_argument('--eps_H', type=float, default=0.10, help="tolérance de HPBW (fraction de H_ref)")
    p.add_argument('--eps_theta', type=float, default=1.0, help="tolérance de direction (deg)")
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_min', type=float, default=0.05)
    p.add_argument('--eps_decay', type=float, default=0.9997)
    p.add_argument('--penalty', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=46)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--tag', type=str, default='', help="suffixe des fichiers de sortie")
    args = p.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''
```
**Options en ligne de commande** : `argparse.ArgumentParser()` crée l'analyseur ; chaque `p.add_argument('--nom', type=..., default=...)` déclare un paramètre avec sa valeur par défaut. Sans rien préciser, ce sont les valeurs du rapport :
- tolérances : `--eps_G` (1 dB), `--eps_S` (2 dB), `--eps_H` (0,10 = 10 %), `--eps_theta` (1°) ;
- apprentissage : `--episodes` (20 000), `--alpha` (0,1), `--gamma` (1) ;
- exploration : `--eps_start` (1), `--eps_min` (0,05), `--eps_decay` (0,9997 : ε est multiplié par 0,9997 à chaque épisode) ;
- `--penalty` (2), `--seed` (46), `--eval_every` (évaluation sur la validation tous les 1000 épisodes) ;
- `--tag` : suffixe ajouté aux fichiers de sortie (utile pour tester d'autres tolérances sans écraser les résultats). Exemple : `python training.py --eps_G 2 --tag eG2` ;
- `p.parse_args()` lit la ligne de commande ; `suffix` vaut `_eG2` si un tag est donné, vide sinon.

```python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tol = dict(eps_G=args.eps_G, eps_S=args.eps_S, eps_H_rel=args.eps_H, eps_theta=args.eps_theta)
    rng = np.random.default_rng(args.seed)
```
- `script_dir` : dossier du script ;
- `tol` : dictionnaire des tolérances ;
- `rng = np.random.default_rng(seed)` : générateur aléatoire à graine 46 (reproductibilité).

```python
    Mi_tr = np.load(os.path.join(script_dir, 'Minput_training.npy'))
    Mo_tr = np.load(os.path.join(script_dir, 'Moutput_training.npy'))
    Mi_va = np.load(os.path.join(script_dir, 'Minput_validation.npy'))
    Mo_va = np.load(os.path.join(script_dir, 'Moutput_validation.npy'))

    train_all = build_scenarios(Mi_tr, Mo_tr, tol)
    val_all = build_scenarios(Mi_va, Mo_va, tol)
```
Charge les matrices d'entraînement et de validation produites par `repartition.py`, puis construit un `Scenario` pour chaque colonne. **Le test n'est jamais chargé ici.**

```python
    # contrôle de cohérence : performances de référence chargées vs recalculées
    ecart = max(np.max(np.abs(np.array(sc.ref[:3]) - Mi_tr[:3, j])) for j, sc in enumerate(train_all))
    print(f"Écart max entre Minput chargé et diagramme recalculé : {ecart:.2e} (attendu < 1e-8)")
    assert ecart < 1e-8
```
Contrôle de cohérence : on compare les performances de référence **chargées** (`Mi_tr`) aux performances **recalculées** par notre modèle physique. `max(... for ...)` prend le plus grand écart sur tous les scénarios. Il doit être inférieur à 10⁻⁸ ; sinon le script s'arrête (`assert`).

```python
    # scénarios sans décision possible (r_max = 0) exclus de l'apprentissage
    train = [s for s in train_all if s.r_max >= 1]
    print(f"Training : {len(train)} scénarios utilisables sur {len(train_all)} (r_max >= 1) ; validation : {len(val_all)}")
    print(f"Tolérances : eps_G={tol['eps_G']} dB, eps_S={tol['eps_S']} dB, eps_H={100*tol['eps_H_rel']:.0f} % de H_ref, "
          f"eps_theta={tol['eps_theta']} deg ; r_max moyen (training) = {np.mean([s.r_max for s in train]):.2f}")
```
- `train` : on **écarte** les scénarios dont `r_max = 0` (aucune suppression possible : ils n'apportent aucune décision à apprendre). Ils restent présents au test ;
- on affiche les effectifs et les tolérances utilisées.

### 11.1 Initialisation

```python
    Q = np.zeros((N_OBS, N_ACTIONS))                      # initialisation neutre
    visits = np.zeros((N_OBS, N_ACTIONS), dtype=np.int64)
    best_seen = np.zeros(len(train_all), dtype=int)       # meilleur r admissible rencontré PENDANT l'entraînement
    id_in_all = {id(s): i for i, s in enumerate(train_all)}
```
- `Q` : table de 54 lignes (observations) × 6 colonnes (actions), remplie de zéros : l'agent ne sait rien au départ ;
- `visits` : même taille, compteur d'utilisations (entiers) ;
- `best_seen` : pour chaque scénario d'entraînement, le plus grand nombre de suppressions admissibles **rencontré pendant l'entraînement** (une « trouvaille » obtenue avec exploration, à ne pas confondre avec la politique apprise) ;
- `id_in_all` : dictionnaire qui retrouve la position d'un scénario dans la liste complète à partir de son identité (`id`).

```python
    H = {k: np.zeros(args.episodes) for k in ['ret', 'r', 'violation', 'stop', 'eps']}
    val_eps, val_r, val_viol, val_ret = [], [], [], []
    t0 = time.time()
```
- `H` : historique par épisode (retour, éléments éteints, violation oui/non, arrêt volontaire oui/non, valeur de ε) ;
- listes vides pour les mesures faites sur la validation ;
- `t0` : instant de départ pour mesurer le temps écoulé.

### 11.2 Boucle d'entraînement

```python
    for ep in range(args.episodes):
        eps = max(args.eps_min, args.eps_start * args.eps_decay ** ep)
        sc = train[rng.integers(len(train))]
        res = run_episode(AntennaEnv(sc, args.penalty), Q, visits, args.alpha, args.gamma, eps, rng)
        i = id_in_all[id(sc)]
        best_seen[i] = max(best_seen[i], res['r'])
        H['ret'][ep], H['r'][ep], H['eps'][ep] = res['ret'], res['r'], eps
        H['violation'][ep] = res['end'] == 'violation'
        H['stop'][ep] = res['end'] == 'stop'
```
Pour chaque épisode `ep` de 0 à 19 999 :
- `eps` : valeur de ε qui **décroît** de 1 vers 0,05 (`max(eps_min, eps_start × decay^ep)`) : beaucoup d'exploration au début, davantage d'exploitation ensuite ;
- `sc` : un scénario tiré **au hasard** dans l'entraînement ;
- on joue un épisode avec apprentissage (`run_episode`) ;
- on met à jour la meilleure trouvaille pour ce scénario ;
- on enregistre retour, nombre d'éléments éteints et ε ; et on note si l'épisode s'est fini par violation ou par STOP.

```python
        if (ep + 1) % args.eval_every == 0:
            v = evaluate_greedy(val_all, Q, args.penalty)
            val_eps.append(ep + 1)
            val_r.append(np.mean([x['r'] / max(1, s.r_max) for x, s in zip(v, val_all) if s.r_max >= 1]))
            val_viol.append(np.mean([x['end'] == 'violation' for x in v]))
            val_ret.append(np.mean([x['ret'] for x in v]))
            w = slice(max(0, ep - 999), ep + 1)
            print(f"ep {ep+1:6d} | eps {eps:.3f} | train (1000 derniers, avec exploration) : retour {H['ret'][w].mean():+.3f}, "
                  f"violations {100*H['violation'][w].mean():5.1f} % | validation gloutonne : r/r_max {val_r[-1]:.3f}, "
                  f"tentatives de violation {100*val_viol[-1]:5.1f} % | {time.time()-t0:5.0f} s")
```
Tous les `eval_every` (1000) épisodes :
- on évalue la politique **gloutonne** sur la **validation**, sans apprendre (`evaluate_greedy`) ;
- on retient : le rapport moyen r / r_max (part de la suppression maximale réalisée), la part des épisodes qui se terminent par une tentative de violation, et le retour moyen ;
- `w` : fenêtre des 1000 derniers épisodes d'entraînement ;
- on affiche une ligne de suivi. Attention : les chiffres d'entraînement incluent l'exploration, donc ne mesurent pas la qualité de la politique finale ; ceux de validation, oui.

### 11.3 Affichage de la table Q

```python
    print("\nCases (observation, action) visitées :", int((visits > 0).sum()), "/", visits.size)
    print("Q-table finale (arrondie, lignes = observation (l,qS,qH), colonnes = anneau1..5, STOP) :")
    np.set_printoptions(linewidth=140, suppress=True)
    for o in range(N_OBS):
        if visits[o].sum() > 0:
            print(f"  l={o//9} qS={(o%9)//3} qH={o%3} :", np.round(Q[o], 3), " visites", visits[o])
```
- nombre de cases (observation, action) visitées sur les 324 possibles ;
- `np.set_printoptions` règle l'affichage des tableaux ;
- pour chaque observation visitée, on affiche ses valeurs Q (arrondies à 3 décimales) et le nombre de visites. `o // 9` retrouve `l`, `(o % 9) // 3` retrouve `q_S`, `o % 3` retrouve `q_H` (division entière et reste).

### 11.4 Enregistrement

```python
    np.save(os.path.join(script_dir, f'q_table{suffix}.npy'), Q)
    scipy.io.savemat(os.path.join(script_dir, f'q_table{suffix}.mat'), {'q_table': Q})
    np.save(os.path.join(script_dir, f'visites_Q{suffix}.npy'), visits)
```
- `q_table.npy` et `q_table.mat` : la table Q apprise (c'est la « politique » de l'agent) ;
- `visites_Q.npy` : les compteurs de visites.

```python
    hist = dict(retour=H['ret'], r_supprimes=H['r'], violation=H['violation'], stop=H['stop'], epsilon=H['eps'],
                val_episodes=np.array(val_eps), val_r_sur_rmax=np.array(val_r),
                val_tentatives_violation=np.array(val_viol), val_retour=np.array(val_ret),
                meilleur_r_vu_en_training=best_seen)
    np.save(os.path.join(script_dir, f'historique_training{suffix}.npy'), hist, allow_pickle=True)
    scipy.io.savemat(os.path.join(script_dir, f'historique_training{suffix}.mat'), hist)
```
- `hist` : dictionnaire de tout l'historique (retours, violations, ε, courbes de validation, meilleures trouvailles) ;
- sauvegardé en `.npy` (`allow_pickle=True` car c'est un dictionnaire) et en `.mat`.

```python
    np.savez(os.path.join(script_dir, f'sarsa_config{suffix}.npz'), eps_G=tol['eps_G'], eps_S=tol['eps_S'],
             eps_H_rel=tol['eps_H_rel'], eps_theta=tol['eps_theta'], penalty=args.penalty, alpha=args.alpha,
             gamma=args.gamma, episodes=args.episodes, seed=args.seed)
```
`sarsa_config.npz` : les tolérances et hyperparamètres utilisés, pour que `test.py` évalue avec **exactement les mêmes réglages**.

### 11.5 Courbes

```python
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def mm(x, w=500):
        return np.convolve(x, np.ones(w) / w, mode='valid')
```
- `matplotlib.use('Agg')` : dessiner sans ouvrir de fenêtre (utile sur un serveur) ;
- `mm` : moyenne glissante sur 500 épisodes (`np.convolve` avec une fenêtre de poids égaux) pour lisser les courbes bruitées.

```python
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(mm(H['ret'])); ax[0].set_title("Retour par épisode (moyenne glissante, avec exploration)")
    ax[0].set_xlabel("épisode")
    ax[1].plot(mm(H['violation']) * 100, label="violations (entraînement)")
    ax[1].plot(np.array(val_eps) - 1, np.array(val_viol) * 100, 'o-', label="tentatives de violation (validation gloutonne)")
    ax[1].set_title("Violations (%)"); ax[1].set_xlabel("épisode"); ax[1].legend()
    ax[2].plot(val_eps, val_r, 'o-'); ax[2].set_title("Validation gloutonne : r / r_max moyen"); ax[2].set_xlabel("épisode")
    ax[2].set_ylim(0, 1.05)
    for a_ in ax: a_.grid(alpha=0.3)
```
Trois graphiques côte à côte : (1) retour par épisode (avec exploration) ; (2) violations : part des épisodes d'entraînement qui s'achèvent par une violation, et tentatives de violation de la politique gloutonne sur la validation ; (3) validation gloutonne : r / r_max moyen au fil de l'entraînement, c'est **la** courbe qui montre ce que la politique a réellement appris. `grid(alpha=0.3)` ajoute une grille discrète.

```python
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'courbes_SARSA{suffix}.png'), dpi=150)
    print(f"\n✅ q_table{suffix}, visites_Q{suffix}, historique_training{suffix}, sarsa_config{suffix}, "
          f"courbes_SARSA{suffix}.png enregistrés dans le dossier du script.")
```
Enregistre la figure `courbes_SARSA.png` et affiche la confirmation.

```python
if __name__ == '__main__':
    main()
```
Garde-fou : `main()` n'est exécuté que si on lance ce fichier directement (`python training.py`), pas quand `test.py` l'importe.

---

## Résumé du flux

```
Minput/Moutput_training  ──►  scénarios (810)           Minput/Moutput_validation ──► scénarios (90)
        │ pour chaque épisode (20 000 fois)                           │ tous les 1000 épisodes
        ▼                                                             ▼
 tirer un scénario ► ε-glouton ► supprimer un élément          évaluation gloutonne, sans apprentissage
 ► calcul du diagramme ► récompense ► mise à jour SARSA
        ▼
 q_table.npy, historique_training, sarsa_config.npz, courbes_SARSA.png
```
