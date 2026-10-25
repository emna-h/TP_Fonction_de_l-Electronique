# Explication ligne par ligne de `test_model_cGAN.py`

**Rôle du script (étape 4/4)** : vérifier si le générateur entraîné fait bien son travail, sur des exemples **qu'il n'a jamais vus** (le jeu de test).

## Le principe

1. On choisit une spécification du jeu de test (Gmax, SSL, HPBW, theta0).
2. Le générateur propose **K architectures candidates** (en changeant le bruit `z`).
3. Pour chaque candidat, on **calcule le vrai diagramme de rayonnement** avec la physique (le même code que pour fabriquer le dataset) : on ne fait pas confiance au réseau pour juger ses propres propositions.
4. On compare les performances obtenues à celles de l'architecture de référence, avec les mêmes erreurs que dans `test_model_FFNN.py`.

Pourquoi plusieurs candidats ? Le problème inverse a **plusieurs solutions** : deux architectures différentes peuvent donner des performances presque identiques. Le cGAN en propose plusieurs, et on garde la meilleure après vérification physique.

---

## 1. Imports

```python
import torch
import torch.nn as nn
```
PyTorch et son module de réseaux (pour recréer le générateur).

```python
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import savemat
```
NumPy (tableaux), `os` (chemins), Matplotlib (tracés), `find_peaks` (repérer les lobes), `savemat` (écrire un `.mat`).

---

## 2. Bannière

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
clear_console()
```
Affiche la barre de séparation « NOUVELLE EXÉCUTION ».

---

## 3. Constantes physiques

```python
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5
```
Fréquence (2,45 GHz), vitesse de la lumière, longueur d'onde, rayon du premier anneau (0,2 λ), écart entre anneaux (0,5 λ) et nombre d'anneaux. **Ces valeurs doivent être identiques à celles de `dataset.py`**, sinon on comparerait des antennes différentes.

---

## 4. Paramètres du test

```python
indice_test = 10
```
Numéro de l'exemple de test analysé en détail (le 11ᵉ, car on compte à partir de 0).

```python
K = 50
```
Nombre de candidats générés pour cet exemple.

```python
EVALUER_TOUT_LE_TEST = True
K_test_complet = 20
```
Si `True`, on évalue aussi les 100 exemples de test (avec 20 candidats chacun). Mets `False` pour un test rapide.

```python
seed = 46
```
Graine aléatoire.

```python
w1, w2, w3 = 0.33, 0.33, 0.34
```
Poids de l'erreur globale pour le gain, le SSL et le HPBW (leur somme vaut 1).

```python
np.random.seed(seed)
torch.manual_seed(seed)
```
Fixent les graines : le bruit `z` tiré sera le même à chaque exécution.

---

## 5. Chargement des données

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
```
Dossier du script.

```python
Minput_cGAN_test = np.load(os.path.join(script_dir, 'Minput_test.npy')).T
Moutput_cGAN_test = np.load(os.path.join(script_dir, 'Moutput_test.npy')).T
```
Charge les spécifications et les architectures de **test** (4 × 100 et 5 × 100), puis les **transpose** (`.T`) : on obtient 100 × 4 et 100 × 5, **une ligne par exemple**.

```python
Minput_complete = np.load(os.path.join(script_dir, 'Minput.npy'))
```
Charge le dataset complet : on y cherchera les valeurs maximales pour calculer des erreurs relatives.

```python
norm_params = np.load(os.path.join(script_dir, 'normalization_params.npz'))
Minput_min = norm_params['Minput_min']
Minput_max = norm_params['Minput_max']
Moutput_min = norm_params['Moutput_min']
Moutput_max = norm_params['Moutput_max']
```
Récupère les min/max de normalisation **calculés pendant l'entraînement**. Il faut utiliser les mêmes qu'à l'entraînement : le réseau ne comprend que les données mises à la même échelle.

```python
Minput_norm_test = (Minput_cGAN_test - Minput_min) / (Minput_max - Minput_min + 1e-8)
```
Normalise les spécifications de test (valeurs entre 0 et 1, environ).

```python
cfg = np.load(os.path.join(script_dir, 'cgan_config.npz'))
d_z = int(cfg['d_z']); cond_size = int(cfg['cond_size']); x_size = int(cfg['x_size'])
G_hidden = tuple(int(h) for h in cfg['G_hidden'])
```
Relit la configuration du générateur (taille du bruit, de la condition, de la sortie, couches cachées). `int(...)` convertit les valeurs NumPy en entiers Python ; `tuple(...)` reconstruit le tuple des couches.

---

## 6. Recréation du générateur

```python
def build_mlp(in_size, hidden, out_size, act):
    layers, prev = [], in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)
```
Même fonction que dans l'entraînement : elle construit un réseau de couches denses avec activation (voir `cGAN_training_explication.md`).

```python
class Generator(nn.Module):
    def __init__(self, d_z, cond_size, hidden, x_size):
        super(Generator, self).__init__()
        self.net = build_mlp(d_z + cond_size, hidden, x_size, nn.ReLU)

    def forward(self, z, c):
        return torch.sigmoid(self.net(torch.cat([z, c], dim=1)))
```
**Doit être identique à la classe d'entraînement** : PyTorch ne sauvegarde que les poids, pas la structure. Sinon le chargement échouerait.

```python
G = Generator(d_z, cond_size, G_hidden, x_size)
```
Crée un générateur « vide » (poids aléatoires).

```python
G.load_state_dict(torch.load(os.path.join(script_dir, 'cgan_generator.pth'), map_location='cpu'))
```
`torch.load` lit le fichier de poids ; `map_location='cpu'` permet de le charger même s'il a été entraîné sur carte graphique ; `load_state_dict` injecte ces poids dans le générateur.

```python
G.eval()
```
Passe en **mode évaluation** (comportement figé, utile si le modèle contenait du dropout ou de la batch-normalisation ; ce n'est pas le cas ici, mais c'est la bonne habitude).

---

## 7. Génération de candidats

```python
def generer_candidats(c_norm, K):
```
Fonction qui reçoit **une** spécification normalisée (4 valeurs) et renvoie K architectures.

```python
    cond = torch.tensor(c_norm, dtype=torch.float32).unsqueeze(0).repeat(K, 1)
```
- `torch.tensor(...)` : convertit en tenseur ;
- `.unsqueeze(0)` : ajoute une dimension (1 × 4) ;
- `.repeat(K, 1)` : copie K fois (K × 4). Ainsi les K candidats partent de **la même spécification**.

```python
    z = torch.randn(K, d_z)
```
K vecteurs de bruit **différents** : c'est ce qui rend les K candidats différents.

```python
    with torch.no_grad():
        x_hat = G(z, cond).numpy()
```
Passe dans G sans calcul de gradient (inutile en test) et convertit le résultat en tableau NumPy.

```python
    x = x_hat * (Moutput_max - Moutput_min + 1e-8) + Moutput_min
```
**Dénormalisation** : on repasse de [0, 1] à des nombres d'éléments (0 à 10).

```python
    x = np.clip(np.round(x), Moutput_min, Moutput_max).astype(int)
```
- `np.round` arrondit à l'entier le plus proche (on ne peut pas poser 3,4 antennes) ;
- `np.clip` force les valeurs entre le minimum et le maximum ;
- `.astype(int)` convertit en entiers.

```python
    return x
```
Renvoie la matrice K × 5.

---

## 8. Constantes de calcul

```python
k = 2 * np.pi / lambda_
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)
```
Nombre d'onde k, angles d'azimut de pointage, et les 1000 directions où l'on calcule le diagramme (0 à 360°), en radians puis en degrés.

---

## 9. Fonction physique `calcul_AF_performance_metrics`

C'est **ton code**, repris tel quel. Seul changement : `theta0` devient un argument de la fonction (au lieu d'une variable globale), pour pouvoir traiter des spécifications différentes. La logique (somme des ondes, normalisation, dB, HPBW, pics, SSL) est expliquée ligne par ligne dans `dataset_explication.md`, sections 5.3 à 5.8. Rappel de ce que la fonction fait :

```python
def calcul_AF_performance_metrics(elements_per_ring, theta0):
```
Entrées : le nombre d'éléments par anneau (5 nombres) et la direction visée (en radians).

```python
    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain
```
Sorties : le **diagramme en dB** (1000 valeurs), le **HPBW**, le **gain max** et le **SSL**.

---

## 10. Valeurs maximales pour les erreurs relatives

```python
max_main_lobe_gain = np.max(Minput_complete[0, :])
max_true_SSL_gain = np.max(np.abs(Minput_complete[1, :]))
max_HPBW = np.max(Minput_complete[2, :])
```
Maximum du gain, du SSL (en valeur absolue, car il est négatif) et du HPBW dans tout le dataset. Ces valeurs servent à exprimer une erreur en **pourcentage** d'une échelle de référence. (`max_HPBW` est calculé mais, comme dans ton code, c'est 180° qui sert au calcul de l'erreur relative du HPBW.)

---

## 11. Fonction `erreurs`

```python
def erreurs(perf_pred, perf_ref):
```
Compare deux jeux de performances `(gain, SSL, HPBW)` : celui du candidat (`perf_pred`) et celui de la référence (`perf_ref`).

```python
    g_p, s_p, h_p = perf_pred
    g_r, s_r, h_r = perf_ref
```
Sépare les trois valeurs de chaque jeu (« déballage » d'un tuple).

```python
    mae = (abs(g_p - g_r), abs(s_p - s_r), abs(h_p - h_r))
```
**Erreurs absolues** (MAE, *mean absolute error*) : écart en valeur absolue pour chaque grandeur (dB, dB, degrés).

```python
    rel = (100 * abs(g_p - g_r) / (abs(max_main_lobe_gain) + 1e-8),
           100 * abs(s_p - s_r) / (abs(max_true_SSL_gain) + 1e-8),
           100 * abs(h_p - h_r) / (180 + 1e-8))
```
**Erreurs relatives en %** : chaque écart divisé par l'échelle de la grandeur (gain max du dataset, SSL max du dataset, 180° pour le HPBW), fois 100. Le `1e-8` évite une division par zéro.

```python
    glob = w1 * rel[0] + w2 * rel[1] + w3 * rel[2]
    return mae, rel, glob
```
**Erreur globale pondérée** : moyenne pondérée des trois erreurs relatives (0,33 / 0,33 / 0,34). Les trois résultats sont renvoyés.

---

## 12. Fonction `evaluer_spec`

```python
def evaluer_spec(idx, K):
```
Traite complètement **une spécification** de test (numéro `idx`) avec K candidats.

```python
    theta0 = np.deg2rad(Minput_cGAN_test[idx, 3])
```
Direction visée de cet exemple (colonne 3), convertie en radians.

```python
    ref_arch = Moutput_cGAN_test[idx].astype(int)
```
L'architecture de **référence** (celle qui a servi à fabriquer l'exemple).

```python
    AF_ref, h_r, g_r, s_r = calcul_AF_performance_metrics(ref_arch, theta0)
```
Performances de la référence, par la physique.

```python
    cands = generer_candidats(Minput_norm_test[idx], K)
```
Le générateur propose K architectures.

```python
    cands = [cd for cd in cands if cd.sum() > 0]
```
On **élimine les candidats vides** (zéro élément partout) : une antenne sans élément n'existe pas. C'est une compréhension de liste avec filtre.

```python
    cands_unique = list(dict.fromkeys(tuple(cd) for cd in cands))
```
Supprime les **doublons** en conservant l'ordre : on transforme chaque candidat en tuple (hashable), `dict.fromkeys` garde les clés uniques dans l'ordre d'apparition, puis `list` reconvertit. Évite de recalculer deux fois le même diagramme.

```python
    resultats = []
    for cd in cands_unique:
        AF_c, h_c, g_c, s_c = calcul_AF_performance_metrics(np.array(cd), theta0)
        mae, rel, glob = erreurs((g_c, s_c, h_c), (g_r, s_r, h_r))
        resultats.append(dict(arch=np.array(cd), AF=AF_c, perf=(g_c, s_c, h_c), mae=mae, rel=rel, glob=glob))
```
Pour chaque candidat unique : on calcule son diagramme et ses performances (la physique), on les compare à la référence, puis on range tout dans un dictionnaire (architecture, diagramme, performances, erreurs).

```python
    return ref_arch, AF_ref, (g_r, s_r, h_r), resultats, len(cands)
```
Renvoie : l'architecture de référence, son diagramme, ses performances, la liste des résultats, et le nombre de candidats valides.

---

## 13. Exemple détaillé

```python
ref_arch, AF_ref_dB, perf_ref, resultats, nb_valides = evaluer_spec(indice_test, K)
```
Évalue l'exemple numéro `indice_test` avec K candidats.

```python
theta0deg = Minput_cGAN_test[indice_test, 3]
gain_main_ref, SSL_ref, HPBW_ref = perf_ref[0], perf_ref[1], perf_ref[2]
```
Direction visée (en degrés) et performances de référence, mises dans des variables lisibles.

```python
premier = resultats[0]
```
Le **premier candidat** : ce que donnerait un seul tirage (K = 1), donc la comparaison la plus équitable avec le FFNN, qui ne propose qu'une réponse.

```python
meilleur = min(resultats, key=lambda r: r['glob'])
```
Le **meilleur candidat** : celui dont l'erreur globale est la plus faible. `min(..., key=...)` cherche le minimum selon un critère ; `lambda r: r['glob']` est une mini-fonction qui lit l'erreur globale du résultat `r`.

```python
print(f"Spécification (échantillon test n°{indice_test}) :")
print(f"  Gmax = ... | SSL = ... | HPBW = ... | theta0 = ...")
print(f"\nArchitecture référence           : {ref_arch}")
print(f"Candidat n°1 (K=1)               : {premier['arch']}")
print(f"Meilleur des {K} candidats générés : {meilleur['arch']}")
print(f"({nb_valides} candidats valides sur {K}, dont {len(resultats)} architectures distinctes)")
```
Affichent la spécification demandée, l'architecture de référence, le premier candidat, le meilleur candidat, et combien de candidats étaient valides et distincts.

```python
def afficher(titre, r):
```
Fonction d'affichage qui imprime, pour un résultat `r` : les performances obtenues (à côté de celles de la référence), les erreurs absolues, les erreurs relatives et l'erreur globale.

```python
afficher("Candidat n°1 (K = 1)", premier)
afficher(f"Meilleur des {K} candidats", meilleur)
```
Appelle cette fonction pour les deux candidats.

---

## 14. Tracés

```python
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
```
Nouvelle figure, avec un graphique en **coordonnées polaires** (comme un radar : l'angle est la direction, la distance au centre est le niveau).

```python
for r in resultats:
    ax.plot(theta, r['AF'], color='gray', alpha=0.15, linewidth=0.8)
```
Dessine **tous les candidats en gris très transparent** (`alpha=0.15`) pour visualiser la diversité des solutions proposées.

```python
ax.plot(theta, AF_ref_dB, label='Référence', linewidth=2)
ax.plot(theta, meilleur['AF'], label='Meilleur candidat cGAN', linewidth=2, linestyle='--')
```
Par-dessus : le diagramme de **référence** (trait plein) et celui du **meilleur candidat** (pointillés).

```python
ax.set_title("Diagramme de rayonnement (polaire)", fontsize=13)
ax.set_rlim([-40, 0])
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
```
Titre, échelle radiale de −40 à 0 dB, légende placée en haut à droite.

```python
plt.figure(figsize=(10, 5))
for r in resultats:
    plt.plot(theta_deg, r['AF'], color='gray', alpha=0.15, linewidth=0.8)
plt.plot(theta_deg, AF_ref_dB, label='Référence', linewidth=2)
plt.plot(theta_deg, meilleur['AF'], label='Meilleur candidat cGAN', linewidth=2, linestyle='--')
```
Même tracé en coordonnées **cartésiennes** : l'azimut en abscisse, le gain en dB en ordonnée.

```python
plt.xlabel("Azimut (°)")
plt.ylabel("Gain (dB)")
plt.title("Diagramme de rayonnement - Azimut (candidats en gris)")
plt.xlim([0, 360]); plt.ylim([-40, 0])
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'diagramme_cGAN_exemple.png'), dpi=150)
```
Étiquettes des axes, titre, limites (0–360° et −40–0 dB), grille, légende, puis enregistrement de l'image `diagramme_cGAN_exemple.png`.

---

## 15. Évaluation sur tout le jeu de test

```python
if EVALUER_TOUT_LE_TEST:
```
Ce bloc ne s'exécute que si l'option est activée.

```python
    n_test = Minput_cGAN_test.shape[0]
```
Nombre d'exemples de test (100).

```python
    res = np.zeros((n_test, 7))
```
Tableau de résultats : une ligne par exemple, 7 colonnes : erreur globale du premier candidat, erreur globale du meilleur, erreurs relatives gain / SSL / HPBW du meilleur, nombre d'architectures distinctes, et « architecture de référence retrouvée exactement ? » (1 ou 0).

```python
    for i in range(n_test):
        ref_a, _, _, rs, _ = evaluer_spec(i, K_test_complet)
        b = min(rs, key=lambda r: r['glob'])
        res[i] = [rs[0]['glob'], b['glob'], *b['rel'], len(rs), float(np.array_equal(b['arch'], ref_a))]
```
Pour chaque exemple : on évalue 20 candidats (`_` ignore les valeurs inutiles), on prend le meilleur `b`, et on remplit la ligne. `*b['rel']` « déballe » les trois erreurs relatives dans la liste ; `np.array_equal` vaut vrai si l'architecture trouvée est exactement la référence.

```python
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_test} spécifications évaluées")
```
Affiche l'avancement tous les 20 exemples (ce calcul prend quelques minutes).

```python
    print(f"... Erreur globale moyenne, K = 1 ... : {res[:, 0].mean():.2f} % ...")
    print(f"... meilleur des {K_test_complet} ... : {res[:, 1].mean():.2f} % ...")
```
Affichent la moyenne et la médiane (`np.median`) de l'erreur globale, pour le premier candidat puis pour le meilleur. La médiane est moins sensible aux quelques mauvais cas.

```python
    print(f"  dont erreur relative moyenne gain / SSL / HPBW : ...")
    print(f"Nombre moyen d'architectures distinctes par spécification : ...")
    print(f"Architecture de référence retrouvée exactement : {100 * res[:, 6].mean():.1f} % ...")
```
Moyenne des trois erreurs relatives, diversité moyenne des candidats, et pourcentage de cas où la référence est retrouvée à l'identique. `res[:, 0]` désigne la **colonne** 0 de `res` pour toutes les lignes.

```python
    np.save(os.path.join(script_dir, 'resultats_test_cGAN.npy'), res)
    savemat(os.path.join(script_dir, 'resultats_test_cGAN.mat'), {'resultats_test_cGAN': res})
```
Sauvegarde le tableau de résultats en `.npy` et en `.mat`.

```python
plt.show()
```
Affiche les figures à l'écran.

---

## Comment interpréter les résultats

- **Erreur du candidat n°1** : performance d'un seul tirage, comparable à celle du FFNN.
- **Erreur du meilleur des K** : gain apporté par la vérification physique de plusieurs candidats.
- **Référence retrouvée exactement : 0 %** n'est pas un échec en soi : il peut exister plusieurs architectures donnant des performances voisines. Le critère pertinent est l'erreur de performance, pas l'égalité des architectures.
