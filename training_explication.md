# Explication ligne par ligne de `cGAN_training.py`

**Rôle du script (étape 3/4)** : entraîner le cGAN.

## L'idée en quelques phrases

Un **GAN conditionnel** met deux réseaux de neurones en compétition :

- le **Générateur G** (le « faussaire ») reçoit les spécifications voulues (la *condition* `c` : Gmax, SSL, HPBW, theta0) plus un **bruit aléatoire `z`**, et propose une architecture `x` (5 nombres, un par anneau) ;
- le **Discriminateur D** (le « contrôleur ») reçoit une architecture **et** sa condition, et doit dire si c'est un **vrai** exemple du dataset ou un faux produit par G.

À force de jouer, G apprend à fabriquer des architectures que D ne distingue plus des vraies. Le bruit `z` fait que, pour une même spécification, G peut proposer **plusieurs architectures différentes** : c'est utile, car plusieurs architectures peuvent donner des performances proches.

## Vocabulaire

| Terme | Signification simple |
|---|---|
| Epoch | Un passage complet sur tout le dataset d'entraînement. |
| Batch | Un petit paquet d'exemples traité d'un coup (ici 64). |
| Loss (perte) | Un nombre qui mesure « à quel point le réseau se trompe ». On cherche à le diminuer. |
| Gradient / backpropagation | Le calcul qui indique dans quel sens modifier chaque poids du réseau pour diminuer la perte. |
| Optimiseur (Adam) | L'algorithme qui applique ces modifications. |
| Learning rate (lr) | La taille du pas de modification. Trop grand : instable. Trop petit : très lent. |
| Normalisation | Ramener les données dans [0, 1] pour que le réseau apprenne plus facilement. |

---

## 1. Imports

```python
import torch
```
**PyTorch** : la bibliothèque d'apprentissage profond (tenseurs, réseaux, gradients).

```python
import torch.nn as nn
```
Le sous-module `nn` (*neural networks*) : couches, fonctions d'activation, fonctions de perte.

```python
from torch.utils.data import DataLoader, TensorDataset
```
- `TensorDataset` : associe plusieurs tenseurs ligne à ligne (ici architecture + condition) ;
- `DataLoader` : découpe en batchs et mélange.

```python
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.io import savemat
```
NumPy (tableaux), `os` (chemins), `time` (mesurer la durée), `matplotlib.pyplot` (courbes), `savemat` (écrire des `.mat`).

---

## 2. Bannière

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
clear_console()
```
Affiche la barre de séparation « NOUVELLE EXÉCUTION » (comme dans tes autres scripts).

---

## 3. Hyperparamètres

Les **hyperparamètres** sont les réglages que l'on choisit *avant* l'entraînement (contrairement aux poids, appris automatiquement).

```python
d_z = 16
```
Taille du vecteur de bruit `z` (16 nombres aléatoires fournis à G).

```python
G_hidden = (128, 128, 64)
D_hidden = (128, 128, 64)
```
Nombre de neurones des **couches cachées** de G et de D : trois couches de 128, 128 et 64 neurones.

```python
batch_size = 64
```
Taille d'un paquet d'exemples.

```python
num_epochs = 2000
```
Nombre de passages sur tout le dataset.

```python
lr_G = 2e-4
lr_D = 2e-4
```
Learning rate de G et de D (`2e-4` = 0,0002).

```python
adam_betas = (0.5, 0.999)
```
Deux réglages internes d'Adam (mémoire des gradients passés). Le `0.5` au lieu du `0.9` habituel est une valeur classique pour stabiliser les GAN.

```python
real_label = 0.9
```
**Lissage des étiquettes** : au lieu de dire à D « un vrai exemple vaut 1 », on lui dit « 0,9 ». Cela l'empêche d'être trop sûr de lui, ce qui aide l'équilibre du jeu. Mettre `1.0` désactive le lissage.

```python
dequantization = True
```
Les vraies architectures sont des **entiers** (3, 7…) alors que G sort des nombres **continus** (0,43…). D pourrait les distinguer juste pour cette raison. En mettant `True`, on ajoute un petit bruit (± un demi-élément) aux vraies architectures pour qu'elles soient aussi continues.

```python
use_aux_loss = True
```
Active un **terme de perte supplémentaire** (voir plus loin).

```python
lambda_aux = 1.0
```
Poids de ce terme supplémentaire.

```python
N_tot_scale = 50.0
```
Valeur de normalisation de ce terme : le nombre maximal d'éléments est 5 anneaux × 10 = 50.

```python
seed = 46
```
Graine aléatoire.

---

## 4. Reproductibilité et matériel

```python
np.random.seed(seed)
torch.manual_seed(seed)
```
Fixent les graines de NumPy et de PyTorch : deux exécutions donnent (à peu près) les mêmes résultats.

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Choisit où calculer : la **carte graphique** (`cuda`) si elle est disponible, sinon le **processeur** (`cpu`). Pour ce petit réseau, le CPU suffit.

```python
print(f"Device utilisé : {device}")
```
Affiche le choix.

---

## 5. Chargement et préparation des données

```python
script_dir = os.path.dirname(os.path.abspath(__file__))
```
Dossier du script.

```python
Minput_training = np.load(os.path.join(script_dir, 'Minput_training.npy'))
Moutput_training = np.load(os.path.join(script_dir, 'Moutput_training.npy'))
```
Charge les matrices de training (4 × 900 et 5 × 900) créées par `repartition_training_test.py`.

```python
if Minput_training.shape[0] < Minput_training.shape[1]:
    Minput_training = Minput_training.T
if Moutput_training.shape[0] < Moutput_training.shape[1]:
    Moutput_training = Moutput_training.T
```
Si la matrice a moins de lignes que de colonnes (cas 4 × 900), on la **transpose** (`.T`) pour obtenir 900 × 4. En apprentissage automatique, on veut **une ligne par exemple**.

### Normalisation min-max

```python
Minput_min = Minput_training.min(axis=0)
Minput_max = Minput_training.max(axis=0)
Moutput_min = Moutput_training.min(axis=0)
Moutput_max = Moutput_training.max(axis=0)
```
Minimum et maximum **de chaque colonne** (`axis=0` = on parcourt les exemples) : un min/max par grandeur (Gmax, SSL, HPBW, theta0 ; et un par anneau). **Calculés sur le training uniquement.**

```python
Minput_norm = (Minput_training - Minput_min) / (Minput_max - Minput_min + 1e-8)
Moutput_norm = (Moutput_training - Moutput_min) / (Moutput_max - Moutput_min + 1e-8)
```
Formule `(valeur − min) / (max − min)` : chaque valeur est ramenée entre **0 et 1**. Le `1e-8` évite une division par zéro si min = max. Cela compte pour le cGAN : la sortie de G (sigmoïde) est forcément entre 0 et 1, et les 4 spécifications ont des échelles très différentes (degrés, dB…).

```python
C = torch.tensor(Minput_norm, dtype=torch.float32)
X = torch.tensor(Moutput_norm, dtype=torch.float32)
```
Convertit les tableaux NumPy en **tenseurs PyTorch** (`float32`). `C` est la condition (spécifications), `X` la donnée réelle (architecture).

```python
cond_size = C.shape[1]
x_size = X.shape[1]
```
Nombre de colonnes : 4 (taille de la condition) et 5 (taille de l'architecture).

```python
dataset = TensorDataset(X, C)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
```
- `TensorDataset(X, C)` : lie chaque architecture à sa condition ;
- `DataLoader(...)` : fournit des batchs de 64 ; `shuffle=True` mélange à chaque epoch ; `drop_last=True` ignore le dernier batch s'il est incomplet.

```python
Moutput_range = torch.tensor(Moutput_max - Moutput_min + 1e-8, dtype=torch.float32, device=device)
Moutput_min_t = torch.tensor(Moutput_min, dtype=torch.float32, device=device)
Gmax_min, Gmax_range = float(Minput_min[0]), float(Minput_max[0] - Minput_min[0] + 1e-8)
```
Constantes gardées sous forme de tenseurs pour **revenir aux vraies valeurs** (dénormaliser) à l'intérieur de l'entraînement :
- étendue (max − min) et minimum de chaque anneau ;
- minimum et étendue du gain Gmax (première ligne de `Minput`).

---

## 6. Définition des réseaux

```python
def build_mlp(in_size, hidden, out_size, act):
```
Fonction utilitaire qui construit un **MLP** (réseau de couches denses). Paramètres : taille d'entrée, tailles des couches cachées, taille de sortie, fonction d'activation.

```python
    layers, prev = [], in_size
```
`layers` : liste des couches (vide au départ) ; `prev` : taille de la couche précédente (au départ, l'entrée).

```python
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
```
Pour chaque taille de couche cachée `h` : on ajoute une couche **linéaire** `nn.Linear(prev, h)` (chaque neurone calcule une somme pondérée de ses entrées) suivie d'une **activation** (qui introduit la non-linéarité, sans laquelle empiler des couches ne servirait à rien). Puis `prev = h`.

```python
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)
```
Dernière couche linéaire vers la sortie, **sans activation**. `nn.Sequential(*layers)` enchaîne toutes les couches dans l'ordre ; `*` « déballe » la liste.

### Le générateur

```python
class Generator(nn.Module):
```
Définit une classe de réseau, qui hérite de `nn.Module` (la base de tous les modèles PyTorch).

```python
    def __init__(self, d_z, cond_size, hidden, x_size):
        super(Generator, self).__init__()
        self.net = build_mlp(d_z + cond_size, hidden, x_size, nn.ReLU)
```
- `__init__` construit le réseau ; `super().__init__()` active la mécanique PyTorch (gestion des poids et des gradients) ;
- l'entrée a la taille `d_z + cond_size` = 16 + 4 = 20, car on **concatène** bruit et condition ;
- la sortie a 5 valeurs (une par anneau) ;
- l'activation est **ReLU** : `max(0, x)`.

```python
    def forward(self, z, c):
        return torch.sigmoid(self.net(torch.cat([z, c], dim=1)))
```
`forward` décrit le calcul : `torch.cat([z, c], dim=1)` colle `z` et `c` côte à côte ; le réseau les traite ; `torch.sigmoid` écrase le résultat dans **[0, 1]** (cohérent avec les architectures normalisées).

### Le discriminateur

```python
class Discriminator(nn.Module):
    def __init__(self, x_size, cond_size, hidden):
        super(Discriminator, self).__init__()
        self.net = build_mlp(x_size + cond_size, hidden, 1, lambda: nn.LeakyReLU(0.2))
```
Même principe. L'entrée a la taille 5 + 4 = 9 (architecture + condition) ; la sortie est **un seul nombre**. L'activation est **LeakyReLU(0,2)** : comme ReLU mais avec une petite pente (0,2) pour les valeurs négatives, ce qui évite des neurones « morts » (usage classique pour D). `lambda: ...` est une mini-fonction qui crée une activation neuve à chaque appel.

```python
    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))
```
Colle architecture et condition, et renvoie un **logit** (un score brut, non borné). Plus il est grand, plus D pense que c'est un vrai exemple. La probabilité est `sigmoid(logit)`.

### Création des instances

```python
G = Generator(d_z, cond_size, G_hidden, x_size).to(device)
D = Discriminator(x_size, cond_size, D_hidden).to(device)
```
Crée un générateur et un discriminateur, et les envoie sur le processeur/la carte choisis.

```python
criterion = nn.BCEWithLogitsLoss()
```
La **perte** : entropie croisée binaire (BCE) appliquée à un logit. Elle mesure l'écart entre la décision de D et l'étiquette voulue (0 ou 1). La version « WithLogits » applique la sigmoïde en interne, de façon plus stable numériquement.

```python
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=adam_betas)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=adam_betas)
```
Un **optimiseur Adam** par réseau. Chacun ne modifie que les paramètres de « son » réseau (`.parameters()`).

```python
print(f"Paramètres G : {sum(p.numel() for p in G.parameters())} | "
      f"Paramètres D : {sum(p.numel() for p in D.parameters())}")
```
Compte le nombre total de poids (`numel` = nombre d'éléments de chaque tenseur de paramètres) : environ 27 800 pour G et 26 100 pour D.

### Nombre total d'éléments

```python
def N_total_from_xhat(x_hat):
    return (x_hat * Moutput_range + Moutput_min_t).sum(dim=1)
```
Prend la sortie de G (normalisée), la **dénormalise** (`× étendue + min`) puis additionne les 5 anneaux (`sum(dim=1)`) : le **nombre total d'éléments** proposé, avant arrondi. Sert au terme auxiliaire.

---

## 7. Boucle d'entraînement

```python
history = {'loss_D': [], 'loss_G': [], 'D_real': [], 'D_fake': [], 'aux': []}
```
Dictionnaire de listes vides pour garder l'historique de chaque epoch (pour les courbes).

```python
start_time = time.time()
```
Heure de début.

```python
for epoch in range(num_epochs):
```
Boucle sur les 2000 epochs.

```python
    s_lD = s_lG = s_Dr = s_Df = s_aux = 0.0
    n_batches = 0
```
Remet à zéro les cumuls de l'epoch (pour calculer ensuite des moyennes).

```python
    for real_x, c in dataloader:
```
Boucle sur les batchs : `real_x` = 64 vraies architectures, `c` = leurs 64 conditions.

```python
        real_x, c = real_x.to(device), c.to(device)
        B = real_x.shape[0]
```
Envoie le batch sur le bon matériel ; `B` est sa taille.

### Dé-quantification

```python
        if dequantization:
            noise = (torch.rand_like(real_x) - 0.5) / Moutput_range
            real_x_in = (real_x + noise).clamp(0.0, 1.0)
        else:
            real_x_in = real_x
```
- `torch.rand_like` tire des nombres entre 0 et 1 de même forme que `real_x` ; en retirant 0,5, on obtient du bruit entre −0,5 et +0,5 ;
- diviser par `Moutput_range` (≈ 10) exprime ce bruit en unité normalisée, donc ± un demi-élément ;
- `.clamp(0, 1)` ramène le résultat dans [0, 1] ;
- si l'option est désactivée, on garde les données telles quelles.

### Étape 1 : entraîner le discriminateur

```python
        z = torch.randn(B, d_z, device=device)
```
Tire `B × d_z` nombres selon une loi **normale** (gaussienne) : le bruit `z`.

```python
        with torch.no_grad():
            fake_x = G(z, c)
```
G fabrique de fausses architectures. `torch.no_grad()` dit « ne calcule pas de gradients ici » : on ne veut pas modifier G pendant l'étape de D.

```python
        logit_real = D(real_x_in, c)
        logit_fake = D(fake_x, c)
```
D donne son score pour les vraies et pour les fausses.

```python
        loss_D = criterion(logit_real, torch.full_like(logit_real, real_label)) + \
                 criterion(logit_fake, torch.zeros_like(logit_fake))
```
La perte de D est la somme de :
- l'erreur sur les **vraies** (étiquette cible 0,9, `torch.full_like` crée un tenseur rempli de cette valeur) ;
- l'erreur sur les **fausses** (étiquette cible 0, `zeros_like`).
La barre oblique `\` en fin de ligne continue l'instruction à la ligne suivante.

```python
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()
```
Le rituel de PyTorch, dans cet ordre :
1. `zero_grad()` efface les gradients précédents (sinon ils s'accumulent) ;
2. `backward()` calcule les gradients de la perte par rapport aux poids de D (**rétropropagation**) ;
3. `step()` met à jour les poids de D.

### Étape 2 : entraîner le générateur

```python
        z = torch.randn(B, d_z, device=device)
        fake_x = G(z, c)
```
Nouveau bruit, nouvelles fausses architectures. Cette fois **sans** `no_grad` : on veut que le gradient remonte jusqu'à G.

```python
        logit_fake = D(fake_x, c)
        loss_adv = criterion(logit_fake, torch.ones_like(logit_fake))
```
D juge les fausses ; G veut que D réponde « vrai » : la cible est donc **1**. C'est la perte « non saturante », qui donne de bons gradients à G même quand D le démasque facilement.

```python
        gain_dB = c[:, 0] * Gmax_range + Gmax_min
```
Retrouve le gain Gmax **en dB** demandé (colonne 0 de la condition), en inversant la normalisation.

```python
        N_target = 10.0 ** (gain_dB / 20.0)
```
Nombre total d'éléments qui correspond à ce gain. Raison physique : au pointage, toutes les ondes sont en phase donc Gmax = 20·log10(N_total), d'où N_total = 10^(Gmax/20).

```python
        loss_aux = (((N_total_from_xhat(fake_x) - N_target) / N_tot_scale) ** 2).mean()
```
Écart au carré entre le nombre d'éléments proposé et celui attendu, normalisé par 50, puis moyenné sur le batch. C'est le **terme auxiliaire** : une aide physique qui guide G.

```python
        loss_G = loss_adv + (lambda_aux * loss_aux if use_aux_loss else 0.0)
```
Perte totale de G = perte du jeu + (poids × terme auxiliaire) si l'option est active.

```python
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
```
Même rituel que pour D, appliqué à G. Les gradients traversent D mais **seuls les poids de G sont modifiés**, car `optimizer_G` ne contrôle que G.

### Suivi

```python
        s_lD += loss_D.item(); s_lG += loss_G.item()
        s_Dr += torch.sigmoid(logit_real).mean().item()
        s_Df += torch.sigmoid(logit_fake).mean().item()
        s_aux += loss_aux.item()
        n_batches += 1
```
Cumule, pour cet epoch :
- les deux pertes (`.item()` convertit un tenseur à un seul nombre en nombre Python) ;
- la **probabilité moyenne** que D attribue aux vrais (`D_real`) et aux faux (`D_fake`) ;
- le terme auxiliaire ;
- le nombre de batchs.

```python
    history['loss_D'].append(s_lD / n_batches)
    history['loss_G'].append(s_lG / n_batches)
    history['D_real'].append(s_Dr / n_batches)
    history['D_fake'].append(s_Df / n_batches)
    history['aux'].append(s_aux / n_batches)
```
Calcule les moyennes de l'epoch et les ajoute à l'historique.

```python
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Loss_D: ...")
```
Affiche le suivi tous les 50 epochs (et à la première). `%` est le reste de la division : `(epoch+1) % 50 == 0` est vrai pour 50, 100, 150…

**Comment lire ces chiffres** : si D(x) et D(G(z)) restent proches de 0,5, le jeu est équilibré ; si D(G(z)) tombe vers 0, D gagne trop (G n'apprend plus) ; la perte de G d'un GAN n'a pas à descendre vers 0.

```python
end_time = time.time()
print(f"\n=== Training finished in {end_time - start_time:.2f} seconds ===")
```
Affiche la durée totale (`:.2f` = 2 décimales).

---

## 8. Sauvegarde

```python
torch.save(G.state_dict(), os.path.join(script_dir, 'cgan_generator.pth'))
torch.save(D.state_dict(), os.path.join(script_dir, 'cgan_discriminator.pth'))
```
`state_dict()` est le dictionnaire des **poids appris**. On l'enregistre dans des fichiers `.pth`. Le script de test n'utilise que le générateur.

```python
np.savez(os.path.join(script_dir, 'normalization_params.npz'),
         Minput_min=Minput_min, Minput_max=Minput_max,
         Moutput_min=Moutput_min, Moutput_max=Moutput_max)
```
Sauvegarde les min/max de normalisation. Le test doit appliquer **exactement les mêmes** transformations.

```python
np.savez(os.path.join(script_dir, 'cgan_config.npz'),
         d_z=d_z, cond_size=cond_size, x_size=x_size,
         G_hidden=np.array(G_hidden), D_hidden=np.array(D_hidden))
```
Sauvegarde l'architecture du réseau (tailles), pour que le test reconstruise un générateur **identique** avant d'y charger les poids.

```python
hist_array = np.array([history[k] for k in ['loss_D', 'loss_G', 'D_real', 'D_fake', 'aux']])
```
Empile les 5 listes d'historique en une matrice 5 × 2000 (compréhension de liste : une boucle écrite sur une ligne).

```python
np.save(os.path.join(script_dir, 'loss_history_cGAN.npy'), hist_array)
savemat(os.path.join(script_dir, 'loss_history_cGAN.mat'), {'loss_history_cGAN': hist_array})
```
Enregistre l'historique en `.npy` et en `.mat`.

---

## 9. Courbes

```python
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
```
Crée une figure avec **3 graphiques côte à côte** (1 ligne, 3 colonnes) de 15 × 4 pouces. `ax` est la liste des 3 graphiques.

```python
ax[0].plot(history['loss_D'], label='Loss D'); ax[0].plot(history['loss_G'], label='Loss G')
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Pertes du cGAN"); ax[0].legend(); ax[0].grid()
```
Graphique 1 : pertes de D et de G en fonction de l'epoch ; légende (`legend`), grille (`grid`).

```python
ax[1].plot(history['D_real'], label='D(x,c)'); ax[1].plot(history['D_fake'], label='D(G(z,c),c)')
ax[1].axhline(0.5, color='gray', linestyle=':')
ax[1].set_xlabel("Epoch"); ax[1].set_title("Sorties moyennes de D"); ax[1].legend(); ax[1].grid()
```
Graphique 2 : probabilités moyennes de D, avec une ligne pointillée à 0,5 (équilibre idéal).

```python
ax[2].semilogy(history['aux']); ax[2].set_xlabel("Epoch"); ax[2].set_title("Terme auxiliaire L_N"); ax[2].grid()
```
Graphique 3 : terme auxiliaire en échelle **logarithmique** (`semilogy`), car il varie sur plusieurs ordres de grandeur.

```python
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'courbes_cGAN.png'), dpi=150)
plt.show()
```
`tight_layout` évite les chevauchements ; `savefig` enregistre l'image (résolution 150 dpi) ; `show` l'affiche à l'écran (le script attend alors que tu fermes la fenêtre).

---

## Résultat

`cgan_generator.pth`, `cgan_discriminator.pth`, `normalization_params.npz`, `cgan_config.npz`, `loss_history_cGAN.npy/.mat` et `courbes_cGAN.png`, tous dans le dossier du script.
