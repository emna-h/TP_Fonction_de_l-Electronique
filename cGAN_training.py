# =============================================================================
#  cGAN_training.py  --  Étape 3/4 : entraînement du cGAN (PyTorch)
#
#  Problème inverse : spécifications c = [Gmax, SSL, HPBW, theta0]
#                     -> architecture x = [N1..N5] (éléments par anneau, 0..10)
#
#  Générateur     G([z | c]) -> x_hat dans [0,1]^5   (sigmoïde ; x_norm = (N-min)/(max-min))
#  Discriminateur D([x | c]) -> logit (probabilité "réel" après sigmoïde)
#  Perte : entropie croisée binaire (BCE), perte "non saturante" pour G,
#          + terme auxiliaire optionnel L_N = (somme des N_hat - 10^(Gmax/20))^2
#          (car Gmax = 20 log10(N_total) au pointage theta0).
#
#  Entrées  (dossier du script) : Minput_training.npy, Moutput_training.npy
#  Sorties  (dossier du script) : cgan_generator.pth, cgan_discriminator.pth,
#                                 normalization_params.npz, cgan_config.npz,
#                                 loss_history_cGAN.npy / .mat, courbes_cGAN.png
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.io import savemat


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")

# Appel immédiat au début du script
clear_console()

# =============================================================================
# === Hyperparamètres modifiables ===
# =============================================================================
d_z = 16                    # dimension du bruit z
G_hidden = (128, 128, 64)   # couches cachées du générateur
D_hidden = (128, 128, 64)   # couches cachées du discriminateur
batch_size = 64
num_epochs = 2000
lr_G = 2e-4                 # learning rate (Adam)
lr_D = 2e-4
adam_betas = (0.5, 0.999)
real_label = 0.9            # lissage unilatéral des étiquettes "réel" (1.0 = désactivé)
dequantization = True       # bruit uniforme ±0.5 élément sur les données réelles (x continu)
use_aux_loss = True         # terme auxiliaire L_N (lien Gmax <-> nombre total d'éléments)
lambda_aux = 1.0            # poids de L_N
N_tot_scale = 50.0          # normalisation de L_N (nombre max d'éléments = 5 x 10)
seed = 46
# =============================================================================

np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé : {device}")

# === Chargement des données du dossier qui contient le script ===
script_dir = os.path.dirname(os.path.abspath(__file__))

Minput_training = np.load(os.path.join(script_dir, 'Minput_training.npy'))
Moutput_training = np.load(os.path.join(script_dir, 'Moutput_training.npy'))

# Transposer si nécessaire (-> une ligne par échantillon)
if Minput_training.shape[0] < Minput_training.shape[1]:
    Minput_training = Minput_training.T
if Moutput_training.shape[0] < Moutput_training.shape[1]:
    Moutput_training = Moutput_training.T

# === Normalisation (Min-Max), statistiques calculées sur le TRAINING uniquement ===
Minput_min = Minput_training.min(axis=0)
Minput_max = Minput_training.max(axis=0)
Moutput_min = Moutput_training.min(axis=0)
Moutput_max = Moutput_training.max(axis=0)

Minput_norm = (Minput_training - Minput_min) / (Minput_max - Minput_min + 1e-8)
Moutput_norm = (Moutput_training - Moutput_min) / (Moutput_max - Moutput_min + 1e-8)

# Conversion en tenseurs PyTorch
C = torch.tensor(Minput_norm, dtype=torch.float32)    # condition c (spécifications)
X = torch.tensor(Moutput_norm, dtype=torch.float32)   # donnée réelle x (architecture)

cond_size = C.shape[1]   # 4
x_size = X.shape[1]      # 5

# === DataLoader ===
dataset = TensorDataset(X, C)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Constantes (tenseurs) pour L_N et pour la dé-quantification
Moutput_range = torch.tensor(Moutput_max - Moutput_min + 1e-8, dtype=torch.float32, device=device)
Moutput_min_t = torch.tensor(Moutput_min, dtype=torch.float32, device=device)
Gmax_min, Gmax_range = float(Minput_min[0]), float(Minput_max[0] - Minput_min[0] + 1e-8)


# === Définition des modèles ===
def build_mlp(in_size, hidden, out_size, act):
    layers, prev = [], in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """G([z | c]) -> x_hat dans [0,1]^5 (sigmoïde en sortie)."""
    def __init__(self, d_z, cond_size, hidden, x_size):
        super(Generator, self).__init__()
        self.net = build_mlp(d_z + cond_size, hidden, x_size, nn.ReLU)

    def forward(self, z, c):
        return torch.sigmoid(self.net(torch.cat([z, c], dim=1)))


class Discriminator(nn.Module):
    """D([x | c]) -> logit ; la probabilité 'réel' est sigmoid(logit)."""
    def __init__(self, x_size, cond_size, hidden):
        super(Discriminator, self).__init__()
        self.net = build_mlp(x_size + cond_size, hidden, 1, lambda: nn.LeakyReLU(0.2))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], dim=1))


G = Generator(d_z, cond_size, G_hidden, x_size).to(device)
D = Discriminator(x_size, cond_size, D_hidden).to(device)

criterion = nn.BCEWithLogitsLoss()   # = sigmoïde + BCE (numériquement stable)
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr_G, betas=adam_betas)
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr_D, betas=adam_betas)

print(f"Paramètres G : {sum(p.numel() for p in G.parameters())} | "
      f"Paramètres D : {sum(p.numel() for p in D.parameters())}")


def N_total_from_xhat(x_hat):
    """Nombre total d'éléments prédit (valeurs dé-normalisées, avant arrondi)."""
    return (x_hat * Moutput_range + Moutput_min_t).sum(dim=1)


# === Entraînement ===
history = {'loss_D': [], 'loss_G': [], 'D_real': [], 'D_fake': [], 'aux': []}
start_time = time.time()

for epoch in range(num_epochs):
    s_lD = s_lG = s_Dr = s_Df = s_aux = 0.0
    n_batches = 0
    for real_x, c in dataloader:
        real_x, c = real_x.to(device), c.to(device)
        B = real_x.shape[0]

        # Dé-quantification : les données réelles sont des entiers -> on les rend
        # continues (±0.5 élément) pour que D ne distingue pas "entier/continu".
        if dequantization:
            noise = (torch.rand_like(real_x) - 0.5) / Moutput_range
            real_x_in = (real_x + noise).clamp(0.0, 1.0)
        else:
            real_x_in = real_x

        # ---------- 1) Mise à jour du discriminateur ----------
        z = torch.randn(B, d_z, device=device)
        with torch.no_grad():
            fake_x = G(z, c)
        logit_real = D(real_x_in, c)
        logit_fake = D(fake_x, c)
        loss_D = criterion(logit_real, torch.full_like(logit_real, real_label)) + \
                 criterion(logit_fake, torch.zeros_like(logit_fake))
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # ---------- 2) Mise à jour du générateur (perte non saturante) ----------
        z = torch.randn(B, d_z, device=device)
        fake_x = G(z, c)
        logit_fake = D(fake_x, c)
        loss_adv = criterion(logit_fake, torch.ones_like(logit_fake))

        # terme auxiliaire : sum(N_hat) ~ 10^(Gmax/20)
        gain_dB = c[:, 0] * Gmax_range + Gmax_min
        N_target = 10.0 ** (gain_dB / 20.0)
        loss_aux = (((N_total_from_xhat(fake_x) - N_target) / N_tot_scale) ** 2).mean()

        loss_G = loss_adv + (lambda_aux * loss_aux if use_aux_loss else 0.0)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        s_lD += loss_D.item(); s_lG += loss_G.item()
        s_Dr += torch.sigmoid(logit_real).mean().item()
        s_Df += torch.sigmoid(logit_fake).mean().item()
        s_aux += loss_aux.item()
        n_batches += 1

    history['loss_D'].append(s_lD / n_batches)
    history['loss_G'].append(s_lG / n_batches)
    history['D_real'].append(s_Dr / n_batches)
    history['D_fake'].append(s_Df / n_batches)
    history['aux'].append(s_aux / n_batches)

    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}/{num_epochs} | Loss_D: {history['loss_D'][-1]:.4f} | "
              f"Loss_G: {history['loss_G'][-1]:.4f} | D(x): {history['D_real'][-1]:.3f} | "
              f"D(G(z)): {history['D_fake'][-1]:.3f} | L_N: {history['aux'][-1]:.5f}")

end_time = time.time()
print(f"\n=== Training finished in {end_time - start_time:.2f} seconds ===")

# === Sauvegarde du modèle, des paramètres de normalisation et de la configuration ===
torch.save(G.state_dict(), os.path.join(script_dir, 'cgan_generator.pth'))
torch.save(D.state_dict(), os.path.join(script_dir, 'cgan_discriminator.pth'))

np.savez(os.path.join(script_dir, 'normalization_params.npz'),
         Minput_min=Minput_min,
         Minput_max=Minput_max,
         Moutput_min=Moutput_min,
         Moutput_max=Moutput_max)

np.savez(os.path.join(script_dir, 'cgan_config.npz'),
         d_z=d_z, cond_size=cond_size, x_size=x_size,
         G_hidden=np.array(G_hidden), D_hidden=np.array(D_hidden))

hist_array = np.array([history[k] for k in ['loss_D', 'loss_G', 'D_real', 'D_fake', 'aux']])
np.save(os.path.join(script_dir, 'loss_history_cGAN.npy'), hist_array)
savemat(os.path.join(script_dir, 'loss_history_cGAN.mat'), {'loss_history_cGAN': hist_array})
print("✅ Modèles (.pth), normalization_params.npz, cgan_config.npz et historique enregistrés dans le dossier du script.")

# === Tracé des courbes ===
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(history['loss_D'], label='Loss D'); ax[0].plot(history['loss_G'], label='Loss G')
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Pertes du cGAN"); ax[0].legend(); ax[0].grid()
ax[1].plot(history['D_real'], label='D(x,c)'); ax[1].plot(history['D_fake'], label='D(G(z,c),c)')
ax[1].axhline(0.5, color='gray', linestyle=':')
ax[1].set_xlabel("Epoch"); ax[1].set_title("Sorties moyennes de D"); ax[1].legend(); ax[1].grid()
ax[2].semilogy(history['aux']); ax[2].set_xlabel("Epoch"); ax[2].set_title("Terme auxiliaire L_N"); ax[2].grid()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'courbes_cGAN.png'), dpi=150)
plt.show()
