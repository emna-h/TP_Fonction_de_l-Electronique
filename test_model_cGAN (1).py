# =============================================================================
#  test_model_cGAN.py  --  Étape 4/4 : test du cGAN entraîné (PyTorch)
#
#  Pour une spécification c = [Gmax, SSL, HPBW, theta0] du jeu de test :
#    1) G génère K candidats (K bruits z différents) ; 10*x_hat est arrondi en entiers
#    2) chaque candidat est évalué avec le modèle physique (fonction
#       calcul_AF_performance_metrics du code fourni : diagramme, Gmax, SSL, HPBW)
#    3) erreurs MAE et erreurs relatives (%), erreur globale pondérée (w = 0.33/0.33/0.34),
#       comme dans test_model_FFNN.py
#  Le problème inverse étant "un-à-plusieurs", on affiche :
#       - le candidat n°1 (K = 1, comparable au FFNN)
#       - le meilleur des K candidats (plus petite erreur globale)
#  Puis (EVALUER_TOUT_LE_TEST) la même évaluation sur TOUT le jeu de test.
#
#  Entrées : Minput_test.npy, Moutput_test.npy, Minput.npy, normalization_params.npz,
#            cgan_generator.pth, cgan_config.npz   (dossier du script)
#  Sorties : resultats_test_cGAN.npy / .mat, diagramme_cGAN_exemple.png
# =============================================================================
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.io import savemat


def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
clear_console()

# === Paramètres modifiables : doivent être identiques à ceux de dataset.py ===
###########################
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
max_rings = 5
###########################

# === Paramètres du test ===
indice_test = 10               # exemple détaillé (comme dans test_model_FFNN.py)
K = 50                         # nombre de candidats générés par spécification
EVALUER_TOUT_LE_TEST = True    # évaluer aussi tout le jeu de test (plus long)
K_test_complet = 20            # candidats par spécification pour l'évaluation complète
seed = 46
w1, w2, w3 = 0.33, 0.33, 0.34  # pondérations de l'erreur globale

np.random.seed(seed)
torch.manual_seed(seed)

# === Chargement des données ===
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput_cGAN_test = np.load(os.path.join(script_dir, 'Minput_test.npy')).T    # une ligne par échantillon
Moutput_cGAN_test = np.load(os.path.join(script_dir, 'Moutput_test.npy')).T
Minput_complete = np.load(os.path.join(script_dir, 'Minput.npy'))

norm_params = np.load(os.path.join(script_dir, 'normalization_params.npz'))
Minput_min = norm_params['Minput_min']
Minput_max = norm_params['Minput_max']
Moutput_min = norm_params['Moutput_min']
Moutput_max = norm_params['Moutput_max']

Minput_norm_test = (Minput_cGAN_test - Minput_min) / (Minput_max - Minput_min + 1e-8)

cfg = np.load(os.path.join(script_dir, 'cgan_config.npz'))
d_z = int(cfg['d_z']); cond_size = int(cfg['cond_size']); x_size = int(cfg['x_size'])
G_hidden = tuple(int(h) for h in cfg['G_hidden'])


# === Définition du générateur === (identique à cGAN_training.py)
def build_mlp(in_size, hidden, out_size, act):
    layers, prev = [], in_size
    for h in hidden:
        layers += [nn.Linear(prev, h), act()]
        prev = h
    layers.append(nn.Linear(prev, out_size))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, d_z, cond_size, hidden, x_size):
        super(Generator, self).__init__()
        self.net = build_mlp(d_z + cond_size, hidden, x_size, nn.ReLU)

    def forward(self, z, c):
        return torch.sigmoid(self.net(torch.cat([z, c], dim=1)))


# === Chargement du générateur entraîné ===
G = Generator(d_z, cond_size, G_hidden, x_size)
G.load_state_dict(torch.load(os.path.join(script_dir, 'cgan_generator.pth'), map_location='cpu'))
G.eval()


def generer_candidats(c_norm, K):
    """K architectures (entiers) pour une spécification normalisée c_norm (4,)."""
    cond = torch.tensor(c_norm, dtype=torch.float32).unsqueeze(0).repeat(K, 1)
    z = torch.randn(K, d_z)
    with torch.no_grad():
        x_hat = G(z, cond).numpy()
    x = x_hat * (Moutput_max - Moutput_min + 1e-8) + Moutput_min      # dénormalisation
    x = np.clip(np.round(x), Moutput_min, Moutput_max).astype(int)    # arrondi en entiers
    return x


# === Constantes ===
k = 2 * np.pi / lambda_
phi0 = 0
phi = 0
theta = np.linspace(0, 2 * np.pi, 1000)
theta_deg = np.rad2deg(theta)


# === Fonction : calcul du diagramme et des métriques (code fourni) ===
# Seule différence avec test_model_FFNN.py : theta0 est passé en argument
# (au lieu d'être une variable globale) afin de pouvoir boucler sur plusieurs spécifications.
def calcul_AF_performance_metrics(elements_per_ring, theta0):
    rings = len(elements_per_ring)
    radii = r0 + delta_r * np.arange(max_rings)

    AF_az = np.zeros_like(theta, dtype=complex)

    for ring in range(rings):
        a = radii[ring]
        N = elements_per_ring[ring]
        if N == 0:
            continue
        phi_n = 2 * np.pi * np.arange(N) / N
        for n in range(N):
            phase = k * a * (np.sin(theta) * np.cos(phi - phi_n[n]) -
                             np.sin(theta0) * np.cos(phi0 - phi_n[n]))
            AF_az += np.exp(1j * phase)

    AF_norm_az = np.abs(AF_az) / (np.max(np.abs(AF_az)) + np.finfo(float).eps)
    AF_dB_az = 20 * np.log10(AF_norm_az + np.finfo(float).eps)
    AF_dB_az[AF_dB_az < -40] = -40
    theta_deg = np.rad2deg(theta)

    # calcul du gain max non normalisé du lobe principal
    AF_abs_az = np.abs(AF_az)
    maxVal = np.max(AF_abs_az)
    maxVal_non_norm = 20 * np.log10(maxVal + np.finfo(float).eps)

    # --- HPBW ---
    maxVal_dB = np.max(AF_dB_az)
    maxIdx = np.argmax(AF_dB_az)
    halfPower = maxVal_dB - 3
    AF_dB_ext = np.concatenate((AF_dB_az, AF_dB_az, AF_dB_az))
    theta_deg_ext = np.concatenate((theta_deg - 360, theta_deg, theta_deg + 360))
    maxIdx_ext = maxIdx + len(theta_deg)

    leftIdx_ext = np.where(AF_dB_ext[:maxIdx_ext] <= halfPower)[0][-1] if np.any(
        AF_dB_ext[:maxIdx_ext] <= halfPower) else None

    rightIdx_ext = np.where(AF_dB_ext[maxIdx_ext:] <= halfPower)[0]
    rightIdx_ext = rightIdx_ext[0] + maxIdx_ext if len(rightIdx_ext) > 0 else None

    if leftIdx_ext is None or rightIdx_ext is None:
        HPBW = 180
    else:
        HPBW = theta_deg_ext[rightIdx_ext] - theta_deg_ext[leftIdx_ext]

    # --- Gain lobe principal & SSL ---
    responseLin = AF_norm_az
    peaks, _ = find_peaks(responseLin, distance=5)

    pk = responseLin[peaks]
    if len(pk) == 0:
        main_lobe_gain = maxVal_non_norm
        true_SSL_gain = 0
    else:
        sorted_idx = np.argsort(pk)[::-1]
        sorted_pk = pk[sorted_idx]
        sorted_angles = theta_deg[peaks][sorted_idx]

        threshold_dB = 1
        main_lobes_idx = np.where(
            20 * np.log10(sorted_pk + np.finfo(float).eps) >=
            20 * np.log10(sorted_pk[0] + np.finfo(float).eps) - threshold_dB)[0]
        side_lobe_idx = np.setdiff1d(np.arange(len(sorted_pk)), main_lobes_idx)

        main_lobe_gain = maxVal_non_norm

        true_SSL_gain = 20 * np.log10(sorted_pk[side_lobe_idx[0]] + np.finfo(float).eps) if len(
            side_lobe_idx) > 0 else 0

    return AF_dB_az, HPBW, maxVal_non_norm, true_SSL_gain


# === Valeurs maximales pour les erreurs relatives (comme test_model_FFNN.py) ===
max_main_lobe_gain = np.max(Minput_complete[0, :])
max_true_SSL_gain = np.max(np.abs(Minput_complete[1, :]))
max_HPBW = np.max(Minput_complete[2, :])


def erreurs(perf_pred, perf_ref):
    """perf = (gain, SSL, HPBW). Retourne MAE (3), erreurs relatives % (3) et erreur globale %."""
    g_p, s_p, h_p = perf_pred
    g_r, s_r, h_r = perf_ref
    mae = (abs(g_p - g_r), abs(s_p - s_r), abs(h_p - h_r))
    rel = (100 * abs(g_p - g_r) / (abs(max_main_lobe_gain) + 1e-8),
           100 * abs(s_p - s_r) / (abs(max_true_SSL_gain) + 1e-8),
           100 * abs(h_p - h_r) / (180 + 1e-8))
    glob = w1 * rel[0] + w2 * rel[1] + w3 * rel[2]
    return mae, rel, glob


def evaluer_spec(idx, K):
    """Génère K candidats pour l'échantillon de test idx ; renvoie les candidats uniques valides,
    leurs performances et erreurs."""
    theta0 = np.deg2rad(Minput_cGAN_test[idx, 3])
    ref_arch = Moutput_cGAN_test[idx].astype(int)
    AF_ref, h_r, g_r, s_r = calcul_AF_performance_metrics(ref_arch, theta0)
    cands = generer_candidats(Minput_norm_test[idx], K)
    cands = [cd for cd in cands if cd.sum() > 0]                 # au moins un élément
    cands_unique = list(dict.fromkeys(tuple(cd) for cd in cands))  # supprime les doublons (ordre conservé)
    resultats = []
    for cd in cands_unique:
        AF_c, h_c, g_c, s_c = calcul_AF_performance_metrics(np.array(cd), theta0)
        mae, rel, glob = erreurs((g_c, s_c, h_c), (g_r, s_r, h_r))
        resultats.append(dict(arch=np.array(cd), AF=AF_c, perf=(g_c, s_c, h_c), mae=mae, rel=rel, glob=glob))
    return ref_arch, AF_ref, (g_r, s_r, h_r), resultats, len(cands)


# =============================================================================
# === 1) Exemple détaillé (indice_test) ===
# =============================================================================
ref_arch, AF_ref_dB, perf_ref, resultats, nb_valides = evaluer_spec(indice_test, K)
theta0deg = Minput_cGAN_test[indice_test, 3]
gain_main_ref, SSL_ref, HPBW_ref = perf_ref[0], perf_ref[1], perf_ref[2]

premier = resultats[0]                                       # candidat n°1 (K = 1)
meilleur = min(resultats, key=lambda r: r['glob'])           # meilleur des K

print(f"Spécification (échantillon test n°{indice_test}) :")
print(f"  Gmax = {Minput_cGAN_test[indice_test, 0]:.2f} dB | SSL = {Minput_cGAN_test[indice_test, 1]:.2f} dB | "
      f"HPBW = {Minput_cGAN_test[indice_test, 2]:.2f}° | theta0 = {theta0deg:.2f}°")
print(f"\nArchitecture référence           : {ref_arch}")
print(f"Candidat n°1 (K=1)               : {premier['arch']}")
print(f"Meilleur des {K} candidats générés : {meilleur['arch']}")
print(f"({nb_valides} candidats valides sur {K}, dont {len(resultats)} architectures distinctes)")


def afficher(titre, r):
    print(f"\n--- {titre} ---")
    print(f"Gain du lobe principal : {r['perf'][0]:.2f} dB   (réf. {gain_main_ref:.2f})")
    print(f"True Side Lobe Gain   : {r['perf'][1]:.2f} dB   (réf. {SSL_ref:.2f})")
    print(f"HPBW                  : {r['perf'][2]:.2f}°    (réf. {HPBW_ref:.2f})")
    print(f"Erreurs absolues (MAE): gain {r['mae'][0]:.2f} dB | SSL {r['mae'][1]:.2f} dB | HPBW {r['mae'][2]:.2f}°")
    print(f"Erreurs relatives (%) : gain {r['rel'][0]:.2f} | SSL {r['rel'][1]:.2f} | HPBW {r['rel'][2]:.2f}")
    print(f"Erreur globale de prédiction pondérée : {r['glob']:.2f} %")

print(f"\n--- Performances Référence ---")
print(f"Gain du lobe principal : {gain_main_ref:.2f} dB")
print(f"True Side Lobe Gain   : {SSL_ref:.2f} dB")
print(f"HPBW                  : {HPBW_ref:.2f}°")
afficher("Candidat n°1 (K = 1)", premier)
afficher(f"Meilleur des {K} candidats", meilleur)

# === Tracés ===
plt.figure(figsize=(8, 6))
ax = plt.subplot(111, polar=True)
for r in resultats:
    ax.plot(theta, r['AF'], color='gray', alpha=0.15, linewidth=0.8)
ax.plot(theta, AF_ref_dB, label='Référence', linewidth=2)
ax.plot(theta, meilleur['AF'], label='Meilleur candidat cGAN', linewidth=2, linestyle='--')
ax.set_title("Diagramme de rayonnement (polaire)", fontsize=13)
ax.set_rlim([-40, 0])
ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.figure(figsize=(10, 5))
for r in resultats:
    plt.plot(theta_deg, r['AF'], color='gray', alpha=0.15, linewidth=0.8)
plt.plot(theta_deg, AF_ref_dB, label='Référence', linewidth=2)
plt.plot(theta_deg, meilleur['AF'], label='Meilleur candidat cGAN', linewidth=2, linestyle='--')
plt.xlabel("Azimut (°)")
plt.ylabel("Gain (dB)")
plt.title("Diagramme de rayonnement - Azimut (candidats en gris)")
plt.xlim([0, 360]); plt.ylim([-40, 0])
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'diagramme_cGAN_exemple.png'), dpi=150)

# =============================================================================
# === 2) Évaluation sur tout le jeu de test ===
# =============================================================================
if EVALUER_TOUT_LE_TEST:
    n_test = Minput_cGAN_test.shape[0]
    print(f"\n=== Évaluation sur les {n_test} échantillons de test (K = {K_test_complet} candidats) ===")
    # colonnes : [glob(K=1), glob(best-of-K), rel gain, rel SSL, rel HPBW (meilleur), nb archis distinctes, exact?]
    res = np.zeros((n_test, 7))
    for i in range(n_test):
        ref_a, _, _, rs, _ = evaluer_spec(i, K_test_complet)
        b = min(rs, key=lambda r: r['glob'])
        res[i] = [rs[0]['glob'], b['glob'], *b['rel'], len(rs), float(np.array_equal(b['arch'], ref_a))]
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_test} spécifications évaluées")
    print(f"\nErreur globale moyenne, K = 1 (candidat n°1)   : {res[:, 0].mean():.2f} %  (médiane {np.median(res[:, 0]):.2f} %)")
    print(f"Erreur globale moyenne, meilleur des {K_test_complet}          : {res[:, 1].mean():.2f} %  (médiane {np.median(res[:, 1]):.2f} %)")
    print(f"  dont erreur relative moyenne gain / SSL / HPBW : {res[:, 2].mean():.2f} / {res[:, 3].mean():.2f} / {res[:, 4].mean():.2f} %")
    print(f"Nombre moyen d'architectures distinctes par spécification : {res[:, 5].mean():.1f}")
    print(f"Architecture de référence retrouvée exactement : {100 * res[:, 6].mean():.1f} % des spécifications")
    np.save(os.path.join(script_dir, 'resultats_test_cGAN.npy'), res)
    savemat(os.path.join(script_dir, 'resultats_test_cGAN.mat'), {'resultats_test_cGAN': res})

plt.show()
