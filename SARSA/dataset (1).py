# =============================================================================
#  dataset.py  --  Étape 1/4 : banque de scénarios (projet SARSA)
#  Réseau d'antennes circulaire multi-anneaux (5 anneaux, 0 à 10 éléments/anneau)
#
#  Le modèle physique (facteur de réseau, Gmax, SSL, HPBW) et la graine (46) sont
#  STRICTEMENT ceux du dataset.py du FFNN et du cGAN : on obtient donc exactement
#  les mêmes matrices Minput / Moutput.
#
#  Ici, chaque colonne est un SCÉNARIO : une configuration de référence à alléger
#  (Moutput) et ses performances de référence (Minput), qui définissent les
#  contraintes de l'environnement SARSA. Il n'y a pas d'étiquette à prédire.
#
#  Sorties (dans le dossier de ce script) :
#     Minput.mat / Minput.npy    (4 x N)  [Gmax (dB) ; SSL (dB) ; HPBW (deg) ; theta0 (deg)]
#     Moutput.mat / Moutput.npy  (5 x N)  [nombre d'éléments des anneaux 1 à 5]
# =============================================================================

import numpy as np
from scipy.signal import find_peaks
import scipy.io
import os


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")

# Appel immédiat au début du script
clear_console()


# === Paramètres modifiables ===
nb_samples = 1000          # (le cGAN profite d'un dataset plus grand : ex. 20000 ou 50000)
max_rings = 5
max_elements = 10
theta0_max_deg = 180
carrierFreq = 2.45e9
c = 3e8
lambda_ = c / carrierFreq
k = 2 * np.pi / lambda_
r0 = 0.2 * lambda_
delta_r = 0.5 * lambda_
np.random.seed(46)

# === Initialisation des matrices ===
Moutput = np.zeros((max_rings, nb_samples))  # Architecture
Minput = np.zeros((4, nb_samples))           # [MainLobe, SSL, HPBW, theta0]

# === Boucle principale ===
for sample_idx in range(nb_samples):
    # --- Architecture complètement aléatoire ---
    while True:
        elements_per_ring = np.random.randint(0, max_elements + 1, size=max_rings)
        if np.sum(elements_per_ring > 0) > 0:  # S'assurer qu'au moins un ring est actif
            break
    Moutput[:, sample_idx] = elements_per_ring

    # --- Theta0 aléatoire ---
    theta0deg = np.random.uniform(0, theta0_max_deg)
    theta0 = np.deg2rad(theta0deg)
    phi0 = 0

    # --- Rayons des anneaux ---
    radii = r0 + delta_r * np.arange(max_rings)

    # --- Diagramme 2D ---
    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = 0
    AF_az = np.zeros_like(theta, dtype=complex)

    for ring in range(max_rings):
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

    # --- Stockage ---
    Minput[0, sample_idx] = main_lobe_gain
    Minput[1, sample_idx] = true_SSL_gain
    Minput[2, sample_idx] = HPBW
    Minput[3, sample_idx] = theta0deg


# === Affichage d'une partie du Minput et Moutput ===
print("\nCaractéristiques de la matrice Minput :")
print(f"- Shape (lignes, colonnes) : {Minput.shape}")
print("→ Premières colonnes (extraits) :")
print(np.round(Minput[:, :5], 4))

print("\nCaractéristiques de la matrice Moutput :")
print(f"- Shape (lignes, colonnes) : {Moutput.shape}")
print("→ Premières colonnes (extraits) :")
print(Moutput[:, :5].astype(int))


# === Validation de la banque de scénarios (voir rapport, section 8.1.3) ===
eps_G_defaut = 1.0   # dB, tolérance de gain par défaut (sert seulement à l'affichage)
assert Minput.shape == (4, nb_samples) and Moutput.shape == (max_rings, nb_samples)
assert np.all(Moutput == np.round(Moutput)) and Moutput.min() >= 0 and Moutput.max() <= max_elements
N_ref_all = Moutput.sum(axis=0)
assert N_ref_all.min() >= 1, "une configuration sans aucun élément"
assert Minput[3].min() >= 0 and Minput[3].max() <= theta0_max_deg
# Loi du gain : Gmax = 20 log10(N_ref) (amplitudes unitaires, phases de pointage)
ecart_loi_gain = np.max(np.abs(Minput[0] - 20 * np.log10(N_ref_all)))
r_max_all = np.floor(N_ref_all * (1 - 10 ** (-eps_G_defaut / 20)) + 1e-9).astype(int)
print("\nValidation de la banque de scénarios :")
print(f"- N_ref : min {int(N_ref_all.min())}, médiane {np.median(N_ref_all):.0f}, max {int(N_ref_all.max())}")
print(f"- Écart maximal à la loi du gain 20*log10(N_ref) : {ecart_loi_gain:.2e} dB")
print(f"- Valeurs de repli (HPBW = 180 ou SSL = 0) : {int(np.sum((Minput[2] == 180) | (Minput[1] == 0)))}")
print(f"- Architectures distinctes : {np.unique(Moutput.T, axis=0).shape[0]} / {nb_samples}")
print(f"- r_max (nombre maximal de suppressions) pour eps_G = {eps_G_defaut} dB : "
      f"moyenne {r_max_all.mean():.2f}, max {r_max_all.max()}, "
      f"part de scénarios avec r_max = 0 : {100 * np.mean(r_max_all == 0):.1f} %")


# === Enregistrement de Minput et Moutput (.mat et .npy) dans le dossier contenant le code ===
script_dir = os.path.dirname(os.path.abspath(__file__))

scipy.io.savemat(os.path.join(script_dir, 'Minput.mat'), {'Minput': Minput})
scipy.io.savemat(os.path.join(script_dir, 'Moutput.mat'), {'Moutput': Moutput})

np.save(os.path.join(script_dir, 'Minput.npy'), Minput)
np.save(os.path.join(script_dir, 'Moutput.npy'), Moutput)

print("✅ Minput.mat/.npy et Moutput.mat/.npy ont été enregistrés dans le dossier du script.")
