# =============================================================================
#  repartition_training_test.py  --  Étape 2/4 : répartition training / test
#
#  Charge Minput.npy et Moutput.npy (générés par dataset.py), mélange les
#  colonnes (graine 46), garde 10 % pour le test et le reste pour le training.
#
#  Sorties (dossier du script, formats .mat ET .npy) :
#     Minput_training, Moutput_training, Minput_test, Moutput_test
# =============================================================================
import numpy as np
import scipy.io
import os


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")

# Appel immédiat au début du script
clear_console()

np.random.seed(46)

# === Chargement des matrices Minput et Moutput (dossier du script) ===
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput = np.load(os.path.join(script_dir, 'Minput.npy'))
Moutput = np.load(os.path.join(script_dir, 'Moutput.npy'))

# === Mélange aléatoire des indices ===
total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)

# === Séparation 10 % test / le reste en training ===
nb_test = round(0.10 * total_cols)
test_indices = rand_indices[:nb_test]
train_indices = rand_indices[nb_test:]

# === Sélection des colonnes ===
Minput_test = Minput[:, test_indices]
Minput_training = Minput[:, train_indices]
Moutput_test = Moutput[:, test_indices]
Moutput_training = Moutput[:, train_indices]

# === Affichage ===
for nom, M in [('Minput_training', Minput_training), ('Moutput_training', Moutput_training),
               ('Minput_test', Minput_test), ('Moutput_test', Moutput_test)]:
    print(f"\nCaractéristiques de la matrice {nom} :")
    print(f"- Shape (lignes, colonnes) : {M.shape}")
    print("→ Premières colonnes (extraits) :")
    print(np.round(M[:, :5], 4) if nom.startswith('Minput') else M[:, :5].astype(int))

# === Enregistrement .mat et .npy dans le dossier contenant le code ===
matrices = {'Minput_training': Minput_training, 'Moutput_training': Moutput_training,
            'Minput_test': Minput_test, 'Moutput_test': Moutput_test}
for nom, M in matrices.items():
    scipy.io.savemat(os.path.join(script_dir, nom + '.mat'), {nom: M})   # .mat
    np.save(os.path.join(script_dir, nom + '.npy'), M)                   # .npy

print("\n✅ Minput/Moutput _training et _test enregistrées en .mat et .npy dans le dossier du script.")
