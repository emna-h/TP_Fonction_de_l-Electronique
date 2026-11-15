# =============================================================================
#  repartition.py  --  Étape 2/4 : répartition training / validation / test
#
#  On sépare des CONFIGURATIONS entières (colonnes), jamais des épisodes :
#  une configuration vue à l'apprentissage ne doit pas se retrouver au test.
#
#  - même permutation (graine 46) que repartition_training_test.py du FFNN et du cGAN
#    => les 100 PREMIERS indices de la permutation forment le même ensemble de TEST
#  - validation : les 90 indices suivants de la même permutation (10 % des 900 restants)
#  - training   : les 810 derniers
#
#  Sorties (dossier du script, formats .mat ET .npy) :
#     Minput_training, Moutput_training, Minput_validation, Moutput_validation,
#     Minput_test, Moutput_test, indices_split (indices dans Minput/Moutput d'origine)
# =============================================================================
import numpy as np
import scipy.io
import os


### Affichage visuel de démarcation (alternative à l'effacement console)
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")

clear_console()

np.random.seed(46)

# === Chargement des matrices Minput et Moutput (dossier du script) ===
script_dir = os.path.dirname(os.path.abspath(__file__))
Minput = np.load(os.path.join(script_dir, 'Minput.npy'))
Moutput = np.load(os.path.join(script_dir, 'Moutput.npy'))

# === Mélange aléatoire des indices (identique au FFNN / cGAN) ===
total_cols = Minput.shape[1]
rand_indices = np.random.permutation(total_cols)

# === Séparation : 10 % test, 10 % des restants en validation, le reste en training ===
nb_test = round(0.10 * total_cols)                  # 100
nb_val = round(0.10 * (total_cols - nb_test))       # 90
test_indices = rand_indices[:nb_test]
val_indices = rand_indices[nb_test:nb_test + nb_val]
train_indices = rand_indices[nb_test + nb_val:]

# === Contrôle de fuite : aucune configuration commune aux trois ensembles ===
assert len(set(test_indices) & set(val_indices)) == 0
assert len(set(test_indices) & set(train_indices)) == 0
assert len(set(val_indices) & set(train_indices)) == 0
assert len(test_indices) + len(val_indices) + len(train_indices) == total_cols

# === Fuite approchée : même architecture présente dans deux ensembles différents ===
nom_ens = {}
for nom, idx in [('training', train_indices), ('validation', val_indices), ('test', test_indices)]:
    for i in idx:
        nom_ens[i] = nom
archi = {}
for i in range(total_cols):
    archi.setdefault(tuple(int(x) for x in Moutput[:, i]), []).append(i)
doublons = {a: ids for a, ids in archi.items() if len(ids) > 1}
print(f"Architectures apparaissant plusieurs fois : {len(doublons)}")
for a, ids in doublons.items():
    print(f"  architecture {a} : colonnes {ids} -> ensembles {[nom_ens[i] for i in ids]}, "
          f"theta0 = {[round(float(Minput[3, i]), 2) for i in ids]}")
    if len({nom_ens[i] for i in ids}) > 1:
        print("  ATTENTION : fuite faible (même architecture, theta0 différent, ensembles différents).")

# === Sélection des colonnes ===
matrices = {
    'Minput_training': Minput[:, train_indices], 'Moutput_training': Moutput[:, train_indices],
    'Minput_validation': Minput[:, val_indices], 'Moutput_validation': Moutput[:, val_indices],
    'Minput_test': Minput[:, test_indices], 'Moutput_test': Moutput[:, test_indices],
}
for nom, M in matrices.items():
    print(f"\nMatrice {nom} : shape {M.shape}")
    print(np.round(M[:, :3], 4) if nom.startswith('Minput') else M[:, :3].astype(int))


print(f"\nTest : {nb_test} configurations, validation : {nb_val}, training : {len(train_indices)}")

# === Enregistrement .mat et .npy dans le dossier contenant le code ===
matrices['indices_split'] = np.array([np.concatenate([train_indices, val_indices, test_indices]),
                                      np.concatenate([np.zeros(len(train_indices)), np.ones(len(val_indices)),
                                                      2 * np.ones(len(test_indices))])])
for nom, M in matrices.items():
    scipy.io.savemat(os.path.join(script_dir, nom + '.mat'), {nom: M})   # .mat
    np.save(os.path.join(script_dir, nom + '.npy'), M)                   # .npy

print("\n✅ Matrices training / validation / test enregistrées en .mat et .npy dans le dossier du script.")
print("   indices_split : ligne 1 = indice d'origine, ligne 2 = ensemble (0 training, 1 validation, 2 test).")
