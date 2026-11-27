# Explication de `test.py`

`test.py` utilise l'agent entraîné par `training.py` (il relit `q_table.npy` et `sarsa_config.npz`). Il a **deux modes** :

```bash
# 1) un réseau donné : nombre d'antennes par anneau + direction visée (deg)
python test.py --arch 8 6 10 3 7 --theta0 60 --show

# 2) sans --arch : comparaison sur des réseaux aléatoires
python test.py --n_bench 60 --n_random 30 --sa_iters 2500
```

Options : `--arch N1..N5` (réseau plein), `--theta0`, `--n_bench`, `--n_random`, `--sa_iters`, `--seed`, `--tag` (utiliser `q_table_<tag>.npy`), `--show` (afficher les figures).

Il faut que `q_table.npy` et `sarsa_config.npz` soient dans le même dossier ; la fenêtre de gain, l'intervalle HPBW et la tolérance de direction sont relus depuis la configuration de l'entraînement.

## 1. Les méthodes (fonctions de calcul)

| Fonction | Rôle |
|:--|:--|
| `sarsa_solve(rs, Qm=None)` | **Méthode principale.** Lance un épisode avec la politique gloutonne (ε = 0, sans apprentissage, égalités = première action) et `tabu=True` (pas de retour sur une configuration déjà visitée). Renvoie le SSL et la liste d'antennes éteintes de la **meilleure configuration admissible** rencontrée. Avec `Qm = Q0` (table de zéros), on obtient la « table non entraînée ». |
| `fixed_policy(rs, prefer)` | Glouton sans apprentissage : applique toujours le même critère (liste `prefer`) jusqu'à 18 antennes, garde la meilleure configuration admissible. Sert à `glouton SSL` (`OFF_SSL`) et `glouton HPBW-sûr` (`OFF_SAFE`, puis `OFF_HPBW`). |
| `random_policy(rs, n_try, rng)` | Politique aléatoire : `n_try` épisodes (30) avec actions valides tirées au hasard ; garde le meilleur résultat. |
| `annealing(rs, rng, iters)` | **Recuit simulé** de référence (2500 itérations) : part de 20 antennes actives au hasard, échange / retire / ajoute des antennes, accepte une dégradation avec une probabilité qui décroît avec la « température ». La fonction coût est `SSL + 60 × violation du HPBW`. Il évalue le diagramme des milliers de fois : c'est une référence, **pas l'optimum**. |

## 2. Mode « un réseau » : le résultat

1. Construit `Reseau(arch, theta0, spec)`. Si le réseau a moins de 18 antennes, le programme s'arrête (le gain minimal est impossible).
2. Appelle `sarsa_solve`. Si aucune configuration admissible n'est trouvée, il le dit et s'arrête.
3. `report(...)` **affiche** :
   - l'ancien réseau (plein) et le nouveau (ex. `[8,6,10,3,7]` → `[5,5,4,2,3]`) ;
   - SSL, gain, HPBW et direction du pic, avant → après, avec les bornes imposées ;
   - le **tableau de toutes les antennes de l'ancien réseau** : anneau, numéro, angle (°), rayon (λ), x, y (λ), état avant (ON), état après (ON/OFF), action (DÉSACTIVER ou `-`) ;
   - le résumé : liste des antennes à désactiver (anneau, numéro, **angle**) et nombre d'antennes par anneau avant → après.
   - Remarque : le réseau part plein, donc toute antenne est active au départ. Le programme n'a ainsi aucune antenne « à activer » à indiquer ; les réactivations ne servent que pendant la recherche.
4. `plot_one(...)` **écrit** :
   - `avant_apres_reseau.png` : positions des antennes (bleu = active, croix rouge = désactivée) avant / après ;
   - `diagrammes_avant_apres.png` : diagramme de rayonnement avant / après (complet, puis zoom autour du lobe principal) ;
   - `resultat_reseau.csv` : une ligne de métriques avant / après ;
   - `positions_antennes.csv` : une ligne par antenne (anneau, numéro, angle, rayon, x, y, état avant / après, action).

## 3. Mode « benchmark » (sans `--arch`)

Tire `n_bench` réseaux aléatoires (même famille que l'entraînement), applique **6 méthodes** : SARSA, table non entraînée, glouton SSL, glouton HPBW-sûr, aléatoire, recuit simulé. Pour chacune, il affiche le nombre de réseaux admissibles et la baisse du SSL (moyenne, médiane, max), et écrit `resultats_benchmark.csv` et `benchmark_SARSA.png` (baisse moyenne du SSL et % de réseaux admissibles).

## 4. Comment lire les résultats

- Une configuration est **admissible** si le gain est dans [25 ; 27] dB, le HPBW dans `H_ref ± 10 %` et l'écart de direction ≤ 1°.
- **Baisse du SSL** = SSL du réseau plein − SSL de la configuration finale (en dB, plus grand = meilleur).
- Pour comparer les méthodes équitablement, comparer la baisse sur les réseaux où elles réussissent toutes.
- Résultat mesuré : SARSA ≈ table non entraînée ≈ glouton HPBW-sûr ; le recuit simulé descend en moyenne plus bas ; la fiabilité baisse sur les très grands réseaux.
