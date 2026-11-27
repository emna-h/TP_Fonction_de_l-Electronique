# Explication de `training.py`

`training.py` entraîne un agent **SARSA tabulaire** qui apprend à désactiver / réactiver des antennes d'un réseau circulaire à 5 anneaux afin de **minimiser le SSL** (niveau du plus haut lobe secondaire), sous contraintes. **Il n'utilise aucun jeu de données** : il apprend en jouant dans un simulateur, avec un réseau plein tiré au hasard à chaque épisode.

Exécution : `python training.py` (≈ 9 min, 15 000 épisodes). Sorties : `q_table.npy/.mat`, `visites_Q.npy`, `historique_training.npy`, `sarsa_config.npz`, `courbes_SARSA.png`.

## 1. Cahier des charges (constante `DEFAULT_SPEC`)

| Élément | Valeur | Traduction |
|:--|:--|:--|
| Gain | 25 à 27 dB | `G = 20·log10(N_actives)` donc **18 ≤ N_actives ≤ 22** (hypothèse à confirmer) |
| HPBW | `H_ref ± 10 %` | `H_ref` = HPBW du réseau plein |
| Direction | `|θ_pic − θ0| ≤ 1°` | le lobe principal ne doit pas bouger |
| Objectif | SSL le plus bas possible | |

## 2. Constantes et actions (haut du fichier)

- Physique : fréquence 2,45 GHz, `R0 = 0,2λ` (rayon de l'anneau 1), `DR = 0,5λ` (pas entre anneaux), 1000 angles `THETA`.
- **6 actions**, chacune étant un *critère de choix* de l'antenne (et non une antenne précise) : `OFF_SSL`, `OFF_SAFE`, `OFF_HPBW`, `ON_SSL`, `ON_HPBW`, `STOP`.
- `N_OBS = 6·4·5·3 = 360` observations possibles, donc une table Q de taille 360 × 6.

## 3. Modèle physique (section 1 du fichier)

| Fonction | Rôle |
|:--|:--|
| `layout(arch)` | place les antennes : anneau `m` de rayon `R0 + m·DR`, `N` éléments régulièrement espacés (angle `2πn/N`). Retourne rayons, angles, numéro d'anneau. |
| `contributions(a, phi, theta0)` | matrice `E` (1000 × N) : contribution complexe de chaque antenne dans chaque direction, avec la phase qui pointe vers `θ0`. |
| `metrics_from_AF(AF)` | à partir du facteur de réseau (somme des colonnes de `E`) calcule **gain, SSL, HPBW, direction du pic**. Normalise en dB (plancher −40 dB), HPBW à −3 dB, SSL = plus haut pic hors lobes principaux (les pics à moins de 1 dB du maximum comptent comme principaux). |
| `delta_theta(pic, theta0)` | écart de direction, en tenant compte des deux lobes symétriques `θ0` et `180° − θ0`. |

**Idée clé :** désactiver une antenne = retirer sa colonne de `E`. Le diagramme d'une configuration se calcule donc très vite (≈ 0,15 ms).

## 4. Classe `Reseau` (réseau + contraintes)

Construit un réseau plein et précalcule `E`, `AF_full` et ses métriques de référence `ref = (G, SSL, HPBW, pic)`. Il fixe aussi les bornes : `h_lo/h_hi` (intervalle HPBW) et `n_lo = 18`, `n_hi = 22`.

- `metrics(removed)` : métriques de la configuration où l'ensemble `removed` est éteint. **Mémoïsé** (cache indexé par `frozenset`) car les mêmes configurations reviennent souvent.
- `hpbw_ok(m)` : HPBW dans l'intervalle ?
- `admissible(m)` : gain dans la fenêtre **et** HPBW dans l'intervalle **et** direction respectée.
- `feasible_start()` : le réseau plein a-t-il au moins 18 antennes ?

## 5. Classe `Env` (l'environnement)

Un épisode : on part du réseau plein (`removed = ∅`) et on modifie une antenne par pas, jusqu'à `STOP`.

- `max_steps = max(12, N − 22 + 10)` : assez de pas pour retirer l'excédent d'antennes puis ajuster.
- `_refresh()` : après chaque changement, recalcule les métriques courantes, mémorise la **meilleure configuration admissible** vue (`best_ssl`, `best_removed`), évalue **tous** les retraits et réactivations possibles (un pas d'anticipation) et construit le dictionnaire `choice` : pour chaque action, l'antenne qu'elle désignerait.
  - `OFF_SSL` : le retrait qui donne le SSL le plus bas.
  - `OFF_SAFE` : idem, mais seulement parmi les retraits dont le HPBW reste dans l'intervalle.
  - `OFF_HPBW` : le retrait qui rapproche le plus le HPBW de `H_ref`.
  - `ON_SSL`, `ON_HPBW` : mêmes critères pour **réactiver** une antenne éteinte.
- `observation()` : résume l'état en **un entier de 0 à 359**, indépendant du réseau, à partir de quatre quantités discrétisées :
  1. `lv` (6 niveaux) : combien d'antennes il reste à retirer pour atteindre 22 (ou position dans la fenêtre 18–22) ;
  2. `hs` (4) : statut du HPBW (sous l'intervalle, entre `0,9·H_ref` et `H_ref`, entre `H_ref` et `1,1·H_ref`, au-dessus) ;
  3. `qc` (5) : baisse de SSL déjà obtenue (≤ 0, < 3, < 6, < 9, ≥ 9 dB) ;
  4. `bd` (3) : baisse offerte par le meilleur prochain retrait (≤ 0, < 1, ≥ 1 dB).
- `valid_actions()` : actions autorisées. Un retrait n'existe que si `N−1 ≥ 18` ; une réactivation que si une antenne est éteinte et `N+1 ≤ 22` ; **`STOP` seulement si `N ≤ 22`** (le gain est donc une contrainte *dure*). Avec `tabu=True` (utilisé au test), on interdit de revenir sur une configuration déjà visitée, pour éviter les cycles.
- `step(action)` : applique l'action et renvoie `(récompense, terminé, cause)`.
  - Récompense d'un changement = **baisse de SSL** `SSL_avant − SSL_après` (en dB) moins `step_cost` (0 par défaut).
  - `STOP` : 0 si la configuration est admissible, sinon **−penalty** (−5).
  - Si `max_steps` est atteint : fin avec la même pénalité si non admissible.
  - La somme des récompenses vaut donc `SSL_ref − SSL_final − pénalité`.

HPBW et direction sont des contraintes **douces** (pénalité finale) ; le gain est une contrainte **dure** (actions invalides).

## 6. Agent SARSA (section 4)

- `choose_action(Q, o, valid, eps, rng)` : politique **ε-gloutonne**. Avec probabilité `eps`, action valide au hasard ; sinon action de plus grande valeur `Q[o, a]` (égalités tirées au hasard, ou première action avec `tie='first'`).
- `run_episode(...)` : cœur de SARSA.
  ```text
  o ← observation ; a ← choose_action
  répéter :
      (r, fini) ← step(a)
      si fini : cible = r
      sinon   : o' ← observation ; a' ← choose_action ; cible = r + γ·Q[o', a']
      Q[o, a] ← Q[o, a] + α·(cible − Q[o, a])        # mise à jour SARSA
      (o, a) ← (o', a')
  ```
  C'est « on-policy » : la cible utilise l'action `a'` **réellement choisie** (exploration comprise), pas le maximum. La fonction renvoie le retour, la meilleure configuration admissible (`removed_best`), la liste d'actions et la cause de fin.
- `random_arch`, `random_reseaux` : tirent des architectures (1 à 10 antennes par anneau, au moins 18 au total) et des `θ0` dans [5°, 175°].
- `evaluate_greedy` : évalue la politique **gloutonne** (ε = 0, sans apprentissage) sur des réseaux fixes, pour la validation.

## 7. `main()` : la boucle d'entraînement

1. **Paramètres** (`argparse`) : `--episodes 15000`, `--alpha 0.1`, `--gamma 1.0`, `--eps_start 1.0`, `--eps_min 0.02`, `--penalty 5`, `--step_cost 0`, `--seed 46`, `--n_val 30`, `--eval_every 500`, fenêtre de gain `--g_lo/--g_hi`, `--h_rel`, `--eps_theta`, famille de réseaux `--n_min/--n_max/--n_tot_min`, `--tag` (suffixe des fichiers).
2. **ε décroissant** : `eps = max(eps_min, eps_start · eps_decay^ep)`, de 1 à 0,02 en 60 % des épisodes (exploration d'abord, exploitation ensuite).
3. **Validation** : 30 réseaux fixes (graine `seed+1`).
4. **Pour chaque épisode** : tirer un réseau plein au hasard, créer l'`Env`, lancer `run_episode`, enregistrer le retour, l'admissibilité et la baisse de SSL.
5. **Tous les 500 épisodes** : évaluer la politique gloutonne sur la validation et afficher la progression.
6. **À la fin** : sauvegarder la table Q (`.npy` et `.mat`), les visites par couple (observation, action), l'historique, la configuration (`sarsa_config.npz`, relue par `test.py`) et tracer `courbes_SARSA.png` (épisodes admissibles, baisse de SSL en validation, % de réseaux admissibles).

## 8. Points à retenir (et limites)

- La table Q est **indépendante du réseau** : c'est ce qui permet d'accepter n'importe quel réseau plein après l'entraînement.
- Les actions sont des **critères** évalués par anticipation d'un pas : une partie de la qualité vient du simulateur, pas de la table.
- Résultat mesuré : la table entraînée est à peine meilleure qu'une table vide (`Q = 0`), et le recuit simulé fait en moyenne mieux (voir le rapport, chapitres 9 et 10).
