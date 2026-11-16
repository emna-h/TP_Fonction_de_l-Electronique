# Explication ligne par ligne de `test.py`

**Rôle du script (étape 4/4)** : évaluer la politique SARSA apprise sur des configurations **qu'elle n'a jamais vues**, la comparer à d'autres méthodes, et enregistrer les résultats.

## L'idée en quelques phrases

Pendant l'entraînement, l'agent a exploré au hasard : ses résultats d'entraînement ne prouvent rien sur la qualité de ce qu'il a appris. Ici :
- **aucune mise à jour** de la table Q (le script vérifie que Q est strictement inchangée) ;
- l'agent joue **toujours son meilleur choix** (politique gloutonne, ε = 0) ;
- on mesure sur les **100 configurations de test**, jamais utilisées pendant l'entraînement.

On présente toujours **séparément** trois choses qu'il ne faut pas confondre :

| Ce qu'on mesure | Statut |
|---|---|
| Meilleure configuration admissible rencontrée **pendant** l'entraînement (avec exploration) | une trouvaille, pas une preuve d'apprentissage |
| Politique finale sur les configurations d'**entraînement** | mesure de mémorisation |
| Politique finale sur le **test** | seule base pour parler de généralisation |

## Les méthodes de comparaison

| Méthode | Principe | A-t-elle un « oracle » ? |
|---|---|---|
| Recherche exhaustive | Essaie **toutes** les façons de retirer r éléments et garde la meilleure : donne l'**optimum exact** | oui (calcule tout) |
| Heuristique gloutonne | À chaque pas, retire l'élément qui laisse le plus de marge, tant que c'est admissible | oui (teste chaque candidat) |
| Suppression aléatoire admissible | Retire au hasard un élément parmi ceux dont la suppression reste admissible | oui (teste chaque candidat) |
| Politique aléatoire sans oracle | Même environnement que SARSA, actions tirées au hasard (STOP compris) | non |
| SARSA | Applique la table Q apprise | non : il ne teste pas les candidats avant d'agir |

« Oracle » veut dire que la méthode peut **tester** la suppression avant de la décider, ce que SARSA ne fait pas.

---

## 1. Imports

```python
import argparse
import csv
import itertools
import math
import os

import numpy as np
import scipy.io

import training as T
```
- `argparse` : options en ligne de commande ; `csv` : écrire un fichier CSV (tableur) ;
- `itertools` : générer toutes les combinaisons d'éléments (pour la recherche exhaustive) ;
- `math` : fonctions mathématiques (on utilise `math.comb`, le nombre de combinaisons) ;
- `os`, `numpy`, `scipy.io` : comme dans les autres scripts ;
- `import training as T` : on **réutilise** les fonctions de `training.py` (modèle physique, environnement, scénarios) en les appelant `T.nom_de_la_fonction`. C'est possible car `training.py` ne lance l'entraînement que s'il est exécuté directement.

## 2. Bannière

```python
def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")
```
Affiche la séparation « NOUVELLE EXÉCUTION ».

## 3. Contrôle du modèle physique

`training.py` calcule le diagramme avec un **raccourci** (somme de colonnes précalculées) pour aller vite. Pour être sûr qu'il ne donne pas des résultats différents du calcul d'origine, on le compare à un calcul « à la main », lent mais identique à `dataset.py`.

```python
def direct_metrics(arch, theta0_deg, removed):
    """Calcul 'à la main', anneau par anneau, exactement comme le dataset.py, en sautant les éléments retirés."""
    th0 = np.deg2rad(theta0_deg)
    AF = np.zeros(len(T.THETA), dtype=complex)
    k = 0
    for m, N in enumerate(arch):
        for n in range(int(N)):
            if k not in removed:
                a = T.R0 + T.DR * m
                phi_n = 2 * np.pi * n / N
                AF += np.exp(1j * T.K_WAVE * a * (np.sin(T.THETA) * np.cos(0 - phi_n) - np.sin(th0) * np.cos(0 - phi_n)))
            k += 1
    return T.metrics_from_AF(AF)
```
`direct_metrics` calcule les métriques en additionnant, élément par élément, les ondes des éléments **non retirés** :
- `AF` : facteur de réseau, initialisé à zéro ;
- `k` : compteur du numéro d'élément (0, 1, 2, ...) dans l'ordre anneau par anneau ;
- pour chaque anneau `m` (avec `N` éléments) et chaque élément `n` : si `k` n'est pas dans `removed` (l'ensemble des éléments retirés), on ajoute son onde (même formule que `dataset.py`) ;
- `k += 1` passe à l'élément suivant (même quand l'élément est retiré, pour garder la numérotation) ;
- on renvoie les métriques via `T.metrics_from_AF`.

```python
def verif_physique(Mi, Mo, tol, rng, n=40):
    err = 0.0
    for _ in range(n):
        j = rng.integers(Mi.shape[1])
        sc = T.Scenario(Mo[:, j], Mi[3, j], tol)
        r = int(rng.integers(0, min(4, sc.n_ref - 1) + 1))
        rem = tuple(sorted(rng.choice(sc.n_ref, r, replace=False).tolist()))
        a = np.array(sc.metrics(rem)[:3])
        b = np.array(direct_metrics(sc.arch, sc.theta0, set(rem))[:3])
        err = max(err, np.max(np.abs(a - b)))
    return err
```
`verif_physique` tire 40 cas au hasard et mesure le plus grand écart entre les deux méthodes :
- `err` : plus grand écart trouvé, initialisé à 0 ;
- `j` : indice d'une configuration tirée au hasard ; `sc` : le scénario ; `r` : nombre d'éléments à retirer (entre 0 et 4) ;
- `rem` : `r` éléments retirés au hasard sans répétition (`rng.choice(..., replace=False)`), triés dans un n-uplet ;
- `a` : métriques par le calcul rapide (gain, SSL, HPBW) ; `b` : par le calcul direct ;
- `err = max(err, plus grand |a − b|)` ; on renvoie `err`. Dans nos essais : 7·10⁻¹⁵, c'est-à-dire des arrondis de l'ordinateur.

## 4. Méthodes de référence

```python
def admissible(sc, removed):
    return sc.u_max(tuple(sorted(removed))) <= 1.0
```
`admissible` : vrai si la configuration où l'on retire les éléments `removed` respecte toutes les tolérances (`u_max ≤ 1`).

```python
def exhaustive(sc, norm, cap):
    """Optimum exact : plus grand r admissible, puis plus petite erreur globale. None si > cap évaluations."""
    count = 0
    for r in range(sc.r_max, -1, -1):
        if count + math.comb(sc.n_ref, r) > cap:      # niveau trop grand : abandon immédiat
            return None
```
**Recherche exhaustive** (`exhaustive`) : trouve l'optimum exact.
- `count` : nombre de configurations déjà évaluées ;
- on essaie d'abord le **plus grand** nombre de suppressions possible (`r_max`), puis on diminue (`range(r_max, -1, -1)`) ;
- `math.comb(n, r)` : nombre de façons de choisir `r` éléments parmi `n`. Si le total dépasse la limite `cap`, on **abandonne** tout de suite (`return None`) : l'optimum n'est alors pas calculé (trop long).

```python
        best, best_e = None, np.inf
        for comb in itertools.combinations(range(sc.n_ref), r):
            count += 1
            if count > cap:
                return None
            if admissible(sc, comb):
                e = T.error_global(sc, comb, norm)
                if e < best_e:
                    best, best_e = comb, e
        if best is not None:
            return r, best
    return 0, ()
```
- `best`, `best_e` : meilleure configuration trouvée et son erreur (infinie au départ) ;
- `itertools.combinations(range(n), r)` génère **toutes** les combinaisons de `r` éléments à retirer ;
- pour chacune : on compte, on vérifie la limite, puis, **si elle est admissible**, on calcule l'erreur globale pondérée ;
- si cette erreur est plus petite que la meilleure, on la garde : parmi les solutions de même nombre d'éléments, on départage donc par l'erreur globale ;
- dès qu'un niveau `r` contient au moins une solution admissible, on s'arrête : c'est le plus grand `r` possible, donc l'optimum, et on renvoie `(r, configuration)` ;
- si aucun `r ≥ 1` ne convient, on renvoie `(0, ())` : le réseau complet.

```python
def greedy_heuristic(sc):
    """À chaque pas, retire l'élément qui laisse la plus grande marge (plus petit u_max), tant que c'est admissible."""
    removed = []
    while len(removed) < sc.n_ref - 1:
        cand = [(sc.u_max(tuple(sorted(removed + [e]))), e) for e in range(sc.n_ref) if e not in removed]
        cand = [c for c in cand if c[0] <= 1.0]
        if not cand:
            break
        removed.append(min(cand)[1])
    return len(removed)
```
**Heuristique gloutonne** (`greedy_heuristic`) :
- `removed` : éléments retirés ; la boucle continue tant qu'il reste plus d'un élément ;
- `cand` : liste de couples (pire écart normalisé, élément) pour **chaque** élément encore actif si on le retirait ;
- on ne garde que les candidats admissibles (`u ≤ 1`) ; s'il n'y en a aucun, `break` ;
- sinon on retire celui dont le pire écart est le plus petit (`min` sur les couples compare d'abord le premier nombre) ;
- on renvoie le nombre d'éléments retirés.

```python
def random_admissible(sc, rng):
    removed = []
    while len(removed) < sc.n_ref - 1:
        cand = [e for e in range(sc.n_ref) if e not in removed and admissible(sc, removed + [e])]
        if not cand:
            break
        removed.append(int(rng.choice(cand)))
    return len(removed)
```
**Suppression aléatoire admissible** (`random_admissible`) : même boucle, mais parmi les candidats admissibles on en tire un **au hasard** (`rng.choice`). Elle représente « ce qu'on obtient avec la contrainte vérifiée mais sans aucune intelligence ».

```python
def random_policy(sc, penalty, rng):
    """Politique aléatoire sans oracle : même environnement, action valide tirée au hasard (STOP compris)."""
    env = T.AntennaEnv(sc, penalty)
    env.reset()
    while True:
        a = int(rng.choice(env.valid_actions()))
        _, done, end = env.step(a)
        if done:
            return len(env.last_admissible), end == 'violation'
```
**Politique aléatoire sans oracle** (`random_policy`) : on joue dans le **même environnement** que SARSA, mais chaque action valide (STOP compris) est tirée au hasard. On renvoie le nombre d'éléments éteints et un booléen « s'est terminé par une violation ». Elle montre ce qu'on obtient **sans aucun apprentissage** et sans pouvoir tester à l'avance.

## 5. Programme principal : préparation

```python
def main():
    clear_console()
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--cap_exh', type=int, default=100000, help="nombre maximal d'évaluations de la recherche exhaustive")
    p.add_argument('--n_random', type=int, default=20)
    args = p.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''
```
- options : `--tag` (pour tester une table entraînée avec `--tag`), `--cap_exh` (limite de la recherche exhaustive, 100 000 évaluations), `--n_random` (nombre de tirages pour les méthodes aléatoires : 20) ;
- `suffix` : suffixe des fichiers, comme dans `training.py`.

```python
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ld = lambda f: np.load(os.path.join(script_dir, f))
```
- `script_dir` : dossier du script ;
- `ld` : petite fonction raccourci (`lambda`) qui charge un fichier `.npy` du dossier du script.

```python
    cfg = np.load(os.path.join(script_dir, f'sarsa_config{suffix}.npz'))
    tol = dict(eps_G=float(cfg['eps_G']), eps_S=float(cfg['eps_S']), eps_H_rel=float(cfg['eps_H_rel']),
               eps_theta=float(cfg['eps_theta']))
    penalty = float(cfg['penalty'])
    Q = ld(f'q_table{suffix}.npy')
    hist = np.load(os.path.join(script_dir, f'historique_training{suffix}.npy'), allow_pickle=True).item()
```
Chargement de tout ce qu'a produit l'entraînement :
- `cfg` : fichier de configuration (tolérances, pénalité...), pour tester avec **exactement** les mêmes réglages ;
- `tol` : dictionnaire des tolérances (les `float(...)` convertissent les valeurs lues) ; `penalty` ;
- `Q` : la table Q apprise ;
- `hist` : l'historique d'entraînement (`.item()` extrait le dictionnaire stocké dans le fichier).

```python
    Mi_all = ld('Minput.npy')
    norm = (np.max(np.abs(Mi_all[0])), np.max(np.abs(Mi_all[1])), 180.0)
    rng = np.random.default_rng(46)
```
- `Mi_all` : le `Minput` complet ; `norm` : valeurs maximales du gain et du SSL du jeu de données, et 180 pour le HPBW : ce sont les normalisations de l'erreur relative utilisées dans les tests du FFNN et du cGAN ;
- `rng` : générateur aléatoire à graine 46.

```python
    Mi_te, Mo_te = ld('Minput_test.npy'), ld('Moutput_test.npy')
    Mi_tr, Mo_tr = ld('Minput_training.npy'), ld('Moutput_training.npy')
    print(f"Tolérances : {tol} ; pénalité {penalty}")
```
Chargement des matrices de **test** et d'**entraînement**, puis affichage des réglages.

```python
    e = verif_physique(Mi_te, Mo_te, tol, rng)
    print(f"Contrôle physique : écart max (gain, SSL, HPBW) entre calcul rapide et calcul direct = {e:.2e}")
    assert e < 1e-8
```
Lance le contrôle du modèle physique (section 3) ; si l'écart dépasse 10⁻⁸, le script s'arrête.

## 6. Évaluation sur le test (sans apprentissage)

```python
    test = T.build_scenarios(Mi_te, Mo_te, tol)
    Q_avant = Q.copy()
    res = T.evaluate_greedy(test, Q, penalty)
    assert np.array_equal(Q, Q_avant), "la table Q ne doit pas changer au test"
```
- `test` : liste de 100 `Scenario` ;
- `Q_avant = Q.copy()` : copie de sécurité de la table Q ;
- `res` : résultats de la politique gloutonne sur chaque scénario ;
- `assert np.array_equal(Q, Q_avant)` : **preuve** que la table Q n'a pas changé pendant le test.

```python
    rows, activations, pos_rows = [], [], []
    for j, (sc, rs) in enumerate(zip(test, res)):
        cfgs = rs['config']
        act = np.ones(sc.n_ref, dtype=int)
        act[list(cfgs)] = 0
        activations.append(act)
        g, s, h, pk = sc.metrics(tuple(cfgs))
        u = sc.deviations(tuple(cfgs))
        ex = exhaustive(sc, norm, args.cap_exh)
```
Boucle sur les 100 configurations de test :
- `rows` : lignes du tableau de résultats ; `activations` : vecteurs d'activation (1 = allumé, 0 = éteint) ;
- `cfgs` : éléments éteints par SARSA ; `act` : vecteur d'activation correspondant ;
- `g, s, h, pk` : métriques de la configuration produite ; `u` : écarts normalisés ;
- `ex` : résultat de la recherche exhaustive (ou `None` si trop longue).

```python
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(test)} configurations traitées (la recherche exhaustive est la partie lente)", flush=True)
```
Toutes les 10 configurations, on affiche une ligne d'avancement normale (la recherche exhaustive est la partie lente). `flush=True` force l'affichage immédiat. (Une ancienne version écrasait la ligne avec `\r`, ce qui ne s'affichait pas dans certaines consoles.)

```python
        rows.append(dict(
            config=j, N_ref=sc.n_ref, theta0=sc.theta0, r_max=sc.r_max,
            r_SARSA=rs['r'], pct_desactive=100 * rs['r'] / sc.n_ref, N_actives=sc.n_ref - rs['r'],
            fin_episode=rs['end'], tentative_violation=int(rs['end'] == 'violation'),
            G_ref=sc.ref[0], G=g, SSL_ref=sc.ref[1], SSL=s, HPBW_ref=sc.ref[2], HPBW=h,
            dG=g - sc.ref[0], dSSL=s - sc.ref[1], dHPBW=h - sc.ref[2], dtheta=T.delta_theta(pk, sc.theta0),
            admissible=int(max(u) <= 1.0), erreur_globale_pct=T.error_global(sc, cfgs, norm),
            r_optimum=np.nan if ex is None else ex[0],
            r_glouton=greedy_heuristic(sc),
            r_aleatoire_admissible=np.mean([random_admissible(sc, rng) for _ in range(args.n_random)]),
        ))
```
`rows.append(dict(...))` ajoute une ligne de résultats par configuration :
- identité : numéro `config`, `N_ref` (nombre d'éléments du réseau complet), `theta0`, `r_max` ;
- résultat de SARSA : `r_SARSA` (nombre d'éléments éteints), `pct_desactive` (en % de `N_ref`), `N_actives`, `fin_episode` (`stop` ou `violation`) et `tentative_violation` (1 ou 0) ;
- performances : référence (`G_ref`, `SSL_ref`, `HPBW_ref`), obtenues (`G`, `SSL`, `HPBW`) et écarts (`dG`, `dSSL`, `dHPBW`, `dtheta`) ;
- `admissible` : 1 si tous les écarts normalisés sont ≤ 1 ; `erreur_globale_pct` : erreur globale pondérée ;
- méthodes de comparaison : `r_optimum` (`nan` = non calculé car trop long), `r_glouton`, `r_aleatoire_admissible` (moyenne de 20 tirages).

## 7. Quelles antennes éteindre ? Colonnes lisibles

Ces lignes répondent à la question « **quelles antennes** dois-je désactiver ? ». Chaque élément est repéré par son **anneau**, son **numéro dans l'anneau** et son **angle**.

```python
        # --- lisible : quelles antennes sont éteintes (anneau n°, position n° dans l'anneau, angle) ---
        off = []
        par_anneau = [0] * T.N_RINGS
        for e_ in sorted(cfgs):
            m_ = int(sc.ring[e_])
            pos = e_ - int(np.where(sc.ring == m_)[0][0])            # 0, 1, 2... dans l'anneau
            off.append(f"anneau{m_ + 1}#{pos + 1}({360 * pos / sc.arch[m_]:.0f}deg)")
            par_anneau[m_] += 1
```
- `off` : liste de textes décrivant chaque antenne éteinte ; `par_anneau` : compteur d'éléments éteints dans chacun des 5 anneaux ;
- `sorted(cfgs)` : on parcourt les éléments éteints dans l'ordre ; `e_` est le numéro **global** de l'élément (0, 1, 2... anneau par anneau) ;
- `m_` : numéro de son anneau (0 à 4) ; `pos` : sa position dans l'anneau = numéro global moins le numéro du premier élément de cet anneau (`np.where(sc.ring == m_)[0][0]`) ;
- le texte écrit par exemple `anneau1#3(90deg)` : 3e antenne de l'anneau 1, située à 90° (`360 × pos / nombre d'éléments de l'anneau`, car les éléments d'un anneau sont régulièrement espacés) ; `{:.0f}` arrondit à l'unité ;
- `par_anneau[m_] += 1` : on compte une antenne éteinte de plus dans cet anneau.

```python
        rows[-1]['antennes_eteintes'] = ' '.join(off) if off else 'aucune'
        rows[-1]['eteintes_par_anneau'] = '-'.join(str(x) for x in par_anneau)
        rows[-1]['architecture_ref'] = '-'.join(str(int(x)) for x in sc.arch)
        rows[-1]['architecture_finale'] = '-'.join(str(int(x) - c_) for x, c_ in zip(sc.arch, par_anneau))
```
On enregistre dans la dernière ligne du tableau (`rows[-1]`) :
- `antennes_eteintes` : le texte ci-dessus, ou `aucune` ;
- `eteintes_par_anneau` : par exemple `3-0-0-0-0` (3 éteintes dans l'anneau 1, aucune ailleurs) ;
- `architecture_ref` et `architecture_finale` : nombre d'éléments par anneau avant et après (par exemple `8-4-10-0-9` → `5-4-10-0-9`), la seconde étant la première moins les éteintes de chaque anneau (`zip` parcourt deux listes ensemble).

## 8. Positions de toutes les antennes

```python
        # --- positions de toutes les antennes de la configuration (les antennes ne bougent pas) ---
        for e_ in range(sc.n_ref):
            m_ = int(sc.ring[e_])
            pos = e_ - int(np.where(sc.ring == m_)[0][0])
            ang = float(np.rad2deg(sc.phi[e_]))
            pos_rows.append([j, e_ + 1, m_ + 1, pos + 1, sc.a[e_], sc.a[e_] / T.LAMBDA, ang,
                             sc.a[e_] * np.cos(sc.phi[e_]), sc.a[e_] * np.sin(sc.phi[e_]), int(act[e_])])
```
Pour chaque élément du réseau (`for e_ in range(sc.n_ref)`), on ajoute une ligne à `pos_rows` :
- `m_` et `pos` : anneau et position dans l'anneau (comme ci-dessus) ; `ang` : son angle en degrés (`np.rad2deg` convertit les radians) ;
- la ligne contient : numéro de configuration, numéro d'antenne (à partir de 1), anneau, numéro dans l'anneau, rayon en mètres, rayon en longueurs d'onde (`/ T.LAMBDA`), angle, coordonnées cartésiennes `x = a cos(φ)` et `y = a sin(φ)`, et `actif` (1 allumée, 0 éteinte).
Ici **les antennes ne bougent pas** : on ne fait qu'en éteindre certaines. Pour les **déplacer**, il faut le repositionnement (scripts `redistribution.py` ou `training_repositionnement.py`).

```python
        rp = [random_policy(sc, penalty, rng) for _ in range(args.n_random)]
        rows[-1]['r_politique_aleatoire'] = np.mean([x[0] for x in rp])
        rows[-1]['viol_politique_aleatoire'] = np.mean([x[1] for x in rp])
```
- `rp` : 20 parties de la politique aléatoire sans oracle ; `x[0]` est le nombre d'éléments éteints et `x[1]` indique si la partie s'est terminée par une violation ;
- on enregistre la moyenne du nombre d'éléments éteints et la part d'épisodes terminés par une violation.

```python
    TEXTE = ('fin_episode', 'antennes_eteintes', 'eteintes_par_anneau', 'architecture_ref', 'architecture_finale')
    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k not in TEXTE}
    R_end = [r['fin_episode'] for r in rows]
```
- `TEXTE` : colonnes qui contiennent du texte (non convertibles en nombres) ;
- `R` : on convertit chaque colonne numérique du tableau en vecteur NumPy ;
- `R_end` : liste des raisons de fin d'épisode.

## 9. Résumé affiché

```python
    use = R['r_max'] >= 1
    print(f"\n=== TEST : {len(test)} configurations jamais vues ({int(use.sum())} avec r_max >= 1) ===")
    print(f"Configurations finales admissibles : {int(R['admissible'].sum())} / {len(test)} "
          f"(par construction : retour à la dernière configuration admissible en cas de violation)")
    print(f"Épisodes terminés par STOP (la politique s'arrête d'elle-même) : {R_end.count('stop')}, "
          f"par tentative de violation (annulée) : {R_end.count('violation')}, plus d'action possible : {R_end.count('plus_daction')}")
    print(f"Antennes désactivées : moyenne {R['r_SARSA'].mean():.2f} (soit {R['pct_desactive'].mean():.2f} % de N_ref), "
          f"borne r_max moyen {R['r_max'].mean():.2f}")
    print(f"Retour relatif r / r_max (r_max >= 1) : {np.mean(R['r_SARSA'][use] / R['r_max'][use]):.3f}")
```
- `use` : masque des configurations avec `r_max ≥ 1` (les seules où il y a une décision) ;
- on affiche : combien de configurations finales sont admissibles (toutes, par construction : une violation est annulée), combien d'épisodes s'arrêtent par STOP et combien par tentative de violation, le nombre moyen d'antennes éteintes, la borne `r_max` moyenne, et le rapport moyen `r / r_max`.

```python
    ok = ~np.isnan(R['r_optimum'])
    print(f"Optimum exact (recherche exhaustive, <= {args.cap_exh} évaluations) disponible pour {int(ok.sum())} / {len(test)} configurations")
    if ok.any():
        gap = R['r_optimum'][ok] - R['r_SARSA'][ok]
        print(f"  SARSA atteint l'optimum : {int((gap == 0).sum())} / {int(ok.sum())} ; écart moyen à l'optimum {gap.mean():.3f} antenne(s) ; "
              f"réduction moyenne optimum {R['r_optimum'][ok].mean():.2f} vs SARSA {R['r_SARSA'][ok].mean():.2f}")
        print(f"  (SARSA ne dépasse jamais l'optimum : {bool(np.all(gap >= 0))})")
```
- `ok` : configurations pour lesquelles l'optimum exact est connu (`~` inverse un booléen) ;
- `gap = optimum − SARSA` : l'écart en nombre d'antennes (toujours ≥ 0 puisque l'optimum est le meilleur possible) ;
- on affiche la part de configurations où SARSA atteint l'optimum, l'écart moyen, les moyennes, et la vérification que SARSA ne dépasse jamais l'optimum (ce qui prouverait une erreur).

```python
    print("\nMéthodes de référence (r moyen sur le test) :")
    print(f"  SARSA (politique finale)            : {R['r_SARSA'].mean():.3f}")
    print(f"  Heuristique gloutonne (oracle)      : {np.nanmean(R['r_glouton']):.3f}")
    print(f"  Aléatoire admissible (oracle)       : {R['r_aleatoire_admissible'].mean():.3f}")
    print(f"  Politique aléatoire sans oracle     : {R['r_politique_aleatoire'].mean():.3f}  "
          f"(tentatives de violation : {100*R['viol_politique_aleatoire'].mean():.1f} % des épisodes)")
    if ok.any():
        print(f"  Optimum exact (sous-ensemble)       : {R['r_optimum'][ok].mean():.3f}")
```
Tableau de comparaison : nombre moyen d'antennes éteintes par SARSA, par l'heuristique gloutonne, par la suppression aléatoire admissible, par la politique aléatoire sans oracle (avec son taux de tentatives de violation), et par l'optimum exact (quand il est connu).

```python
    print("\nPerformances (test) : écarts moyens |dG| {:.3f} dB, |dSSL| {:.3f} dB, |dHPBW| {:.3f} deg, dtheta {:.3f} deg ; "
          "erreur globale pondérée {:.3f} %".format(np.abs(R['dG']).mean(), np.abs(R['dSSL']).mean(),
                                                      np.abs(R['dHPBW']).mean(), R['dtheta'].mean(), R['erreur_globale_pct'].mean()))
```
Écarts moyens aux performances de référence (valeurs absolues de ΔG, ΔSSL, ΔHPBW et écart de direction) et erreur globale pondérée moyenne, comme dans les scripts de test du FFNN et du cGAN.

```python
    print("\nExemples (5 premières configurations de test) : quelles antennes éteindre")
    for r_ in rows[:5]:
        print(f"  config {r_['config']} : N_ref = {r_['N_ref']}, architecture {r_['architecture_ref']} -> {r_['architecture_finale']} ; "
              f"éteindre {r_['r_SARSA']} antenne(s) : {r_['antennes_eteintes']}")
    print("  (liste complète dans resultats_test_SARSA.csv, colonnes antennes_eteintes et architecture_finale)")
    print("  Positions de toutes les antennes (actives : actif = 1) : positions_antennes_test_SARSA.csv")
```
Affiche, pour les 5 premières configurations, l'architecture avant → après et **la liste des antennes à éteindre** (colonne `antennes_eteintes`). La liste complète est dans `resultats_test_SARSA.csv` ; les positions de toutes les antennes sont dans `positions_antennes_test_SARSA.csv`.

## 10. Entraînement : trouvaille vs politique finale

```python
    train_all = T.build_scenarios(Mi_tr, Mo_tr, tol)
    res_tr = T.evaluate_greedy(train_all, Q, penalty)
    r_final = np.array([x['r'] for x in res_tr])
    r_best = np.array(hist['meilleur_r_vu_en_training'])
    u2 = np.array([s.r_max >= 1 for s in train_all])
    print(f"\n=== TRAINING (configurations vues) ===")
    print(f"Meilleure configuration admissible rencontrée PENDANT l'entraînement (avec exploration) : r moyen {r_best[u2].mean():.3f}")
    print(f"Politique finale gloutonne sur les mêmes configurations                                  : r moyen {r_final[u2].mean():.3f}")
    print(f"r_max moyen sur ces configurations                                                        : {np.mean([s.r_max for s in train_all if s.r_max >= 1]):.3f}")
    print("(la première valeur est une trouvaille ; seule la deuxième mesure la politique apprise.)")
```
- on évalue la politique **finale** sur les configurations d'entraînement (sans apprendre) ;
- `r_final` : éléments éteints par la politique finale ; `r_best` : meilleure trouvaille pendant l'entraînement (lue dans l'historique) ; `u2` : configurations avec `r_max ≥ 1` ;
- on affiche les deux moyennes côte à côte, avec `r_max` moyen. La première est une trouvaille obtenue avec exploration ; **seule la deuxième** mesure ce que la politique a appris.

## 11. Enregistrement des résultats

```python
    keys = list(rows[0].keys())
    with open(os.path.join(script_dir, f'resultats_test_SARSA{suffix}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=';')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
```
- `keys` : noms des colonnes ;
- `open(..., 'w', newline='', encoding='utf-8')` ouvre le fichier CSV en écriture ; `with` le referme automatiquement ;
- `csv.DictWriter(..., delimiter=';')` : séparateur `;` (pratique pour Excel en français) ; `writeheader()` écrit la ligne de titres ;
- pour chaque ligne, les nombres décimaux sont arrondis à 6 chiffres.

```python
    # --- positions des antennes : nouveau fichier lisible (une ligne par antenne) ---
    # colonnes : config, antenne (numéro global), anneau, numéro dans l'anneau, rayon (m), rayon (en lambda),
    #            angle (deg), x (m), y (m), actif (1 = allumée, 0 = éteinte)
    noms_pos = ['config', 'antenne', 'anneau', 'num_dans_anneau', 'rayon_m', 'rayon_lambda', 'angle_deg', 'x_m', 'y_m', 'actif']
    with open(os.path.join(script_dir, f'positions_antennes_test_SARSA{suffix}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f, delimiter=';')
        w.writerow(noms_pos)
        for r_ in pos_rows:
            w.writerow([r_[0], r_[1], r_[2], r_[3]] + [round(float(v), 10) for v in r_[4:9]] + [r_[9]])   # 10 décimales : les rayons sont en mètres (~0,02 m)
    P = np.array(pos_rows, dtype=float)
    np.save(os.path.join(script_dir, f'positions_antennes_test_SARSA{suffix}.npy'), P)
    scipy.io.savemat(os.path.join(script_dir, f'positions_antennes_test_SARSA{suffix}.mat'), {'positions': P, 'colonnes': np.array(noms_pos, dtype=object)})
```
Fichier des **positions** (une ligne par antenne) :
- `noms_pos` : noms des 10 colonnes (config, antenne, anneau, num_dans_anneau, rayon_m, rayon_lambda, angle_deg, x_m, y_m, actif) ;
- écrit en CSV avec **10 décimales** : les rayons sont de l'ordre de 0,02 m, donc un arrondi à 6 décimales fausserait les positions ;
- `P` : le même tableau en matrice NumPy, enregistré en `.npy` et en `.mat` (MATLAB, avec les noms de colonnes).

```python
    nmax = int(Mo_te.sum(axis=0).max())
    act_mat = -np.ones((nmax, len(test)))                  # -1 = case vide (longueur variable)
    for j, a in enumerate(activations):
        act_mat[:len(a), j] = a
    out = {k: v for k, v in R.items()}
    out['activation'] = act_mat
    scipy.io.savemat(os.path.join(script_dir, f'resultats_test_SARSA{suffix}.mat'), out)
    np.save(os.path.join(script_dir, f'resultats_test_SARSA{suffix}.npy'), out, allow_pickle=True)
```
- `nmax` : plus grand nombre d'éléments d'une configuration de test ;
- `act_mat` : matrice `nmax × 100` remplie de −1 (case vide) ; chaque colonne reçoit le vecteur d'activation de la configuration (de longueur variable) : on complète donc par −1 pour avoir un tableau rectangulaire ;
- `out` : dictionnaire de tous les résultats plus la matrice d'activation, écrit en `.mat` (MATLAB) et en `.npy` (Python).

## 12. Figures

```python
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = ['SARSA', 'Glouton', 'Aléatoire\nadmissible', 'Aléatoire\nsans oracle']
    vals = [R['r_SARSA'].mean(), np.nanmean(R['r_glouton']), R['r_aleatoire_admissible'].mean(), R['r_politique_aleatoire'].mean()]
```
- `matplotlib.use('Agg')` : dessiner sans fenêtre ;
- `fig, ax = plt.subplots(1, 2, ...)` : une figure à deux graphiques ;
- `names`, `vals` : noms et valeurs des barres ; on ajoute l'optimum quand il est connu (`\n` passe à la ligne dans une étiquette).

```python
    if ok.any():
        names.append('Optimum\n(exhaustif)')
        vals.append(R['r_optimum'][ok].mean())
    ax[0].bar(names, vals); ax[0].set_ylabel("antennes désactivées (moyenne, test)")
    ax[0].axhline(R['r_max'].mean(), color='k', ls='--', label='borne r_max (gain)'); ax[0].legend(); ax[0].grid(alpha=0.3, axis='y')
    ax[0].set_title("Test : méthodes comparées")
```
Premier graphique : diagramme en barres des méthodes ; la ligne horizontale en pointillés est la borne `r_max` due au gain (`axhline`) ; `legend` affiche la légende, `grid` la grille.

```python
    if ok.any():
        ax[1].hist(gap, bins=np.arange(-0.5, gap.max() + 1.5, 1), rwidth=0.8)
        ax[1].set_xlabel("optimum - SARSA (antennes)"); ax[1].set_ylabel("configurations de test")
        ax[1].set_title("Écart à l'optimum exact"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(script_dir, f'resultats_SARSA{suffix}.png'), dpi=150); plt.close()
```
Deuxième graphique : histogramme de l'écart à l'optimum (combien de configurations sont à 0, 1, 2... antennes de l'optimum). `plt.tight_layout()` ajuste les marges, `savefig` enregistre `resultats_SARSA.png`, `plt.close()` libère la figure.

```python
    j = int(np.argmax(R['r_SARSA'])) if R['r_SARSA'].max() > 0 else 0
    sc = test[j]; cfgs = tuple(res[j]['config'])
    a_, p_, ring_ = sc.a, sc.phi, sc.ring
    AF0 = sc.E.sum(axis=1); AF1 = AF0 - (sc.E[:, list(cfgs)].sum(axis=1) if cfgs else 0)
    d = lambda AF: np.maximum(20 * np.log10(np.abs(AF) / np.abs(AF).max() + 1e-16), -40)
```
Figure d'exemple :
- `j` : on choisit la configuration de test où SARSA éteint le plus d'éléments (ou la 0 si aucun n'est éteint) ; `sc`, `cfgs` : son scénario et sa solution ;
- `a_`, `p_`, `ring_` : rayons, angles et anneaux des éléments ;
- `AF0` : diagramme du réseau complet ; `AF1` : diagramme sans les éléments éteints (`sc.E[:, cfgs]` : colonnes des éléments retirés) ;
- `d` : petite fonction qui convertit un facteur de réseau en dB normalisé (plancher −40 dB).

```python
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(T.THETA_DEG, d(AF0), label=f"référence ({sc.n_ref} éléments)")
    ax[0].plot(T.THETA_DEG, d(AF1), '--', label=f"SARSA ({sc.n_ref - len(cfgs)} éléments)")
    ax[0].set_xlabel("theta (deg)"); ax[0].set_ylabel("|AF| normalisé (dB)"); ax[0].set_ylim(-40, 2); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Test, configuration {j} : theta0 = {sc.theta0:.1f}°")
```
Graphique de gauche : diagramme de rayonnement du réseau de référence et du réseau allégé superposés (courbe pleine et pointillée), avec axes, limites verticales, légende et titre indiquant θ0.

```python
    x = a_ * np.cos(p_); y = a_ * np.sin(p_)
    act = np.ones(sc.n_ref, dtype=bool); act[list(cfgs)] = False
    ax[1].scatter(x[act], y[act], c='tab:blue', label='actif'); ax[1].scatter(x[~act], y[~act], c='tab:red', marker='x', label='désactivé')
    ax[1].set_aspect('equal'); ax[1].legend(); ax[1].set_title("Géométrie (plan du réseau)"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(script_dir, f'diagramme_SARSA_exemple{suffix}.png'), dpi=150); plt.close()
```
Graphique de droite : plan du réseau. `x = a cos(φ)`, `y = a sin(φ)` donnent les coordonnées de chaque élément ; `scatter` dessine les éléments actifs en bleu et les éléments éteints en rouge (croix) ; `set_aspect('equal')` garde un cercle rond. On enregistre `diagramme_SARSA_exemple.png`.

```python
    print(f"\n✅ resultats_test_SARSA{suffix}.csv/.mat/.npy, resultats_SARSA{suffix}.png, diagramme_SARSA_exemple{suffix}.png enregistrés.")
```
Message de confirmation.

```python
if __name__ == '__main__':
    main()
```
Garde-fou : `main()` ne s'exécute que si on lance ce fichier directement (`python test.py`).

---

## Résumé du flux

```
q_table.npy + sarsa_config.npz + historique_training.npy + Minput/Moutput_test.npy
        │ contrôle physique (calcul rapide = calcul direct)
        ▼
 politique gloutonne sur 100 configurations de test (Q inchangée)
        │ + recherche exhaustive, glouton, aléatoires (comparaison)
        ▼
 résumé affiché + resultats_test_SARSA.csv/.mat/.npy
 + positions_antennes_test_SARSA.csv/.mat/.npy + figures
```

## Comment lire les résultats

- **Quelles antennes éteindre ?** colonne `antennes_eteintes` de `resultats_test_SARSA.csv` (par exemple `anneau1#1(0deg) anneau1#3(90deg)`), et `architecture_finale` pour le nombre d'éléments restant par anneau.
- **Où sont les antennes ?** `positions_antennes_test_SARSA.csv` : une ligne par antenne, avec `actif = 1` si elle reste allumée. Elles gardent leur position d'origine.
- **Optimum** : nombre maximal d'antennes qu'on peut éteindre en respectant les tolérances (recherche exhaustive).
- **Écart à l'optimum** = optimum − SARSA : plus il est proche de 0, mieux c'est.
- **Tentatives de violation** : parties où l'agent a voulu éteindre un élément de trop (annulé ensuite).
- Si SARSA ne fait pas mieux que la suppression aléatoire admissible, c'est le résultat honnête à rapporter : on ne cache pas un résultat décevant.
