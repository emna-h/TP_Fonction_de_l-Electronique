# =============================================================================
#  test.py  --  Étape 4/4 : évaluation de la politique SARSA apprise (projet SARSA)
#
#  AUCUNE mise à jour de la table Q ici : politique purement gloutonne (epsilon = 0).
#  Les configurations de test n'ont jamais été vues à l'entraînement.
#
#  Trois résultats sont toujours présentés SÉPARÉMENT :
#     - la meilleure configuration admissible rencontrée PENDANT l'entraînement
#       (avec exploration : une trouvaille, pas une preuve d'apprentissage) ;
#     - la configuration produite par la politique FINALE sur les configurations d'entraînement ;
#     - la configuration produite par la politique FINALE sur le TEST (seule base de généralisation).
#
#  Comparaison avec trois méthodes de référence (rapport, section 3.4) :
#     recherche exhaustive (optimum exact quand elle est faisable), suppression aléatoire
#     admissible, heuristique gloutonne ; plus une politique aléatoire sans oracle.
#
#  Entrées  : q_table.npy, sarsa_config.npz, historique_training.npy, Minput/Moutput_*.npy
#  Sorties  : resultats_test_SARSA.csv / .mat / .npy, resultats_SARSA.png, diagramme_SARSA_exemple.png
#  Options  : python test.py --tag eG2   (pour une table entraînée avec --tag eG2)
# =============================================================================
import argparse
import csv
import itertools
import math
import os

import numpy as np
import scipy.io

import training as T


def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")


# --- Vérification du modèle physique rapide (matrice de contributions) contre le calcul direct ---
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


# --- Méthodes de référence ---
def admissible(sc, removed):
    return sc.u_max(tuple(sorted(removed))) <= 1.0


def exhaustive(sc, norm, cap):
    """Optimum exact : plus grand r admissible, puis plus petite erreur globale. None si > cap évaluations."""
    count = 0
    for r in range(sc.r_max, -1, -1):
        if count + math.comb(sc.n_ref, r) > cap:      # niveau trop grand : abandon immédiat
            return None
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


def random_admissible(sc, rng):
    removed = []
    while len(removed) < sc.n_ref - 1:
        cand = [e for e in range(sc.n_ref) if e not in removed and admissible(sc, removed + [e])]
        if not cand:
            break
        removed.append(int(rng.choice(cand)))
    return len(removed)


def random_policy(sc, penalty, rng):
    """Politique aléatoire sans oracle : même environnement, action valide tirée au hasard (STOP compris)."""
    env = T.AntennaEnv(sc, penalty)
    env.reset()
    while True:
        a = int(rng.choice(env.valid_actions()))
        _, done, end = env.step(a)
        if done:
            return len(env.last_admissible), end == 'violation'


def main():
    clear_console()
    p = argparse.ArgumentParser()
    p.add_argument('--tag', type=str, default='')
    p.add_argument('--cap_exh', type=int, default=100000, help="nombre maximal d'évaluations de la recherche exhaustive")
    p.add_argument('--n_random', type=int, default=20)
    args = p.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ld = lambda f: np.load(os.path.join(script_dir, f))

    cfg = np.load(os.path.join(script_dir, f'sarsa_config{suffix}.npz'))
    tol = dict(eps_G=float(cfg['eps_G']), eps_S=float(cfg['eps_S']), eps_H_rel=float(cfg['eps_H_rel']),
               eps_theta=float(cfg['eps_theta']))
    penalty = float(cfg['penalty'])
    Q = ld(f'q_table{suffix}.npy')
    hist = np.load(os.path.join(script_dir, f'historique_training{suffix}.npy'), allow_pickle=True).item()
    Mi_all = ld('Minput.npy')
    norm = (np.max(np.abs(Mi_all[0])), np.max(np.abs(Mi_all[1])), 180.0)
    rng = np.random.default_rng(46)

    Mi_te, Mo_te = ld('Minput_test.npy'), ld('Moutput_test.npy')
    Mi_tr, Mo_tr = ld('Minput_training.npy'), ld('Moutput_training.npy')
    print(f"Tolérances : {tol} ; pénalité {penalty}")

    # --- 0. contrôle du modèle physique ---
    e = verif_physique(Mi_te, Mo_te, tol, rng)
    print(f"Contrôle physique : écart max (gain, SSL, HPBW) entre calcul rapide et calcul direct = {e:.2e}")
    assert e < 1e-8

    # --- 1. politique finale sur le TEST (sans mise à jour) ---
    test = T.build_scenarios(Mi_te, Mo_te, tol)
    Q_avant = Q.copy()
    res = T.evaluate_greedy(test, Q, penalty)
    assert np.array_equal(Q, Q_avant), "la table Q ne doit pas changer au test"

    rows, activations = [], []
    for j, (sc, rs) in enumerate(zip(test, res)):
        cfgs = rs['config']
        act = np.ones(sc.n_ref, dtype=int)
        act[list(cfgs)] = 0
        activations.append(act)
        g, s, h, pk = sc.metrics(tuple(cfgs))
        u = sc.deviations(tuple(cfgs))
        ex = exhaustive(sc, norm, args.cap_exh)
        if (j + 1) % 10 == 0:
            print(f"  {j+1}/{len(test)} configurations traitées (la recherche exhaustive est la partie lente)", flush=True)
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
        rp = [random_policy(sc, penalty, rng) for _ in range(args.n_random)]
        rows[-1]['r_politique_aleatoire'] = np.mean([x[0] for x in rp])
        rows[-1]['viol_politique_aleatoire'] = np.mean([x[1] for x in rp])
    R = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != 'fin_episode'}
    R_end = [r['fin_episode'] for r in rows]

    # --- 2. résumé ---
    use = R['r_max'] >= 1
    print(f"\n=== TEST : {len(test)} configurations jamais vues ({int(use.sum())} avec r_max >= 1) ===")
    print(f"Configurations finales admissibles : {int(R['admissible'].sum())} / {len(test)} "
          f"(par construction : retour à la dernière configuration admissible en cas de violation)")
    print(f"Épisodes terminés par STOP (la politique s'arrête d'elle-même) : {R_end.count('stop')}, "
          f"par tentative de violation (annulée) : {R_end.count('violation')}, plus d'action possible : {R_end.count('plus_daction')}")
    print(f"Antennes désactivées : moyenne {R['r_SARSA'].mean():.2f} (soit {R['pct_desactive'].mean():.2f} % de N_ref), "
          f"borne r_max moyen {R['r_max'].mean():.2f}")
    print(f"Retour relatif r / r_max (r_max >= 1) : {np.mean(R['r_SARSA'][use] / R['r_max'][use]):.3f}")
    ok = ~np.isnan(R['r_optimum'])
    print(f"Optimum exact (recherche exhaustive, <= {args.cap_exh} évaluations) disponible pour {int(ok.sum())} / {len(test)} configurations")
    if ok.any():
        gap = R['r_optimum'][ok] - R['r_SARSA'][ok]
        print(f"  SARSA atteint l'optimum : {int((gap == 0).sum())} / {int(ok.sum())} ; écart moyen à l'optimum {gap.mean():.3f} antenne(s) ; "
              f"réduction moyenne optimum {R['r_optimum'][ok].mean():.2f} vs SARSA {R['r_SARSA'][ok].mean():.2f}")
        print(f"  (SARSA ne dépasse jamais l'optimum : {bool(np.all(gap >= 0))})")
    print("\nMéthodes de référence (r moyen sur le test) :")
    print(f"  SARSA (politique finale)            : {R['r_SARSA'].mean():.3f}")
    print(f"  Heuristique gloutonne (oracle)      : {np.nanmean(R['r_glouton']):.3f}")
    print(f"  Aléatoire admissible (oracle)       : {R['r_aleatoire_admissible'].mean():.3f}")
    print(f"  Politique aléatoire sans oracle     : {R['r_politique_aleatoire'].mean():.3f}  "
          f"(tentatives de violation : {100*R['viol_politique_aleatoire'].mean():.1f} % des épisodes)")
    if ok.any():
        print(f"  Optimum exact (sous-ensemble)       : {R['r_optimum'][ok].mean():.3f}")
    print("\nPerformances (test) : écarts moyens |dG| {:.3f} dB, |dSSL| {:.3f} dB, |dHPBW| {:.3f} deg, dtheta {:.3f} deg ; "
          "erreur globale pondérée {:.3f} %".format(np.abs(R['dG']).mean(), np.abs(R['dSSL']).mean(),
                                                      np.abs(R['dHPBW']).mean(), R['dtheta'].mean(), R['erreur_globale_pct'].mean()))

    # --- 3. entraînement : meilleure trouvaille vs politique finale ---
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

    # --- 4. enregistrement ---
    keys = list(rows[0].keys())
    with open(os.path.join(script_dir, f'resultats_test_SARSA{suffix}.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=';')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
    nmax = int(Mo_te.sum(axis=0).max())
    act_mat = -np.ones((nmax, len(test)))                  # -1 = case vide (longueur variable)
    for j, a in enumerate(activations):
        act_mat[:len(a), j] = a
    out = {k: v for k, v in R.items()}
    out['activation'] = act_mat
    scipy.io.savemat(os.path.join(script_dir, f'resultats_test_SARSA{suffix}.mat'), out)
    np.save(os.path.join(script_dir, f'resultats_test_SARSA{suffix}.npy'), out, allow_pickle=True)

    # --- 5. figures ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    names = ['SARSA', 'Glouton', 'Aléatoire\nadmissible', 'Aléatoire\nsans oracle']
    vals = [R['r_SARSA'].mean(), np.nanmean(R['r_glouton']), R['r_aleatoire_admissible'].mean(), R['r_politique_aleatoire'].mean()]
    if ok.any():
        names.append('Optimum\n(exhaustif)')
        vals.append(R['r_optimum'][ok].mean())
    ax[0].bar(names, vals); ax[0].set_ylabel("antennes désactivées (moyenne, test)")
    ax[0].axhline(R['r_max'].mean(), color='k', ls='--', label='borne r_max (gain)'); ax[0].legend(); ax[0].grid(alpha=0.3, axis='y')
    ax[0].set_title("Test : méthodes comparées")
    if ok.any():
        ax[1].hist(gap, bins=np.arange(-0.5, gap.max() + 1.5, 1), rwidth=0.8)
        ax[1].set_xlabel("optimum - SARSA (antennes)"); ax[1].set_ylabel("configurations de test")
        ax[1].set_title("Écart à l'optimum exact"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(script_dir, f'resultats_SARSA{suffix}.png'), dpi=150); plt.close()

    j = int(np.argmax(R['r_SARSA'])) if R['r_SARSA'].max() > 0 else 0
    sc = test[j]; cfgs = tuple(res[j]['config'])
    a_, p_, ring_ = sc.a, sc.phi, sc.ring
    AF0 = sc.E.sum(axis=1); AF1 = AF0 - (sc.E[:, list(cfgs)].sum(axis=1) if cfgs else 0)
    d = lambda AF: np.maximum(20 * np.log10(np.abs(AF) / np.abs(AF).max() + 1e-16), -40)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(T.THETA_DEG, d(AF0), label=f"référence ({sc.n_ref} éléments)")
    ax[0].plot(T.THETA_DEG, d(AF1), '--', label=f"SARSA ({sc.n_ref - len(cfgs)} éléments)")
    ax[0].set_xlabel("theta (deg)"); ax[0].set_ylabel("|AF| normalisé (dB)"); ax[0].set_ylim(-40, 2); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Test, configuration {j} : theta0 = {sc.theta0:.1f}°")
    x = a_ * np.cos(p_); y = a_ * np.sin(p_)
    act = np.ones(sc.n_ref, dtype=bool); act[list(cfgs)] = False
    ax[1].scatter(x[act], y[act], c='tab:blue', label='actif'); ax[1].scatter(x[~act], y[~act], c='tab:red', marker='x', label='désactivé')
    ax[1].set_aspect('equal'); ax[1].legend(); ax[1].set_title("Géométrie (plan du réseau)"); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(script_dir, f'diagramme_SARSA_exemple{suffix}.png'), dpi=150); plt.close()
    print(f"\n✅ resultats_test_SARSA{suffix}.csv/.mat/.npy, resultats_SARSA{suffix}.png, diagramme_SARSA_exemple{suffix}.png enregistrés.")


if __name__ == '__main__':
    main()
