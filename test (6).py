# =============================================================================
#  test.py -- On donne un réseau PLEIN quelconque (N1..N5 antennes dans les 5 anneaux + theta0) à l'agent SARSA
#             entraîné. Il indique quelles antennes DÉSACTIVER (et réactiver) pour minimiser le SSL en gardant
#             gain dans [25 ; 27] dB, HPBW dans l'intervalle « best » et theta0.
#
#  Usage :
#     python test.py --arch 8 6 10 3 7 --theta0 60 [--show]     un réseau donné
#     python test.py                                            benchmark sur des réseaux aléatoires
#  Sorties (un réseau) : resultat_reseau.csv, positions_antennes.csv, avant_apres_reseau.png, diagrammes_avant_apres.png
#  Sorties (benchmark) : resultats_benchmark.csv, benchmark_SARSA.png
# =============================================================================
import argparse
import csv
import os

import numpy as np

import training as T

ap = argparse.ArgumentParser()
ap.add_argument('--arch', type=int, nargs=5, default=None, help="antennes dans les anneaux 1..5 (réseau PLEIN)")
ap.add_argument('--theta0', type=float, default=45.0, help="direction visée (deg, 0..180)")
ap.add_argument('--n_bench', type=int, default=60)
ap.add_argument('--n_random', type=int, default=30, help="essais de la politique aléatoire par réseau")
ap.add_argument('--sa_iters', type=int, default=2500, help="itérations du recuit simulé de référence")
ap.add_argument('--seed', type=int, default=123)
ap.add_argument('--tag', type=str, default='')
ap.add_argument('--show', action='store_true')
args = ap.parse_args()
suffix = f'_{args.tag}' if args.tag else ''
d = os.path.dirname(os.path.abspath(__file__))
Q = np.load(os.path.join(d, f'q_table{suffix}.npy'))
cfg = np.load(os.path.join(d, f'sarsa_config{suffix}.npz'))
spec = dict(g_lo=float(cfg['g_lo']), g_hi=float(cfg['g_hi']), h_rel=float(cfg['h_rel']), eps_theta=float(cfg['eps_theta']))

import matplotlib
if not args.show:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ méthodes
def sarsa_solve(rs, Qm=None):
    r = T.run_episode(T.Env(rs), Q if Qm is None else Qm, None, 0.0, 1.0, 0.0, np.random.default_rng(0), learn=False, tie='first')
    return r['ssl_best'], r['removed_best']


def fixed_policy(rs, prefer):
    """Glouton sans apprentissage : on applique toujours le même critère jusqu'à 18 antennes ; on garde le meilleur admissible."""
    env = T.Env(rs)
    while True:
        for a in prefer:
            if a in env.choice:
                break
        else:
            break
        _, done, _ = env.step(a)
        if done or env.n_act <= rs.n_lo:
            break
    return (env.best_ssl, env.best_removed) if env.best_removed is not None else (np.nan, None)


def random_policy(rs, n_try, rng):
    best = (np.nan, None)
    for _ in range(n_try):
        env = T.Env(rs)
        while True:
            v = env.valid_actions()
            _, done, _ = env.step(int(v[rng.integers(len(v))]))
            if done:
                break
        if env.best_removed is not None and (best[1] is None or env.best_ssl < best[0]):
            best = (env.best_ssl, env.best_removed)
    return best


def annealing(rs, rng, iters):
    """Recuit simulé (méthode de référence, très coûteuse : milliers d'évaluations du diagramme par réseau)."""
    n = rs.n
    on = np.ones(n, bool)
    on[rng.permutation(n)[:max(0, n - 20)]] = False

    def ev(mask):
        g, s, h, p = T.metrics_from_AF(rs.E[:, mask].sum(axis=1))
        viol = 0.0 if rs.hpbw_ok((g, s, h, p)) else (abs(h - np.clip(h, rs.h_lo, rs.h_hi)) / rs.ref[2])
        return s, viol, p

    s, v, p = ev(on)
    f = s + 60 * v
    best = (np.nan, None)
    for it in range(iters):
        temp = 2.0 * (1 - it / iters) + 0.02
        m = on.copy()
        r = rng.random()
        if r < 0.7 and (~m).any():
            m[rng.choice(np.flatnonzero(m))] = False
            m[rng.choice(np.flatnonzero(~m))] = True
        elif r < 0.85 and m.sum() > rs.n_lo:
            m[rng.choice(np.flatnonzero(m))] = False
        elif m.sum() < rs.n_hi and (~m).any():
            m[rng.choice(np.flatnonzero(~m))] = True
        else:
            continue
        s2, v2, p2 = ev(m)
        f2 = s2 + 60 * v2
        if f2 <= f or rng.random() < np.exp((f - f2) / temp):
            on, f, s, v = m, f2, s2, v2
            if v == 0 and T.delta_theta(p2, rs.theta0) <= rs.spec['eps_theta'] and (np.isnan(best[0]) or s < best[0]):
                best = (s, tuple(int(i) for i in np.flatnonzero(~on)))
    return best


# ------------------------------------------------------------------ rapport pour un réseau
def names(rs):
    out, cnt = [], {}
    for e in range(rs.n):
        m = int(rs.ring[e]); cnt[m] = cnt.get(m, 0) + 1
        out.append((m + 1, cnt[m], float(np.rad2deg(rs.phi[e]))))
    return out


def report(rs, removed):
    nm = names(rs)
    new_arch = rs.arch.copy()
    for e in removed:
        new_arch[nm[e][0] - 1] -= 1
    m0, m1 = rs.metrics(()), rs.metrics(removed)
    print("\n" + "=" * 72)
    print(f"Ancien réseau (plein) : {[int(x) for x in rs.arch]} = {rs.n} antennes, theta0 = {rs.theta0:.1f}°")
    print(f"Nouveau réseau        : {[int(x) for x in new_arch]} = {rs.n - len(removed)} antennes actives")
    print(f"SSL  : {m0[1]:.2f} dB -> {m1[1]:.2f} dB   (baisse {m0[1]-m1[1]:.2f} dB)")
    print(f"Gain : {m0[0]:.2f} dB -> {m1[0]:.2f} dB   (fenêtre imposée [{spec['g_lo']} ; {spec['g_hi']}] dB)")
    print(f"HPBW : {m0[2]:.2f}° -> {m1[2]:.2f}°   (intervalle « best » [{rs.h_lo:.2f}° ; {rs.h_hi:.2f}°])")
    print(f"Pic  : {m0[3]:.2f}° -> {m1[3]:.2f}°   (écart à theta0 : {T.delta_theta(m1[3], rs.theta0):.2f}°, tolérance {spec['eps_theta']}°)")
    print("\nANCIEN RÉSEAU : toutes les antennes (position) et NOUVELLE CONFIGURATION (état, action)")
    print(f"{'anneau':>6} {'n°':>3} {'angle (deg)':>12} {'rayon (λ)':>10} {'x (λ)':>8} {'y (λ)':>8}   {'avant':>5} {'après':>6}   action")
    for e in range(rs.n):
        ring, num, ang = nm[e]
        off = e in removed
        print(f"{ring:6d} {num:3d} {ang:12.1f} {rs.a[e]/T.LAMBDA:10.2f} {rs.a[e]*np.cos(rs.phi[e])/T.LAMBDA:8.3f} {rs.a[e]*np.sin(rs.phi[e])/T.LAMBDA:8.3f}   "
              f"{'ON':>5} {'OFF' if off else 'ON':>6}   {'DÉSACTIVER' if off else '-'}")
    print(f"\nRésumé : {len(removed)} antenne(s) à DÉSACTIVER, 0 à ACTIVER (tout le réseau part actif ; une antenne ne peut être"
          f" « à activer » que si elle était éteinte au départ).")
    for e in sorted(removed):
        ring, num, ang = nm[e]
        print(f"   DÉSACTIVER : anneau {ring}, antenne n°{num}, angle {ang:.1f}°")
    print("Par anneau (avant -> après) : " + ", ".join(f"anneau {m+1} : {int(rs.arch[m])} -> {int(new_arch[m])}" for m in range(T.N_RINGS)))
    return new_arch, m0, m1, nm


def plot_one(rs, removed, new_arch, m0, m1, nm):
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    for k, (ttl, rem) in enumerate([("Avant (réseau plein)", set()), ("Après (SARSA)", set(removed))]):
        for m in range(T.N_RINGS):
            ax[k].add_patch(plt.Circle((0, 0), (T.R0 + T.DR * m) / T.LAMBDA, fill=False, color='0.8', lw=0.8))
        for e in range(rs.n):
            x, y = rs.a[e] * np.cos(rs.phi[e]) / T.LAMBDA, rs.a[e] * np.sin(rs.phi[e]) / T.LAMBDA
            ax[k].plot(x, y, 'x' if e in rem else 'o', color='crimson' if e in rem else 'tab:blue', ms=9 if e in rem else 6, mew=2)
        ax[k].set_aspect('equal'); ax[k].set_title(ttl); ax[k].set_xlabel("x / λ"); ax[k].grid(alpha=0.3)
    ax[1].plot([], [], 'o', color='tab:blue', label='active'); ax[1].plot([], [], 'x', color='crimson', label='désactivée'); ax[1].legend(loc='upper right')
    plt.tight_layout(); plt.savefig(os.path.join(d, 'avant_apres_reseau.png'), dpi=150)
    AF0 = rs.AF_full; AF1 = rs.AF_full - rs.E[:, list(removed)].sum(axis=1) if removed else AF0
    dbn = lambda AF: np.maximum(20 * np.log10(np.abs(AF) / np.abs(AF).max() + T.EPS_MACH), -40)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a_ in ax:
        a_.plot(T.THETA_DEG, dbn(AF0), color='tab:gray', label=f"avant (SSL {m0[1]:.2f} dB, HPBW {m0[2]:.1f}°)")
        a_.plot(T.THETA_DEG, dbn(AF1), color='tab:red', label=f"après (SSL {m1[1]:.2f} dB, HPBW {m1[2]:.1f}°)")
        a_.set_ylim(-40, 3); a_.set_xlabel("θ (deg)"); a_.grid(alpha=0.3)
    ax[0].axvline(rs.theta0, color='k', ls=':', lw=1); ax[0].axvline(180 - rs.theta0, color='k', ls=':', lw=1)
    ax[0].set_xlim(0, 360); ax[0].set_ylabel("dB normalisé"); ax[0].set_title("Diagramme de rayonnement"); ax[0].legend(fontsize=8)
    ax[1].set_xlim(max(0, rs.theta0 - 30), min(180, rs.theta0 + 30)); ax[1].set_title("Zoom autour du lobe principal")
    plt.tight_layout(); plt.savefig(os.path.join(d, 'diagrammes_avant_apres.png'), dpi=150)
    with open(os.path.join(d, 'resultat_reseau.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ancien_reseau', 'nouveau_reseau', 'theta0', 'SSL_avant', 'SSL_apres', 'gain_avant', 'gain_apres', 'HPBW_avant', 'HPBW_apres',
                    'HPBW_min_admis', 'HPBW_max_admis', 'pic_avant', 'pic_apres'])
        w.writerow([[int(x) for x in rs.arch], [int(x) for x in new_arch], rs.theta0, round(m0[1], 3), round(m1[1], 3), round(m0[0], 3), round(m1[0], 3),
                    round(m0[2], 3), round(m1[2], 3), round(rs.h_lo, 3), round(rs.h_hi, 3), round(m0[3], 3), round(m1[3], 3)])
    with open(os.path.join(d, 'positions_antennes.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['anneau', 'numero', 'angle_deg', 'rayon_en_lambda', 'x_en_lambda', 'y_en_lambda', 'etat_avant', 'etat_apres', 'action'])
        for e in range(rs.n):
            ring, num, ang = nm[e]
            w.writerow([ring, num, round(ang, 2), round(rs.a[e] / T.LAMBDA, 3), round(rs.a[e] * np.cos(rs.phi[e]) / T.LAMBDA, 4), round(rs.a[e] * np.sin(rs.phi[e]) / T.LAMBDA, 4), 'ON', 'OFF' if e in removed else 'ON', 'DÉSACTIVER' if e in removed else '-'])
    if args.show:
        plt.show()
    plt.close('all')


rng = np.random.default_rng(args.seed)
if args.arch is not None:
    rs = T.Reseau(args.arch, args.theta0, spec)
    if not rs.feasible_start():
        raise SystemExit(f"Réseau trop petit : {rs.n} antennes. Un gain de {spec['g_lo']} dB demande au moins {rs.n_lo} antennes actives.")
    ssl, removed = sarsa_solve(rs)
    if removed is None:
        raise SystemExit("SARSA n'a trouvé aucune configuration admissible pour ce réseau (HPBW ou gain hors cahier des charges).")
    new_arch, m0, m1, nm = report(rs, removed)
    plot_one(rs, removed, new_arch, m0, m1, nm)
    print("\nFichiers : resultat_reseau.csv, positions_antennes.csv, avant_apres_reseau.png, diagrammes_avant_apres.png")
else:
    reseaux = T.random_reseaux(rng, args.n_bench, spec, int(cfg['n_min']), int(cfg['n_max']), int(cfg['n_tot_min']))
    Q0 = np.zeros_like(Q)
    meths = ['SARSA', 'table non entraînée', 'glouton SSL', 'glouton HPBW-sûr', 'aléatoire (30 essais)', 'recuit simulé (référence)']
    rows = []
    for rs in reseaux:
        res = [sarsa_solve(rs), sarsa_solve(rs, Q0), fixed_policy(rs, [T.OFF_SSL]), fixed_policy(rs, [T.OFF_SAFE, T.OFF_HPBW]),
               random_policy(rs, args.n_random, rng), annealing(rs, rng, args.sa_iters)]
        rows.append([[int(x) for x in rs.arch], round(rs.theta0, 1), rs.n, rs.ref[1]] + [x[0] for x in res])
    S = np.array([[r[3] - r[4 + k] for k in range(6)] for r in rows])             # baisse (nan si échec)
    print(f"\nRéseaux de test : {len(rows)}. Pour chaque méthode : réseaux admissibles, baisse moyenne du SSL (sur les admissibles)")
    for k, nm_ in enumerate(meths):
        ok = ~np.isnan(S[:, k])
        print(f"  {nm_:28s} admissible {int(ok.sum()):3d}/{len(rows)} | baisse moyenne {np.nanmean(S[:, k]):.2f} dB | médiane {np.nanmedian(S[:, k]):.2f} | max {np.nanmax(S[:, k]):.2f}")
    with open(os.path.join(d, 'resultats_benchmark.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['reseau', 'theta0', 'N', 'SSL_plein'] + ['SSL_' + m for m in meths])
        w.writerows(rows)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    cols = ['tab:red', 'tab:gray', 'tab:blue', 'tab:cyan', 'tab:orange', 'tab:green']
    ax[0].bar(range(6), [np.nanmean(S[:, k]) for k in range(6)], color=cols); ax[0].set_xticks(range(6)); ax[0].set_xticklabels([m.replace(' (', '\n(') for m in meths], fontsize=7)
    ax[0].set_ylabel("baisse moyenne du SSL (dB)"); ax[0].set_title("Baisse du SSL (réseaux admissibles)"); ax[0].grid(alpha=0.3, axis='y')
    ax[1].bar(range(6), [100 * (~np.isnan(S[:, k])).mean() for k in range(6)], color=cols); ax[1].set_xticks(range(6)); ax[1].set_xticklabels([m.replace(' (', '\n(') for m in meths], fontsize=7)
    ax[1].set_ylabel("% de réseaux avec solution admissible"); ax[1].set_ylim(0, 105); ax[1].grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(d, 'benchmark_SARSA.png'), dpi=150)
    if args.show:
        plt.show()
