# =============================================================================
#  training.py  --  Étape 3/4 : entraînement SARSA tabulaire (projet SARSA)
#
#  Objectif : apprendre une politique qui désactive le plus d'antennes possible
#  d'un réseau circulaire multi-anneaux tout en gardant gain, SSL, HPBW et
#  direction theta0 proches de la référence (tolérances eps_G, eps_S, eps_H, eps_theta).
#
#  Tout est écrit explicitement ici (aucune bibliothèque d'apprentissage par
#  renforcement) :
#     1. le modèle physique (facteur de réseau, Gmax, SSL, HPBW), identique à
#        celui du dataset.py (calcul_AF_performance_metrics) ;
#     2. l'environnement (état, actions, récompense, fin d'épisode) ;
#     3. l'agent SARSA : politique epsilon-glouton + mise à jour
#        Q(o,a) <- Q(o,a) + alpha [ R + gamma Q(o',a') - Q(o,a) ].
#
#  Ce fichier est aussi importé par test.py (les fonctions sont réutilisées).
#
#  Entrées  (dossier du script) : Minput_training/validation.npy, Moutput_training/validation.npy
#  Sorties  (dossier du script) : q_table.npy/.mat, visites_Q.npy, historique_training.npy/.mat,
#           sarsa_config.npz, courbes_SARSA.png
#
#  Les valeurs "à valider" du rapport sont des paramètres modifiables ci-dessous
#  ou en ligne de commande (ex. : python training.py --eps_G 2 --tag eG2).
# =============================================================================
import argparse
import itertools
import os
import time

import numpy as np
import scipy.io
from scipy.signal import find_peaks

# --- Constantes physiques (identiques au dataset.py) ---
C0 = 3e8
F0 = 2.45e9
LAMBDA = C0 / F0
K_WAVE = 2 * np.pi / LAMBDA
R0 = 0.2 * LAMBDA            # rayon de l'anneau 1
DR = 0.5 * LAMBDA            # pas entre anneaux
THETA = np.linspace(0, 2 * np.pi, 1000)
THETA_DEG = np.rad2deg(THETA)
EPS_MACH = np.finfo(float).eps

# --- Constantes de l'agent (voir rapport, chapitres 5 à 7) ---
N_RINGS = 5
N_ACTIONS = N_RINGS + 1      # actions 0..4 : amincir l'anneau 1..5 ; action 5 : STOP
STOP = N_RINGS
L_MAX = 5                    # l = min(5, r_max - r) : budget de suppressions restant (0..5)
N_OBS = (L_MAX + 1) * 3 * 3  # 54 observations (l, q_S, q_H)
Q_THRESH = (0.5, 0.8)        # seuils de discrétisation des écarts normalisés u
W_ERR = (0.33, 0.33, 0.34)   # pondérations de l'erreur globale (comme test_model_FFNN / cGAN)


def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")


# =============================================================================
#  1. MODÈLE PHYSIQUE
# =============================================================================
def layout(arch):
    """Position de TOUS les éléments de référence : rayon, angle dans l'anneau, numéro d'anneau."""
    a, phi, ring = [], [], []
    for m, N in enumerate(arch):
        N = int(N)
        for n in range(N):
            a.append(R0 + DR * m)
            phi.append(2 * np.pi * n / N)
            ring.append(m)
    return np.array(a), np.array(phi), np.array(ring)


def contributions(a, phi, theta0_deg):
    """Matrice E (1000 x N) : contribution complexe de chaque élément au facteur de réseau.
    AF(theta) = somme des colonnes de E pour les éléments actifs."""
    th0 = np.deg2rad(theta0_deg)
    phase = K_WAVE * a[None, :] * (np.sin(THETA)[:, None] * np.cos(0 - phi)[None, :]
                                   - np.sin(th0) * np.cos(0 - phi)[None, :])
    return np.exp(1j * phase)


def metrics_from_AF(AF):
    """Gmax (non normalisé), SSL, HPBW et direction du maximum : même calcul que le dataset.py."""
    mag = np.abs(AF)
    mx = mag.max()
    n = mag / (mx + EPS_MACH)
    dB = 20 * np.log10(n + EPS_MACH)
    dB[dB < -40] = -40
    gain = 20 * np.log10(mx + EPS_MACH)
    # HPBW : croisement à -3 dB, sur un diagramme répété 3 fois
    half = dB.max() - 3
    ext = np.concatenate((dB, dB, dB))
    th_ext = np.concatenate((THETA_DEG - 360, THETA_DEG, THETA_DEG + 360))
    imax = np.argmax(dB) + len(THETA_DEG)
    left = np.where(ext[:imax] <= half)[0]
    right = np.where(ext[imax:] <= half)[0]
    hpbw = 180.0 if (len(left) == 0 or len(right) == 0) else th_ext[right[0] + imax] - th_ext[left[-1]]
    # SSL : plus haut pic hors de 1 dB du sommet
    pk, _ = find_peaks(n, distance=5)
    pv = n[pk]
    if len(pv) == 0:
        ssl = 0.0
    else:
        sp = np.sort(pv)[::-1]
        main = np.where(20 * np.log10(sp + EPS_MACH) >= 20 * np.log10(sp[0] + EPS_MACH) - 1)[0]
        side = np.setdiff1d(np.arange(len(sp)), main)
        ssl = 20 * np.log10(sp[side[0]] + EPS_MACH) if len(side) else 0.0
    return gain, ssl, hpbw, THETA_DEG[int(np.argmax(dB))]


def delta_theta(peak_deg, theta0_deg):
    """Écart de direction : le diagramme a deux lobes principaux symétriques, theta0 et 180-theta0."""
    return min(abs(peak_deg - theta0_deg), abs(peak_deg - (180.0 - theta0_deg)))


# =============================================================================
#  2. ENVIRONNEMENT
# =============================================================================
class Scenario:
    """Une configuration de référence (une colonne de Minput / Moutput) et ses contraintes."""

    def __init__(self, arch, theta0_deg, tol):
        self.arch = np.array(arch).astype(int)
        self.theta0 = float(theta0_deg)
        self.a, self.phi, self.ring = layout(self.arch)
        self.n_ref = len(self.a)
        self.tol = tol                                    # dict eps_G, eps_S, eps_H_rel, eps_theta
        self._E = None                                    # calculée au premier besoin
        self.cache = {}                                   # {tuple des éléments retirés : métriques}
        self.ref = self.metrics(())                       # performances de référence (réseau complet)
        self.eps_H = tol['eps_H_rel'] * self.ref[2]
        self.r_max = int(np.floor(self.n_ref * (1 - 10 ** (-tol['eps_G'] / 20)) + 1e-9))

    @property
    def E(self):
        if self._E is None:
            self._E = contributions(self.a, self.phi, self.theta0)
        return self._E

    def metrics(self, removed):
        """(G, SSL, HPBW, pic) du réseau privé des éléments 'removed' (n-uplet trié)."""
        if removed not in self.cache:
            AF = self.E.sum(axis=1) if not removed else self.E.sum(axis=1) - self.E[:, list(removed)].sum(axis=1)
            self.cache[removed] = metrics_from_AF(AF)
        return self.cache[removed]

    def deviations(self, removed):
        """Écarts normalisés (u_G, u_S, u_H, u_theta) ; admissible <=> max <= 1."""
        g, s, h, pk = self.metrics(removed)
        rg, rs, rh, _ = self.ref
        return (abs(g - rg) / self.tol['eps_G'], abs(s - rs) / self.tol['eps_S'],
                abs(h - rh) / self.eps_H, delta_theta(pk, self.theta0) / self.tol['eps_theta'])

    def u_max(self, removed):
        return max(self.deviations(removed))


class AntennaEnv:
    """Environnement SARSA : un épisode = suite de suppressions d'éléments sur un scénario."""

    def __init__(self, scenario, penalty=2.0):
        self.sc = scenario
        self.penalty = penalty
        self.reset()

    def reset(self):
        self.removed = []                                 # éléments retirés (dans l'ordre)
        self.active = np.ones(self.sc.n_ref, dtype=bool)
        self.last_admissible = tuple()
        return self.observation()

    # --- observation compacte o = (l, q_S, q_H) -> indice 0..53 ---
    def observation(self):
        sc = self.sc
        u = sc.deviations(tuple(sorted(self.removed)))
        l = min(L_MAX, max(0, sc.r_max - len(self.removed)))
        qs = int(u[1] >= Q_THRESH[0]) + int(u[1] >= Q_THRESH[1])
        qh = int(u[2] >= Q_THRESH[0]) + int(u[2] >= Q_THRESH[1])
        return l * 9 + qs * 3 + qh

    def valid_actions(self):
        """STOP toujours valide ; amincir un anneau est valide s'il lui reste un élément actif
        et s'il reste au moins 2 éléments actifs au total (les suppressions non admissibles
        ne sont PAS masquées : l'agent doit apprendre à les éviter)."""
        acts = []
        if self.active.sum() > 1:
            for m in range(N_RINGS):
                if np.any(self.active & (self.sc.ring == m)):
                    acts.append(m)
        acts.append(STOP)
        return acts

    def element_to_remove(self, ring):
        """Règle fixe dans l'anneau : on retire l'élément actif dont la somme des deux écarts
        angulaires aux voisins actifs est la plus petite (égalité -> plus petit indice)."""
        idx = np.where(self.active & (self.sc.ring == ring))[0]
        if len(idx) == 1:
            return int(idx[0])
        ang = self.sc.phi[idx]                            # angles croissants (construits ainsi)
        nxt = np.roll(ang, -1)
        prv = np.roll(ang, 1)
        gap = ((nxt - ang) % (2 * np.pi)) + ((ang - prv) % (2 * np.pi))
        return int(idx[int(np.argmin(gap))])              # argmin renvoie le premier minimum

    def step(self, action):
        """Retourne (récompense, terminé, info). Récompenses : STOP 0 ; suppression admissible
        +1/r_max ; violation -penalty (retour à la dernière configuration admissible)."""
        if action == STOP:
            return 0.0, True, 'stop'
        e = self.element_to_remove(action)
        self.active[e] = False
        self.removed.append(e)
        u = self.sc.u_max(tuple(sorted(self.removed)))
        if u > 1.0:
            self.active[e] = True                         # annulation : la solution reste la config précédente
            self.removed.pop()
            return -self.penalty, True, 'violation'
        self.last_admissible = tuple(sorted(self.removed))
        if len(self.valid_actions()) == 1:                # plus aucune action de suppression possible
            return 1.0 / max(1, self.sc.r_max), True, 'plus_daction'
        return 1.0 / max(1, self.sc.r_max), False, 'ok'


# =============================================================================
#  3. AGENT SARSA TABULAIRE
# =============================================================================
def choose_action(Q, o, valid, eps, rng, tie='random'):
    """Politique epsilon-glouton restreinte aux actions valides."""
    if eps > 0 and rng.random() < eps:
        return int(valid[rng.integers(len(valid))])
    q = np.array([Q[o, a] for a in valid])
    best = np.flatnonzero(q == q.max())
    return int(valid[best[rng.integers(len(best))]]) if tie == 'random' else int(valid[best[0]])


def run_episode(env, Q, visits, alpha, gamma, eps, rng, learn=True, tie='random'):
    """Un épisode SARSA complet. Si learn=False : aucune mise à jour (évaluation)."""
    o = env.reset()
    a = choose_action(Q, o, env.valid_actions(), eps, rng, tie)
    ret, n_steps, end = 0.0, 0, ''
    while True:
        r, done, end = env.step(a)
        ret += r
        n_steps += 1
        if done:
            target = r                                    # état terminal : Q(s', .) = 0
        else:
            o2 = env.observation()
            a2 = choose_action(Q, o2, env.valid_actions(), eps, rng, tie)
            target = r + gamma * Q[o2, a2]
        if learn:
            Q[o, a] += alpha * (target - Q[o, a])         # mise à jour SARSA
            visits[o, a] += 1
        if done:
            break
        o, a = o2, a2
        assert n_steps <= env.sc.r_max + 1, "épisode trop long : la loi du gain devrait l'interdire"
    n_removed = len(env.last_admissible)
    return dict(ret=ret, r=n_removed, end=end, config=env.last_admissible, steps=n_steps)


def error_global(sc, removed, norm):
    """Erreur globale pondérée (%) comme dans test_model_FFNN / cGAN (départage des égalités)."""
    g, s, h, _ = sc.metrics(tuple(sorted(removed)))
    rg, rs, rh, _ = sc.ref
    return (W_ERR[0] * 100 * abs(g - rg) / norm[0] + W_ERR[1] * 100 * abs(s - rs) / norm[1]
            + W_ERR[2] * 100 * abs(h - rh) / norm[2])


def build_scenarios(Minput, Moutput, tol):
    return [Scenario(Moutput[:, j], Minput[3, j], tol) for j in range(Minput.shape[1])]


def evaluate_greedy(scenarios, Q, penalty):
    """Politique gloutonne (eps = 0), SANS mise à jour, sur une liste de scénarios."""
    rng = np.random.default_rng(0)
    out = []
    for sc in scenarios:
        res = run_episode(AntennaEnv(sc, penalty), Q, None, 0.0, 1.0, 0.0, rng, learn=False, tie='first')
        out.append(res)
    return out


# =============================================================================
#  PROGRAMME PRINCIPAL
# =============================================================================
def main():
    clear_console()
    p = argparse.ArgumentParser()
    p.add_argument('--eps_G', type=float, default=1.0, help="tolérance de gain (dB)")
    p.add_argument('--eps_S', type=float, default=2.0, help="tolérance de SSL (dB)")
    p.add_argument('--eps_H', type=float, default=0.10, help="tolérance de HPBW (fraction de H_ref)")
    p.add_argument('--eps_theta', type=float, default=1.0, help="tolérance de direction (deg)")
    p.add_argument('--episodes', type=int, default=20000)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_min', type=float, default=0.05)
    p.add_argument('--eps_decay', type=float, default=0.9997)
    p.add_argument('--penalty', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=46)
    p.add_argument('--eval_every', type=int, default=1000)
    p.add_argument('--tag', type=str, default='', help="suffixe des fichiers de sortie")
    args = p.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''

    script_dir = os.path.dirname(os.path.abspath(__file__))
    tol = dict(eps_G=args.eps_G, eps_S=args.eps_S, eps_H_rel=args.eps_H, eps_theta=args.eps_theta)
    rng = np.random.default_rng(args.seed)

    Mi_tr = np.load(os.path.join(script_dir, 'Minput_training.npy'))
    Mo_tr = np.load(os.path.join(script_dir, 'Moutput_training.npy'))
    Mi_va = np.load(os.path.join(script_dir, 'Minput_validation.npy'))
    Mo_va = np.load(os.path.join(script_dir, 'Moutput_validation.npy'))

    train_all = build_scenarios(Mi_tr, Mo_tr, tol)
    val_all = build_scenarios(Mi_va, Mo_va, tol)
    # contrôle de cohérence : performances de référence chargées vs recalculées
    ecart = max(np.max(np.abs(np.array(sc.ref[:3]) - Mi_tr[:3, j])) for j, sc in enumerate(train_all))
    print(f"Écart max entre Minput chargé et diagramme recalculé : {ecart:.2e} (attendu < 1e-8)")
    assert ecart < 1e-8
    # scénarios sans décision possible (r_max = 0) exclus de l'apprentissage
    train = [s for s in train_all if s.r_max >= 1]
    print(f"Training : {len(train)} scénarios utilisables sur {len(train_all)} (r_max >= 1) ; validation : {len(val_all)}")
    print(f"Tolérances : eps_G={tol['eps_G']} dB, eps_S={tol['eps_S']} dB, eps_H={100*tol['eps_H_rel']:.0f} % de H_ref, "
          f"eps_theta={tol['eps_theta']} deg ; r_max moyen (training) = {np.mean([s.r_max for s in train]):.2f}")

    Q = np.zeros((N_OBS, N_ACTIONS))                      # initialisation neutre
    visits = np.zeros((N_OBS, N_ACTIONS), dtype=np.int64)
    best_seen = np.zeros(len(train_all), dtype=int)       # meilleur r admissible rencontré PENDANT l'entraînement
    id_in_all = {id(s): i for i, s in enumerate(train_all)}

    H = {k: np.zeros(args.episodes) for k in ['ret', 'r', 'violation', 'stop', 'eps']}
    val_eps, val_r, val_viol, val_ret = [], [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        eps = max(args.eps_min, args.eps_start * args.eps_decay ** ep)
        sc = train[rng.integers(len(train))]
        res = run_episode(AntennaEnv(sc, args.penalty), Q, visits, args.alpha, args.gamma, eps, rng)
        i = id_in_all[id(sc)]
        best_seen[i] = max(best_seen[i], res['r'])
        H['ret'][ep], H['r'][ep], H['eps'][ep] = res['ret'], res['r'], eps
        H['violation'][ep] = res['end'] == 'violation'
        H['stop'][ep] = res['end'] == 'stop'
        if (ep + 1) % args.eval_every == 0:
            v = evaluate_greedy(val_all, Q, args.penalty)
            val_eps.append(ep + 1)
            val_r.append(np.mean([x['r'] / max(1, s.r_max) for x, s in zip(v, val_all) if s.r_max >= 1]))
            val_viol.append(np.mean([x['end'] == 'violation' for x in v]))
            val_ret.append(np.mean([x['ret'] for x in v]))
            w = slice(max(0, ep - 999), ep + 1)
            print(f"ep {ep+1:6d} | eps {eps:.3f} | train (1000 derniers, avec exploration) : retour {H['ret'][w].mean():+.3f}, "
                  f"violations {100*H['violation'][w].mean():5.1f} % | validation gloutonne : r/r_max {val_r[-1]:.3f}, "
                  f"tentatives de violation {100*val_viol[-1]:5.1f} % | {time.time()-t0:5.0f} s")

    # --- Politique finale gloutonne : action préférée par observation ---
    print("\nCases (observation, action) visitées :", int((visits > 0).sum()), "/", visits.size)
    print("Q-table finale (arrondie, lignes = observation (l,qS,qH), colonnes = anneau1..5, STOP) :")
    np.set_printoptions(linewidth=140, suppress=True)
    for o in range(N_OBS):
        if visits[o].sum() > 0:
            print(f"  l={o//9} qS={(o%9)//3} qH={o%3} :", np.round(Q[o], 3), " visites", visits[o])

    # --- Enregistrement ---
    np.save(os.path.join(script_dir, f'q_table{suffix}.npy'), Q)
    scipy.io.savemat(os.path.join(script_dir, f'q_table{suffix}.mat'), {'q_table': Q})
    np.save(os.path.join(script_dir, f'visites_Q{suffix}.npy'), visits)
    hist = dict(retour=H['ret'], r_supprimes=H['r'], violation=H['violation'], stop=H['stop'], epsilon=H['eps'],
                val_episodes=np.array(val_eps), val_r_sur_rmax=np.array(val_r),
                val_tentatives_violation=np.array(val_viol), val_retour=np.array(val_ret),
                meilleur_r_vu_en_training=best_seen)
    np.save(os.path.join(script_dir, f'historique_training{suffix}.npy'), hist, allow_pickle=True)
    scipy.io.savemat(os.path.join(script_dir, f'historique_training{suffix}.mat'), hist)
    np.savez(os.path.join(script_dir, f'sarsa_config{suffix}.npz'), eps_G=tol['eps_G'], eps_S=tol['eps_S'],
             eps_H_rel=tol['eps_H_rel'], eps_theta=tol['eps_theta'], penalty=args.penalty, alpha=args.alpha,
             gamma=args.gamma, episodes=args.episodes, seed=args.seed)

    # --- Courbes ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def mm(x, w=500):
        return np.convolve(x, np.ones(w) / w, mode='valid')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(mm(H['ret'])); ax[0].set_title("Retour par épisode (moyenne glissante, avec exploration)")
    ax[0].set_xlabel("épisode")
    ax[1].plot(mm(H['violation']) * 100, label="violations (entraînement)")
    ax[1].plot(np.array(val_eps) - 1, np.array(val_viol) * 100, 'o-', label="tentatives de violation (validation gloutonne)")
    ax[1].set_title("Violations (%)"); ax[1].set_xlabel("épisode"); ax[1].legend()
    ax[2].plot(val_eps, val_r, 'o-'); ax[2].set_title("Validation gloutonne : r / r_max moyen"); ax[2].set_xlabel("épisode")
    ax[2].set_ylim(0, 1.05)
    for a_ in ax: a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, f'courbes_SARSA{suffix}.png'), dpi=150)
    print(f"\n✅ q_table{suffix}, visites_Q{suffix}, historique_training{suffix}, sarsa_config{suffix}, "
          f"courbes_SARSA{suffix}.png enregistrés dans le dossier du script.")


if __name__ == '__main__':
    main()
