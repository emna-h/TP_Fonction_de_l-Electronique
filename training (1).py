# =============================================================================
#  training.py -- SARSA tabulaire : réduire au maximum le SSL d'un réseau circulaire à 5 anneaux
#                 en ACTIVANT / DÉSACTIVANT des antennes, en partant d'un réseau PLEIN.
#
#  Cahier des charges (version 3) :
#    * départ : réseau plein (toutes les antennes actives), nombre d'antennes quelconque par anneau
#    * objectif : minimiser le SSL (niveau du plus haut lobe secondaire, en dB : plus bas = meilleur)
#    * gain final dans la fenêtre [25 dB ; 27 dB]  (G = 20 log10(N_actives)  ->  18 <= N_actives <= 22)
#    * HPBW final dans l'intervalle « best » : [(1-0,10).H_ref ; (1+0,10).H_ref], H_ref = HPBW du réseau plein
#    * direction du lobe principal : |theta_pic - theta0| <= 1 deg
#
#  Pas de dataset : l'agent apprend en jouant dans un simulateur (réseaux pleins tirés au hasard à chaque épisode).
#  Sorties : q_table.npy/.mat, visites_Q.npy, historique_training.npy, sarsa_config.npz, courbes_SARSA.png
#  Exemple : python training.py --episodes 15000
# =============================================================================
import argparse
import os
import time

import numpy as np
import scipy.io
from scipy.signal import find_peaks

C0 = 3e8
F0 = 2.45e9
LAMBDA = C0 / F0
K_WAVE = 2 * np.pi / LAMBDA
R0 = 0.2 * LAMBDA            # rayon de l'anneau 1
DR = 0.5 * LAMBDA            # pas entre anneaux
THETA = np.linspace(0, 2 * np.pi, 1000)
THETA_DEG = np.rad2deg(THETA)
EPS_MACH = np.finfo(float).eps
N_RINGS = 5

# --- 6 actions (chaque action = un CRITÈRE de choix de l'antenne à changer) + STOP
OFF_SSL, OFF_SAFE, OFF_HPBW, ON_SSL, ON_HPBW, STOP = 0, 1, 2, 3, 4, 5
N_ACTIONS = 6
ACTION_NAMES = ['DÉSACTIVER (SSL min.)', 'DÉSACTIVER (SSL min., HPBW ok)', 'DÉSACTIVER (HPBW le plus proche)',
                'RÉACTIVER (SSL min.)', 'RÉACTIVER (HPBW le plus proche)', 'STOP']
# --- Observation : niveau de réduction (6) x statut HPBW (4) x amélioration déjà obtenue (5) x meilleure baisse offerte (3)
N_OBS = 6 * 4 * 5 * 3


def clear_console():
    print("\n" + "=" * 80 + "\n" + " " * 30 + "NOUVELLE EXÉCUTION\n" + "=" * 80 + "\n")


# =============================================================================
#  1. MODÈLE PHYSIQUE (identique à calcul_AF_performance_metrics du projet)
# =============================================================================
def layout(arch):
    a, phi, ring = [], [], []
    for m, N in enumerate(arch):
        N = int(N)
        for n in range(N):
            a.append(R0 + DR * m)
            phi.append(2 * np.pi * n / N)
            ring.append(m)
    return np.array(a), np.array(phi), np.array(ring)


def contributions(a, phi, theta0_deg):
    th0 = np.deg2rad(theta0_deg)
    phase = K_WAVE * a[None, :] * (np.sin(THETA)[:, None] * np.cos(0 - phi)[None, :]
                                   - np.sin(th0) * np.cos(0 - phi)[None, :])
    return np.exp(1j * phase)


def metrics_from_AF(AF):
    """(gain dB non normalisé, SSL dB, HPBW deg, direction du maximum deg)."""
    mag = np.abs(AF)
    mx = mag.max()
    n = mag / (mx + EPS_MACH)
    dB = 20 * np.log10(n + EPS_MACH)
    dB[dB < -40] = -40
    gain = 20 * np.log10(mx + EPS_MACH)
    half = dB.max() - 3
    ext = np.concatenate((dB, dB, dB))
    th_ext = np.concatenate((THETA_DEG - 360, THETA_DEG, THETA_DEG + 360))
    imax = np.argmax(dB) + len(THETA_DEG)
    left = np.where(ext[:imax] <= half)[0]
    right = np.where(ext[imax:] <= half)[0]
    hpbw = 180.0 if (len(left) == 0 or len(right) == 0) else th_ext[right[0] + imax] - th_ext[left[-1]]
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
    """Écart de direction (deux lobes principaux symétriques : theta0 et 180 - theta0)."""
    return min(abs(peak_deg - theta0_deg), abs(peak_deg - (180.0 - theta0_deg)))


# =============================================================================
#  2. LE RÉSEAU ET SES CONTRAINTES
# =============================================================================
DEFAULT_SPEC = dict(g_lo=25.0, g_hi=27.0, h_rel=0.10, eps_theta=1.0)


class Reseau:
    """Réseau plein + cahier des charges (spec). La liste `removed` (antennes éteintes) décrit une configuration."""

    def __init__(self, arch, theta0_deg, spec=DEFAULT_SPEC):
        self.arch = np.array(arch).astype(int)
        self.theta0 = float(theta0_deg)
        self.spec = spec
        self.a, self.phi, self.ring = layout(self.arch)
        self.n = len(self.a)
        self.E = contributions(self.a, self.phi, self.theta0)
        self.AF_full = self.E.sum(axis=1)
        self.ref = metrics_from_AF(self.AF_full)                          # (G, SSL, HPBW, pic) du réseau plein
        self.h_lo = (1 - spec['h_rel']) * self.ref[2]
        self.h_hi = (1 + spec['h_rel']) * self.ref[2]
        self.n_lo = int(np.ceil(10 ** (spec['g_lo'] / 20) - 1e-9))        # 25 dB -> 18 antennes
        self.n_hi = int(np.floor(10 ** (spec['g_hi'] / 20) + 1e-9))       # 27 dB -> 22 antennes
        self.cache = {}

    def metrics(self, removed):
        key = frozenset(removed)
        if key not in self.cache:
            AF = self.AF_full - self.E[:, list(key)].sum(axis=1) if key else self.AF_full
            self.cache[key] = metrics_from_AF(AF)
        return self.cache[key]

    def hpbw_ok(self, m):
        return self.h_lo <= m[2] <= self.h_hi

    def admissible(self, m):
        """Configuration finale admissible : gain dans la fenêtre, HPBW dans l'intervalle, theta0 respecté."""
        return (self.spec['g_lo'] - 1e-9 <= m[0] <= self.spec['g_hi'] + 1e-9 and self.hpbw_ok(m)
                and delta_theta(m[3], self.theta0) <= self.spec['eps_theta'])

    def feasible_start(self):
        return self.n >= self.n_lo


# =============================================================================
#  3. ENVIRONNEMENT
# =============================================================================
class Env:
    """Un épisode : on part du réseau PLEIN et on change l'état d'antennes jusqu'à STOP."""

    def __init__(self, reseau, step_cost=0.0, penalty=5.0, extra_steps=10):
        self.rs = reseau
        self.step_cost = step_cost
        self.penalty = penalty
        self.max_steps = max(12, (reseau.n - reseau.n_hi) + extra_steps)
        self.reset()

    def reset(self):
        self.removed = set()
        self.steps = 0
        self.best_ssl, self.best_removed = np.inf, None
        self._refresh()
        return self.observation()

    @property
    def n_act(self):
        return self.rs.n - len(self.removed)

    def _refresh(self):
        rs = self.rs
        self.cur = rs.metrics(self.removed)
        if rs.admissible(self.cur) and self.cur[1] < self.best_ssl:
            self.best_ssl, self.best_removed = self.cur[1], tuple(sorted(self.removed))
        self.off = []                                              # (SSL, |HPBW - H_ref|, HPBW ok, élément)
        if self.n_act - 1 >= rs.n_lo:
            for e in range(rs.n):
                if e not in self.removed:
                    m = rs.metrics(self.removed | {e})
                    self.off.append((m[1], abs(m[2] - rs.ref[2]), rs.hpbw_ok(m), e))
        self.on = []
        if self.removed and self.n_act + 1 <= rs.n_hi:
            for e in self.removed:
                m = rs.metrics(self.removed - {e})
                self.on.append((m[1], abs(m[2] - rs.ref[2]), rs.hpbw_ok(m), e))
        # choix de l'antenne pour chaque action
        self.choice = {}
        if self.off:
            self.choice[OFF_SSL] = min(self.off)[3]
            safe = [c for c in self.off if c[2]]
            if safe:
                self.choice[OFF_SAFE] = min(safe)[3]
            self.choice[OFF_HPBW] = min(self.off, key=lambda c: (c[1], c[0]))[3]
        if self.on:
            self.choice[ON_SSL] = min(self.on)[3]
            self.choice[ON_HPBW] = min(self.on, key=lambda c: (c[1], c[0]))[3]

    def observation(self):
        rs = self.rs
        forced = self.n_act - rs.n_hi
        if forced > 6: lv = 0
        elif forced >= 3: lv = 1
        elif forced >= 1: lv = 2
        elif self.n_act >= rs.n_hi - 1 or self.n_act > rs.n_lo + 2: lv = 3
        elif self.n_act > rs.n_lo: lv = 4
        else: lv = 5
        h = self.cur[2]
        hs = 0 if h < rs.h_lo else (1 if h <= rs.ref[2] else (2 if h <= rs.h_hi else 3))
        cum = rs.ref[1] - self.cur[1]
        qc = 0 if cum <= 0 else (1 if cum < 3 else (2 if cum < 6 else (3 if cum < 9 else 4)))
        key = OFF_SAFE if OFF_SAFE in self.choice else OFF_SSL
        if key in self.choice:
            m = rs.metrics(self.removed | {self.choice[key]})
            d = self.cur[1] - m[1]
        else:
            d = 0.0
        bd = 0 if d <= 0 else (1 if d < 1 else 2)
        return ((lv * 4 + hs) * 5 + qc) * 3 + bd

    def valid_actions(self):
        acts = [a for a in (OFF_SSL, OFF_SAFE, OFF_HPBW, ON_SSL, ON_HPBW) if a in self.choice]
        if self.n_act <= self.rs.n_hi:
            acts.append(STOP)                                       # STOP seulement si le gain est dans la fenêtre
        return acts

    def step(self, action):
        """Récompense = baisse de SSL (dB) - petit coût ; pénalité terminale si la configuration finale n'est pas admissible."""
        rs = self.rs
        before = self.cur[1]
        if action == STOP:
            r = 0.0 if rs.admissible(self.cur) else -self.penalty
            return r, True, 'stop'
        e = self.choice[action]
        if action in (ON_SSL, ON_HPBW):
            self.removed.remove(e)
        else:
            self.removed.add(e)
        self.steps += 1
        self._refresh()
        r = (before - self.cur[1]) - self.step_cost
        if self.steps >= self.max_steps:
            if not rs.admissible(self.cur):
                r -= self.penalty
            return r, True, 'max_pas'
        return r, False, 'ok'


# =============================================================================
#  4. AGENT SARSA TABULAIRE
# =============================================================================
def choose_action(Q, o, valid, eps, rng, tie='random'):
    if eps > 0 and rng.random() < eps:
        return int(valid[rng.integers(len(valid))])
    q = np.array([Q[o, a] for a in valid])
    best = np.flatnonzero(q == q.max())
    return int(valid[best[rng.integers(len(best))]]) if tie == 'random' else int(valid[best[0]])


def run_episode(env, Q, visits, alpha, gamma, eps, rng, learn=True, tie='random'):
    o = env.reset()
    a = choose_action(Q, o, env.valid_actions(), eps, rng, tie)
    ret, actions, end = 0.0, [], ''
    while True:
        actions.append(a)
        r, done, end = env.step(a)
        ret += r
        if done:
            target = r
        else:
            o2 = env.observation()
            a2 = choose_action(Q, o2, env.valid_actions(), eps, rng, tie)
            target = r + gamma * Q[o2, a2]
        if learn:
            Q[o, a] += alpha * (target - Q[o, a])                  # mise à jour SARSA
            visits[o, a] += 1
        if done:
            break
        o, a = o2, a2
    ok = env.best_removed is not None
    return dict(ret=ret, ssl_ref=env.rs.ref[1], ok=ok, ssl_best=env.best_ssl if ok else np.nan,
                removed_best=env.best_removed, actions=actions, end=end, final_ok=env.rs.admissible(env.cur))


def random_arch(rng, n_min, n_max, n_tot_min):
    while True:
        arch = rng.integers(n_min, n_max + 1, size=N_RINGS)
        if arch.sum() >= n_tot_min:
            return arch


def random_reseaux(rng, n, spec, n_min=1, n_max=10, n_tot_min=18, th_lo=5.0, th_hi=175.0):
    out = []
    while len(out) < n:
        rs = Reseau(random_arch(rng, n_min, n_max, n_tot_min), rng.uniform(th_lo, th_hi), spec)
        if rs.feasible_start():
            out.append(rs)
    return out


def evaluate_greedy(reseaux, Q):
    rng = np.random.default_rng(0)
    return [run_episode(Env(rs), Q, None, 0.0, 1.0, 0.0, rng, learn=False, tie='first') for rs in reseaux]


def main():
    clear_console()
    p = argparse.ArgumentParser()
    p.add_argument('--g_lo', type=float, default=25.0); p.add_argument('--g_hi', type=float, default=27.0)
    p.add_argument('--h_rel', type=float, default=0.10, help="demi-largeur relative de l'intervalle HPBW autour de H_ref")
    p.add_argument('--eps_theta', type=float, default=1.0)
    p.add_argument('--n_min', type=int, default=1); p.add_argument('--n_max', type=int, default=10)
    p.add_argument('--n_tot_min', type=int, default=18)
    p.add_argument('--episodes', type=int, default=15000)
    p.add_argument('--alpha', type=float, default=0.1); p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--eps_start', type=float, default=1.0); p.add_argument('--eps_min', type=float, default=0.02)
    p.add_argument('--step_cost', type=float, default=0.0); p.add_argument('--penalty', type=float, default=5.0)
    p.add_argument('--seed', type=int, default=46); p.add_argument('--eval_every', type=int, default=500)
    p.add_argument('--n_val', type=int, default=30); p.add_argument('--tag', type=str, default='')
    args = p.parse_args()
    suffix = f'_{args.tag}' if args.tag else ''
    d = os.path.dirname(os.path.abspath(__file__))
    spec = dict(g_lo=args.g_lo, g_hi=args.g_hi, h_rel=args.h_rel, eps_theta=args.eps_theta)
    rng = np.random.default_rng(args.seed)
    eps_decay = (args.eps_min / args.eps_start) ** (1.0 / (0.6 * args.episodes))
    val = random_reseaux(np.random.default_rng(args.seed + 1), args.n_val, spec, args.n_min, args.n_max, args.n_tot_min)
    print(f"Cahier des charges : gain {spec['g_lo']}-{spec['g_hi']} dB, HPBW dans H_ref ± {100*spec['h_rel']:.0f} %, theta0 ± {spec['eps_theta']} deg")
    Q = np.zeros((N_OBS, N_ACTIONS)); visits = np.zeros((N_OBS, N_ACTIONS), dtype=np.int64)
    H = dict(ret=np.zeros(args.episodes), red=np.full(args.episodes, np.nan), ok=np.zeros(args.episodes), eps=np.zeros(args.episodes))
    val_eps, val_red, val_ok = [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        eps = max(args.eps_min, args.eps_start * eps_decay ** ep)
        rs = Reseau(random_arch(rng, args.n_min, args.n_max, args.n_tot_min), rng.uniform(5, 175), spec)
        if not rs.feasible_start():
            continue
        res = run_episode(Env(rs, args.step_cost, args.penalty), Q, visits, args.alpha, args.gamma, eps, rng)
        H['ret'][ep], H['ok'][ep], H['eps'][ep] = res['ret'], float(res['ok']), eps
        if res['ok']:
            H['red'][ep] = res['ssl_ref'] - res['ssl_best']
        if (ep + 1) % args.eval_every == 0:
            v = evaluate_greedy(val, Q)
            okv = [x for x in v if x['ok']]
            val_eps.append(ep + 1); val_ok.append(len(okv) / len(v))
            val_red.append(np.mean([x['ssl_ref'] - x['ssl_best'] for x in okv]) if okv else np.nan)
            w = slice(max(0, ep - 499), ep + 1)
            print(f"ep {ep+1:6d} | eps {eps:.3f} | entraînement : admissible {100*H['ok'][w].mean():5.1f} % | validation gloutonne : "
                  f"admissible {100*val_ok[-1]:5.1f} %, baisse SSL {val_red[-1]:.2f} dB | {time.time()-t0:5.0f} s", flush=True)
    np.save(os.path.join(d, f'q_table{suffix}.npy'), Q)
    scipy.io.savemat(os.path.join(d, f'q_table{suffix}.mat'), {'q_table': Q})
    np.save(os.path.join(d, f'visites_Q{suffix}.npy'), visits)
    np.save(os.path.join(d, f'historique_training{suffix}.npy'),
            dict(retour=H['ret'], baisse_ssl=H['red'], admissible=H['ok'], epsilon=H['eps'], val_episodes=np.array(val_eps),
                 val_baisse_ssl=np.array(val_red), val_admissible=np.array(val_ok)), allow_pickle=True)
    np.savez(os.path.join(d, f'sarsa_config{suffix}.npz'), g_lo=spec['g_lo'], g_hi=spec['g_hi'], h_rel=spec['h_rel'],
             eps_theta=spec['eps_theta'], alpha=args.alpha, gamma=args.gamma, episodes=args.episodes, seed=args.seed,
             step_cost=args.step_cost, penalty=args.penalty, n_min=args.n_min, n_max=args.n_max, n_tot_min=args.n_tot_min)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mm = lambda x, w=300: np.convolve(np.nan_to_num(x, nan=0.0), np.ones(w) / w, mode='valid')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(mm(H['ok']) * 100); ax[0].set_title("Épisodes admissibles (%, moyenne glissante)"); ax[0].set_xlabel("épisode")
    ax[1].plot(val_eps, val_red, 'o-'); ax[1].set_title("Validation : baisse moyenne du SSL (dB)"); ax[1].set_xlabel("épisode")
    ax[2].plot(val_eps, 100 * np.array(val_ok), 'o-'); ax[2].set_title("Validation : % de réseaux avec solution admissible"); ax[2].set_xlabel("épisode")
    for a_ in ax: a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(d, f'courbes_SARSA{suffix}.png'), dpi=150)
    print(f"\n✅ q_table{suffix}, visites_Q{suffix}, historique_training{suffix}, sarsa_config{suffix}, courbes_SARSA{suffix}.png enregistrés.")


if __name__ == '__main__':
    main()
