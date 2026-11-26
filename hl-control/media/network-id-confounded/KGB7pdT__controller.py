"""Identification of a linear dynamic network with unmeasured hubs: sparse node equations + structural EM.

This file is an identification procedure, not a feedback law: `controller(obs)` receives one measured record and
returns the direct links among the measured nodes and the simulated response to the query signals.

Model (the class of the task): every measured node j has  x_j(t) = a_j x_j(t-1..2) + sum_i b_ji w_i(t-l) + sum_u c_ju z_u(t-l) + e_j,
x_j = w_j - r_j, and every unmeasured node u is a latent signal  z_u(t) = a_u z_u(t-1..2) + sum_i b_ui w_i(t-l) + e_u.

Procedure (least squares, Gaussian likelihood, BIC-type selection; no neural networks):
  1. each measured node gets a sparse equation on the lagged signals of the other measured nodes (forward selection
     with a backward pass on the Gram matrix; a source enters with one lag and may get one adjacent lag);
  2. pairs of nodes with correlated equation residuals propose a latent node. Its signal is estimated by a smoother
     (banded normal equations of the Gaussian model; stationary posterior covariance), every equation is re-selected
     with the latent signal as a candidate regressor on the expected Gram matrix (EM), measured parents of the latent
     node are searched by likelihood ratio, and the latent node is kept if the BIC of the measured record drops.
     Latent nodes without parents model the innovation groups (same-instant noise correlation);
  3. clean-up moves, each accepted on the BIC of the measured record: a measured "parent" of a latent node at lag 0/1
     may be an early child acting as a proxy (flip), direct links to the children may belong to the latent node
     (transfer); then EM to convergence;
  4. the fitted full network (latent nodes included) is simulated from rest for the queries. The measured parents of
     the latent nodes have the largest leverage on the response, so their coefficients are weighted by the
     two-model posterior weight 1 / (1 + exp(dBIC / 2)) instead of a hard in/out decision;
  5. links: i -> j if w_i is a regressor of node j, or i is a parent of a latent node that is a regressor of j.

PARAMS (penalties are thresholds on -2 log-likelihood per coefficient, in the role of BIC's log N):
  pen_src   a new source node in a measured equation        pen_tap   a second, adjacent lag / the second AR lag
  pen_zsrc  a measured parent in a latent equation          pen_bic   penalty of the record-level BIC used to accept moves
  max_src   in-degree cap during selection                  force_ar2 latent nodes always get two AR lags
  corr_min  smallest residual correlation that proposes a latent node;  kmax / max_prop  caps on latent nodes / proposals
  em_iters, rounds, tr_iters, final_iters  EM iterations (per round, rounds per proposal, per trial move, at the end)
  score_min, count_bonus, max_try  candidates for a latent node's parents (score threshold, bonus per child that the
            candidate drives directly, trials per latent node);  max_soft  extra candidates in the soft weighting
  soft, flip, drop, prune_extra  switches of the soft parent weights and of the clean-up moves
  margin    required BIC drop for a new latent node;  cpu_budget  processor seconds for the search (the call stays
            well inside the 10 s limit; later stages have their own small allowances).
"""
import time
import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

LMAX = 4          # largest lag of a measured neighbour in a measured node's equation (delay 3 + second tap)
LW = 6            # lags of w kept in the design (a latent node's equation may use all of them)
LZ = 5            # lags of a latent signal kept (children of a latent node)
T0 = max(LW, LZ)


class Prep:
    """Lagged design of the measured signals and its Gram matrix (rows t = T0..N-1).

    Columns of D:  w_i(t-l) -> (l-1)*n+i (l = 1..LW);  x_j(t-k) -> LW*n+(k-1)*n+j (k = 1, 2);  x_j(t) -> (LW+2)*n+j
    (the targets);  w_i(t) -> (LW+3)*n+i (lag 0, used only in latent equations).
    Latent columns (appended by the E-step): z_u(t-l) -> nD + l*K + u  (l = 0..LZ; l = 0 is the latent node's own target)."""

    def __init__(self, w, r):
        self.w, self.r = w, r
        self.N, self.n = w.shape
        self.x = w - r
        self.rows = np.arange(T0, self.N)
        self.nr = len(self.rows)
        n = self.n
        self.D = np.hstack([w[self.rows - l] for l in range(1, LW + 1)] + [self.x[self.rows - k] for k in (1, 2)] + [self.x[self.rows]] + [w[self.rows]])
        self.M = self.D.T @ self.D
        self.nD = (LW + 4) * n

    def w0(self, i):
        return (LW + 3) * self.n + i

    def wl(self, i, l):
        return (l - 1) * self.n + i

    def xl(self, j, k):
        return LW * self.n + (k - 1) * self.n + j

    def xt(self, j):
        return (LW + 2) * self.n + j

    def zl(self, u, l, K):
        return self.nD + l * K + u


def select2(G, tgt, forced, cand, src_id, lag, nr, pen_src, pen_tap, max_src=4, max_add=12):
    """Forward selection with a backward pass on a Gram matrix, with the structure of the model class:
    a source (a node) enters with one lag (penalty pen_src) and may get one adjacent lag (penalty pen_tap);
    candidates with src_id < 0 are single terms (the second AR lag) that enter with pen_tap.
    Returns (active list, beta, sigma2)."""
    act = list(forced)
    cand = np.asarray(cand, dtype=int)
    src_id = np.asarray(src_id, dtype=int)
    lag = np.asarray(lag, dtype=int)
    yy = G[tgt, tgt]
    chosen = {}                                  # source -> list of (lag, column)
    pen_of = {}                                  # column -> penalty it entered with

    def fit(a):
        if not a:
            return np.zeros(0), yy, None
        Pm = np.linalg.inv(G[np.ix_(a, a)] + 1e-9 * np.eye(len(a)))
        b = Pm @ G[a, tgt]
        return b, max(yy - b @ G[a, tgt], 1e-12), Pm

    b, rss, Pm = fit(act)
    alive = np.ones(len(cand), dtype=bool)
    pen = np.where(src_id < 0, pen_tap, pen_src).astype(float)
    for _ in range(max_add):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        C = cand[idx]
        if act:
            GCA = G[np.ix_(C, act)]
            num = G[C, tgt] - GCA @ b
            den = G[C, C] - np.einsum("ij,jk,ik->i", GCA, Pm, GCA)
        else:
            num, den = G[C, tgt], G[C, C].copy()
        den = np.maximum(den, 1e-9 * np.maximum(G[C, C], 1e-12))
        delta = np.minimum(num ** 2 / den, rss * (1 - 1e-9))
        stat = nr * np.log(rss / (rss - delta))
        gain = stat - pen[idx]
        k = int(np.argmax(gain))
        if gain[k] < 0:
            break
        c = idx[k]
        act.append(int(cand[c]))
        pen_of[int(cand[c])] = float(pen[c])
        alive[c] = False
        s = src_id[c]
        if s >= 0:
            chosen.setdefault(s, []).append(int(lag[c]))
            same = src_id == s
            if len(chosen[s]) == 1:
                adj = same & (np.abs(lag - lag[c]) == 1)
                alive[same & ~adj] = False
                pen[adj] = pen_tap
                if len(chosen) >= max_src:
                    alive[(src_id >= 0) & ~np.isin(src_id, list(chosen))] = False
            else:
                alive[same] = False
        b, rss, Pm = fit(act)
    # backward pass: a term must be worth the penalty it entered with (a lone lag of a source: pen_src)
    while len(act) > len(forced):
        s2 = rss / nr
        t2 = b ** 2 / np.maximum(s2 * np.diag(Pm), 1e-300)
        need = np.array([0.0] * len(forced) + [pen_of[c] for c in act[len(forced):]])
        short = t2 - need
        short[: len(forced)] = np.inf
        k = int(np.argmin(short))
        if short[k] >= 0:
            break
        del act[k]
        b, rss, Pm = fit(act)
    return act, b, rss / nr


class Model:
    """act[q]: global column indices of equation q (q < n: measured node, q >= n: latent q-n); beta[q]; sig2[q]."""

    def __init__(self, n, K):
        self.n, self.K = n, K
        self.act = [[] for _ in range(n + K)]
        self.beta = [np.zeros(0) for _ in range(n + K)]
        self.sig2 = np.ones(n + K)


def estep(P, mod, need_ll=False):
    """Posterior mean of the latent signals (N, K), stationary posterior covariance of [z_u(t-l)] (S order, index l*K+u)
    and, on request, the marginal log-likelihood of the measured record (up to a constant)."""
    n, K, nD, nr = mod.n, mod.K, P.nD, P.nr
    nS = (LZ + 1) * K
    N = P.N
    rows_s, rows_d, wts = [], [], []
    ll_acc = 0.0
    for q in range(n + K):
        act = np.asarray(mod.act[q], dtype=int)
        b = mod.beta[q]
        isz = act >= nD
        if q < n and not isz.any():
            if need_ll:
                a = list(act)
                rss = P.M[P.xt(q), P.xt(q)] - 2.0 * b @ P.M[a, P.xt(q)] + b @ P.M[np.ix_(a, a)] @ b
                ll_acc += nr * np.log(mod.sig2[q]) + rss / mod.sig2[q]
            continue
        s = np.zeros(nS)
        zi = act[isz] - nD
        loc = (LZ - zi // K) * K + zi % K                 # time-ordered local index of z_u(t-l)
        am = act[~isz]
        if q < n:
            s[loc] = b[isz]
            d = P.D[:, P.xt(q)] - (P.D[:, am] @ b[~isz] if len(am) else 0.0)
        else:
            s[loc] = -b[isz]
            s[LZ * K + (q - n)] = 1.0
            d = P.D[:, am] @ b[~isz] if len(am) else np.zeros(nr)
        rows_s.append(s); rows_d.append(d); wts.append(1.0 / mod.sig2[q])
        if need_ll:
            ll_acc += nr * np.log(mod.sig2[q]) + float(d @ d) / mod.sig2[q]
    Sm = np.array(rows_s)
    wS = Sm * np.array(wts)[:, None]
    Bk = Sm.T @ wS
    H = np.column_stack(rows_d) @ wS
    # banded normal equations: the block Bk repeats at every row t (shift K per sample)
    bw = nS - 1
    pp, kk = _band_index(nS)
    Bpad = np.zeros((nS, 2 * nS))
    Bpad[:, :nS] = Bk
    cs = Bpad[pp, pp + kk].reshape(nS, LZ + 1, K).cumsum(axis=1)       # [k, m, rho]
    full = cs[:, -1, :]
    V = np.empty((nS, nr + LZ, K))
    V[:] = full[:, None, :]
    V[:, :LZ, :] = cs[:, :LZ, :]
    V[:, nr:, :] = full[:, None, :] - cs[:, :LZ, :]
    V = V.reshape(nS, -1)
    ab = np.zeros((bw + 1, N * K))
    off = (T0 - LZ) * K
    for k in range(nS):
        m = min(V.shape[1], N * K - off - k)
        ab[bw - k, off + k: off + k + m] = V[k, :m]
    ab[bw] += 1e-5
    h = np.zeros((N, K))
    for l in range(LZ + 1):
        h[P.rows - l] += H[:, (LZ - l) * K:(LZ - l + 1) * K]
    cb = cholesky_banded(ab, check_finite=False)
    z = cho_solve_banded((cb, False), h.reshape(-1), check_finite=False).reshape(N, K)
    ll = 0.0
    if need_ll:
        ll = -0.5 * (ll_acc - float(h.reshape(-1) @ z.reshape(-1)) + 2.0 * float(np.sum(np.log(cb[-1]))))
        # latent samples before the first equation that involves them are held by the ridge alone: take them out of the determinant
        used = (np.abs(np.diag(Bk)).reshape(LZ + 1, K) > 0)          # time-ordered: row m <-> lag LZ - m
        first = np.array([np.argmax(used[:, u]) if used[:, u].any() else LZ for u in range(K)])
        ll += 0.5 * float(np.sum(T0 - LZ + first)) * np.log(1e-5)
    # stationary posterior covariance from a window of the (block-Toeplitz) precision matrix
    nt = 4 * (LZ + 1) + 1
    Lw = np.zeros((nt * K, nt * K))
    for t in range(nt - LZ):
        Lw[t * K:t * K + nS, t * K:t * K + nS] += Bk
    Lw[np.diag_indices_from(Lw)] += 1e-5
    Cw = np.linalg.inv(Lw)
    c = ((nt - (LZ + 1)) // 2) * K
    Ct = Cw[c:c + nS, c:c + nS]
    perm = np.array([(LZ - l) * K + u for l in range(LZ + 1) for u in range(K)])
    Cz = Ct[np.ix_(perm, perm)]
    return z, Cz, ll


_BAND_CACHE = {}


def _band_index(nS):
    if nS not in _BAND_CACHE:
        kk, pp = np.meshgrid(np.arange(nS), np.arange(nS), indexing="ij")
        _BAND_CACHE[nS] = (pp, kk)
    return _BAND_CACHE[nS]


def ext_gram(P, z, Cz):
    K = z.shape[1]
    S = np.hstack([z[P.rows - l] for l in range(LZ + 1)])
    DS = P.D.T @ S
    G = np.block([[P.M, DS], [DS.T, S.T @ S + P.nr * Cz]])
    return G


def mstep(P, mod, G):
    """Least squares of every equation that involves a latent signal, on the expected Gram matrix."""
    n, K, nr = mod.n, mod.K, P.nr
    for q in range(n + K):
        act = mod.act[q]
        if q < n and (not act or max(act) < P.nD) and len(mod.beta[q]) == len(act):
            continue                                   # no latent regressor: the fit does not change
        tgt = P.xt(q) if q < n else P.zl(q - n, 0, K)
        if not act:
            mod.beta[q] = np.zeros(0)
            mod.sig2[q] = max(G[tgt, tgt] / nr, 1e-8)
            continue
        Gaa = G[np.ix_(act, act)]
        Gaa = Gaa + 1e-9 * np.trace(Gaa) / len(act) * np.eye(len(act))
        b = np.linalg.solve(Gaa, G[act, tgt])
        mod.beta[q] = b
        mod.sig2[q] = max((G[tgt, tgt] - b @ G[act, tgt]) / nr, 1e-8)


def run_em(P, mod, iters=30):
    for _ in range(iters):
        z, Cz, _ = estep(P, mod)
        G = ext_gram(P, z, Cz)
        mstep(P, mod, G)
    return z, Cz


def bic(P, mod, pen):
    """-2 log marginal likelihood of the measured record + pen per coefficient."""
    if mod.K == 0:
        ll = 0.0
        for q in range(mod.n):
            a, b = list(mod.act[q]), mod.beta[q]
            rss = P.M[P.xt(q), P.xt(q)] - 2.0 * b @ P.M[a, P.xt(q)] + b @ P.M[np.ix_(a, a)] @ b
            ll += -0.5 * (P.nr * np.log(mod.sig2[q]) + rss / mod.sig2[q])
    else:
        ll = estep(P, mod, need_ll=True)[2]
    return -2.0 * ll + pen * sum(len(a) for a in mod.act)


def decode(P, mod):
    """Lag matrices of the fitted full network: Aown (2, n) on x_j(t-k); Wc (LS+1, n+K, n+K) on v(t-l), l = 0..LS, v = [w, z]."""
    n, K, nD = mod.n, mod.K, P.nD
    LS = max(LW, LZ)
    Aown = np.zeros((2, n))
    Wc = np.zeros((LS + 1, n + K, n + K))
    for q in range(n + K):
        for c, b in zip(mod.act[q], mod.beta[q]):
            if c < LW * n:
                l, i = c // n + 1, c % n
                Wc[l, q, i] += b
            elif c < (LW + 2) * n:
                k, j = (c - LW * n) // n + 1, (c - LW * n) % n
                Aown[k - 1, j] += b
            elif c >= nD:
                l, u = (c - nD) // K, (c - nD) % K
                Wc[l, q, n + u] += b
            elif c >= (LW + 3) * n:
                Wc[0, q, c - (LW + 3) * n] += b
    return Aown, Wc


def simulate_queries(Aown, Wc, n, K, qnode, qsig):
    """Noise-free response from rest; lag-0 terms (a latent aligned with its parent or child) are solved algebraically."""
    Q, T = qsig.shape
    nt = n + K
    LS = Wc.shape[0] - 1
    inv0 = np.linalg.inv(np.eye(nt) - Wc[0])
    v = np.zeros((Q, T + LS, nt))
    x = np.zeros((Q, T + LS, n))
    qi = np.arange(Q)
    for t in range(T):
        acc = np.zeros((Q, nt))
        for l in range(1, LS + 1):
            acc += v[:, t + LS - l] @ Wc[l].T
        acc[:, :n] += Aown[0] * x[:, t + LS - 1] + Aown[1] * x[:, t + LS - 2]
        acc[qi, qnode] += qsig[:, t]
        vt = acc @ inv0.T
        np.clip(vt, -1e3, 1e3, out=vt)
        v[:, t + LS] = vt
        x[:, t + LS] = vt[:, :n]
        x[qi, t + LS, qnode] -= qsig[:, t]
    return v[:, LS:, :n]


def links_of(P, mod):
    n, K, nD = mod.n, mod.K, P.nD
    par = [set() for _ in range(n + K)]
    for q in range(n + K):
        for c in mod.act[q]:
            if c < LW * n:
                par[q].add(c % n)
            elif (LW + 3) * n <= c < nD:
                par[q].add(c - (LW + 3) * n)
            elif c >= nD:
                u = (c - nD) % K
                if q < n:
                    par[q].add(n + u)
    out = set()
    for j in range(n):
        for p in par[j]:
            if p < n:
                if p != j:
                    out.add((p, j))
            else:
                for i in par[p]:
                    if i < n and i != j:
                        out.add((i, j))
    return sorted(out)


LPAR = 4          # largest lag of a measured parent in a latent node's equation that the search may use
PARAMS = {"pen_src": 14.0, "pen_tap": 4.0, "pen_zsrc": 10.0, "max_src": 4, "pen_bic": 12.0, "force_ar2": 1, "prune_extra": -1.0, "max_soft": 2, "soft": 1, "flip": 1, "drop": 0, "corr_min": 0.15, "kmax": 12, "max_prop": 22, "margin": 0.0, "cpu_budget": 3.5, "em_iters": 6, "rounds": 2, "final_iters": 40, "tr_iters": 10, "score_min": 5.0, "count_bonus": 3.0, "max_try": 2}


def cand_measured(P, j, K):
    """Candidate columns of a measured node's equation with their source ids and lags."""
    n = P.n
    cols, sid, lag = [P.xl(j, 2)], [-1], [2]
    for i in range(n):
        if i != j:
            for l in range(1, LMAX + 1):
                cols.append(P.wl(i, l)); sid.append(i); lag.append(l)
    for u in range(K):
        for l in range(1, LZ + 1):
            cols.append(P.zl(u, l, K)); sid.append(n + u); lag.append(l)
    return cols, sid, lag


def cand_latent(P, u, K):
    n = P.n
    cols, sid, lag = [P.zl(u, 2, K)], [-1], [2]
    for i in range(n):
        for l in range(0, LPAR + 1):
            cols.append(P.w0(i) if l == 0 else P.wl(i, l)); sid.append(i); lag.append(l)
    return cols, sid, lag


def reselect(P, mod, G, params, nodes=None):
    n, K = mod.n, mod.K
    for q in (range(n + K) if nodes is None else nodes):
        if q < n:
            cols, sid, lag = cand_measured(P, q, K)
            act, b, s2 = select2(G, P.xt(q), [P.xl(q, 1)], cols, sid, lag, P.nr, params["pen_src"], params["pen_tap"], params["max_src"])
        else:
            u = q - n
            cols, sid, lag = cand_latent(P, u, K)
            forced = [P.zl(u, 1, K), P.zl(u, 2, K)] if params["force_ar2"] else [P.zl(u, 1, K)]
            act, b, s2 = select2(G, P.zl(u, 0, K), forced, cols, sid, lag, P.nr, params["pen_zsrc"], params["pen_tap"], 3)
        mod.act[q], mod.beta[q], mod.sig2[q] = act, b, max(s2, 1e-8)


def residuals(P, mod, z):
    n, K = mod.n, mod.K
    S = np.hstack([z[P.rows - l] for l in range(LZ + 1)]) if K else np.zeros((P.nr, 0))
    DS = np.hstack([P.D, S])
    E = np.zeros((P.nr, n))
    for j in range(n):
        E[:, j] = P.D[:, P.xt(j)] - DS[:, mod.act[j]] @ mod.beta[j]
    return E


def lat_parents(P, mod, u):
    n = mod.n
    return {c % n for c in mod.act[n + u] if c < LW * n} | {c - (LW + 3) * n for c in mod.act[n + u] if (LW + 3) * n <= c < P.nD}


def children(P, mod, u):
    n, K = mod.n, mod.K
    return [j for j in range(n) if any(c >= P.nD and (c - P.nD) % K == u for c in mod.act[j])]


def remap(P, mod, keep):
    """Drop latents not in `keep` (list of old indices) and renumber the latent columns."""
    n, K = mod.n, mod.K
    new = Model(n, len(keep))
    pos = {u: k for k, u in enumerate(keep)}
    for q in list(range(n)) + [n + u for u in keep]:
        act, beta = [], []
        for c, b in zip(mod.act[q], mod.beta[q]):
            if c >= P.nD:
                l, u = (c - P.nD) // K, (c - P.nD) % K
                if u not in pos:
                    continue
                c = P.nD + l * len(keep) + pos[u]
            act.append(c); beta.append(b)
        qn = q if q < n else n + pos[q - n]
        new.act[qn], new.beta[qn], new.sig2[qn] = act, np.array(beta), mod.sig2[q]
        if q < n and len(act) < len(mod.act[q]) and (not act or max(act) < P.nD):
            refit_plain(P, new, q)
    return new


def refit_plain(P, mod, q):
    """Least squares of a measured equation without latent regressors."""
    act, tgt = mod.act[q], P.xt(q)
    if act:
        b = np.linalg.solve(P.M[np.ix_(act, act)] + 1e-9 * np.eye(len(act)), P.M[act, tgt])
        mod.beta[q] = b
        mod.sig2[q] = max((P.M[tgt, tgt] - b @ P.M[act, tgt]) / P.nr, 1e-8)
    else:
        mod.beta[q] = np.zeros(0)
        mod.sig2[q] = max(P.M[tgt, tgt] / P.nr, 1e-8)


def add_latent(P, mod, z, Cz, zinit, kids):
    """New model with one more latent (signal guess zinit (N,), kids = [(node, lag)])."""
    n, K = mod.n, mod.K
    new = Model(n, K + 1)
    for q in range(n + K):
        act = []
        for c in mod.act[q]:
            if c >= P.nD:
                l, u = (c - P.nD) // K, (c - P.nD) % K
                c = P.nD + l * (K + 1) + u
            act.append(c)
        new.act[q], new.beta[q], new.sig2[q] = act, mod.beta[q].copy(), mod.sig2[q]
    for j, l in kids:
        new.act[j] = new.act[j] + [P.zl(K, l, K + 1)]
    new.act[n + K] = [P.zl(K, 1, K + 1), P.zl(K, 2, K + 1)]
    znew = np.column_stack([z, zinit]) if K else zinit[:, None]
    nS_old, nS = (LZ + 1) * K, (LZ + 1) * (K + 1)
    Cn = np.zeros((nS, nS))
    old_idx = np.array([l * (K + 1) + u for l in range(LZ + 1) for u in range(K)], dtype=int)
    if K:
        Cn[np.ix_(old_idx, old_idx)] = Cz
    new_idx = np.array([l * (K + 1) + K for l in range(LZ + 1)])
    Cn[new_idx, new_idx] = 0.3 * np.var(zinit)
    G = ext_gram(P, znew, Cn)
    mstep(P, new, G)
    return new


def copy_model(mod):
    new = Model(mod.n, mod.K)
    new.act = [list(a) for a in mod.act]
    new.beta = [b.copy() for b in mod.beta]
    new.sig2 = mod.sig2.copy()
    return new


def parent_scores(P, mod, G, u):
    """Score of every measured node as a parent of latent u: best single-lag likelihood-ratio statistic and its lag."""
    n, K, nr = mod.n, mod.K, P.nr
    act, tgt = mod.act[n + u], P.zl(u, 0, K)
    Pm = np.linalg.inv(G[np.ix_(act, act)] + 1e-9 * np.eye(len(act)))
    b = Pm @ G[act, tgt]
    rss = max(G[tgt, tgt] - b @ G[act, tgt], 1e-12)
    cols = np.array([[P.w0(i)] + [P.wl(i, l) for l in range(1, LPAR + 1)] for i in range(n)])      # (n, LPAR+1)
    C = cols.ravel()
    GCA = G[np.ix_(C, act)]
    num = G[C, tgt] - GCA @ b
    den = np.maximum(G[C, C] - np.einsum("ij,jk,ik->i", GCA, Pm, GCA), 1e-9 * G[C, C])
    d = np.minimum(num ** 2 / den, rss * (1 - 1e-9))
    stat = (nr * np.log(rss / (rss - d))).reshape(n, LPAR + 1)
    return stat


def transfer_parents(P, mod, z, Cz, params, verbose=False, only=None, deadline=None):
    """Search measured parents of the latent nodes: a candidate i (scored on the expected Gram matrix, or a node that
    drives the children directly) gets lags in the latent equation, its direct links to the children are dropped,
    EM is run, the equations concerned are re-selected, and the change is kept if the BIC drops."""
    n = mod.n
    base = bic(P, mod, params["pen_bic"])
    for u in (range(mod.K) if only is None else only):
        if deadline is not None and time.process_time() > deadline:
            break
        K = mod.K
        kids = children(P, mod, u)
        have = lat_parents(P, mod, u)
        stat = parent_scores(P, mod, ext_gram(P, z, Cz), u)
        best = stat.max(axis=1)
        count = np.zeros(n)
        for j in kids:
            for i in {c % n for c in mod.act[j] if c < LW * n}:
                count[i] += 1
        merit = best + params["count_bonus"] * count
        order = [int(i) for i in np.argsort(-merit) if merit[i] >= params["score_min"] and i not in have]
        for i in order[: params["max_try"]]:
            new = copy_model(mod)
            l0 = int(np.argmax(stat[i]))
            lags = [l for l in (l0 - 1, l0, l0 + 1) if 0 <= l <= LPAR]
            cols = [P.w0(i) if l == 0 else P.wl(i, l) for l in lags]
            cols = [c for c in cols if c not in new.act[n + u]]
            new.act[n + u] = new.act[n + u] + cols
            new.beta[n + u] = np.concatenate([new.beta[n + u], np.zeros(len(cols))])
            for j in kids:
                keep = [k for k, c in enumerate(new.act[j]) if not (c < LW * n and c % n == i)]
                new.act[j] = [new.act[j][k] for k in keep]
                new.beta[j] = new.beta[j][keep]
            zc, Cc = run_em(P, new, iters=params["tr_iters"])
            Gc = ext_gram(P, zc, Cc)
            reselect(P, new, Gc, params, nodes=kids + [n + u])
            zc, Cc = run_em(P, new, iters=params["em_iters"])
            val = bic(P, new, params["pen_bic"])
            if verbose:
                print("    parent", i, "-> latent", u, "score", round(float(best[i]), 1), "bic", round(base, 1), "->", round(val, 1))
            if val < base - 1e-6:
                mod, z, Cz, base = new, zc, Cc, val
    return mod, z, Cz


def prune_parents(P, mod, z, Cz, params, deadline=None):
    """Backward check of the measured parents of every latent node: a parent stays only if removing it costs more
    than (pen_bic + prune_extra) per coefficient in -2 log-likelihood of the measured record."""
    n = mod.n
    pen = params["pen_bic"] + params["prune_extra"]
    base = bic(P, mod, pen)
    for u in range(mod.K):
        for i in sorted(lat_parents(P, mod, u)):
            if deadline is not None and time.process_time() > deadline:
                return mod, z, Cz
            new = copy_model(mod)
            keep = [k for k, c in enumerate(new.act[n + u]) if not ((c < LW * n and c % n == i) or c == P.w0(i))]
            new.act[n + u] = [new.act[n + u][k] for k in keep]
            new.beta[n + u] = new.beta[n + u][keep]
            zc, Cc = run_em(P, new, iters=params["tr_iters"])
            val = bic(P, new, pen)
            if val < base:
                mod, z, Cz, base = new, zc, Cc, val
    return mod, z, Cz


def soft_parents(P, mod, z, Cz, params, deadline=None):
    """Soft decisions on the measured parents of the latent nodes (the terms with the largest leverage on the response).

    For a parent in the model, and for the best-scoring candidates outside it, the BIC difference D between the model
    with and without that parent gives the weight 1 / (1 + exp(D / 2)) (model averaging over the two models); the
    parent's coefficients enter the simulated network multiplied by that weight.
    Returns (scale {(u, i): weight}, extra [(u, i, columns, weighted coefficients, weight)])."""
    n = mod.n
    pen = params["pen_bic"]
    base = bic(P, mod, pen)
    scale, extra = {}, []
    G = ext_gram(P, z, Cz)
    for u in range(mod.K):
        have = lat_parents(P, mod, u)
        for i in sorted(have):
            if deadline is not None and time.process_time() > deadline:
                return scale, extra
            new = copy_model(mod)
            keep = [k for k, c in enumerate(new.act[n + u]) if not ((c < LW * n and c % n == i) or c == P.w0(i))]
            new.act[n + u] = [new.act[n + u][k] for k in keep]
            new.beta[n + u] = new.beta[n + u][keep]
            run_em(P, new, iters=params["tr_iters"])
            d = base - bic(P, new, pen)                         # BIC(with) - BIC(without)
            scale[(u, i)] = float(1.0 / (1.0 + np.exp(np.clip(d / 2.0, -50, 50))))
        stat = parent_scores(P, mod, G, u)
        best = stat.max(axis=1)
        cands = [int(i) for i in np.argsort(-best) if i not in have and best[i] >= params["score_min"]][: params["max_soft"]]
        for i in cands:
            if deadline is not None and time.process_time() > deadline:
                return scale, extra
            new = copy_model(mod)
            l0 = int(np.argmax(stat[i]))
            nb = [l for l in (l0 - 1, l0 + 1) if 0 <= l <= LPAR]
            lags = [l0] + ([max(nb, key=lambda l: stat[i, l])] if nb else [])
            cols = [P.w0(i) if l == 0 else P.wl(i, l) for l in lags]
            new.act[n + u] = new.act[n + u] + cols
            new.beta[n + u] = np.concatenate([new.beta[n + u], np.zeros(len(cols))])
            run_em(P, new, iters=params["tr_iters"])
            d = bic(P, new, pen) - base
            wgt = float(1.0 / (1.0 + np.exp(np.clip(d / 2.0, -50, 50))))
            if wgt > 0.1:
                extra.append((u, i, cols, new.beta[n + u][-len(cols):] * wgt, wgt))
    return scale, extra


def apply_soft(P, mod, Wc, scale, extra):
    n = mod.n
    for (u, i), wgt in scale.items():
        Wc[:, n + u, i] *= wgt
    for u, i, cols, coefs, wgt in extra:
        for c, b in zip(cols, coefs):
            l = 0 if c >= (LW + 3) * n else c // n + 1
            Wc[l, n + u, i] += b
    return Wc


def soft_links(P, mod, scale, extra):
    """Links of the immersed network with the soft parent decisions (a parent counts if its weight is at least 1/2)."""
    n, K, nD = mod.n, mod.K, P.nD
    par = [set() for _ in range(n + K)]
    for q in range(n + K):
        for c in mod.act[q]:
            if c < LW * n:
                par[q].add(c % n)
            elif (LW + 3) * n <= c < nD:
                par[q].add(c - (LW + 3) * n)
            elif c >= nD and q < n:
                par[q].add(n + (c - nD) % K)
    for (u, i), wgt in scale.items():
        if wgt < 0.5:
            par[n + u].discard(i)
    for u, i, cols, coefs, wgt in extra:
        if wgt >= 0.5:
            par[n + u].add(i)
    out = set()
    for j in range(n):
        for p in par[j]:
            if p < n:
                if p != j:
                    out.add((p, j))
            else:
                out.update((i, j) for i in par[p] if i != j)
    return sorted(out)


def flip_parents(P, mod, z, Cz, params, deadline=None, verbose=False):
    """A measured node that enters a latent equation at lag 0 or 1 may be an early child of that latent node acting as
    its proxy: try the reverse orientation (the node becomes a child) and keep it if the BIC drops."""
    n = mod.n
    base = bic(P, mod, params["pen_bic"])
    for u in range(mod.K):
        K = mod.K
        kids = children(P, mod, u)
        for i in sorted(lat_parents(P, mod, u)):
            if deadline is not None and time.process_time() > deadline:
                return mod, z, Cz
            lags_i = [0 if c == P.w0(i) else c // n + 1 for c in mod.act[n + u] if c == P.w0(i) or (c < LW * n and c % n == i)]
            if i in kids or min(lags_i) > 1:
                continue
            new = copy_model(mod)
            keep = [k for k, c in enumerate(new.act[n + u]) if not ((c < LW * n and c % n == i) or c == P.w0(i))]
            new.act[n + u] = [new.act[n + u][k] for k in keep]
            new.beta[n + u] = new.beta[n + u][keep]
            cols = [P.zl(u, l, K) for l in (1, 2)]
            new.act[i] = new.act[i] + cols
            new.beta[i] = np.concatenate([new.beta[i], np.zeros(len(cols))])
            zc, Cc = run_em(P, new, iters=params["tr_iters"])
            Gc = ext_gram(P, zc, Cc)
            reselect(P, new, Gc, params, nodes=[i, n + u])
            zc, Cc = run_em(P, new, iters=params["em_iters"])
            val = bic(P, new, params["pen_bic"])
            if verbose:
                print("    flip", i, "of latent", u, "bic", round(base, 1), "->", round(val, 1))
            if val < base - 1e-6:
                mod, z, Cz, base = new, zc, Cc, val
    return mod, z, Cz


def drop_latents(P, mod, z, Cz, params, deadline=None, verbose=False):
    """Backward check of the latent nodes: one of two latent nodes that describe the same hub (or a latent node made
    redundant by later changes) is removed if the BIC of the measured record drops without it."""
    n = mod.n
    base = bic(P, mod, params["pen_bic"])
    u = 0
    while u < mod.K and mod.K > 1:
        if deadline is not None and time.process_time() > deadline:
            break
        kids = children(P, mod, u)
        new = remap(P, mod, [v for v in range(mod.K) if v != u])
        zc, Cc = run_em(P, new, iters=params["em_iters"])
        Gc = ext_gram(P, zc, Cc)
        reselect(P, new, Gc, params, nodes=kids)
        zc, Cc = run_em(P, new, iters=params["tr_iters"])
        val = bic(P, new, params["pen_bic"])
        if verbose:
            print("    drop latent", u, "kids", kids, "bic", round(base, 1), "->", round(val, 1))
        if val < base - 1e-6:
            mod, z, Cz, base = new, zc, Cc, val
        else:
            u += 1
    return mod, z, Cz


def _propose(P, mod, z, Cz, E, j, k, s, sg, params, verbose):
    """Candidate model with a new latent node suggested by the residual pair (j, k) at shift s; None if it is not viable."""
    n = P.n
    zi = np.zeros(P.N)
    common = np.zeros(P.nr)
    common[: P.nr - s] = 0.5 * (E[: P.nr - s, j] + sg * E[s:, k])
    zi[P.rows[0] - 2: P.N - 2] = common                 # z(t-2) = common(t): room for an earlier child at lag 1
    base_mod = copy_model(mod)
    for a, b_ in ((j, k), (k, j)):                       # a sibling is a proxy of the latent signal: drop it, selection may re-add it
        keep = [m for m, c in enumerate(base_mod.act[a]) if not (c < LW * n and c % n == b_)]
        base_mod.act[a] = [base_mod.act[a][m] for m in keep]
        base_mod.beta[a] = base_mod.beta[a][keep]
    cand = add_latent(P, base_mod, z, Cz, zi, [(j, 2), (k, 2 + s)])
    K1 = cand.K
    zc, Cc = run_em(P, cand, iters=params["em_iters"])
    for _ in range(params["rounds"]):
        Gc = ext_gram(P, zc, Cc)
        reselect(P, cand, Gc, params)
        zc, Cc = run_em(P, cand, iters=params["em_iters"])
    kids = children(P, cand, K1 - 1)
    if verbose:
        print("  proposal", (j, k, s), "-> kids", kids, "parents", sorted(lat_parents(P, cand, K1 - 1)))
    if len(kids) < 2:
        return None
    keep = [u for u in range(K1) if len(children(P, cand, u)) >= 2]
    if len(keep) < K1:
        cand = remap(P, cand, keep)
        zc, Cc = run_em(P, cand, iters=params["em_iters"])
    cand, zc, Cc = transfer_parents(P, cand, zc, Cc, params, verbose, only=[cand.K - 1])
    return cand, zc, Cc, bic(P, cand, params["pen_bic"])


def identify(w, r, params=PARAMS, deadline=None, verbose=False):
    """Structure and coefficients of the network behind one record. Returns (Prep, Model, latent signals, their covariance).
    Every search move is guarded: a numerical failure discards that move, not the model found so far."""
    P = Prep(w, r)
    n = P.n
    mod = Model(n, 0)
    reselect(P, mod, P.M, params)
    P.mod0 = copy_model(mod)                               # measured-only fit: the fallback of the caller
    z = np.zeros((P.N, 0)); Cz = np.zeros((0, 0))
    tried = set()
    base = bic(P, mod, params["pen_bic"])
    n_prop = 0
    while mod.K < params["kmax"] and n_prop < params["max_prop"]:
        if deadline is not None and time.process_time() > deadline:
            break
        n_prop += 1
        E = residuals(P, mod, z)
        E = E / np.maximum(E.std(0), 1e-9)
        best = (0.0, None)
        for s in range(0, 3):
            R = E[: P.nr - s].T @ E[s:] / (P.nr - s)      # R[j, k] = corr(e_j(t), e_k(t+s))
            np.fill_diagonal(R, 0.0)
            if s == 0:
                R = np.triu(R)
            for (j, k, s_) in tried:
                if s_ == s:
                    R[j, k] = 0.0
            idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
            if abs(R[idx]) > best[0]:
                best = (abs(R[idx]), (int(idx[0]), int(idx[1]), s, float(np.sign(R[idx]))))
        if best[1] is None or best[0] < params["corr_min"]:
            break
        j, k, s, sg = best[1]
        tried.add((j, k, s))
        try:
            out = _propose(P, mod, z, Cz, E, j, k, s, sg, params, verbose)
        except Exception:
            out = None
        if out is not None and np.isfinite(out[3]) and out[3] < base - params["margin"]:
            mod, z, Cz, base = out
    if mod.K:
        late = (lambda extra: None) if deadline is None else (lambda extra: deadline + extra)
        moves = []
        if params["flip"]:
            moves.append(lambda m, a, b: flip_parents(P, m, a, b, params, deadline=late(0.5), verbose=verbose))
        if params["drop"]:
            moves.append(lambda m, a, b: drop_latents(P, m, a, b, params, deadline=late(0.8), verbose=verbose))
        moves.append(lambda m, a, b: transfer_parents(P, m, a, b, params, verbose, deadline=late(1.0)))
        if params["prune_extra"] >= 0:
            moves.append(lambda m, a, b: prune_parents(P, m, a, b, params, deadline=late(1.8)))

        def polish(m, a, b):
            m = copy_model(m)
            for _ in range(2):
                reselect(P, m, ext_gram(P, a, b), params)
                a, b = run_em(P, m, iters=params["em_iters"])
            a, b = run_em(P, m, iters=params["final_iters"])
            return m, a, b

        moves.append(polish)
        for move in moves:
            try:
                m2, z2, C2 = move(mod, z, Cz)
                if np.all(np.isfinite(z2)) and np.all(np.isfinite(C2)):
                    mod, z, Cz = m2, z2, C2
            except Exception:
                pass
    return P, mod, z, Cz


def _fallback(obs):
    out = np.zeros((int(obs["n_queries"]), int(obs["query_length"]), int(obs["n_measured"])))
    for q, node in enumerate(np.asarray(obs["query_node"], dtype=int)):
        out[q, :, node] = np.asarray(obs["query_signal"], dtype=float)[q]
    return {"links": [], "response": out}


def _answer(P, mod, obs, z=None, Cz=None, soft_deadline=None):
    Aown, Wc = decode(P, mod)
    links = links_of(P, mod)
    if z is not None and PARAMS["soft"] and mod.K:
        try:
            scale, extra = soft_parents(P, mod, z, Cz, PARAMS, deadline=soft_deadline)
            Wc = apply_soft(P, mod, Wc, scale, extra)
            links = soft_links(P, mod, scale, extra)
        except Exception:
            Aown, Wc = decode(P, mod)
            links = links_of(P, mod)
    resp = simulate_queries(Aown, Wc, P.n, mod.K, np.asarray(obs["query_node"], dtype=int),
                            np.asarray(obs["query_signal"], dtype=float))
    if not np.all(np.isfinite(resp)) or np.max(np.abs(resp)) > 100.0:
        raise FloatingPointError("unstable fit")
    return {"links": [[int(a), int(b)] for a, b in links], "response": resp}


def controller(obs: dict) -> dict:
    """Identify the hidden network from one record; return its direct links and its response to the queries.
    Fallbacks if a stage fails or the fit is unstable: the measured-only sparse fit, then the trivial predictor."""
    start = time.process_time()
    P = None
    try:
        w = np.asarray(obs["w"], dtype=float)
        r = np.asarray(obs["r"], dtype=float)
        budget = min(PARAMS["cpu_budget"], 0.4 * float(obs.get("time_limit_s", 10.0)))
        P, mod, z, Cz = identify(w, r, params=PARAMS, deadline=start + budget)
        return _answer(P, mod, obs, z, Cz, soft_deadline=start + budget + 2.5)
    except Exception:
        pass
    try:
        if P is None:
            P = Prep(np.asarray(obs["w"], dtype=float), np.asarray(obs["r"], dtype=float))
            mod0 = Model(P.n, 0)
            reselect(P, mod0, P.M, PARAMS)
        else:
            mod0 = P.mod0
        return _answer(P, mod0, obs)
    except Exception:
        return _fallback(obs)
