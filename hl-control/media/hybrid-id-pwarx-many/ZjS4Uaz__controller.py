"""Identification of a piecewise-affine ARX system (4-6 regions) from one input-output record.

This is an identification procedure, not a feedback law: `controller(obs)` receives a measured record and a
query input and returns the free-run output of the query, the mode of every record sample and the number of modes.
Only least squares, linear separation, combinatorial/local search and a statistical test are used.

Model class (the one the systems are drawn from): an ordered list of hyperplane cuts in the (standardised)
regressor space, cut k moving the points of one existing region `parent` with h'z > b into a new region, and one
affine ARX submodel per region fitted by least squares.  A cut earlier in the list has priority.

Procedure
  1. Peeling (first guess of the cuts).  The family is a central region with caps cut off it in priority order,
     so the cap of highest priority is a whole half-space that a single affine model fits.  A hill climb over
     directions finds a half-space cap of `n0` samples with the smallest least-squares residual (a pure core);
     the cap is grown to the largest half-space one model still fits (minimise RSS - tau2 * sigma^2 * size over
     the offset, all sizes at once from cumulative normal equations, and over the direction by local search) and
     removed; repeat on the remaining samples.  Run for a few (n0, tau2) settings.
  2. Partition fit.  The caps become cuts of the centre; their priority order is chosen by RSS over all orders,
     then coordinate descent on the RSS: with the submodels fixed every cut is re-optimised as a cost-sensitive
     linear classification (cost = squared one-step residual of the region a sample would end in; exact 1-D scan
     of the offset, hinge directions and random local search for the normal), then the submodels are refitted.
  3. Extra cut.  The best two-way split of one region (Breiman's hinging-hyperplane iteration from random starts)
     is added: this is how the two regions with nearly equal parameters are separated.
  4. Structure search, for every number of regions K in the allowed range: swap moves (drop one cut, add the best
     new one, keep if the RSS drops) and consistency passes (the K-region model must be at least as good as the
     (K-1)-region model plus its best cut and as the (K+1)-region model minus its least useful cut).
  5. Model selection.  The number of regions is the smallest K after which one more region does not buy a
     significant RSS drop; a model with a region far below half an equal share of the record is rejected.
  6. Free run of the query: mean of the free runs of the best model of every K and of the other candidates nearly
     as good (averaging limits the damage of one misplaced cut); outputs are clipped to a range a little wider
     than the record's so that a badly fitted submodel cannot diverge.  Mode labels come from the selected model.

Tuning constants (PARAMS)
  n_start, n_pert      random starts of a split search; directions tried per step of a local search on a normal
  k_min, k_min_split   fewest samples on either side of a cut (cap cuts; extra cut), split_share: fewest as a
                       share of the region divided by the extra cut
  rounds               refinement rounds of the partition fit
  n_rest               samples the peeling always leaves in the centre
  core_starts, core_pert, n_grow, grow_pert   effort of the pure-core search and of growing the best cores
  core_corr            correction of the noise variance read off the best core (selection bias)
  restarts             number of (n0, tau2) peeling settings used
  swap_moves           maximum number of accepted swap moves per model
  gain_thr             RSS drop, in residual variances, one more region must buy to be accepted
  size_share           a model is rejected if a region holds less than this share of N / n_modes_max samples
  ens_max, ens_tol     free-run ensemble: at most ens_max models, each within ens_tol of the best RSS
  soft_time, hard_time shares of the time limit after which no new peeling run / no further structure search
                       is started (the answer is always returned well inside the limit)
"""

import itertools
import time

import numpy as np

LAG = 3
PARAMS = {"n_start": 30, "n_pert": 40, "k_min": 22, "k_min_split": 12, "split_share": 0.25, "rounds": 4,
          "n_rest": 25, "core_starts": 30, "core_pert": 8, "n_grow": 3, "grow_pert": 10, "core_corr": 1.3,
          "restarts": 3, "swap_moves": 6, "gain_thr": 55.0, "size_share": 0.42, "ens_max": 12, "ens_tol": 1.3,
          "soft_time": 0.3, "hard_time": 0.6}
_DEBUG = {}  # diagnostics of the last call, for development only; the procedure does not read it


def _regdata(y, u, na, nb):
    n = len(y)
    cols = [y[LAG - k:n - k] for k in range(1, na + 1)] + [u[LAG - k:n - k] for k in range(1, nb + 1)]
    return np.column_stack(cols), y[LAG:].copy()


def _ls(X, y):
    """Least squares with a tiny ridge; returns (theta, rss)."""
    A = X.T @ X
    A[np.diag_indices_from(A)] += 1e-9 * (np.trace(A) / len(A) + 1e-12)
    th = np.linalg.solve(A, X.T @ y)
    r = y - X @ th
    return th, float(r @ r)


def _scan_batch(P, g, kmin, kmax):
    """P: (n, m) projections on m directions, g: (n,) cost of moving each sample.  Best (value, direction, b)
    minimising the sum of g over {proj > b} with kmin <= #moved <= kmax."""
    order = np.argsort(-P, axis=0)
    cs = np.cumsum(g[order], axis=0)
    sub = cs[kmin - 1:kmax]
    r, j = np.unravel_index(int(np.argmin(sub)), sub.shape)
    k = r + kmin
    ps = P[order[:, j], j]
    b = 0.5 * (ps[k - 1] + ps[k])
    return float(sub[r, j]), int(j), float(b)


def _optimize_cut(Zs, g, h, b, cur, kmin, rng, n_pert, extra=()):
    """Local search for the half-space {h'z > b} minimising the sum of g inside it, from (h, b) of value cur."""
    n, d = Zs.shape
    best_v, best_h, best_b = cur, h, b
    if n < 2 * kmin:
        return best_h, best_b, best_v
    first = True
    for step in (0.5, 0.25, 0.12, 0.06, 0.03):
        H = best_h[None, :] + step * rng.standard_normal((n_pert, d))
        if first:
            H = np.vstack([best_h[None, :]] + [np.asarray(e)[None, :] for e in extra] + [H])
            first = False
        H /= np.linalg.norm(H, axis=1, keepdims=True) + 1e-12
        v, j, bb = _scan_batch(Zs @ H.T, g, kmin, n - kmin)
        if v < best_v - 1e-12:
            best_v, best_h, best_b = v, H[j].copy(), bb
    return best_h, best_b, best_v


def _hh_local(Xr, yr, h, b, kmin, iters=12):
    """Hinging-hyperplane iteration from the cut (h, b): fit both sides, re-cut where the two fits intersect
    (offset by an exact scan).  Returns the best (rss, h, b) met on the way."""
    n, d1 = Xr.shape
    d = d1 - 1
    Zr = Xr[:, :d]
    best = (np.inf, None, None)
    mask = (Zr @ h) > b
    for _it in range(iters):
        k = int(mask.sum())
        if k < kmin or n - k < kmin:
            break
        thA, rA = _ls(Xr[mask], yr[mask])
        thB, rB = _ls(Xr[~mask], yr[~mask])
        if rA + rB < best[0]:
            best = (rA + rB, h, b)
        delta = thA - thB
        nn = float(np.linalg.norm(delta[:d]))
        if nn < 1e-12:
            break
        hn = delta[:d] / nn
        g = (yr - Xr @ thA) ** 2 - (yr - Xr @ thB) ** 2
        p = Zr @ hn
        _, j, b = _scan_batch(np.column_stack([p, -p]), g, kmin, n - kmin)
        h = hn if j == 0 else -hn
        newmask = (Zr @ h) > b
        if np.array_equal(newmask, mask):
            break
        mask = newmask
    return best


def _split_search(Xr, yr, rng, n_start, kmin):
    """Best two-way split of a sample set by a hyperplane: hinging-hyperplane iteration from random starts."""
    n, d1 = Xr.shape
    d = d1 - 1
    best = (np.inf, None, None)
    if n < 2 * kmin:
        return best
    for _ in range(n_start):
        h = rng.standard_normal(d)
        h /= np.linalg.norm(h)
        proj = Xr[:, :d] @ h
        b = float(np.quantile(proj, rng.uniform(0.2, 0.8)))
        res = _hh_local(Xr, yr, h, b, kmin)
        if res[0] < best[0]:
            best = res
    return best


def _polish_split(Xr, yr, h, b, rng, kmin, n_pert, rounds=3):
    d = Xr.shape[1] - 1
    Zr = Xr[:, :d]
    rss = np.inf
    for _ in range(rounds):
        mask = (Zr @ h) > b
        thA, rA = _ls(Xr[mask], yr[mask])
        thB, rB = _ls(Xr[~mask], yr[~mask])
        rss = rA + rB
        g = (yr - Xr @ thA) ** 2 - (yr - Xr @ thB) ** 2
        cur = float(g[mask].sum())
        h2, b2, v2 = _optimize_cut(Zr, g, h, b, cur, kmin, rng, n_pert)
        if v2 >= cur - 1e-12:
            break
        h, b = h2, b2
        mask = (Zr @ h) > b
        _, rA = _ls(Xr[mask], yr[mask])
        _, rB = _ls(Xr[~mask], yr[~mask])
        rss = rA + rB
    return rss, h, b


def _assign(cuts, Z, labels=None):
    lab = np.zeros(len(Z), dtype=int) if labels is None else labels
    for (p, c, h, b) in cuts:
        lab[(lab == p) & (Z @ h > b)] = c
    return lab


def _fit_regions(X, y, labels, K):
    d1 = X.shape[1]
    thetas = np.zeros((K, d1))
    rss = np.zeros(K)
    for r in range(K):
        m = labels == r
        if m.sum() > 0:
            thetas[r], rss[r] = _ls(X[m], y[m])
    return thetas, rss


def _refine(cuts, Z, X, y, K, rng, rounds, n_pert, kmin):
    """Coordinate descent on the RSS over the cuts (submodels fixed), then refit; a few rounds."""
    d = Z.shape[1]
    cuts = list(cuts)
    for _ in range(rounds):
        labels = _assign(cuts, Z)
        thetas, _ = _fit_regions(X, y, labels, K)
        E2 = (y[:, None] - X @ thetas.T) ** 2
        changed = False
        for i in range(len(cuts)):
            p, c, h, b = cuts[i]
            before = _assign(cuts[:i], Z)
            S = np.flatnonzero(before == p)
            if len(S) < 2 * kmin:
                continue
            Zs = Z[S]
            stay = np.full(len(S), p)
            move = np.full(len(S), c)
            for (p2, c2, h2, b2) in cuts[i + 1:]:
                pr = Zs @ h2 > b2
                stay[(stay == p2) & pr] = c2
                move[(move == p2) & pr] = c2
            g = E2[S, move] - E2[S, stay]
            cur = float(g[Zs @ h > b].sum())
            delta = thetas[c, :d] - thetas[p, :d]
            nn = np.linalg.norm(delta)
            extra = (delta / nn, -delta / nn) if nn > 1e-12 else ()
            h2, b2, v2 = _optimize_cut(Zs, g, h, b, cur, kmin, rng, n_pert, extra)
            if v2 < cur - 1e-9:
                cuts[i] = (p, c, h2, b2)
                changed = True
        if not changed:
            break
    labels = _assign(cuts, Z)
    thetas, rss = _fit_regions(X, y, labels, K)
    return cuts, labels, thetas, float(rss.sum())


def _star(hinges, Z, X, y, rng, P):
    """Partition with the half-spaces as caps cut off a central region; the priority order of the caps (which cap
    owns an overlap) is chosen by RSS over all orders, then the cuts are refined."""
    m = len(hinges)
    K = m + 1
    d1 = X.shape[1]
    cuts = [(0, i + 1, h, b) for i, (h, b) in enumerate(hinges)]
    res = None
    for _pass in range(2):
        best = None
        for perm in itertools.permutations(range(m)):
            trial = [(0, i + 1, cuts[k][2], cuts[k][3]) for i, k in enumerate(perm)]
            lab = _assign(trial, Z)
            if np.bincount(lab, minlength=K).min() < d1 + 2:
                continue
            _, rr = _fit_regions(X, y, lab, K)
            v = float(rr.sum())
            if best is None or v < best[0] - 1e-9:
                best = (v, trial, perm)
        if best is not None:
            if _pass > 0 and best[2] == tuple(range(m)):
                break
            cuts = best[1]
        res = _refine(cuts, Z, X, y, K, rng, int(P["rounds"]), int(P["n_pert"]), int(P["k_min"]))
        cuts = res[0]
    return res


def _add_split(model, Z, X, y, K, rng, P, cache=None):
    """Best extra cut dividing one region of a (K-1)-region model into two comparable parts."""
    cuts, labels = model[0], model[1]
    _, rss_r = _fit_regions(X, y, labels, K - 1)
    best = None
    for r in range(K - 1):
        idx = np.flatnonzero(labels == r)
        kmin = max(int(P["k_min_split"]), int(P["split_share"] * len(idx)))
        if len(idx) < 2 * kmin:
            continue
        key = idx.tobytes()
        if cache is not None and key in cache:
            rs, h, b = cache[key]
        else:
            rs, h, b = _split_search(X[idx], y[idx], rng, int(P["n_start"]), kmin)
            if h is not None:
                rs, h, b = _polish_split(X[idx], y[idx], h, b, rng, kmin, int(P["n_pert"]))
            if cache is not None:
                cache[key] = (rs, h, b)
        if h is None:
            continue
        gain = rss_r[r] - rs
        if best is None or gain > best[0]:
            best = (gain, r, h, b)
    if best is None:
        return None
    _, r, h, b = best
    new = list(cuts) + [(r, K - 1, h, b)]
    return _refine(new, Z, X, y, K, rng, 2, int(P["n_pert"]), int(P["k_min_split"]))


def _remove_cut(cuts, i):
    """The cut list without cut i (its region merges back into its parent); None if that region was cut again."""
    c = cuts[i][1]
    if any(q[0] == c for q in cuts):
        return None
    return [(p2 - (p2 > c), c2 - (c2 > c), h2, b2) for j, (p2, c2, h2, b2) in enumerate(cuts) if j != i]


def _swap_search(model, Z, X, y, K, rng, P, cache):
    """Local search over the structure: drop one cut, add the best new one, keep the change if the RSS drops."""
    best = model
    for _ in range(int(P["swap_moves"])):
        if time.process_time() > P["t_hard"]:
            break
        improved = False
        for i in range(len(best[0])):
            red = _remove_cut(best[0], i)
            if red is None:
                continue
            res = _add_split((red, _assign(red, Z)), Z, X, y, K, rng, P, cache)
            if res is not None and res[3] < best[3] * (1.0 - 1e-3):
                best, improved = res, True
                break
        if not improved:
            break
    return best


def _prefix_rss(Xs, ys):
    """RSS of the least-squares fit on the first n rows, for every n (rows already ordered)."""
    d1 = Xs.shape[1]
    A = np.cumsum(Xs[:, :, None] * Xs[:, None, :], axis=0)
    A += 1e-8 * np.eye(d1)[None, :, :]
    c = np.cumsum(Xs * ys[:, None], axis=0)
    th = np.linalg.solve(A, c[:, :, None])[:, :, 0]
    return np.cumsum(ys * ys) - np.einsum("ni,ni->n", th, c)


def _core_search(Xw, yw, rng, n0, n_start, n_pert):
    """Half-space caps of n0 samples that one affine model fits best (hill climbing on the direction).
    Returns [(rss, h)] sorted by rss."""
    d = Xw.shape[1] - 1
    Zw = Xw[:, :d]

    def val(h):
        idx = np.argpartition(-(Zw @ h), n0 - 1)[:n0]
        return _ls(Xw[idx], yw[idx])[1]

    out = []
    for _ in range(n_start):
        h = rng.standard_normal(d)
        h /= np.linalg.norm(h)
        v = val(h)
        for step in (0.5, 0.35, 0.25, 0.15, 0.1, 0.05):
            for h2 in h[None, :] + step * rng.standard_normal((n_pert, d)):
                h2 = h2 / np.linalg.norm(h2)
                v2 = val(h2)
                if v2 < v:
                    h, v = h2, v2
        out.append((v, h))
    out.sort(key=lambda t: t[0])
    return out


def _grow(Xw, yw, h, lam, n_lo, n_hi, rng, n_pert):
    """Largest half-space {h'z > b} of the working set that one affine model fits: minimise RSS - lam * size
    over the offset (all sizes at once) and, by local search, over the direction."""
    d = Xw.shape[1] - 1
    Zw = Xw[:, :d]

    def val(h):
        p = Zw @ h
        order = np.argsort(-p)
        rss = _prefix_rss(Xw[order], yw[order])
        J = rss[n_lo - 1:n_hi] - lam * np.arange(n_lo, n_hi + 1)
        i = int(np.argmin(J))
        n = n_lo + i
        return float(J[i]), n, 0.5 * float(p[order[n - 1]] + p[order[n]])

    J, n, b = val(h)
    for step in (0.3, 0.2, 0.12, 0.07, 0.04):
        for h2 in h[None, :] + step * rng.standard_normal((n_pert, d)):
            h2 = h2 / np.linalg.norm(h2)
            J2, n2, b2 = val(h2)
            if J2 < J:
                J, n, b, h = J2, n2, b2, h2
    return J, h, b, n


def _peel(Z, X, y, rng, P, max_caps, n0, tau2):
    """Decision-list peeling: repeatedly find the largest half-space of the remaining samples that a single
    affine model fits (a cap) and remove it.  Returns the caps in priority order."""
    N, d1 = X.shape
    rest = int(P["n_rest"])
    W = np.arange(N)
    caps, sig2 = [], None
    while len(caps) < max_caps and len(W) >= n0 + rest:
        cores = _core_search(X[W], y[W], rng, n0, int(P["core_starts"]), int(P["core_pert"]))
        if sig2 is None:
            sig2 = P["core_corr"] * cores[0][0] / (n0 - d1)
        lam = tau2 * sig2
        best = None
        for v, h in cores[:int(P["n_grow"])]:
            res = _grow(X[W], y[W], h, lam, n0, len(W) - rest, rng, int(P["grow_pert"]))
            if best is None or res[0] < best[0]:
                best = res
        J, h, b, n = best
        if n >= len(W) - rest:
            break
        caps.append((h, b))
        W = W[(Z[W] @ h) <= b]
    return caps


def _build(Z, X, y, kmin_modes, kmax_modes, rng, P):
    """Several peeling runs (different core sizes and purity thresholds); for every number of regions K the
    candidates are: K-1 caps and a centre, or K-2 caps and a centre plus one extra cut dividing a region.  The
    candidate with the smallest RSS is kept for every K."""
    models, cache, pool = {}, {}, []
    settings = [(30, 4.0), (26, 3.0), (36, 5.0), (30, 6.0), (28, 4.0), (34, 3.5)][:int(P["restarts"])]
    for n0, tau2 in settings:
        if models and time.process_time() > P["t_soft"]:
            break
        caps = _peel(Z, X, y, rng, P, kmax_modes - 1, n0, tau2)
        stars = {}
        for M in range(max(1, kmin_modes - 2), len(caps) + 1):
            stars[M] = _star(caps[:M], Z, X, y, rng, P)
        for K in range(kmin_modes, kmax_modes + 1):
            cands = []
            if K - 1 in stars:
                cands.append(stars[K - 1])
            if K - 2 in stars:
                res = _add_split(stars[K - 2], Z, X, y, K, rng, P, cache)
                if res is not None:
                    cands.append(res)
            for c in cands:
                pool.append(c)
                if K not in models or c[3] < models[K][3]:
                    models[K] = c
    for K in list(models):
        models[K] = _swap_search(models[K], Z, X, y, K, rng, P, cache)
    # consistency of the sequence: a K-region model is at least as good as the (K-1)-region model plus its best
    # extra cut, and a K-region model at least as good as the (K+1)-region model without its least useful cut
    order = sorted(models)
    for K in order[1:]:
        if K - 1 in models and time.process_time() < P["t_hard"]:
            res = _add_split(models[K - 1], Z, X, y, K, rng, P, cache)
            if res is not None and res[3] < models[K][3]:
                models[K] = _swap_search(res, Z, X, y, K, rng, P, cache)
    for K in order[-2::-1]:
        if K + 1 not in models or time.process_time() > P["t_hard"]:
            continue
        best = None
        for i in range(len(models[K + 1][0])):
            red = _remove_cut(models[K + 1][0], i)
            if red is None:
                continue
            res = _refine(red, Z, X, y, K, rng, 2, int(P["n_pert"]), int(P["k_min_split"]))
            if best is None or res[3] < best[3]:
                best = res
        if best is not None and best[3] < models[K][3]:
            models[K] = _swap_search(best, Z, X, y, K, rng, P, cache)
    # models used together for the free run: the best of every K and the other candidates nearly as good
    best_rss = min(m[3] for m in models.values())
    ensemble = list(models.values())
    for c in sorted(pool, key=lambda c: c[3]):
        if len(ensemble) >= int(P["ens_max"]) or c[3] > P["ens_tol"] * best_rss:
            break
        ensemble.append(c)
    return models, ensemble


def _select(models, N, d, kmin_modes, kmax_modes, P):
    """Smallest K after which a further region does not buy a significant RSS drop (drop in units of the
    residual variance of the larger model).  A model with a region far below half an equal share of the
    record is not accepted (every mode of the family holds at least half an equal share)."""
    d1 = d + 1
    stats = {}
    for K in range(kmin_modes, kmax_modes):
        if K in models and (K + 1) in models:
            rss_a, rss_b = models[K][3], models[K + 1][3]
            dof = max(N - (K + 1) * d1 - K * d, 10)
            stat = (rss_a - rss_b) / (rss_b / dof)
            if np.bincount(models[K + 1][1], minlength=K + 1).min() < P["size_share"] * N / kmax_modes:
                stat = min(stat, 0.0)
            stats[K + 1] = stat
    K = kmin_modes
    while K < kmax_modes and stats.get(K + 1, -1.0) > P["gain_thr"]:
        K += 1
    return K, stats


def _simulate(cuts, thetas, mu, sd, u, y_init, na, nb, ylo, yhi):
    n = len(u)
    y = np.zeros(n)
    y[:LAG] = np.asarray(y_init, dtype=float)[:LAG]
    for t in range(LAG, n):
        phi = np.concatenate([y[t - na:t][::-1], u[t - nb:t][::-1]])
        z = (phi - mu) / sd
        r = 0
        for (p, c, h, b) in cuts:
            if r == p and float(z @ h) > b:
                r = c
        v = float(thetas[r, :-1] @ z + thetas[r, -1])
        y[t] = min(max(v, ylo), yhi)
    return y


def controller(obs: dict) -> dict:
    """Called once per hidden system; returns {'y_query', 'modes', 'n_modes'} (see the module docstring)."""
    t_start = time.process_time()
    na, nb = int(obs["na"]), int(obs["nb"])
    y = np.asarray(obs["y"], dtype=float)
    u = np.asarray(obs["u"], dtype=float)
    uq = np.asarray(obs["u_query"], dtype=float)
    kmin_modes, kmax_modes = int(obs["n_modes_min"]), int(obs["n_modes_max"])
    n_train, n_query = len(y), len(uq)
    fallback = {"y_query": np.full(n_query, float(np.mean(y))), "modes": np.zeros(n_train, dtype=int),
                "n_modes": kmin_modes}
    try:
        Phi, tgt = _regdata(y, u, na, nb)
        mu, sd = Phi.mean(axis=0), Phi.std(axis=0) + 1e-9
        Z = (Phi - mu) / sd
        X = np.column_stack([Z, np.ones(len(Z))])
        rng = np.random.default_rng(12345)
        P = dict(PARAMS)
        limit = float(obs.get("time_limit", 10.0))
        P["t_soft"], P["t_hard"] = t_start + P["soft_time"] * limit, t_start + P["hard_time"] * limit
        _DEBUG.clear()
        models, ensemble = _build(Z, X, tgt, kmin_modes, kmax_modes, rng, P)
        K, stats = _select(models, len(tgt), Z.shape[1], kmin_modes, kmax_modes, P)
        _DEBUG.update({"models": models, "stats": stats, "K": K, "Z": Z, "X": X, "tgt": tgt, "ensemble": ensemble})
        Kfit = K if K in models else max(models)
        cuts, labels, thetas, _ = models[Kfit]
        span = float(y.max() - y.min())
        lo, hi = float(y.min()) - 0.3 * span, float(y.max()) + 0.3 * span
        sims = [_simulate(m[0], m[2], mu, sd, uq, obs["y_query_init"], na, nb, lo, hi) for m in ensemble]
        yq = np.mean(sims, axis=0)
        modes = np.zeros(n_train, dtype=int)
        modes[LAG:] = labels
        if not np.all(np.isfinite(yq)):
            yq = fallback["y_query"]
        return {"y_query": yq, "modes": modes, "n_modes": int(K)}
    except Exception as _exc:
        _DEBUG["exc"] = repr(_exc)
        return fallback
