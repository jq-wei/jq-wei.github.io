"""Identification of a piecewise-affine ARX system (4-6 regions, one pair with nearly equal parameters).

`controller(obs)` is an identification routine (nothing is controlled); it is called once per hidden system.

Model class: y(t) = theta_i' [z(t); 1] on region i, where z is the standardised regressor and the regions are a
decision list of half-spaces h'z > b (caps cut off a central region, as the generator does) plus one further
cut that halves one region (the "close pair").

Procedure
1. Several candidate identifications are computed (CANDIDATES: two pipelines, a few settings / random seeds);
   the answer whose partition of the record has the lowest BIC (least squares per mode) is returned.
2. Pipeline "peel" (`identify`):
   a. noise variance from the purest leaf of a greedy likelihood partitioning, refined after a first pass;
   b. the largest pure cap of the remainder is peeled off repeatedly: half-space maximising
      (number of samples - their residual sum of squares / (c_pure * noise variance)); for a direction the best
      threshold is exact (scan over sorted projections with cumulative normal equations), directions come from
      many starts, a random local search and the discriminant between explained / unexplained samples;
   c. refinement: robust least squares per region <-> every cut re-optimised by weighted linear
      classification with the others fixed, and swaps of the priority of neighbouring caps;
   d. consolidation: cuts whose removal costs less than tau_rm noise variances, and regions below min_region
      samples, are removed; regions closer than sep_same (RMS prediction difference / noise std) are one mode;
   e. the family has exactly one close pair: if no balanced pair of regions closer than sep_close exists, the
      largest regions are searched for a split near their median (likelihood-ratio gain) and the best is taken.
3. Pipeline "kplane" (`_identify_kplane`): leaves of the likelihood partitioning -> k-plane regression with
   split-and-merge moves -> models closer than sep_close merged -> decision list fitted to the labels by
   weighted linear classification -> refinement -> the close-pair split as above.
4. Least squares per mode and a free run of the query with the identified partition.

Gains (PARAMS)
  tau_split, tau_extra   likelihood-ratio gain to split a region (first split of the k-plane pipeline / further)
  sep_close, sep_gray    separation below which two models are the close pair; above sep_gray a split is forced
  sep_same               separation below which two regions are the same mode
  nmin, nrest            minimum samples of a cap / of what remains after a cut;  max_leaves, max_caps: limits
  n_polish*, n_class, n_top, n_refine   effort of the local searches (steps, starts kept, refinement rounds)
  split_q                the close-pair cut is searched between the split_q and 1 - split_q quantiles
  c_pure, c_stop         admitted variance of a cap / of the final remainder, in noise variances
  noise_q                quantile of the regions' residual variances used as the noise variance
  kappa                  bisquare scale (noise std) of the robust fits in the refinement (0: plain least squares)
  tau_rm, min_region     consolidation thresholds;  pair_balance, big_region: size rules for the close pair
  nmin_move, n_polish_move, move_rounds   split-and-merge moves of the k-plane pipeline
  bic_weight             weight of the BIC penalty in the selection;  cpu_budget: seconds after which no further
                         candidate is started (the time limit is 10 s per system)
"""

import numpy as np

from hl_control.envs.hybrid_id import LAG, regression_data

PARAMS = {
    "tau_split": 45.0,
    "tau_extra": 70.0,
    "sep_close": 5.6,
    "sep_gray": 6.4,
    "nmin": 22,
    "max_leaves": 9,
    "n_polish": 24,
    "n_polish_close": 60,
    "n_class": 150,
    "n_refine": 4,
    "split_q": 0.3,
    "nmin_move": 15,
    "n_polish_move": 8,
    "move_rounds": 3,
    "bic_weight": 1.0,
    "cpu_budget": 5.0,
    "kappa": 4.0,
    "tau_rm": 50.0,
    "noise_q": 35.0,
    "min_region": 16,
    "pair_balance": 0.5,
    "big_region": 1.3,
    "sep_same": 2.2,
    "c_pure": 4.5,
    "c_stop": 3.5,
    "nrest": 12,
    "max_caps": 8,
    "n_polish_cap": 30,
    "n_top": 6,
}

_RIDGE = 1e-8


def _scan(Z, X, y, h, nmin, q=0.0):
    """Best threshold along h: two affine models with separate variances.  Returns
    (gain, b, theta_low, theta_high, var_low, var_high) or None."""
    n, p0 = X.shape
    proj = Z @ h
    order = np.argsort(proj, kind="stable")
    Xo, yo, po = X[order], y[order], proj[order]
    lo, hi = max(nmin, int(np.ceil(q * n))) - 1, min(n - nmin, int(np.floor((1.0 - q) * n))) - 1
    if hi < lo:
        return None
    A = np.cumsum(Xo[:, :, None] * Xo[:, None, :], axis=0)
    g = np.cumsum(Xo * yo[:, None], axis=0)
    yy = np.cumsum(yo * yo)
    eye = np.eye(p0)
    At, gt, yt = A[-1], g[-1], yy[-1]
    sol_t = np.linalg.solve(At + _RIDGE * (1 + np.trace(At)) * eye, gt)
    sse_t = max(float(yt - gt @ sol_t), 1e-12)
    AL, gL, yL = A[lo:hi + 1], g[lo:hi + 1], yy[lo:hi + 1]
    AR, gR, yR = At - AL, gt - gL, yt - yL
    solL = np.linalg.solve(AL + (_RIDGE * (1 + np.trace(AL, axis1=1, axis2=2)))[:, None, None] * eye, gL[:, :, None])[:, :, 0]
    solR = np.linalg.solve(AR + (_RIDGE * (1 + np.trace(AR, axis1=1, axis2=2)))[:, None, None] * eye, gR[:, :, None])[:, :, 0]
    nL = np.arange(lo + 1, hi + 2).astype(float)
    nR = n - nL
    vL = np.maximum(yL - np.einsum("ni,ni->n", gL, solL), 1e-12) / (nL - p0)
    vR = np.maximum(yR - np.einsum("ni,ni->n", gR, solR), 1e-12) / (nR - p0)
    gain = n * np.log(sse_t / (n - p0)) - nL * np.log(vL) - nR * np.log(vR)
    gap = po[lo + 1:hi + 2] - po[lo:hi + 1]
    gain = np.where(gap > 1e-12, gain, -np.inf)
    j = int(np.argmax(gain))
    if not np.isfinite(gain[j]):
        return None
    i = lo + j
    return float(gain[j]), float(0.5 * (po[i] + po[i + 1])), solL[j], solR[j], float(vL[j]), float(vR[j])


def _find_splits(Z, X, y, nmin, rng, n_polish, n_keep, q=0.0):
    """Candidate cuts (gain, h, b) of one region: distinct local optima of the likelihood gain, best first."""
    n, d = Z.shape
    if n < 2 * nmin + 2:
        return []

    def scan(h):
        h = h / (np.linalg.norm(h) + 1e-12)
        return h, _scan(Z, X, y, h, nmin, q)

    def proposals(res):
        """Directions suggested by the fitted pair of models: the hinge normal, and the discriminant between the
        samples that the purer model explains and those it does not."""
        _, b, thL, thR, vL, vR = res
        out = [(thR - thL)[:d]]
        th, v = (thL, vL) if vL <= vR else (thR, vR)
        e = y - X @ th
        inl = np.abs(e) < 2.5 * np.sqrt(v)
        if nmin <= inl.sum() <= n - nmin:
            m1, m0 = Z[inl].mean(axis=0), Z[~inl].mean(axis=0)
            Zc = np.where(inl[:, None], Z - m1, Z - m0)
            out.append(np.linalg.solve(Zc.T @ Zc / n + 1e-6 * np.eye(d), m1 - m0))
        return [v for v in out if np.linalg.norm(v) > 1e-12]

    scored = []
    for h0 in list(np.eye(d)) + [rng.standard_normal(d) for _ in range(3 * d)]:
        h, res = scan(h0)
        if res is not None:
            scored.append((res[0], h, res))
    scored.sort(key=lambda t: -t[0])
    found = []
    for _, h, res in scored[:5]:
        step = 0.35
        for t in range(n_polish):
            trial = proposals(res) if t % 6 == 0 else []
            trial.append(h + step * rng.standard_normal(d))
            better = False
            for hn in trial:
                hn, r2 = scan(hn)
                if r2 is not None and r2[0] > res[0]:
                    h, res, better = hn, r2, True
            if not better:
                step *= 0.92
        side = Z @ h > res[1]
        found.append((res[0], h, res[1], side))
    found.sort(key=lambda t: -t[0])
    keep = []
    for c in found:
        if all(0.12 < np.mean(c[3] == k[3]) < 0.88 for k in keep):
            keep.append(c)
        if len(keep) >= n_keep:
            break
    return [(c[0], c[1], c[2]) for c in keep]


def _best_halfspace(Zr, w, inits, rng, n_polish, min_in=6):
    """Half-space {h'z > b} maximising the sum of the weights w inside (weighted linear classification):
    exact threshold for a direction, discriminant starts and a random local search over directions."""
    n, d = Zr.shape
    if n < 2 * min_in + 2:
        return -np.inf, None, 0.0

    def ev(h):
        p = Zr @ h
        o = np.argsort(-p, kind="stable")
        cs = np.cumsum(w[o])
        k = int(np.argmax(cs[min_in - 1:n - min_in])) + min_in - 1
        return float(cs[k]), float(0.5 * (p[o[k]] + p[o[k + 1]]))

    starts = [np.asarray(h, dtype=float) for h in inits]
    pos, a = w > 0, np.minimum(np.abs(w), np.percentile(np.abs(w), 75) + 1e-12)
    if pos.sum() >= 2 and (~pos).sum() >= 2:
        m1 = (a[pos, None] * Zr[pos]).sum(axis=0) / a[pos].sum()
        m0 = (a[~pos, None] * Zr[~pos]).sum(axis=0) / a[~pos].sum()
        Zc = np.where(pos[:, None], Zr - m1, Zr - m0)
        S = (a[:, None] * Zc).T @ Zc / a.sum() + 1e-6 * np.eye(d)
        starts += [m1 - m0, np.linalg.solve(S, m1 - m0)]
    best = (-np.inf, None, 0.0)
    for h in starts:
        nh = np.linalg.norm(h)
        if nh < 1e-12:
            continue
        h = h / nh
        val, b = ev(h)
        if val > best[0]:
            best = (val, h, b)
    if best[1] is None:
        return best
    val, h, b = best
    step = 0.3
    for _ in range(n_polish):
        c = h + step * rng.standard_normal(d)
        c /= np.linalg.norm(c)
        v2, b2 = ev(c)
        if v2 > val:
            val, h, b = v2, c, b2
        else:
            step *= 0.96
    return val, h, b


def _geo_labels(Z, dl, center):
    """Decision list: the first cut (m, h, b) with h'z > b gives the region m; no cut: the centre."""
    labels = np.full(len(Z), center, dtype=int)
    free = np.ones(len(Z), dtype=bool)
    for m, h, b in dl:
        hit = free & (Z @ h > b)
        labels[hit] = m
        free &= ~hit
    return labels


def _scan_pure(Z, X, y, h, nmin, nrest, cs2):
    """One-sided scan along h: the cap (either end of the sorted projections) maximising
    J = n_cap - SSE_cap / cs2, the size of the cap minus its residual sum of squares in units of the admitted
    variance.  Returns (J, h or -h, b) with the cap = {h'z > b}, or None."""
    n, p0 = X.shape
    proj = Z @ h
    order = np.argsort(proj, kind="stable")
    Xo, yo, po = X[order], y[order], proj[order]
    lo, hi = min(nmin, nrest) - 1, n - min(nmin, nrest) - 1
    if hi < lo:
        return None
    A = np.cumsum(Xo[:, :, None] * Xo[:, None, :], axis=0)
    g = np.cumsum(Xo * yo[:, None], axis=0)
    yy = np.cumsum(yo * yo)
    eye = np.eye(p0)
    AL, gL, yL = A[lo:hi + 1], g[lo:hi + 1], yy[lo:hi + 1]
    AR, gR, yR = A[-1] - AL, g[-1] - gL, yy[-1] - yL
    solL = np.linalg.solve(AL + (_RIDGE * (1 + np.trace(AL, axis1=1, axis2=2)))[:, None, None] * eye, gL[:, :, None])[:, :, 0]
    solR = np.linalg.solve(AR + (_RIDGE * (1 + np.trace(AR, axis1=1, axis2=2)))[:, None, None] * eye, gR[:, :, None])[:, :, 0]
    nL = np.arange(lo + 1, hi + 2).astype(float)
    nR = n - nL
    sL = np.maximum(yL - np.einsum("ni,ni->n", gL, solL), 0.0)
    sR = np.maximum(yR - np.einsum("ni,ni->n", gR, solR), 0.0)
    gap = po[lo + 1:hi + 2] - po[lo:hi + 1] > 1e-12
    JL = np.where(gap & (nL >= nmin) & (nR >= nrest), nL - sL / cs2, -np.inf)
    JR = np.where(gap & (nR >= nmin) & (nL >= nrest), nR - sR / cs2, -np.inf)
    jl, jr = int(np.argmax(JL)), int(np.argmax(JR))
    if not (np.isfinite(JL[jl]) or np.isfinite(JR[jr])):
        return None
    if JR[jr] >= JL[jl]:
        i = lo + jr
        return float(JR[jr]), h, float(0.5 * (po[i] + po[i + 1])), solR[jr]
    i = lo + jl
    return float(JL[jl]), -h, float(-0.5 * (po[i] + po[i + 1])), solL[jl]


def _find_cap(Z, X, y, nmin, nrest, cs2, rng, n_polish, n_top):
    """The largest pure cap of a region: half-space maximising J (see _scan_pure); several starts, a random
    local search over directions and the discriminant between the samples the cap's model explains or not."""
    n, d = Z.shape

    def scan(h):
        return _scan_pure(Z, X, y, h / (np.linalg.norm(h) + 1e-12), nmin, nrest, cs2)

    def proposal(res):
        e = y - X @ res[3]
        inl = e * e < cs2
        if inl.sum() < 3 or (~inl).sum() < 3:
            return None
        m1, m0 = Z[inl].mean(axis=0), Z[~inl].mean(axis=0)
        Zc = np.where(inl[:, None], Z - m1, Z - m0)
        v = np.linalg.solve(Zc.T @ Zc / n + 1e-6 * np.eye(d), m1 - m0)
        return v if np.linalg.norm(v) > 1e-12 else None

    scored = []
    for h0 in list(np.eye(d)) + [rng.standard_normal(d) for _ in range(5 * d)]:
        res = scan(h0)
        if res is not None:
            scored.append(res)
    scored.sort(key=lambda t: -t[0])
    best = None
    for res in scored[:n_top]:
        step = 0.35
        for t in range(n_polish):
            trial = [res[1] + step * rng.standard_normal(d)]
            if t % 5 == 0:
                v = proposal(res)
                if v is not None:
                    trial.append(v)
            better = False
            for hn in trial:
                r2 = scan(hn)
                if r2 is not None and r2[0] > res[0]:
                    res, better = r2, True
            if not better:
                step *= 0.93
        if best is None or res[0] > best[0]:
            best = res
    return best


def _robust_fit(Xr, yr, s2, kappa, iters=6):
    """Least squares with Tukey bisquare weights at the scale kappa * noise std: the fit of the majority mode
    of a region that still holds samples of its neighbours."""
    th = np.linalg.lstsq(Xr, yr, rcond=None)[0]
    p0 = Xr.shape[1]
    for _ in range(iters):
        u2 = (yr - Xr @ th) ** 2 / (kappa * kappa * s2)
        w = np.where(u2 < 1.0, (1.0 - u2) ** 2, 0.0)
        if (w > 0.05).sum() < p0 + 6:
            break
        sw = np.sqrt(w)
        th = np.linalg.lstsq(Xr * sw[:, None], yr * sw, rcond=None)[0]
    return th


def _region_fits(X, y, labels, K, p0):
    th_all = np.linalg.lstsq(X, y, rcond=None)[0]
    Th, var, cnt = np.zeros((K, X.shape[1])), np.full(K, np.nan), np.zeros(K, dtype=int)
    for m in range(K):
        idx = labels == m
        cnt[m] = idx.sum()
        if cnt[m] >= p0 + 3:
            Th[m] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            e = y[idx] - X[idx] @ Th[m]
            var[m] = float(e @ e) / (cnt[m] - p0)
        else:
            Th[m] = th_all
    return Th, var, cnt


def identify(obs, params=None, debug=None):
    P = dict(PARAMS)
    if params:
        P.update(params)
    na, nb = int(obs["na"]), int(obs["nb"])
    Phi, y = regression_data(obs["y"], obs["u"], na, nb)
    mu, sd = Phi.mean(axis=0), Phi.std(axis=0) + 1e-9
    Z = (Phi - mu) / sd
    N, d = Z.shape
    p0 = d + 1
    X = np.column_stack([Z, np.ones(N)])
    rng = np.random.default_rng(int(P.get("seed", 20240607)))
    smin, smax = int(obs.get("n_modes_min", 4)), int(obs.get("n_modes_max", 6))
    nmin, n_polish = int(P["nmin"]), int(P["n_polish"])

    # ---- A. noise level: the purest leaf of a greedy recursive partitioning
    leaf = np.zeros(N, dtype=int)
    n_leaf = 1

    def search(m):
        idx = np.flatnonzero(leaf == m)
        c = _find_splits(Z[idx], X[idx], y[idx], nmin, rng, n_polish, 1)
        return c[0] if c else None

    cand = {0: search(0)}
    while n_leaf < int(P["max_leaves"]):
        avail = {m: c for m, c in cand.items() if c is not None}
        if not avail:
            break
        m = max(avail, key=lambda k: avail[k][0])
        gain, h, b = avail[m]
        if gain < P["tau_split"] and n_leaf >= smax:
            break
        idx = np.flatnonzero(leaf == m)
        leaf[idx[Z[idx] @ h > b]] = n_leaf
        n_leaf += 1
        cand[m], cand[n_leaf - 1] = search(m), search(n_leaf - 1)
    _, lvar, lcnt = _region_fits(X, y, leaf, n_leaf, p0)
    s2 = float(np.nanmin(np.where(lcnt >= nmin, lvar, np.nan)))
    cs2 = float(P["c_pure"]) * s2

    # ---- B. peel the largest pure caps off the remainder: a decision list of cuts (two passes: the noise
    #         variance of the second pass is the median residual variance of the regions of the first)
    def peel(s2):
        rem = np.ones(N, dtype=bool)
        dl = []
        while len(dl) < int(P["max_caps"]):
            idx = np.flatnonzero(rem)
            n = len(idx)
            if n < nmin + int(P["nrest"]):
                break
            th = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            e = y[idx] - X[idx] @ th
            if float(e @ e) / (n - p0) <= P["c_stop"] * s2 and len(dl) >= smin - 2:
                break
            res = _find_cap(Z[idx], X[idx], y[idx], nmin, int(P["nrest"]), float(P["c_pure"]) * s2, rng,
                            int(P["n_polish_cap"]), int(P["n_top"]))
            if res is None or res[0] <= 0:
                break
            dl.append((len(dl), res[1], res[2]))
            rem[idx[Z[idx] @ res[1] > res[2]]] = False
        return dl

    def total_sse(dl_, center):
        lab = _geo_labels(Z, dl_, center)
        tot = 0.0
        for m in np.unique(lab):
            idx = lab == m
            if idx.sum() >= p0 + 3:
                e = y[idx] - X[idx] @ np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
                tot += float(e @ e)
        return tot

    def refine(dl, center, K, rounds):
        """Least squares per region <-> every cut re-optimised (weighted linear classification), others fixed."""
        for _ in range(rounds):
            labels = _geo_labels(Z, dl, center)
            Th, _, cnt_ = _region_fits(X, y, labels, K, p0)
            if P["kappa"] > 0:
                for m in range(K):
                    if cnt_[m] >= 3 * p0:
                        Th[m] = _robust_fit(X[labels == m], y[labels == m], s2, float(P["kappa"]))
            C = (y[:, None] - X @ Th.T) ** 2
            for k, (m, h, b) in enumerate(dl):
                reach = np.ones(N, dtype=bool)
                for _, h2, b2 in dl[:k]:
                    reach &= ~(Z @ h2 > b2)
                idx = np.flatnonzero(reach)
                later = _geo_labels(Z[idx], dl[k + 1:], center)
                w = C[idx, later] - C[idx, m]
                val, hn, bn = _best_halfspace(Z[idx], w, [h], rng, int(P["n_class"]))
                if hn is not None:
                    dl[k] = (m, hn, bn)
            # priority of the caps: swap neighbours in the list when that lowers the residual sum of squares
            for k in range(len(dl) - 1):
                sw = dl[:k] + [dl[k + 1], dl[k]] + dl[k + 2:]
                if total_sse(sw, center) < total_sse(dl, center) * (1.0 - 1e-3):
                    dl = sw
        return dl

    def med_var(dl, center, K, q=50.0):
        _, var, cnt = _region_fits(X, y, _geo_labels(Z, dl, center), K, p0)
        v = var[(cnt >= 20) & np.isfinite(var)]
        return float(np.percentile(v, q)) if len(v) else None

    dl = peel(s2)
    v1 = med_var(dl, len(dl), len(dl) + 1, float(P["noise_q"]))
    if v1 is not None:
        s2 = v1
        dl = peel(s2)
    center = len(dl)
    K = center + 1
    if debug is not None:
        debug["lab_peel"] = _geo_labels(Z, dl, center)
    dl = refine(dl, center, K, int(P["n_refine"]))

    # ---- consolidation: a cut whose removal does not increase the residual sum of squares is dropped (a mode
    #      that was peeled in pieces), and so is the cut of a region with very few samples
    removed = []
    while len(dl) > smin - 2:
        base = total_sse(dl, center)
        v = med_var(dl, center, K)
        sig2 = min(v, 1.5 * s2) if v is not None else s2
        lab = _geo_labels(Z, dl, center)
        sizes = [int((lab == m).sum()) for m, _, _ in dl]
        incs = [(total_sse(dl[:k] + dl[k + 1:], center) - base, k) for k in range(len(dl))]
        inc, k = min(incs)
        ksmall = int(np.argmin(sizes))
        if sizes[ksmall] < int(P["min_region"]):
            inc, k = incs[ksmall][0], ksmall
        elif inc >= P["tau_rm"] * sig2:
            break
        removed.append(inc / sig2)
        dl = dl[:k] + dl[k + 1:]
        dl = refine(dl, center, K, 1)
    dl = refine(dl, center, K, 2)
    # relabel the regions 0..K-1 (centre last)
    dl = [(k, h, b) for k, (_, h, b) in enumerate(dl)]
    center = len(dl)
    K = center + 1
    regions = _geo_labels(Z, dl, center)
    if debug is not None:
        debug["lab_geo"] = regions.copy()

    # ---- M. pairs of regions: below sep_same they are one mode, below sep_close they are the close pair
    Th, var, cnt = _region_fits(X, y, regions, K, p0)
    v = var[(cnt >= 20) & np.isfinite(var)]
    sig2 = float(np.percentile(v, float(P["noise_q"]))) if len(v) else s2
    mode_of_region = np.arange(K)
    merge_log = []
    while True:
        labels = mode_of_region[regions]
        ids = np.unique(labels)
        fits = {int(m): np.linalg.lstsq(X[labels == m], y[labels == m], rcond=None)[0] for m in ids}
        seps = []
        for ii in range(len(ids)):
            for jj in range(ii + 1, len(ids)):
                i, j = int(ids[ii]), int(ids[jj])
                idx = (labels == i) | (labels == j)
                diff = X[idx] @ (fits[i] - fits[j])
                sep = float(np.sqrt(np.mean(diff ** 2) / sig2))
                if min((labels == i).sum(), (labels == j).sum()) < int(P["min_region"]):
                    sep *= 0.01
                seps.append((sep, i, j))
        seps.sort()
        n_close = sum(1 for s_ in seps if s_[0] < P["sep_close"])
        too_many = len(ids) > smax or (len(ids) == smax and n_close == 0)
        if len(ids) > smin - 1 and seps and (n_close >= 2 or seps[0][0] < P["sep_same"] or too_many):
            merge_log.append(seps[0][0])
            mode_of_region[mode_of_region == seps[0][2]] = seps[0][1]
            continue
        break
    min_sep = seps[0][0] if seps else np.inf
    has_close = bool(seps) and seps[0][0] < P["sep_close"] and len(ids) >= smin
    if has_close:
        ni, nj = int((labels == seps[0][1]).sum()), int((labels == seps[0][2]).sum())
        if min(ni, nj) < P["pair_balance"] * max(ni, nj) and len(ids) > smin - 1:
            merge_log.append(-seps[0][0])
            mode_of_region[mode_of_region == seps[0][2]] = seps[0][1]
            has_close = False
    ids = np.unique(mode_of_region)
    remap = {int(m): k for k, m in enumerate(ids)}
    mode_of_region = np.array([remap[int(m)] for m in mode_of_region])
    labels = mode_of_region[regions]
    K_modes = len(ids)
    if debug is not None:
        debug["lab_strong"] = labels.copy()
        debug["removed"] = removed

    # ---- E. the close pair: a split near the median of one strong mode
    splits = []
    n_modes = K_modes
    info = []

    def region_split(m):
        idx = np.flatnonzero(labels == m)
        c = _find_splits(Z[idx], X[idx], y[idx], nmin, rng, int(P["n_polish_close"]), 1, float(P["split_q"]))
        return c[0] if c else None

    cand = {m: region_split(m) for m in range(K_modes)}
    while n_modes < smax:
        avail = {m: c for m, c in cand.items() if c is not None}
        if not avail:
            break
        size = {k: int((labels == k).sum()) for k in avail}
        big = [k for k in avail if size[k] >= P["big_region"] * np.median([int((labels == k).sum()) for k in range(n_modes)])]
        pool = big if big else list(avail)
        m = max(pool, key=lambda k: avail[k][0])
        gain, h, b = avail[m]
        info.append((gain, {k: round(c[0]) for k, c in avail.items()}))
        ok = (not splits) and (not has_close)
        if not (ok or n_modes < smin):
            break
        idx = np.flatnonzero(labels == m)
        labels[idx[Z[idx] @ h > b]] = n_modes
        splits.append((m, h, b, n_modes))
        n_modes += 1
        cand[m], cand[n_modes - 1] = region_split(m), region_split(n_modes - 1)

    # ---- F. parameters and the free run of the query
    thetas, _, _ = _region_fits(X, y, labels, n_modes, p0)

    def mode_of(z):
        r = center
        for rr, h, b in dl:
            if float(h @ z) > b:
                r = rr
                break
        m = int(mode_of_region[r])
        for parent, h, b, new in splits:
            if m == parent and float(h @ z) > b:
                m = new
        return m

    uq = np.asarray(obs["u_query"], dtype=float)
    nq = len(uq)
    yq = np.zeros(nq)
    yq[:LAG] = np.asarray(obs["y_query_init"], dtype=float)[:LAG]
    ymax = 10.0 * float(np.max(np.abs(obs["y"]))) + 1.0
    phi = np.zeros(d)
    for t in range(LAG, nq):
        for k in range(na):
            phi[k] = yq[t - 1 - k]
        for k in range(nb):
            phi[na + k] = uq[t - 1 - k]
        z = (phi - mu) / sd
        m = mode_of(z)
        val = float(thetas[m, :d] @ z + thetas[m, d])
        yq[t] = min(ymax, max(-ymax, val)) if np.isfinite(val) else 0.0
    modes = np.zeros(len(obs["y"]), dtype=int)
    modes[LAG:] = labels
    if debug is not None:
        debug.update({"L": n_leaf, "K": K, "Km": K_modes, "merges": merge_log, "min_sep": min_sep, "info": info,
                      "n_splits": len(splits), "sig": float(np.sqrt(s2)), "sig_m": float(np.sqrt(sig2)), "var": var / sig2, "cnt": cnt})
    return {"y_query": yq, "modes": modes, "n_modes": int(n_modes)}


def _kplane(X, y, labels, p0, n_iter=40):
    """K-plane regression: least squares per cluster, every sample to the model with the smallest residual."""
    labels = labels.copy()
    Th = None
    for _ in range(n_iter):
        ths = []
        for m in np.unique(labels):
            idx = labels == m
            if idx.sum() >= p0 + 3:
                ths.append(np.linalg.lstsq(X[idx], y[idx], rcond=None)[0])
        Th = np.array(ths)
        R = (y[:, None] - X @ Th.T) ** 2
        new = np.argmin(R, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
    # final consistent fit
    ids = np.unique(labels)
    remap = {int(m): k for k, m in enumerate(ids)}
    labels = np.array([remap[int(m)] for m in labels])
    Th = np.array([np.linalg.lstsq(X[labels == k], y[labels == k], rcond=None)[0] for k in range(len(ids))])
    R = (y[:, None] - X @ Th.T) ** 2
    return Th, labels, R


def _noise_var(R, labels, p0):
    """Robust noise variance: median over the models of their residual variance."""
    v = []
    for m in range(R.shape[1]):
        idx = labels == m
        if idx.sum() > 2 * p0:
            v.append(R[idx, m].sum() / (idx.sum() - p0))
    return float(np.median(v)) if v else float(R.min(axis=1).mean())


def _separations(X, Th, labels, sig2):
    out = []
    for i in range(len(Th)):
        for j in range(i + 1, len(Th)):
            idx = (labels == i) | (labels == j)
            diff = X[idx] @ (Th[i] - Th[j])
            out.append((float(np.sqrt(np.mean(diff ** 2) / sig2)), i, j))
    return sorted(out)


def _identify_kplane(obs, params=None, debug=None):
    P = dict(PARAMS)
    if params:
        P.update(params)
    na, nb = int(obs["na"]), int(obs["nb"])
    Phi, y = regression_data(obs["y"], obs["u"], na, nb)
    mu, sd = Phi.mean(axis=0), Phi.std(axis=0) + 1e-9
    Z = (Phi - mu) / sd
    N, d = Z.shape
    p0 = d + 1
    X = np.column_stack([Z, np.ones(N)])
    rng = np.random.default_rng(int(P.get("seed", 20240607)))
    smin, smax = int(obs.get("n_modes_min", 4)), int(obs.get("n_modes_max", 6))
    nmin, n_polish = int(P["nmin"]), int(P["n_polish"])

    # ---- A. initial regions: greedy recursive partitioning
    leaf = np.zeros(N, dtype=int)
    n_leaf = 1

    def search(m):
        idx = np.flatnonzero(leaf == m)
        c = _find_splits(Z[idx], X[idx], y[idx], nmin, rng, n_polish, 1)
        return c[0] if c else None

    cand = {0: search(0)}
    while n_leaf < int(P["max_leaves"]):
        avail = {m: c for m, c in cand.items() if c is not None}
        if not avail:
            break
        m = max(avail, key=lambda k: avail[k][0])
        gain, h, b = avail[m]
        if gain < P["tau_split"] and n_leaf >= smax:
            break
        idx = np.flatnonzero(leaf == m)
        leaf[idx[Z[idx] @ h > b]] = n_leaf
        n_leaf += 1
        cand[m], cand[n_leaf - 1] = search(m), search(n_leaf - 1)

    # ---- B. strong modes: k-plane regression from the leaves, improved by split-and-merge moves, and merging
    #         of the models closer than sep_close (the close pair is kept as one model here)
    k_top = smax - 1
    nmin_s = int(P["nmin_move"])

    def sse_of(R, labels):
        return float(R[np.arange(N), labels].sum())

    def improve(Th, labels, R, rounds):
        """Split-and-merge moves: drop one model, split another by its best cut, re-run k-plane; keep if better."""
        K = len(Th)
        sse = sse_of(R, labels)
        for _ in range(rounds):
            improved = False
            for b in range(K):
                idx_b = np.flatnonzero(labels == b)
                c = _find_splits(Z[idx_b], X[idx_b], y[idx_b], nmin_s, rng, int(P["n_polish_move"]), 1)
                if not c:
                    continue
                side = Z[idx_b] @ c[0][1] > c[0][2]
                for a in range(K):
                    if a == b:
                        continue
                    lab2 = labels.copy()
                    Ra = R.copy()
                    Ra[:, a] = np.inf
                    ia = labels == a
                    lab2[ia] = np.argmin(Ra[ia], axis=1)
                    lab2[idx_b[side]] = a
                    Th2, lab2, R2 = _kplane(X, y, lab2, p0)
                    if len(Th2) == K and sse_of(R2, lab2) < sse * (1.0 - 1e-4):
                        Th, labels, R, sse, improved = Th2, lab2, R2, sse_of(R2, lab2), True
                        break
            if not improved:
                break
        return Th, labels, R

    labels = leaf
    merge_log = []
    optimised = False
    while True:
        Th, labels, R = _kplane(X, y, labels, p0)
        K = len(Th)
        if K <= k_top:
            Th, labels, R = improve(Th, labels, R, int(P["move_rounds"]) if not optimised else 1)
            optimised = True
        sig2 = _noise_var(R, labels, p0)
        seps = _separations(X, Th, labels, sig2)
        if K > smin - 1 and seps and (seps[0][0] < P["sep_close"] or K > k_top):
            s0, i, j = seps[0]
            merge_log.append(s0)
            labels = np.where(labels == j, i, labels)
            continue
        break
    min_sep = seps[0][0] if seps else np.inf
    if debug is not None:
        debug["lab_leaf"] = leaf.copy()
        debug["lab_kplane"] = labels.copy()

    # ---- C. geometry: peel the strong modes off as caps (decision list) by weighted linear classification
    C = R
    rem = np.ones(N, dtype=bool)
    classes = list(range(K))
    dl = []
    while len(classes) > 1:
        best = None
        idx = np.flatnonzero(rem)
        for m in classes:
            others = [c for c in classes if c != m]
            w = C[idx][:, others].min(axis=1) - C[idx, m]
            val, h, b = _best_halfspace(Z[idx], w, [], rng, int(P["n_class"]))
            if h is None:
                continue
            excess = float(np.maximum(w, 0).sum() - val)
            if best is None or excess < best[0]:
                best = (excess, m, h, b)
        if best is None:
            break
        _, m, h, b = best
        dl.append((m, h, b))
        rem &= ~(Z @ h > b)
        classes.remove(m)
    center = classes[0]
    if debug is not None:
        debug["lab_geo0"] = _geo_labels(Z, dl, center)

    # ---- D. refinement: least squares per region <-> every cut re-optimised with the others fixed
    for _ in range(int(P["n_refine"])):
        labels = _geo_labels(Z, dl, center)
        for m in range(K):
            idx = labels == m
            if idx.sum() >= p0 + 3:
                Th[m] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        C = (y[:, None] - X @ Th.T) ** 2
        for k, (m, h, b) in enumerate(dl):
            reach = np.ones(N, dtype=bool)
            for _, h2, b2 in dl[:k]:
                reach &= ~(Z @ h2 > b2)
            idx = np.flatnonzero(reach)
            later = _geo_labels(Z[idx], dl[k + 1:], center)
            w = C[idx, later] - C[idx, m]
            val, hn, bn = _best_halfspace(Z[idx], w, [h], rng, int(P["n_class"]))
            if hn is not None:
                dl[k] = (m, hn, bn)
    labels = _geo_labels(Z, dl, center)
    if debug is not None:
        debug["lab_geo"] = labels.copy()

    # ---- E. the close pair: a split near the median of one region
    splits = []
    n_modes = K
    info = []

    def region_split(m):
        idx = np.flatnonzero(labels == m)
        c = _find_splits(Z[idx], X[idx], y[idx], nmin, rng, int(P["n_polish_close"]), 1, float(P["split_q"]))
        return c[0] if c else None

    cand = {m: region_split(m) for m in range(K)}
    while n_modes < smax:
        avail = {m: c for m, c in cand.items() if c is not None}
        if not avail:
            break
        m = max(avail, key=lambda k: avail[k][0])
        gain, h, b = avail[m]
        info.append((gain, {k: round(c[0]) for k, c in avail.items()}))
        if splits:
            ok = gain > P["tau_extra"]
        else:
            ok = gain > P["tau_split"] or min_sep > P["sep_gray"]
        if not (ok or n_modes < smin):
            break
        idx = np.flatnonzero(labels == m)
        labels[idx[Z[idx] @ h > b]] = n_modes
        splits.append((m, h, b, n_modes))
        n_modes += 1
        cand[m], cand[n_modes - 1] = None, None
        if P["tau_extra"] < 1e8:
            cand[m], cand[n_modes - 1] = region_split(m), region_split(n_modes - 1)

    # ---- F. parameters and the free run of the query
    thetas = np.zeros((n_modes, p0))
    th_all = np.linalg.lstsq(X, y, rcond=None)[0]
    for m in range(n_modes):
        idx = labels == m
        thetas[m] = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0] if idx.sum() >= p0 + 2 else th_all

    def mode_of(z):
        m = center
        for mm, h, b in dl:
            if float(h @ z) > b:
                m = mm
                break
        for parent, h, b, new in splits:
            if m == parent and float(h @ z) > b:
                m = new
        return m

    uq = np.asarray(obs["u_query"], dtype=float)
    nq = len(uq)
    yq = np.zeros(nq)
    yq[:LAG] = np.asarray(obs["y_query_init"], dtype=float)[:LAG]
    ymax = 10.0 * float(np.max(np.abs(obs["y"]))) + 1.0
    phi = np.zeros(d)
    for t in range(LAG, nq):
        for k in range(na):
            phi[k] = yq[t - 1 - k]
        for k in range(nb):
            phi[na + k] = uq[t - 1 - k]
        z = (phi - mu) / sd
        m = mode_of(z)
        val = float(thetas[m, :d] @ z + thetas[m, d])
        yq[t] = min(ymax, max(-ymax, val)) if np.isfinite(val) else 0.0
    modes = np.zeros(len(obs["y"]), dtype=int)
    modes[LAG:] = labels
    if debug is not None:
        debug.update({"L": n_leaf, "K": K, "merges": merge_log, "seps": [s[0] for s in seps], "info": info,
                      "n_splits": len(splits), "sig": float(np.sqrt(sig2)) * 1.0})
    return {"y_query": yq, "modes": modes, "n_modes": int(n_modes)}


def _bic(obs, answer):
    """Bayesian information criterion of an answer's partition of the record (least squares per mode)."""
    na, nb = int(obs["na"]), int(obs["nb"])
    Phi, y = regression_data(obs["y"], obs["u"], na, nb)
    X = np.column_stack([Phi, np.ones(len(Phi))])
    labels = np.asarray(answer["modes"])[LAG:]
    N, p0 = X.shape
    sse = 0.0
    for m in np.unique(labels):
        idx = labels == m
        if idx.sum() > p0:
            e = y[idx] - X[idx] @ np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
            sse += float(e @ e)
    return N * np.log(max(sse, 1e-12) / N) + PARAMS["bic_weight"] * int(answer["n_modes"]) * (2 * p0 - 1) * np.log(N)


def _fallback(obs):
    """One affine ARX model (used only if the identification raises)."""
    na, nb = int(obs["na"]), int(obs["nb"])
    Phi, y = regression_data(obs["y"], obs["u"], na, nb)
    th = np.linalg.lstsq(np.column_stack([Phi, np.ones(len(Phi))]), y, rcond=None)[0]
    uq = np.asarray(obs["u_query"], dtype=float)
    yq = np.zeros(len(uq))
    yq[:LAG] = np.asarray(obs["y_query_init"], dtype=float)[:LAG]
    for t in range(LAG, len(uq)):
        phi = np.concatenate([yq[t - na:t][::-1], uq[t - nb:t][::-1]])
        v = float(th[:-1] @ phi + th[-1])
        yq[t] = v if np.isfinite(v) and abs(v) < 1e6 else 0.0
    modes = np.zeros(len(obs["y"]), dtype=int)
    k = int(obs.get("n_modes_min", 4))
    modes[LAG:] = np.minimum((np.argsort(np.argsort(Phi[:, 0])) * k) // len(Phi), k - 1)
    return {"y_query": yq, "modes": modes, "n_modes": k}


# candidate identifications: (pipeline, parameter overrides); the answer with the lowest BIC is returned
CANDIDATES = (
    ("peel", {}),
    ("kplane", {}),
    ("peel", {"c_pure": 3.0, "seed": 1}),
    ("peel", {"c_pure": 6.5, "seed": 2}),
    ("peel", {"noise_q": 50.0, "kappa": 0.0, "seed": 4}),
    ("peel", {"c_pure": 3.75, "seed": 5}),
    ("peel", {"c_pure": 5.5, "seed": 6}),
    ("kplane", {"seed": 3}),
)


def controller(obs: dict, debug=None) -> dict:
    import time

    t0 = time.process_time()
    budget = min(float(PARAMS["cpu_budget"]), 0.5 * float(obs.get("time_limit", 10.0)))
    answers = []
    for name, overrides in CANDIDATES:
        if answers and time.process_time() - t0 > budget:
            break
        try:
            ans = (identify if name == "peel" else _identify_kplane)(obs, params=overrides)
            answers.append((_bic(obs, ans), len(answers), ans))
        except Exception:
            pass
    if not answers:
        return _fallback(obs)
    if debug is not None:
        debug["bics"] = [a[0] for a in answers]
        debug["answers"] = [a[2] for a in answers]
    return min(answers, key=lambda a: a[:2])[2]
