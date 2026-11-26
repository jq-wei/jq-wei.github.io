"""Identification of a switched ARX system on a hidden clock-driven schedule (no control law: nothing is fed back).

``controller(obs)`` receives one input-output record and answers with the number of modes, the mode of every
sample and the free run of the query.  Only least squares, random sampling of windows and dynamic programming
are used (no black-box learning).  All costs are squared residuals in units of the common noise variance,
which is estimated robustly from the residuals of the assigned models (all modes share one noise level).

1. Forward search (``_search_step``).  Candidate submodels are least-squares fits to one window, to two or
   three short windows drawn where the present models fit badly, and to unions of segments that the present
   segmentation gives to one model.  A candidate is scored by the cost it saves over *runs* of consecutive
   samples (two-state dynamic programme with a price per run), so chance agreements at isolated samples do
   not count.  The best candidates are re-fitted on their runs, the best one is accepted, and all models are
   re-fitted by hard EM (``_hard_em``: Viterbi segmentation with a switching price, least squares per mode).
   The search runs to one model more than the largest admissible count.
2. Number of modes.  Backward: the model whose removal costs least is dropped while that cost is below
   ``min_gain``.  Extension: new candidates and direct splits of every model (``_split_proposals``, which
   finds a pair of nearly equal modes hidden in one model) are accepted when the cost after a full re-fit
   falls by at least ``min_gain``.  A model must hold a minimum share of the record.  The whole search is
   repeated on several random streams and the solution with the least penalised cost is kept.
3. Answer.  Soft EM (Baum-Welch) polishes the parameters; the labels are the posterior marginals' maxima; the
   query is the mean of the free runs over schedules sampled from the posterior (the schedule is clock-driven
   and shared by the query, and an uncertain switching instant is hedged instead of guessed).

Constants (``PARAMS``):
  tau2         cap on the cost of a sample that no model explains (noise variances)
  pen          switching price of the Viterbi segmentation (noise variances)
  run_pen      price of one run in the candidate score
  min_gain     cost a mode must save to be kept or added: the test that decides the number of modes
  n_pair, n_triple, n_seg   random two-window, three-window and segment-union candidates per search step
  n_top        candidates refined per search step;  n_try: distinct candidates judged by a full re-fit
  extra        models searched beyond the largest admissible count before pruning
  n_starts     independent searches (fewer if the processor time runs short)
  min_share    smallest size of a model, as a share of an equal split of the record among n_modes_max modes
  n_split, n_split_try   starts per model of the split test, and splits judged by a full re-fit
  soft_iters   Baum-Welch iterations;  n_sched: schedules sampled for the query
"""

import time

import numpy as np

LAG = 3
_SEED = 12345      # seed of the procedure's own random stream (candidate windows, sampled schedules)
_DEBUG = {}         # diagnostics of the last call (not part of the answer)

PARAMS = {
    "tau2": 9.0,
    "pen": 8.0,
    "run_pen": 8.0,
    "min_gain": 47.0,
    "n_pair": 2500,
    "n_triple": 2500,
    "n_top": 60,
    "extra": 1,
    "n_seg": 30,
    "n_starts": 3,
    "n_try": 6,
    "min_share": 0.3,
    "n_split": 40,
    "n_split_try": 3,
    "soft_iters": 30,
    "n_sched": 200,
}


def _design(y, u, na, nb):
    n = len(y)
    cols = [y[LAG - k:n - k] for k in range(1, na + 1)] + [u[LAG - k:n - k] for k in range(1, nb + 1)]
    return np.column_stack(cols + [np.ones(n - LAG)]), y[LAG:].copy()


def _viterbi(C, pen):
    """Minimum-cost labelling of the columns of C (K, N) with a price ``pen`` per switch."""
    K, N = C.shape
    back = np.zeros((N, K), dtype=np.int64)
    ar = np.arange(K)
    cost = C[:, 0].copy()
    CT = np.ascontiguousarray(C.T)
    for t in range(1, N):
        j = int(np.argmin(cost))
        thr = cost[j] + pen
        sw = cost > thr
        b = ar.copy()
        b[sw] = j
        back[t] = b
        cost = np.minimum(cost, thr) + CT[t]
    lab = np.empty(N, dtype=np.int64)
    k = int(np.argmin(cost))
    for t in range(N - 1, -1, -1):
        lab[t] = k
        k = back[t, k]
    return lab


def _runs(D, pen):
    """For every row of D (B, N): the set of runs maximising sum(D over runs) - pen * (number of runs)."""
    B, N = D.shape
    out = np.zeros(B)
    inn = np.full(B, -1e18)
    ent = np.zeros((N, B), dtype=bool)
    ext = np.zeros((N, B), dtype=bool)
    DT = np.ascontiguousarray(D.T)
    for t in range(N):
        cand = out - pen
        e = cand > inn
        x = inn > out
        new_in = np.where(e, cand, inn) + DT[t]
        out = np.where(x, inn, out)
        inn = new_in
        ent[t] = e
        ext[t] = x
    gain = np.maximum(out, inn)
    state = inn > out
    mask = np.zeros((N, B), dtype=bool)
    for t in range(N - 1, -1, -1):
        mask[t] = state
        state = np.where(state, ~ent[t], ext[t])
    return gain, mask.T


def _run_gain(D, pen):
    """Gain only (no masks), batched."""
    B, N = D.shape
    out = np.zeros(B)
    inn = np.full(B, -1e18)
    DT = np.ascontiguousarray(D.T)
    for t in range(N):
        new_in = np.maximum(inn, out - pen) + DT[t]
        out = np.maximum(out, inn)
        inn = new_in
    return np.maximum(out, inn)


def _fit_idx(X, Y, idx, ridge, th0):
    """Batched ridge least squares on index sets idx (B, m), shrunk towards th0."""
    A = X[idx]
    b = Y[idx] - A @ th0
    At = A.transpose(0, 2, 1)
    G = At @ A + ridge
    g = (At @ b[..., None])
    return th0[None, :] + np.linalg.solve(G, g)[..., 0]


def _fit_mask(X, Y, W, ridge, th0):
    """Batched ridge least squares with 0/1 weights W (B, N)."""
    Wf = W.astype(float)
    G = np.einsum("bn,ni,nj->bij", Wf, X, X) + ridge
    r0 = Y - X @ th0
    g = np.einsum("bn,ni,n->bi", Wf, X, r0)
    return th0[None, :] + np.linalg.solve(G, g[..., None])[..., 0]


def _refit(X, Y, lab, th, ridge, th0):
    K, p = th.shape
    out = th.copy()
    cnt = np.zeros(K, dtype=int)
    for k in range(K):
        m = lab == k
        cnt[k] = int(m.sum())
        if cnt[k] >= p + 2:
            A = X[m]
            out[k] = th0 + np.linalg.solve(A.T @ A + ridge, A.T @ (Y[m] - A @ th0))
    return out, cnt


def _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=8, min_count=None):
    """Viterbi segmentation / per-mode least squares.  A model that holds fewer than ``min_count`` samples is
    dropped; a model that has just been added gets a few iterations to collect its samples first."""
    p = X.shape[1]
    small = p + 2
    final = small if min_count is None else max(int(min_count), small)
    grow = 3 if final > small else 0
    lab = None
    for it in range(iters + grow):
        new = _viterbi((Y[None, :] - th @ X.T) ** 2 / sig2, pen)
        cnt = np.bincount(new, minlength=len(th))
        need = small if it < grow else final
        if len(th) > 1 and cnt.min() < need:
            keep = cnt >= need
            if not keep.any():
                keep[np.argmax(cnt)] = True
            th = th[keep]
            lab = None
            continue
        if lab is not None and it >= grow and np.array_equal(new, lab):
            break
        lab = new
        th, _ = _refit(X, Y, lab, th, ridge, th0)
    for _ in range(4):
        if lab is not None and lab.max() < len(th):
            cnt = np.bincount(lab, minlength=len(th))
            if len(th) == 1 or cnt.min() >= final:
                break
            keep = cnt >= final
            if not keep.any():
                keep[np.argmax(cnt)] = True
            th = th[keep]
        lab = _viterbi((Y[None, :] - th @ X.T) ** 2 / sig2, pen)
    return th, lab


def _noise(X, Y, th, lab, lo, hi):
    """Robust common noise variance from the residuals of the assigned models: mean square of the 90 % smallest
    squared residuals, rescaled for a Gaussian and for the fitted parameters."""
    r2 = np.sort((Y - np.einsum("ni,ni->n", X, th[lab])) ** 2)
    p = X.shape[1]
    corr = len(Y) / max(len(Y) - len(th) * p, 1.0)
    s2 = float(np.mean(r2[:max(int(0.9 * len(r2)), 1)])) / 0.6230 * corr
    return float(np.clip(np.sqrt(s2), lo, hi)) ** 2


def _candidates(X, Y, w, rng, p, n_pair, n_triple, ridge, th0, dense=True):
    """Least-squares fits to one, two and three windows; window centres are drawn with weights w."""
    N = len(Y)
    out = []
    if dense:
        for L in (p + 2, 2 * p + 1):
            starts = np.arange(0, N - L + 1)
            out.append(_fit_idx(X, Y, starts[:, None] + np.arange(L)[None, :], ridge, th0))
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]

    def centres(shape):
        return np.minimum(np.searchsorted(cdf, rng.random(shape)), N - 1)

    wl2 = max(4, (p + 4) // 2)
    for n_c, n_w, wl in ((n_pair, 2, wl2), (n_triple, 3, 4)):
        if n_c <= 0:
            continue
        c = centres((n_c, n_w))
        idx = (c[:, :, None] + np.arange(wl)[None, None, :] - wl // 2).reshape(n_c, n_w * wl)
        idx = np.clip(idx, 0, N - 1)
        out.append(_fit_idx(X, Y, idx, ridge, th0))
    return np.concatenate(out, axis=0)


def _cost(X, Y, th, lab, sig2, pen):
    r = Y - np.einsum("ni,ni->n", X, th[lab])
    return float(r @ r) / sig2 + pen * float(np.count_nonzero(np.diff(lab)))


def _segment_candidates(X, Y, lab, K, rng, n_per, ridge, th0):
    """Fits to the union of two or three whole segments that the current segmentation gives to the same model:
    if that model merges two modes, such a union is pure with a high probability."""
    N, p = X.shape
    b = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], b])
    ends = np.concatenate([b, [N]])
    seg_lab = lab[starts]
    masks = []
    for k in range(K):
        idx = np.flatnonzero(seg_lab == k)
        if len(idx) < 3:
            continue
        for n_w in (2, 3):
            ch = idx[rng.integers(0, len(idx), size=(n_per, n_w))]
            for row in ch:
                w = np.zeros(N, dtype=bool)
                for c in row:
                    w[starts[c]:ends[c]] = True
                masks.append(w)
    if not masks:
        return np.zeros((0, p))
    W = np.array(masks)
    W = W[W.sum(axis=1) >= p + 3]
    if len(W) == 0:
        return np.zeros((0, p))
    return _fit_mask(X, Y, W, ridge, th0)


def _search_step(X, Y, th, lab, cur, sig2, P, rng, ridge, th0, n_out=1):
    """The best new submodels (up to ``n_out`` distinct ones, best first) given the per-sample cost ``cur``
    of the models already accepted."""
    p = X.shape[1]
    run_pen = P["run_pen"]
    w = cur + 0.05 * P["tau2"]
    cand = _candidates(X, Y, w, rng, p, P["n_pair"], P["n_triple"], ridge, th0)
    if len(th) >= 2 and P["n_seg"] > 0:
        cand = np.concatenate([cand, _segment_candidates(X, Y, lab, len(th), rng, int(P["n_seg"]), ridge, th0)], axis=0)
    R2 = (Y[None, :] - cand @ X.T) ** 2 / sig2
    g = _run_gain(cur[None, :] - R2, run_pen)
    top = np.argsort(-g)[:P["n_top"]]
    ct = cand[top]
    gt = mask = None
    for it in range(4):
        R2 = (Y[None, :] - ct @ X.T) ** 2 / sig2
        gt_new, mask_new = _runs(cur[None, :] - R2, run_pen)
        if gt is None:
            gt, mask, best = gt_new, mask_new, ct.copy()
        else:
            better = gt_new > gt
            gt = np.where(better, gt_new, gt)
            mask = np.where(better[:, None], mask_new, mask)
            best = np.where(better[:, None], ct, best)
        if it == 3:
            break
        ok = mask_new.sum(axis=1) >= p + 2
        new = _fit_mask(X, Y, mask_new, ridge, th0)
        ct = np.where(ok[:, None], new, ct)
    order = np.argsort(-gt)
    chosen = []
    for j in order:
        if gt[j] <= 0 and chosen:
            break
        mj = mask[j]
        nj = max(int(mj.sum()), 1)
        if all((mj & mask[i]).sum() < 0.5 * min(nj, max(int(mask[i].sum()), 1)) for i in chosen):
            chosen.append(int(j))
            if len(chosen) >= n_out:
                break
    return [float(gt[j]) for j in chosen], [best[j].copy() for j in chosen]


def _split_proposals(X, Y, th, lab, sig2, rng, P, ridge, th0, mc):
    """Proposals that split one model in two.  A model that merges two nearly equal modes is hard to see in the
    candidate score (the merged fit is not bad anywhere), so every model is tested directly: two-model fits to
    its own segments (whole segments go to the better model, both are re-fitted), started from unions of two
    or three of its segments.  Returns the model sets of the most promising splits."""
    N, p = X.shape
    b = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], b])
    ends = np.concatenate([b, [N]])
    seg_lab = lab[starts]
    seg_of = np.repeat(np.arange(len(starts)), ends - starts)
    n_per = int(P["n_split"])
    found = []
    for k in range(len(th)):
        segs = np.flatnonzero(seg_lab == k)
        S = len(segs)
        idx = np.flatnonzero(lab == k)
        n_k = len(idx)
        if S < 4 or n_k < 2 * mc:
            continue
        Xk, Yk = X[idx], Y[idx]
        M = np.zeros((n_k, S))
        M[np.arange(n_k), np.searchsorted(segs, seg_of[idx])] = 1.0
        size = M.sum(axis=0)
        assign = np.zeros((2 * n_per, S), dtype=bool)
        assign[np.arange(n_per)[:, None], rng.integers(0, S, size=(n_per, 2))] = True
        assign[(n_per + np.arange(n_per))[:, None], rng.integers(0, S, size=(n_per, 3))] = True
        thA = None
        thB = np.tile(th[k], (len(assign), 1))
        r0 = float(np.sum((Yk - Xk @ th[k]) ** 2))
        for it in range(4):
            W = (assign.astype(float) @ M.T) > 0.5
            nA = W.sum(axis=1)
            newA = _fit_mask(Xk, Yk, W, ridge, th0)
            thA = newA if thA is None else np.where((nA >= p + 2)[:, None], newA, thA)
            if it > 0:
                newB = _fit_mask(Xk, Yk, ~W, ridge, th0)
                thB = np.where((n_k - nA >= p + 2)[:, None], newB, thB)
            SA = ((Yk[None, :] - thA @ Xk.T) ** 2) @ M
            SB = ((Yk[None, :] - thB @ Xk.T) ** 2) @ M
            assign = SA < SB
        nA = assign.astype(float) @ size
        gain = (r0 - np.minimum(SA, SB).sum(axis=1)) / sig2
        gain[(nA < mc) | (n_k - nA < mc)] = -1e9
        j = int(np.argmax(gain))
        if gain[j] > 0:
            found.append((float(gain[j]), k, thA[j].copy(), thB[j].copy()))
    found.sort(key=lambda item: -item[0])
    props = []
    for _, k, a, b_ in found[:int(P["n_split_try"])]:
        t2 = th.copy()
        t2[k] = b_
        props.append(np.vstack([t2, a[None, :]]))
    return props


def _current_cost(X, Y, th, lab, sig2, tau2):
    r2 = (Y - np.einsum("ni,ni->n", X, th[lab])) ** 2 / sig2
    return np.minimum(r2, tau2)


def _identify(X, Y, n_min, n_max, P, rng, sy):
    N, p = X.shape
    ridge = 1e-3 * np.diag(np.concatenate([np.var(X[:, :-1], axis=0), [1.0]]))
    th0 = np.linalg.lstsq(X, Y, rcond=None)[0]
    s_lo, s_hi = 0.045 * sy, 0.115 * sy
    sig2 = (0.07 * sy) ** 2
    tau2, pen = P["tau2"], P["pen"]

    # forward: accept the best candidate until there is one model more than the largest admissible count
    th = np.zeros((0, p))
    lab = np.zeros(N, dtype=np.int64)
    cur = np.full(N, tau2)
    k_cap = n_max + int(P["extra"])
    for _ in range(k_cap + 3):
        if len(th) >= k_cap:
            break
        _, cands = _search_step(X, Y, th, lab, cur, sig2, P, rng, ridge, th0)
        if not cands:
            break
        th = np.vstack([th, cands[0][None, :]])
        th, lab = _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=6)
        if len(th) >= 3:
            sig2 = _noise(X, Y, th, lab, s_lo, s_hi)
        cur = _current_cost(X, Y, th, lab, sig2, tau2)
    # from here on a model must hold a minimum number of samples (every mode of the family holds at least half
    # of an equal share of the record): tiny models that only collect outlying samples are dropped
    mc = max(p + 2, int(P["min_share"] * N / n_max))
    th, lab = _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=12, min_count=mc)
    sig2 = _noise(X, Y, th, lab, s_lo, s_hi)
    th, lab = _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=12, min_count=mc)
    sig2 = _noise(X, Y, th, lab, s_lo, s_hi)
    info = {"drops": [], "adds": []}

    for sweep in range(3):
        # backward: drop the model whose removal costs least, while that cost is below the threshold
        J = _cost(X, Y, th, lab, sig2, pen)
        dropped = False
        while len(th) > n_min:
            best = None
            for k in range(len(th)):
                th_k, lab_k = _hard_em(X, Y, np.delete(th, k, axis=0), sig2, pen, ridge, th0, iters=5, min_count=mc)
                J_k = _cost(X, Y, th_k, lab_k, sig2, pen)
                if best is None or J_k < best[0]:
                    best = (J_k, th_k, lab_k)
            info["drops"].append(best[0] - J)
            if best[0] - J >= P["min_gain"] and len(th) <= n_max:
                break
            _, th, lab = best
            dropped = True
            sig2 = _noise(X, Y, th, lab, s_lo, s_hi)        # fewer models: less over-fitting in the noise estimate
            th, lab = _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=5, min_count=mc)
            J = _cost(X, Y, th, lab, sig2, pen)
        if sweep > 0 and not dropped:
            break
        # extension: the best distinct candidates are judged by the cost after a full re-fit, not by their score
        added = False
        while len(th) < n_max + 1 and sweep < 2:
            cur = _current_cost(X, Y, th, lab, sig2, tau2)
            _, cands = _search_step(X, Y, th, lab, cur, sig2, P, rng, ridge, th0, n_out=int(P["n_try"]))
            props = [np.vstack([th, c[None, :]]) for c in cands]
            props += _split_proposals(X, Y, th, lab, sig2, rng, P, ridge, th0, mc)
            best = None
            for th_p in props:
                th_c, lab_c = _hard_em(X, Y, th_p, sig2, pen, ridge, th0, iters=6, min_count=mc)
                if len(th_c) != len(th) + 1:
                    continue
                sig2_c = _noise(X, Y, th_c, lab_c, s_lo, s_hi)
                gain = _cost(X, Y, th, lab, sig2_c, pen) - _cost(X, Y, th_c, lab_c, sig2_c, pen)
                if best is None or gain > best[0]:
                    best = (gain, th_c, lab_c, sig2_c)
            if best is None:
                break
            info["adds"].append(best[0])
            if best[0] < P["min_gain"] and len(th) >= n_min:
                break
            _, th, lab, sig2 = best
            th, lab = _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=8, min_count=mc)
            added = True
        if not added:
            break
    th, lab = _hard_em(X, Y, th, sig2, pen, ridge, th0, iters=15, min_count=mc)
    return th, lab, sig2, info


def _fb(R2, sig2, q):
    """Forward-backward of the sticky hidden Markov model (switching probability q, uniform over the other
    modes) with Gaussian residuals R2 (K, N).  Returns the posterior marginals and the filtered probabilities."""
    K, N = R2.shape
    E = -0.5 * R2 / sig2
    Bm = np.ascontiguousarray(np.exp(E - E.max(axis=0, keepdims=True)).T) + 1e-300
    off = q / (K - 1)
    diag = 1.0 - q - off
    alpha = np.empty((N, K))
    a = Bm[0] / Bm[0].sum()
    alpha[0] = a
    for t in range(1, N):
        a = (diag * a + off) * Bm[t]
        a = a / a.sum()
        alpha[t] = a
    beta = np.empty((N, K))
    b = np.full(K, 1.0 / K)
    beta[N - 1] = b
    for t in range(N - 1, 0, -1):
        v = Bm[t] * b
        b = diag * v + off * v.sum()
        b = b / b.sum()
        beta[t - 1] = b
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma, alpha


def _soft_em(X, Y, th, sig2, q, ridge, th0, s_lo, s_hi, iters):
    """Baum-Welch for the switched regression: weighted least squares per mode, common noise variance."""
    N, p = X.shape
    K = len(th)
    r0 = Y - X @ th0
    gamma = alpha = None
    for _ in range(iters):
        R2 = (Y[None, :] - th @ X.T) ** 2
        gamma, alpha = _fb(R2, sig2, q)
        W = gamma.T
        G = np.einsum("kn,ni,nj->kij", W, X, X) + ridge
        g = np.einsum("kn,ni,n->ki", W, X, r0)
        th = th0[None, :] + np.linalg.solve(G, g[..., None])[..., 0]
        R2 = (Y[None, :] - th @ X.T) ** 2
        sig2 = float(np.clip((W * R2).sum() / max(N - K * p, 1), s_lo ** 2, s_hi ** 2))
        lab = np.argmax(gamma, axis=1)
        q = float(np.clip(np.count_nonzero(np.diff(lab)) / (N - 1.0), 0.03, 0.45))
    R2 = (Y[None, :] - th @ X.T) ** 2
    gamma, alpha = _fb(R2, sig2, q)
    return th, sig2, q, gamma, alpha


def _sample_schedules(alpha, q, M, rng):
    """M schedules drawn from the posterior of the hidden Markov model (forward filtering, backward sampling)."""
    N, K = alpha.shape
    off = q / (K - 1)
    labs = np.empty((M, N), dtype=np.int64)
    c = np.cumsum(alpha[-1])
    k = np.minimum(np.searchsorted(c / c[-1], rng.random(M)), K - 1)
    labs[:, -1] = k
    ar = np.arange(M)
    for t in range(N - 2, -1, -1):
        Pm = np.tile(alpha[t] * off, (M, 1))
        Pm[ar, k] = alpha[t, k] * (1.0 - q)
        c = np.cumsum(Pm, axis=1)
        r = rng.random(M) * c[:, -1]
        k = np.minimum((c < r[:, None]).sum(axis=1), K - 1)
        labs[:, t] = k
    return labs


def _free_runs(th, labs, uq, y_init, na, nb):
    """Free runs of the query under M schedules (rows of labs), vectorised over the schedules."""
    M, L = labs.shape
    n = len(uq)
    y = np.zeros((n, M))
    y[:LAG] = np.asarray(y_init, dtype=float)[:LAG, None]
    for t in range(LAG, n):
        tk = th[labs[:, min(t - LAG, L - 1)]]
        v = tk[:, -1].copy()
        for i in range(na):
            v += tk[:, i] * y[t - 1 - i]
        for i in range(nb):
            v += tk[:, na + i] * uq[t - 1 - i]
        y[t] = np.clip(np.nan_to_num(v), -1e6, 1e6)
    return y.T


def _free_run(th, lab, uq, y_init, na, nb):
    n = len(uq)
    y = np.zeros(n)
    y[:LAG] = y_init[:LAG]
    L = len(lab)
    for t in range(LAG, n):
        k = lab[min(t - LAG, L - 1)]
        v = th[k, -1]
        for i in range(na):
            v += th[k, i] * y[t - 1 - i]
        for i in range(nb):
            v += th[k, na + i] * uq[t - 1 - i]
        y[t] = min(1e6, max(-1e6, v)) if np.isfinite(v) else 0.0
    return y


def controller(obs: dict) -> dict:
    """Identify one hidden switched ARX system from its record and answer the query (see the module docstring)."""
    t_start = time.process_time()
    y = np.asarray(obs["y"], dtype=float)
    u = np.asarray(obs["u"], dtype=float)
    uq = np.asarray(obs["u_query"], dtype=float)
    y_init = np.asarray(obs["y_query_init"], dtype=float)
    na, nb = int(obs["na"]), int(obs["nb"])
    n_min, n_max = int(obs["n_modes_min"]), int(obs["n_modes_max"])
    budget = 0.3 * float(obs.get("time_limit", 10.0))      # processor seconds this call allows itself
    n = len(y)
    try:
        X, Y = _design(y, u, na, nb)
        N, p = X.shape
        rng = np.random.default_rng(_SEED)
        sy = float(np.std(y))
        ridge = 1e-3 * np.diag(np.concatenate([np.var(X[:, :-1], axis=0), [1.0]]))
        th0 = np.linalg.lstsq(X, Y, rcond=None)[0]

        # several searches on different random streams; keep the one with the least penalised cost
        # (cost of the segmentation in noise variances + min_gain per mode), all judged at one noise level
        sols = []
        for i in range(int(PARAMS["n_starts"])):
            used = time.process_time() - t_start
            if i > 0 and used + used / i > budget:
                break
            sols.append(_identify(X, Y, n_min, n_max, PARAMS, rng, sy))
        sig2 = float(np.median([s[2] for s in sols]))
        best = None
        for th_r, lab_r, _, info_r in sols:
            th_r, lab_r = _hard_em(X, Y, th_r, sig2, PARAMS["pen"], ridge, th0, iters=10)
            info_r["score"] = _cost(X, Y, th_r, lab_r, sig2, PARAMS["pen"]) + PARAMS["min_gain"] * len(th_r)
            if best is None or info_r["score"] < best[0]:
                best = (info_r["score"], th_r, lab_r, info_r)
        _, th, lab, info = best
        yq = _free_run(th, lab, uq, y_init, na, nb)
        try:
            # polish with soft EM, label by the posterior marginals, and answer the query with the posterior
            # mean of the free run over schedules (an uncertain switching instant is hedged, not guessed)
            q = float(np.clip(np.count_nonzero(np.diff(lab)) / (N - 1.0), 0.03, 0.45))
            th_s, sig2_s, q, gamma, alpha = _soft_em(X, Y, th, sig2, q, ridge, th0, 0.045 * sy, 0.115 * sy, int(PARAMS["soft_iters"]))
            lab_s = np.argmax(gamma, axis=1)
            if len(np.unique(lab_s)) == len(th) and np.all(np.isfinite(th_s)):
                labs = _sample_schedules(alpha, q, int(PARAMS["n_sched"]), rng)
                yq_s = _free_runs(th_s, labs, uq, y_init, na, nb).mean(axis=0)
                if np.all(np.isfinite(yq_s)) and np.max(np.abs(yq_s)) < 1e3 * max(sy, 1e-9):
                    yq, lab, th, sig2 = yq_s, lab_s, th_s, sig2_s
        except Exception:
            pass
        _DEBUG.clear()
        _DEBUG.update(drops=info["drops"], adds=info["adds"], sig=float(np.sqrt(sig2)), starts=len(sols),
                      scores=[s[3]["score"] for s in sols])
        modes = np.concatenate([np.full(LAG, lab[0]), lab]).astype(int)
        return {"y_query": np.nan_to_num(yq), "modes": modes, "n_modes": int(len(th))}
    except Exception:  # never fail the call: fall back to one linear model
        X, Y = _design(y, u, na, nb)
        th = np.linalg.lstsq(X, Y, rcond=None)[0][None, :]
        yq = _free_run(th, np.zeros(n - LAG, dtype=int), uq, y_init, na, nb)
        return {"y_query": np.nan_to_num(yq), "modes": np.zeros(n, dtype=int), "n_modes": n_min}
