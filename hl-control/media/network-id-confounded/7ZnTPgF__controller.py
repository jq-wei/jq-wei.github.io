"""Network identification with unmeasured hubs: sparse VARMAX equations fitted node by node, model-averaged.

`controller(obs)` is an identification procedure, not a feedback law. Every measured node j gets the equation

    x_j(t) = sum_k a_k x_j(t-k) + sum b w_i(t-l) + sum c [F_f w_i](t-l) + (innovation terms) + eps_j(t),   x = w - r,

  * a, b: the node's own poles and the direct links (the excitation r_i enters through w_i = x_i + r_i);
  * F_f w_i: the signal of node i passed through one of a few fixed AR filters (BANK). It stands for a path
    i -> unmeasured node -> j, whose dynamics are not those of node j;
  * innovation terms: lagged, filtered and same-instant one-step prediction errors of the nodes. They stand in
    for the noise of an unmeasured hub that several nodes share (and for the correlated innovations), so that a
    sibling's signal is not mistaken for a link.

Terms are chosen by greedy forward selection with a BIC-type penalty; the innovations are then re-estimated and
the selection repeated (extended least squares). The response to a query is the simulation of the a/b/c part
from rest, averaged over a few fits that differ in the selection penalty (hard selection is unstable on 400
samples). Links are the sources of the selected w and filtered-w terms of the reference fit.

PARAMS
  bic_a, bic_w, bic_fw, bic_e, bic_fe, bic_e0: multiples of log(n) charged for one term of each type (extra AR
      lags, signal lags, filtered signals, lagged / filtered / same-instant innovations).
  ar_base, ar_max, lw, lfw, le, lfe: always-in AR order and the lag windows of the candidate terms.
  n_iter: re-estimation passes of the innovations.  t0: first sample used (lets the filter bank settle).
  members: penalty scalings (applied to bic_w and bic_fw) of the averaged fits; the first one gives the links.
  cpu_reserve_s: no further member is started once this much processor time is used.
"""
import time

import numpy as np
from scipy.signal import lfilter


def _bank():
    """Fixed AR filters (a1, a2): three real poles and four complex pairs covering the model class."""
    out = [(a, 0.0) for a in (0.3, 0.5, 0.7)]
    for rad in (0.55, 0.75):
        for ang in (0.5, 1.0):
            out.append((2 * rad * np.cos(ang), -rad * rad))
    return out


BANK = _bank()
PARAMS = {"t0": 12, "ar_base": 2, "ar_max": 6, "lw": 8, "lfw": 6, "le": 4, "lfe": 3,
          "bic_a": 1.5, "bic_w": 2.0, "bic_fw": 2.0, "bic_e": 1.5, "bic_fe": 2.0, "bic_e0": 1.5,
          "n_iter": 4, "max_terms": 24, "members": (1.0, 0.75, 1.3, 0.875, 1.15), "cpu_reserve_s": 5.0}


def forward_select(y, base, cand, pen, max_terms):
    """Greedy forward selection among the columns of `cand` (penalty pen[k] each) given the always-in `base`."""
    n = len(y)
    q, _ = np.linalg.qr(base)
    ry = y - q @ (q.T @ y)
    rc = cand - q @ (q.T @ cand)
    rss = float(ry @ ry)
    chosen = []
    orig = np.einsum('ij,ij->j', cand, cand) + 1e-300
    norms = np.einsum('ij,ij->j', rc, rc)
    alive = np.ones(cand.shape[1], bool)
    while len(chosen) < max_terms and rss > 1e-12:
        ok = alive & (norms > 1e-6 * orig)
        if not ok.any():
            break
        proj = rc.T @ ry
        gain = np.where(ok, proj ** 2 / np.where(ok, norms, 1.0), 0.0)
        gain = np.minimum(gain, rss * (1 - 1e-9))
        score = -n * np.log(1.0 - gain / rss) - pen          # improvement of the criterion
        score[~ok] = -np.inf
        k = int(np.argmax(score))
        if score[k] <= 0:
            break
        chosen.append(k)
        alive[k] = False
        v = rc[:, k] / np.sqrt(norms[k])
        ry = ry - v * (v @ ry)
        rc = rc - np.outer(v, v @ rc)
        norms = np.einsum('ij,ij->j', rc, rc)
        rss = float(ry @ ry)
    return chosen


def fit(w, r, P):
    """Sparse VARMAX by extended least squares. Returns, per node, (terms, coefficients)."""
    x = w - r
    N, m = w.shape
    rows = np.arange(P["t0"], N)
    pen = np.log(len(rows))
    eps = np.zeros((N, m))
    models = [None] * m
    FW = [lfilter([1.0], [1.0, -a1, -a2], w, axis=0) for (a1, a2) in BANK]
    for it in range(P["n_iter"] + 1):
        new_eps = np.zeros((N, m))
        FE = [lfilter([1.0], [1.0, -a1, -a2], eps, axis=0) for (a1, a2) in BANK]
        for j in range(m):
            y = x[rows, j]
            base_terms = [("a", k) for k in range(1, P["ar_base"] + 1)]
            cand_terms, pens = [], []
            for k in range(P["ar_base"] + 1, P["ar_max"] + 1):
                cand_terms.append(("a", k)); pens.append(P["bic_a"])
            for i in range(m):
                if i != j:
                    for l in range(1, P["lw"] + 1):
                        cand_terms.append(("w", i, l)); pens.append(P["bic_w"])
            for i in range(m):
                for f in range(len(BANK)):
                    for l in range(2, P["lfw"] + 1):
                        cand_terms.append(("fw", i, l, f)); pens.append(P["bic_fw"])
            if it > 0:                                   # innovation terms need a first estimate of the innovations
                for s in range(m):
                    for f in range(len(BANK)):
                        for l in range(1, P["lfe"] + 1):
                            cand_terms.append(("fe", s, l, f)); pens.append(P["bic_fe"])
                for s in range(m):
                    for l in range(1, P["le"] + 1):
                        cand_terms.append(("e", s, l)); pens.append(P["bic_e"])
                for s in range(m):
                    if s != j:
                        cand_terms.append(("e", s, 0)); pens.append(P["bic_e0"])

            def col(t):
                if t[0] == "a":
                    return x[rows - t[1], j]
                if t[0] == "w":
                    return w[rows - t[2], t[1]]
                if t[0] == "fw":
                    return FW[t[3]][rows - t[2], t[1]]
                if t[0] == "fe":
                    return FE[t[3]][rows - t[2], t[1]]
                return eps[rows - t[2], t[1]]

            base = np.column_stack([col(t) for t in base_terms])
            cand = np.column_stack([col(t) for t in cand_terms])
            ch = forward_select(y, base, cand, np.array(pens) * pen, P["max_terms"])
            terms = base_terms + [cand_terms[k] for k in ch]
            X = np.column_stack([base] + [cand[:, k] for k in ch]) if ch else base
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            past = np.array([not (t[0] == "e" and t[2] == 0) for t in terms])
            new_eps[rows, j] = y - X[:, past] @ coef[past]          # innovation: prediction from the past only
            models[j] = (terms, coef)
        eps = new_eps
    return models


def simulate(models, m, exc, L=8):
    """Response (Q, T, m) of the deterministic part of the model to the excitations exc (Q, T, m), from rest."""
    virt = sorted({(t[1], t[3]) for terms, _ in models for t in terms if t[0] == "fw"})      # (source, filter)
    vidx = {v: m + k for k, v in enumerate(virt)}
    nt = m + len(virt)
    a_lag = np.zeros((L, nt))
    w_lag = np.zeros((L, nt, nt))
    for (p, f), v in vidx.items():                 # virtual node: v(t) = a1 v(t-1) + a2 v(t-2) + w_p(t-1)
        a_lag[0, v], a_lag[1, v] = BANK[f]
        w_lag[0, v, p] = 1.0
    for j, (terms, coef) in enumerate(models):
        for t, c in zip(terms, coef):
            if t[0] == "a":
                a_lag[t[1] - 1, j] += c
            elif t[0] == "w":
                w_lag[t[2] - 1, j, t[1]] += c
            elif t[0] == "fw":
                w_lag[t[2] - 2, j, vidx[(t[1], t[3])]] += c
    Q, T, _ = exc.shape
    full = np.zeros((Q, T, nt))
    full[:, :, :m] = exc
    xs = np.zeros((Q, T + L, nt))
    ws = np.zeros((Q, T + L, nt))
    for t in range(T):
        acc = np.zeros((Q, nt))
        for l in range(L):
            acc += a_lag[l] * xs[:, t + L - 1 - l] + ws[:, t + L - 1 - l] @ w_lag[l].T
        np.clip(acc, -1e3, 1e3, out=acc)             # an unstable fit must not overflow
        xs[:, t + L] = acc
        ws[:, t + L] = acc + full[:, t]
    return ws[:, L:, :m]


def controller(obs: dict) -> dict:
    start = time.process_time()
    w = np.asarray(obs["w"], float)
    r = np.asarray(obs["r"], float)
    signals = np.asarray(obs["query_signal"], float)
    Q, T = signals.shape
    m = int(obs["n_measured"])
    exc = np.zeros((Q, T, m))
    for q, node in enumerate(np.asarray(obs["query_node"], int)):
        exc[q, :, int(node)] = signals[q]
    responses, links = [], []
    for k, scale in enumerate(PARAMS["members"]):
        if k > 0 and time.process_time() - start > PARAMS["cpu_reserve_s"]:
            break
        P = dict(PARAMS, bic_w=PARAMS["bic_w"] * scale, bic_fw=PARAMS["bic_fw"] * scale)
        models = fit(w, r, P)
        resp = simulate(models, m, exc)
        responses.append(np.nan_to_num(resp, nan=0.0, posinf=0.0, neginf=0.0))
        if k == 0:
            links = sorted({(int(t[1]), int(j)) for j, (terms, _) in enumerate(models)
                            for t in terms if t[0] in ("w", "fw") and t[1] != j})
    return {"links": [list(p) for p in links], "response": np.mean(responses, axis=0)}
