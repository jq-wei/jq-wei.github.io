"""Identification of a linear dynamic network (all nodes measured): Bayesian structure search + joint estimation.

There is no control law here: `controller(obs)` is an identification procedure for one hidden network.

Model class (given): for every node j, with x_j = w_j - r_j,

    x_j(t) = a_1 x_j(t-1) [+ a_2 x_j(t-2)] + sum_{i in N_j} sum_{k<nb} b_k w_i(t-d-k) + e_j(t)

with |N_j| <= 3, delay d in 1..3, nb in 1..2 taps, e white in time and correlated across nodes at the same
instant. Every regressor lies in the past of e_j(t), so least squares on a node equation is consistent; the
difficulty is the structure (which neighbours, which lags, which AR order) at this record length.

Procedure
  1. Exhaustive search. Per node, *every* structure of the class (AR order 1/2 x up to three neighbours x six lag
     patterns each, about 39 000 per node) is fitted by least squares from the Gram matrix of the candidate
     regressors, in batches, and ranked by  N log RSS + bic_weight * k log N - 2 * prior_weight * log prior(structure).
  2. Evidence. For the `shortlist` best structures the marginal likelihood is computed with the coefficient
     distribution of the model class (pole and gain ranges, minimum-phase / inverse-response tap pairs): the Gaussian
     likelihood around the least-squares estimate is integrated against that distribution by Monte Carlo
     (`mc_draws` fixed draws). This gives a posterior probability for every plausible structure of the node.
  3. Noise groups. The innovations of a node are partly revealed by the residuals of the nodes it is correlated
     with (|correlation| > `cf_corr`); steps 1-2 are repeated `cf_rounds` times with those residuals as extra
     regressors, which lowers the noise of the node equation and makes weak links and second taps visible.
  4. Estimation. Given a structure for every node, the coefficients are estimated by least squares and by feasible
     generalised least squares over the whole network (SUR, `gls_iters` passes, residual correlations below
     `corr_floor` set to zero) and blended: least squares + `gls_mix` * (GLS - least squares). GLS is the more
     accurate of the two on average; the blend keeps most of that gain and stays close to least squares.
  5. Answer. The response is the posterior mean over the structures the record does not tell apart: `bma_draws`
     joint structures are drawn node by node in proportion to their posterior probability (tempered by
     `bma_temp`), each is estimated (step 4) and simulated from rest, and the simulations are averaged. A link is
     reported when the structures that contain it carry more than half of the posterior probability of its node.

PARAMS (there are no control gains; these are the settings of the procedure)
  bic_weight, prior_weight  ranking of step 1 (penalty per coefficient, weight of the structure prior).
  shortlist                 number of structures per node whose evidence is computed (step 2).
  mc_draws, margin          Monte Carlo draws of step 2; relative widening of the parameter ranges of the class.
  cf_rounds, cf_corr        passes of step 3 and the residual correlation above which a node is used as a partner.
  gls_iters, corr_floor, gls_mix   step 4.
  bma_draws, bma_temp       step 5 (0 or 1 draws: the single best structure).
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # small matrices: threads only cost time
    os.environ.setdefault(_var, "1")

import time
from itertools import combinations, product
from math import comb

import numpy as np

PARAMS = {"bic_weight": 1.0, "prior_weight": 1.0, "shortlist": 80, "mc_draws": 1500, "margin": 0.03,
          "cf_rounds": 2, "cf_corr": 0.25,
          "gls_iters": 2, "corr_floor": 0.12, "gls_mix": 0.7,
          "bma_draws": 40, "bma_temp": 1.0}

MAX_LAG = 4              # largest lag of a neighbour's signal in a node equation (delay 3, two taps)
MAX_SOURCES = 9          # candidate neighbours searched exhaustively per node (all of them in a ten-node network)
LAG_SETS = [(1,), (2,), (3,), (1, 2), (2, 3), (3, 4)]      # lags of the source signal for (delay, taps) of one link
_DELAY_P = {1: 0.6, 2: 0.25, 3: 0.15}                       # the structure prior of the model class
_LAG_LOGP = [np.log(_DELAY_P[s[0]] * 0.5) for s in LAG_SETS]
_DEGREE_P = {1: 0.35, 2: 0.45, 3: 0.20}

_CLOCK = {"start": 0.0, "limit": 10.0}                      # processor time of the current call
_TABLES = {}
_DRAWS = {}


def _normal_draws(m):
    """A fixed table of standard normal draws (the same for every structure, so comparisons are not noisy)."""
    if m not in _DRAWS:
        _DRAWS[m] = np.random.default_rng(12345).standard_normal((m, 2 + 2 * 3))
    return _DRAWS[m]


def _prior_density(th, p, tap_counts):
    """Density of the model class's coefficient distribution at the rows of `th` (M, k), for a structure with AR order
    `p` and links with `tap_counts` taps: pole in [0.2, 0.8] or complex pair (radius [0.4, 0.8], angle [0.3, 1.3]);
    link DC gain of magnitude [0.3, 0.9]; two taps either minimum phase (zero in [-0.9, -0.3]) or not (zero in [2, 4]).
    The supports are widened by PARAMS["margin"] (relative) so that the approximations made here are not brittle."""
    mg = PARAMS["margin"]
    lo = lambda v: v * (1.0 - mg)
    hi = lambda v: v * (1.0 + mg)
    if p == 1:
        a = th[:, 0]
        dens = ((a >= lo(0.2)) & (a <= hi(0.8))) / (hi(0.8) - lo(0.2))
        a_one = 1.0 - a
    else:
        a1, a2 = th[:, 0], th[:, 1]
        ok = a2 < 0
        rad = np.sqrt(np.where(ok, -a2, 1.0))
        cosang = a1 / (2.0 * rad)
        ok = ok & (rad >= lo(0.4)) & (rad <= hi(0.8)) & (cosang <= np.cos(lo(0.3))) & (cosang >= np.cos(hi(1.3)))
        sinang = np.sqrt(np.clip(1.0 - cosang ** 2, 1e-4, None))
        dens = ok / ((hi(0.8) - lo(0.4)) * (hi(1.3) - lo(0.3)) * 4.0 * rad ** 2 * sinang)
        a_one = 1.0 - a1 - a2
    a_one = np.maximum(a_one, 1e-3)
    g_lo, g_hi = lo(0.3), hi(0.9)
    pos = p
    for nb in tap_counts:
        if nb == 1:
            g = np.abs(th[:, pos]) / a_one
            dens = dens * ((g >= g_lo) & (g <= g_hi)) / (2.0 * (g_hi - g_lo) * a_one)
            pos += 1
        else:
            b0, b1 = th[:, pos], th[:, pos + 1]
            g = np.abs(b0 + b1) / a_one
            okg = (g >= g_lo) & (g <= g_hi)
            ratio = b1 / np.where(b0 == 0.0, 1e-12, b0)
            mp = okg & (ratio >= lo(0.3)) & (ratio <= hi(0.9))
            nmp = okg & (-ratio >= lo(2.0)) & (-ratio <= hi(4.0))
            scale = 2.0 * (g_hi - g_lo) * np.maximum(g, 1e-6) * a_one ** 2
            d_mp = mp * (1.0 + ratio) ** 2 / (scale * (hi(0.9) - lo(0.3)))
            d_nmp = nmp * (-ratio - 1.0) ** 2 / (scale * (hi(4.0) - lo(2.0)))
            dens = dens * (0.6 * d_mp + 0.4 * d_nmp)
            pos += 2
    return dens


def _log_evidence(gram, cvec, yy, n_rows, n_extra, idx, logp, desc):
    """Log marginal likelihood (up to a constant) of one structure: Gaussian approximation of the likelihood around
    the least-squares estimate, integrated against the coefficient distribution of the model class by Monte Carlo."""
    k = len(idx)
    g = gram[np.ix_(idx, idx)]
    c = cvec[idx]
    theta = np.linalg.solve(g, c)
    rss = max(yy - float(c @ theta), 1e-12 * yy)
    cov = np.linalg.inv(g) * rss / max(n_rows - k - n_extra, 1)
    cov = 0.5 * (cov + cov.T)
    chol = np.linalg.cholesky(cov)
    z = _normal_draws(int(PARAMS["mc_draws"]))[:, :k]
    p, _, sets = desc
    dens = _prior_density(theta + z @ chol.T, p, [len(LAG_SETS[s]) for s in sets])
    mean_density = float(np.mean(dens))
    return (logp - 0.5 * n_rows * np.log(rss) + 0.5 * k * np.log(2.0 * np.pi) + float(np.sum(np.log(np.diag(chol))))
            + np.log(mean_density + 1e-30))




def _structure_tables(n_other):
    """All structures of one node, grouped by number of coefficients.

    Columns of the candidate matrix: 0, 1 = x_j(t-1), x_j(t-2); 2 + 4 m + (l - 1) = w of the m-th other node at lag l.
    Returns a list of (index array (M, size), log prior (M,), descriptor list)."""
    if n_other in _TABLES:
        return _TABLES[n_other]
    groups = {}
    for k in range(1, min(3, n_other) + 1):
        base = np.log(_DEGREE_P[k] / comb(n_other, k))
        for sources in combinations(range(n_other), k):
            for sets in product(range(len(LAG_SETS)), repeat=k):
                cols = []
                logp = base
                for m, s in zip(sources, sets):
                    cols += [2 + 4 * m + (lag - 1) for lag in LAG_SETS[s]]
                    logp += _LAG_LOGP[s]
                for p in (1, 2):
                    idx = list(range(p)) + cols
                    g = groups.setdefault(len(idx), ([], [], []))
                    g[0].append(idx)
                    g[1].append(logp + np.log(0.5))
                    g[2].append((p, sources, sets))
    tables = [(np.array(g[0], dtype=int), np.array(g[1]), g[2]) for _, g in sorted(groups.items())]
    _TABLES[n_other] = tables
    return tables


def _screen_sources(w, x, j, rows, limit):
    """At most `limit` candidate neighbours of node j, by greedy forward selection on blocks of four lags
    (only used when the network has more nodes than can be searched exhaustively)."""
    n = w.shape[1]
    others = [i for i in range(n) if i != j]
    if len(others) <= limit:
        return others
    y = x[rows, j]
    base = [x[rows - 1, j], x[rows - 2, j]]
    chosen = []
    while len(chosen) < limit:
        best = (np.inf, None)
        for i in others:
            if i in chosen:
                continue
            d = np.column_stack(base + [w[rows - lag, i] for lag in range(1, MAX_LAG + 1)])
            res = y - d @ np.linalg.lstsq(d, y, rcond=None)[0]
            rss = float(res @ res)
            if rss < best[0]:
                best = (rss, i)
        chosen.append(best[1])
        base += [w[rows - lag, best[1]] for lag in range(1, MAX_LAG + 1)]
    return sorted(chosen)


def _candidates(w, x, j, rows):
    others = _screen_sources(w, x, j, rows, MAX_SOURCES)
    cols = [x[rows - 1, j], x[rows - 2, j]]
    for i in others:
        cols += [w[rows - lag, i] for lag in range(1, MAX_LAG + 1)]
    return np.column_stack(cols), others


def _select_node(w, x, j, rows, extra=None):
    """Posterior over the structures of node j: a list of (probability, (AR order, ((source, lags), ...))), best
    first. `extra` (optional): columns that are always in the equation (residuals of the node's noise partners)."""
    z, others = _candidates(w, x, j, rows)
    y = x[rows, j]
    if extra is not None and extra.shape[1]:
        # partial the fixed columns out of the target and of every candidate
        q, _ = np.linalg.qr(extra)
        z = z - q @ (q.T @ z)
        y = y - q @ (q.T @ y)
    n_rows = len(rows)
    gram = z.T @ z
    cvec = z.T @ y
    yy = float(y @ y)
    log_n = np.log(n_rows)
    keep = int(PARAMS["shortlist"])
    short = []                                    # (score, table number, row)
    tables = _structure_tables(len(others))
    for tn, (idx, logp, desc) in enumerate(tables):
        g = gram[idx[:, :, None], idx[:, None, :]]
        c = cvec[idx]
        sol = np.linalg.solve(g, c[:, :, None])[:, :, 0]
        rss = np.maximum(yy - np.sum(c * sol, axis=1), 1e-12 * yy)
        score = n_rows * np.log(rss / n_rows) + PARAMS["bic_weight"] * idx.shape[1] * log_n - 2.0 * PARAMS["prior_weight"] * logp
        top = np.argsort(score)[:max(keep, 1)]
        short += [(float(score[m]), tn, int(m)) for m in top]
    short.sort()

    def named(desc):
        p, sources, sets = desc
        return p, tuple((others[m], LAG_SETS[s]) for m, s in zip(sources, sets))

    best_desc = tables[short[0][1]][2][short[0][2]]
    posterior = [(1.0, named(best_desc))]
    if keep > 1:
        n_extra = 0 if extra is None else extra.shape[1]
        evs, descs = [], []
        for _, tn, m in short[:keep]:
            idx, logp, desc = tables[tn]
            try:
                evs.append(_log_evidence(gram, cvec, yy, n_rows, n_extra, idx[m], logp[m], desc[m]))
                descs.append(desc[m])
            except np.linalg.LinAlgError:
                continue
        if evs:
            evs = np.array(evs)
            prob = np.exp((evs - evs.max()) / PARAMS["bma_temp"])
            prob /= prob.sum()
            order = np.argsort(-prob)
            posterior = [(float(prob[m]), named(descs[m])) for m in order if prob[m] > 1e-3]
    return posterior


def _design(w, x, j, rows, structure):
    p, links = structure
    cols = [x[rows - k, j] for k in range(1, p + 1)]
    for i, lags in links:
        cols += [w[rows - lag, i] for lag in lags]
    return np.column_stack(cols)


def _fit(w, x, rows, structures):
    """Coefficients of every node for given structures: least squares per node, feasible GLS (SUR) over the whole
    network, and the blend of the two. Returns a list of coefficient vectors (AR terms first, then the taps)."""
    n = w.shape[1]
    designs = [_design(w, x, j, rows, structures[j]) for j in range(n)]
    targets = [x[rows, j] for j in range(n)]
    coefs = [np.linalg.lstsq(designs[j], targets[j], rcond=None)[0] for j in range(n)]
    ols = coefs
    sizes = [d.shape[1] for d in designs]
    offs = np.concatenate([[0], np.cumsum(sizes)])
    for _ in range(int(PARAMS["gls_iters"])):
        resid = np.column_stack([targets[j] - designs[j] @ coefs[j] for j in range(n)])
        cov = resid.T @ resid / len(rows)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        corr[np.abs(corr) < PARAMS["corr_floor"]] = 0.0
        np.fill_diagonal(corr, 1.0)
        evals = np.linalg.eigvalsh(corr)
        if evals[0] < 0.05:                       # keep the thresholded matrix well conditioned
            corr = (corr + (0.05 - evals[0]) * np.eye(n)) / (1.0 + 0.05 - evals[0])
        prec = np.linalg.inv(corr * np.outer(sd, sd))
        total = offs[-1]
        lhs = np.zeros((total, total))
        rhs = np.zeros(total)
        for a in range(n):
            for b in range(n):
                if prec[a, b] == 0.0:
                    continue
                lhs[offs[a]:offs[a + 1], offs[b]:offs[b + 1]] = prec[a, b] * (designs[a].T @ designs[b])
                rhs[offs[a]:offs[a + 1]] += prec[a, b] * (designs[a].T @ targets[b])
        theta = np.linalg.solve(lhs, rhs)
        coefs = [theta[offs[j]:offs[j + 1]] for j in range(n)]
    mix = PARAMS["gls_mix"]
    return [ols[j] + mix * (coefs[j] - ols[j]) for j in range(n)]


def _lag_matrices(n, structures, coefs):
    a_lag = np.zeros((MAX_LAG, n))
    w_lag = np.zeros((MAX_LAG, n, n))
    for j in range(n):
        p, links = structures[j]
        a_lag[:p, j] = coefs[j][:p]
        pos = p
        for i, lags in links:
            for lag in lags:
                w_lag[lag - 1, j, i] += coefs[j][pos]
                pos += 1
    return a_lag, w_lag


def _simulate(a_lag, w_lag, node, signal):
    n_lags, n = a_lag.shape
    steps = len(signal)
    x = np.zeros((steps + n_lags, n))
    w = np.zeros((steps + n_lags, n))
    for t in range(steps):
        acc = np.zeros(n)
        for lag in range(n_lags):
            acc += a_lag[lag] * x[t + n_lags - 1 - lag] + w_lag[lag] @ w[t + n_lags - 1 - lag]
        np.clip(acc, -1e3, 1e3, out=acc)
        x[t + n_lags] = acc
        w[t + n_lags] = acc
        w[t + n_lags, node] += signal[t]
    return w[n_lags:]


def _used():
    """Processor seconds used by this call so far."""
    return time.process_time() - _CLOCK["start"]


def identify(obs):
    """Posterior over the structures of every node: a list (per node) of (probability, structure), best first."""
    w = np.asarray(obs["w"], dtype=float)
    r = np.asarray(obs["r"], dtype=float)
    x = w - r
    n_samples, n = w.shape
    rows = np.arange(MAX_LAG, n_samples)
    posteriors = [_select_node(w, x, j, rows) for j in range(n)]
    first_pass = _used()
    # refinement: the innovations of a node are partly revealed by the residuals of the nodes of its noise group
    # (taken from their best structure); with those residuals as extra regressors the equation of node j has less
    # noise and weak links stand out.
    for _ in range(int(PARAMS["cf_rounds"])):
        if _used() + 1.3 * first_pass > 0.6 * _CLOCK["limit"]:      # keep clear of the time limit on a slow machine
            break
        resid = np.column_stack([_residual(w, x, j, rows, posteriors[j][0][1]) for j in range(n)])
        for j in range(n):
            corr = np.corrcoef(resid.T)[j]
            partners = [k for k in range(n) if k != j and abs(corr[k]) > PARAMS["cf_corr"]]
            if not partners:
                continue
            posteriors[j] = _select_node(w, x, j, rows, extra=resid[:, partners])
            resid[:, j] = _residual(w, x, j, rows, posteriors[j][0][1])
    return posteriors, (w, x, rows)


def _residual(w, x, j, rows, structure):
    d = _design(w, x, j, rows, structure)
    y = x[rows, j]
    return y - d @ np.linalg.lstsq(d, y, rcond=None)[0]


def _joint_structures(posteriors, m):
    """`m` joint structures of the network, each node's structures in proportion to their posterior probability."""
    rng = np.random.default_rng(2024)
    picks = []
    for post in posteriors:
        prob = np.array([p for p, _ in post])
        prob = prob / prob.sum()
        counts = np.floor(prob * m).astype(int)
        short = m - int(counts.sum())
        if short > 0:
            counts[np.argsort(-(prob * m - counts), kind="stable")[:short]] += 1
        assign = np.repeat(np.arange(len(post)), counts)
        rng.shuffle(assign)
        picks.append(assign)
    joint = {}
    for row in np.array(picks).T:
        key = tuple(int(v) for v in row)
        joint[key] = joint.get(key, 0) + 1
    return joint


def controller(obs: dict) -> dict:
    """Identify the network of one record; return its links and its predicted response to the queries."""
    _CLOCK["start"] = time.process_time()
    _CLOCK["limit"] = float(obs.get("time_limit_s", 10.0))
    try:
        posteriors, record = identify(obs)
        return answer(obs, posteriors, record)
    except Exception:                               # never fail a network on a numerical accident
        return _fallback(obs)


def _fallback(obs):
    """A structure-blind answer (dense VARX response, no links); only used if the procedure above fails."""
    shape = (int(obs["n_queries"]), int(obs["query_length"]), int(obs["n_measured"]))
    response = np.zeros(shape)
    try:
        from hl_control.envs.network_id import dense_varx_response
        dense = np.asarray(dense_varx_response(obs), dtype=float)
        if dense.shape == shape and np.all(np.isfinite(dense)) and np.max(np.abs(dense)) < 50.0:
            response = dense
    except Exception:
        pass
    if not np.any(response):
        for q, node in enumerate(np.asarray(obs["query_node"], dtype=int)):
            response[q, :, int(node)] = np.asarray(obs["query_signal"], dtype=float)[q]
    return {"links": [], "response": response}


def answer(obs, posteriors, record):
    """Links and query responses from the posterior over the structures of every node."""
    n = int(obs["n_measured"])
    w, x, rows = record
    signals = np.asarray(obs["query_signal"], dtype=float)
    nodes = np.asarray(obs["query_node"], dtype=int)
    shape = (int(obs["n_queries"]), int(obs["query_length"]), n)

    def respond(structures):
        coefs = _fit(w, x, rows, structures)
        a_lag, w_lag = _lag_matrices(n, structures, coefs)
        return np.stack([_simulate(a_lag, w_lag, int(nodes[q]), signals[q]) for q in range(shape[0])])

    best = [post[0][1] for post in posteriors]
    response = respond(best)
    draws = int(PARAMS["bma_draws"])
    if draws > 1 and _used() < 0.7 * _CLOCK["limit"]:
        # posterior mean of the response over the structures that the record does not tell apart
        total, weight = np.zeros(shape), 0.0
        for key, count in _joint_structures(posteriors, draws).items():
            if _used() > 0.85 * _CLOCK["limit"]:
                break
            sim = respond([posteriors[j][key[j]][1] for j in range(n)])
            if np.all(np.isfinite(sim)) and np.max(np.abs(sim)) < 50.0:
                total += count * sim
                weight += count
        if weight > 0:
            response = total / weight
    # a link is reported when the structures that contain it carry more than half of the posterior probability
    links = []
    for j in range(n):
        mass = {}
        for prob, (_, node_links) in posteriors[j]:
            for i, _ in node_links:
                mass[i] = mass.get(i, 0.0) + prob
        norm = sum(prob for prob, _ in posteriors[j])
        chosen = [i for i, v in mass.items() if v > 0.5 * norm]
        if not chosen and mass:
            chosen = [max(mass, key=mass.get)]
        links += [[int(i), int(j)] for i in sorted(chosen)]
    if not np.all(np.isfinite(response)):
        response = np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)
    return {"links": links, "response": response}
