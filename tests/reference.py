"""
Independent reference implementation of the extended stratification core,
written from the method specification (thesis Chapter 2 as revised at
Checkpoint H) without reusing engine code. Arrays only, no pandas.

Specification:
  coverage      criterion active iff observed for >= 2 alternatives of the
                initial pool; alternatives observed on no active criterion
                are unstratified.
  limits        expert > historical > pool median; the pool median is
                recomputed while >= 2 pool members are observed, otherwise
                the last such median is kept (frozen).
  attainment    range R = observed pool values together with the limit;
                if R is degenerate or mode is binary: 1[x meets L];
                otherwise x' and l' are oriented positions in R and
                a = th*x'/l' (x' <= l'), th + (1-th)(x'-l')/(1-l') (x' > l'),
                with l' = 0 -> th + (1-th) x', l' = 1 -> th x'.
  calibrated    th in [lo, hi] on a 0.01 grid maximizing the population
                variance over the pool, ties to the midpoint.
  K             weighted mean over observed active criteria.
  delta         smallest positive weight among criteria observed in the pool.
  tier          K_max - K < delta - 1e-9, or K_max - K <= 1e-9.
  guard         pool normalization (oriented min max over observed pool
                values; constant criteria noninformative); a member strictly
                weaker than another member on every shared informative
                criterion (at least q_min of them) is demoted; never empty.
  target        min weighted distance to the ideal (1 on every informative
                criterion with an observation) over informative criteria;
                ties (1e-12) by name.
  S             weighted distance to the target over shared informative
                criteria; S_target = 0; ties by name.
  margin        min K in tier - max K of the rest.
"""
import numpy as np

EPS, TIE = 1e-9, 1e-12


def _deg(x, l, th):
    if abs(l) < 1e-12:
        return th + (1 - th) * x
    if abs(l - 1) < 1e-12:
        return th * x
    return th * x / l if x <= l else th + (1 - th) * (x - l) / (1 - l)


def _pos(v, lo, hi, d):
    x = (v - lo) / (hi - lo)
    x = min(max(x, 0.0), 1.0)
    return 1 - x if d < 0 else x


def _calib(xs, l, lo, hi):
    mid = (lo + hi) / 2
    if len(xs) < 2:
        return mid
    best_t, best_v = mid, -1.0
    for t in np.round(np.arange(lo, hi + 1e-9, 0.01), 2):
        v = float(np.var([_deg(x, l, t) for x in xs]))
        if v > best_v + 1e-12 or (abs(v - best_v) <= 1e-12 and abs(t - mid) < abs(best_t - mid)):
            best_v, best_t = v, float(t)
    return best_t


def reference(X, d, w, names, Lexp=None, Lhist=None, mode="graded_fixed", theta=0.3,
              bounds=(0.3, 0.7), guard=True, qmin=1):
    Lexp, Lhist = Lexp or {}, Lhist or {}
    m, n = X.shape
    obs0 = ~np.isnan(X)
    active = [j for j in range(n) if obs0[:, j].sum() >= 2]
    w = np.asarray(w, float)
    wa = np.zeros(n)
    tot = sum(max(w[j], 0) for j in active)
    for j in active:
        wa[j] = (max(w[j], 0) / tot) if tot > 0 else 1 / len(active)
    strat = [i for i in range(m) if any(obs0[i, j] and wa[j] > 0 for j in active)]
    unstrat = [names[i] for i in range(m) if i not in strat]
    pool = list(strat)
    frozen = {}
    tiers, S_all, margins, targets = {}, {}, {}, {}
    t = 0
    while pool:
        t += 1
        L = {}
        for j in active:
            vals = [X[i, j] for i in pool if obs0[i, j]]
            if j in Lexp:
                L[j] = Lexp[j]
            elif j in Lhist:
                L[j] = Lhist[j]
            elif len(vals) >= 2:
                frozen[j] = float(np.median(vals)); L[j] = frozen[j]
            elif j in frozen:
                L[j] = frozen[j]
            else:
                L[j] = float(np.median(vals)) if vals else np.nan
        # attainment
        A = {i: {} for i in pool}
        for j in active:
            vals = [X[i, j] for i in pool if obs0[i, j]]
            if not vals:
                continue
            span = vals + ([L[j]] if not np.isnan(L[j]) else [])
            lo, hi = min(span), max(span)
            graded = mode != "binary" and not np.isnan(L[j]) and not np.isclose(lo, hi)
            if graded:
                lp = _pos(L[j], lo, hi, d[j])
                xs = {i: _pos(X[i, j], lo, hi, d[j]) for i in pool if obs0[i, j]}
                th = theta if mode == "graded_fixed" else _calib(list(xs.values()), lp, *bounds)
                for i, x in xs.items():
                    A[i][j] = _deg(x, lp, th)
            else:
                for i in pool:
                    if obs0[i, j]:
                        A[i][j] = float(X[i, j] >= L[j]) if d[j] > 0 else float(X[i, j] <= L[j])
        K = {}
        for i in pool:
            js = [j for j in A[i] if wa[j] > 0]
            K[i] = sum(wa[j] * A[i][j] for j in js) / sum(wa[j] for j in js) if js else np.nan
        comp = [i for i in pool if not np.isnan(K[i])] or list(pool)
        for i in pool:
            if np.isnan(K[i]):
                K[i] = 0.0 if comp == pool else K[i]
        obs_now = [j for j in active if any(obs0[i, j] for i in pool)]
        delta = min([wa[j] for j in obs_now if wa[j] > 0], default=1 / max(len(active), 1))
        km = max(K[i] for i in comp)
        cand = [i for i in comp if (km - K[i]) < delta - EPS or (km - K[i]) <= EPS]
        # pool normalization
        N, flat = {i: {} for i in pool}, set()
        for j in active:
            vals = [X[i, j] for i in pool if obs0[i, j]]
            if not vals:
                flat.add(j); continue
            lo, hi = min(vals), max(vals)
            if np.isclose(lo, hi):
                flat.add(j)
            for i in pool:
                if obs0[i, j]:
                    N[i][j] = 0.0 if np.isclose(lo, hi) else _pos(X[i, j], lo, hi, d[j])
        inf = [j for j in active if j not in flat]
        mem = list(cand)
        if guard and len(cand) > 1:
            for a in cand:
                for b in cand:
                    if a == b:
                        continue
                    sh = [j for j in inf if j in N[a] and j in N[b]]
                    if len(sh) >= qmin and all(N[a][j] < N[b][j] for j in sh):
                        if a in mem:
                            mem.remove(a)
                        break
            if not mem:
                mem = list(cand)

        def dist(a, ref):
            js = [j for j in inf if wa[j] > 0 and j in N[a] and ref.get(j) is not None]
            if not js:
                return np.nan
            return sum(wa[j] * abs(N[a][j] - ref[j]) for j in js) / sum(wa[j] for j in js)

        ideal = {j: 1.0 for j in inf if any(j in N[i] for i in pool)}

        def key(a, dd):
            v = dd[a]
            return (1, 0.0, names[a]) if np.isnan(v) else (0, round(v / TIE) * TIE, names[a])
        di = {a: dist(a, ideal) for a in mem}
        T = min(mem, key=lambda a: key(a, di))
        S = {a: dist(a, N[T]) for a in mem}
        S[T] = 0.0
        for a in mem:
            tiers[names[a]] = t
            S_all[names[a]] = S[a]
        targets[t] = names[T]
        rest = [i for i in pool if i not in mem and not np.isnan(K[i])]
        if rest:
            margins[t] = min(K[i] for i in mem) - max(K[i] for i in rest)
        pool = [i for i in pool if i not in mem]
    return tiers, S_all, margins, targets, unstrat
