"""
protocol.py -- The censoring-aware inventory / time-on-shelf protocol and the
practitioner baselines it is benchmarked against (MethodsX paper #5).

Pure NumPy/Pandas. ASCII-only. No lifelines: Kaplan-Meier, Greenwood variance,
and RMST are hand-rolled from first principles so the release is self-contained.

Estimators implemented
----------------------
Protocol
  * reconstruct_stock         Step 1: accounting-identity month-end stock.
  * km_dwell                  Step 2: Kaplan-Meier S(t) + Greenwood variance.
  * rmst                      Step 2: restricted mean survival time up to tau.
  * km_median                 Step 2: median time-on-shelf from the KM curve.
  * depletion_forecast        Step 3 (HEADLINE): age-conditioned residual-survival
                              depletion projection of the current censored stock.

Baselines
  * sold_only_dwell           (b1) mean/median time-on-shelf dropping censored items.
  * censor_at_end_dwell       (b2) treat censored items as SOLD at W_end.
  * zero_opening_net_flow     (b3) cumulative arrivals - sales, zero opening stock.
  * age_naive_depletion       (b4) depletion from UNCONDITIONAL S(h), ignoring age.
  * accounting_only_depletion (b5) current stock depletes at the historical
                              average monthly sale count.

Conventions
-----------
Durations are in DAYS. An item's observed record is:
  * SOLD (event=1)     : exit_date <= W_end; duration = exit_date - entry_date.
  * CENSORED (event=0) : exit_date is NaT; duration = W_end - entry_date.
Right-censoring here is administrative (unsold at window end), so the KM
independent-censoring assumption holds by construction in the synthetic design.
"""

import numpy as np
import pandas as pd


DAY_NS = np.int64(86400) * np.int64(1_000_000_000)  # nanoseconds per day


def _to_ns(dt_like):
    """int64 nanoseconds-since-epoch at ns resolution (pandas 2.x/3.x safe)."""
    return pd.DatetimeIndex(dt_like).as_unit("ns").asi8.astype(np.int64)


def _exit_ns_float(exit_col):
    """
    Convert an exit-date column that may contain NaT into float nanoseconds with
    NaT mapped to np.nan. NOTE: DatetimeIndex.asi8 returns the integer sentinel
    iNaT (int64 min) for NaT, NOT NaN -- casting that to float would produce a
    huge negative number, silently mis-classifying censored items as sold. We
    therefore detect NaT explicitly via isna() and overwrite with NaN.
    """
    idx = pd.DatetimeIndex(exit_col).as_unit("ns")
    ns = idx.asi8.astype("float64")
    ns[np.asarray(idx.isna())] = np.nan
    return ns


# ==============================================================================
# Duration extraction
# ==============================================================================

def observed_durations(events, w_end):
    """
    From the analyst's event log, derive observed time-on-shelf durations (days) and
    the event indicator.

    Parameters
    ----------
    events : DataFrame with columns [entry_date, exit_date] (exit_date NaT if
             right-censored).
    w_end  : pd.Timestamp, observation window end.

    Returns
    -------
    durations : float ndarray (days), duration = (exit or W_end) - entry.
    event     : int ndarray, 1 if sold (exit observed), 0 if censored.
    """
    entry_ns = _to_ns(events["entry_date"])
    exit_ns = _exit_ns_float(events["exit_date"])  # NaT -> NaN (correctly)
    w_end_ns = float(pd.Timestamp(w_end).value)

    sold = ~np.isnan(exit_ns)
    end_ns = np.where(sold, exit_ns, w_end_ns)
    durations = (end_ns - entry_ns.astype(float)) / float(DAY_NS)
    durations = np.maximum(durations, 0.0)
    event = sold.astype(int)
    return durations, event


# ==============================================================================
# Step 1 -- Accounting-identity stock reconstruction
# ==============================================================================

def reconstruct_stock(events, month_ends):
    """
    In_Stock(t) = count of items with entry <= t AND (obs_exit is NaT OR
    obs_exit > t), evaluated at each month-end t.

    This is the deterministic stock-flow accounting identity (Step 1). Under
    complete linked events and zero opening stock it equals true stock exactly.

    Parameters
    ----------
    events     : DataFrame [entry_date, exit_date] (exit_date NaT if censored).
    month_ends : DatetimeIndex of evaluation timestamps (month-ends).

    Returns
    -------
    pd.Series indexed by month_ends with the reconstructed stock count.
    """
    entry_ns = _to_ns(events["entry_date"])
    exit_ns = _exit_ns_float(events["exit_date"])  # NaT -> NaN (correctly)
    t_ns = _to_ns(month_ends).astype(float)

    entries_sorted = np.sort(entry_ns.astype(float))
    # sold items only contribute a "removal" at their exit time
    sold_exits = np.sort(exit_ns[~np.isnan(exit_ns)])

    n_entered = np.searchsorted(entries_sorted, t_ns, side="right")
    if sold_exits.size > 0:
        n_exited = np.searchsorted(sold_exits, t_ns, side="right")
    else:
        n_exited = np.zeros_like(t_ns)
    stock = (n_entered - n_exited).astype(int)
    return pd.Series(stock, index=pd.DatetimeIndex(month_ends), name="reconstructed_stock")


# ==============================================================================
# Step 2 -- Kaplan-Meier + Greenwood variance + RMST
# ==============================================================================

def km_dwell(durations, event_observed):
    """
    HAND-ROLLED Kaplan-Meier product-limit estimator with Greenwood variance.

    At each distinct EVENT (death/sale) time t_j:
        n_j = number at risk just before t_j (duration >= t_j),
        d_j = number of events (sales) exactly at t_j,
        S(t_j) = prod_{k<=j} (1 - d_k / n_k).
    Greenwood:
        Var[S(t_j)] = S(t_j)^2 * sum_{k<=j} d_k / (n_k * (n_k - d_k)).

    Censored observations contribute to the risk set until their censoring time
    but never create a step. Ties handled by aggregating events/censorings at the
    same time.

    Parameters
    ----------
    durations      : array of durations (days).
    event_observed : array {1=sale, 0=censored}.

    Returns
    -------
    dict with:
      't'        : ndarray of distinct EVENT times (ascending), prefixed with 0.
      'S'        : ndarray survival at those times (S(0)=1).
      'var'      : ndarray Greenwood variance of S at those times.
      'n_risk'   : ndarray number at risk at each event time.
      'd'        : ndarray number of events at each event time.
      'n'        : total sample size.
      'n_events' : total number of events (sales).
    Represents a right-continuous step function: S is constant on [t_j, t_{j+1}).
    """
    durations = np.asarray(durations, dtype=float)
    event = np.asarray(event_observed, dtype=int)
    n_total = durations.size

    # Distinct EVENT times only (KM steps down only at observed deaths).
    event_times = np.unique(durations[event == 1])
    if event_times.size == 0:
        # No events at all: flat survival at 1.
        return {
            "t": np.array([0.0]),
            "S": np.array([1.0]),
            "var": np.array([0.0]),
            "n_risk": np.array([n_total]),
            "d": np.array([0]),
            "n": n_total,
            "n_events": 0,
        }

    # Vectorised risk-set and death counts (O(n log n) via sorting), giving the
    # same result as the naive per-event-time scan but far faster since km_dwell
    # is called thousands of times across the replications.
    dur_sorted = np.sort(durations)
    # n_j = # with duration >= t_j = n_total - # with duration < t_j
    n_risk_arr = n_total - np.searchsorted(dur_sorted, event_times, side="left")
    # d_j = # of EVENTS exactly at t_j
    ev_times_all = np.sort(durations[event == 1])
    lo = np.searchsorted(ev_times_all, event_times, side="left")
    hi = np.searchsorted(ev_times_all, event_times, side="right")
    d_arr = (hi - lo).astype(float)
    n_risk_f = n_risk_arr.astype(float)

    # Product-limit survival and Greenwood variance.
    # Keep only steps with n_j > 0 (all should satisfy this by construction).
    valid = n_risk_f > 0
    event_times = event_times[valid]
    n_risk_f = n_risk_f[valid]
    d_arr = d_arr[valid]

    factors = 1.0 - d_arr / n_risk_f
    Sv_events = np.cumprod(factors)
    # Greenwood cumulative sum term d_k / (n_k (n_k - d_k)); guard n_k - d_k == 0
    # (the last event time can have n_k == d_k, where the term is undefined and is
    # conventionally treated as 0). Use masked divide to avoid a spurious warning.
    denom = n_risk_f * (n_risk_f - d_arr)
    green_terms = np.zeros_like(denom, dtype=float)
    np.divide(d_arr, denom, out=green_terms, where=denom > 0)
    cum_green = np.cumsum(green_terms)
    var_events = Sv_events * Sv_events * cum_green

    # Prefix with (t=0, S=1).
    t = np.concatenate([[0.0], event_times.astype(float)])
    Sv = np.concatenate([[1.0], Sv_events])
    var = np.concatenate([[0.0], var_events])
    n_risk = np.concatenate([[n_total], n_risk_f.astype(int)])
    d = np.concatenate([[0], d_arr.astype(int)])
    return {
        "t": t, "S": Sv, "var": var, "n_risk": n_risk, "d": d,
        "n": n_total, "n_events": int(np.sum(event == 1)),
    }


def km_survival_at(km, times):
    """
    Evaluate the KM step function S(t) at arbitrary query times (right-continuous:
    S is constant on [t_j, t_{j+1}), so S(query) = S at the largest event time
    <= query). For query < 0 returns 1. For query beyond the last event time we
    return the last KM value (flat-tail extension); callers that need a different
    tail must handle it explicitly.
    """
    t = km["t"]
    S = km["S"]
    times = np.asarray(times, dtype=float)
    # index of largest t_j <= query
    idx = np.searchsorted(t, times, side="right") - 1
    idx = np.clip(idx, 0, len(S) - 1)
    out = S[idx]
    out = np.where(times < 0, 1.0, out)
    return out


def km_median(km):
    """
    Median time-on-shelf = smallest event time t with S(t) <= 0.5. If the KM curve never
    reaches 0.5 (heavy right-censoring), the median is not identified; return NaN
    (honest: do not fabricate a median beyond KM support).
    """
    t, S = km["t"], km["S"]
    below = np.where(S <= 0.5)[0]
    if below.size == 0:
        return np.nan
    return float(t[below[0]])


def rmst(km, tau):
    """
    Restricted mean survival time up to horizon tau:
        RMST(tau) = integral_0^tau S(u) du,
    computed as a step-sum of the right-continuous KM curve. S is constant on
    [t_j, t_{j+1}); the last piece is truncated at tau.

    Parameters
    ----------
    km  : output of km_dwell.
    tau : horizon (days).

    Returns
    -------
    float RMST in day-units.
    """
    t, S = km["t"], km["S"]
    tau = float(tau)
    # Build breakpoints within [0, tau].
    # area = sum over pieces S(t_j) * (min(t_{j+1}, tau) - t_j) for t_j < tau.
    area = 0.0
    for j in range(len(t)):
        left = t[j]
        if left >= tau:
            break
        right = t[j + 1] if (j + 1) < len(t) else tau
        right = min(right, tau)
        if right > left:
            area += S[j] * (right - left)
    return float(area)


# ==============================================================================
# Step 3 (HEADLINE) -- age-conditioned residual-survival depletion forecast
# ==============================================================================

def _weibull_tail_fit(km):
    """
    Fit a 2-parameter Weibull to the KM curve for tail extrapolation beyond KM
    support. We fit by ordinary least squares on the linearised survival:
        ln(-ln S(t)) = k * ln(t) - k * ln(lambda),
    using KM points with 0 < S < 1 and t > 0. Returns (k, lambda) or None if the
    fit is not usable. Kept as a lightweight fallback; the misspecification study
    uses the proper CENSORED-MLE fits below (weibull_mle / lognormal_mle).
    """
    t, S = km["t"], km["S"]
    mask = (t > 0) & (S > 1e-6) & (S < 1.0 - 1e-9)
    if mask.sum() < 3:
        return None
    x = np.log(t[mask])
    y = np.log(-np.log(S[mask]))
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    k = coef[0]
    intercept = coef[1]
    if not np.isfinite(k) or k <= 0:
        return None
    lam = np.exp(-intercept / k)
    if not np.isfinite(lam) or lam <= 0:
        return None
    return float(k), float(lam)


# ==============================================================================
# Parametric survival tails fit by CENSORED MAXIMUM LIKELIHOOD (numpy-only).
# These are the real tail models for the misspecification study. Each fit uses
# BOTH sold (exact) and censored (right-censored) observed durations, so the fitted
# survival is a proper censored-data estimate, not a complete-case fit.
# ==============================================================================

def weibull_mle(durations, event, max_iter=200, tol=1e-8):
    """
    Censored MLE for a Weibull(shape k, scale lam). With sold times t_i (event=1)
    and right-censored times c_j (event=0), the profile likelihood in k has the
    stationary equation
        1/k + mean_sold(ln t) - [sum_all w_i ln t_i] / [sum_all w_i] = 0,
    where w_i = t_i^k and the scale is lam = ( sum_all t_i^k / d )^(1/k), d = #sold.
    Solved by a guarded Newton/bisection on k. Returns dict {k, lam, loglik, n_sold}
    or None if not estimable (needs >=2 sold items).
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(event, dtype=int)
    pos = t > 0
    t, e = t[pos], e[pos]
    d = int(e.sum())
    if d < 2:
        return None
    lt = np.log(t)
    sold = e == 1
    mean_lt_sold = float(np.mean(lt[sold]))

    def g(k):
        tk = np.exp(k * lt)  # t^k, stable
        Stk = float(np.sum(tk))
        num = float(np.sum(tk * lt))
        return 1.0 / k + mean_lt_sold - num / Stk

    # bracket k in [lo, hi] with sign change (g decreasing in k)
    lo, hi = 1e-3, 1e-3
    glo = g(lo)
    hi = 0.05
    ghi = g(hi)
    n_exp = 0
    while ghi > 0 and n_exp < 100:
        hi *= 1.5
        ghi = g(hi)
        n_exp += 1
    if glo < 0:  # degenerate; g should be positive at very small k
        lo = 1e-4
        glo = g(lo)
    if ghi > 0:
        k = hi  # could not bracket; take last
    else:
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            gm = g(mid)
            if abs(gm) < tol:
                lo = hi = mid
                break
            if gm > 0:
                lo = mid
            else:
                hi = mid
        k = 0.5 * (lo + hi)
    if not np.isfinite(k) or k <= 0:
        return None
    tk = np.exp(k * lt)
    lam = (float(np.sum(tk)) / d) ** (1.0 / k)
    if not np.isfinite(lam) or lam <= 0:
        return None
    # log-likelihood (for AIC/BIC)
    z = (t / lam) ** k
    ll_sold = np.log(k) - k * np.log(lam) + (k - 1.0) * lt[sold] - z[sold]
    ll_cens = -z[~sold]
    loglik = float(np.sum(ll_sold) + np.sum(ll_cens))
    return {"k": float(k), "lam": float(lam), "loglik": loglik,
            "n_sold": d, "n_params": 2}


def _norm_pdf(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)


def _norm_sf(z):
    """Standard-normal survival 1-Phi(z), numpy-only via erfc-equivalent series.

    Uses the relation 1-Phi(z) = 0.5*erfc(z/sqrt2). numpy has no erfc, so we use a
    high-accuracy rational approximation (Abramowitz & Stegun 7.1.26 for erf).
    """
    # erf approximation (A&S 7.1.26), max abs error ~1.5e-7
    x = z / np.sqrt(2.0)
    sign = np.sign(x)
    ax = np.abs(x)
    tt = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * tt - 1.453152027) * tt) + 1.421413741) * tt
                - 0.284496736) * tt + 0.254829592) * tt * np.exp(-ax * ax)
    erf = sign * y
    return 0.5 * (1.0 - erf)


def lognormal_mle(durations, event, max_iter=500, tol=1e-9):
    """
    Censored MLE for a Log-normal(mu, sigma) on log-durations. Sold items are exact
    Gaussian observations of ln t; censored items are right-censored Gaussians. The
    log-likelihood is
        sum_sold [ -ln(sigma t) + ln phi((ln t - mu)/sigma) ]
        + sum_cens [ ln SF((ln c - mu)/sigma) ].
    Optimised by coordinate ascent: given sigma, the sold-only closed form gives a
    good mu start; then a short Newton on (mu, sigma) via numerical gradient. To
    stay numpy-only and robust we use a bounded grid + local refine. Returns dict
    {mu, sigma, loglik, n_sold, n_params} or None.
    """
    t = np.asarray(durations, dtype=float)
    e = np.asarray(event, dtype=int)
    pos = t > 0
    t, e = t[pos], e[pos]
    d = int(e.sum())
    if d < 2:
        return None
    y = np.log(t)
    sold = e == 1
    ys = y[sold]
    yc = y[~sold]

    def negll(mu, sigma):
        if sigma <= 1e-6:
            return np.inf
        zs = (ys - mu) / sigma
        # sold: -ln(sigma) - ln t + ln phi(zs); the -ln t is constant in params
        ll_s = np.sum(-np.log(sigma) - 0.5 * zs * zs - np.log(np.sqrt(2 * np.pi)))
        if yc.size:
            zc = (yc - mu) / sigma
            sf = np.clip(_norm_sf(zc), 1e-300, 1.0)
            ll_c = np.sum(np.log(sf))
        else:
            ll_c = 0.0
        return -(ll_s + ll_c)

    # start from sold-only moments
    mu0 = float(np.mean(ys))
    sig0 = float(np.std(ys)) if ys.size > 1 else 1.0
    sig0 = max(sig0, 1e-2)
    # Nelder-Mead-lite: coordinate descent with shrinking steps (numpy-only).
    mu, sigma = mu0, max(sig0, 0.3)
    step_mu, step_sig = max(0.5 * sig0, 0.2), max(0.3 * sig0, 0.1)
    best = negll(mu, sigma)
    for _ in range(max_iter):
        improved = False
        for dmu in (step_mu, -step_mu):
            cand = negll(mu + dmu, sigma)
            if cand < best - tol:
                mu += dmu; best = cand; improved = True; break
        for dsig in (step_sig, -step_sig):
            if sigma + dsig > 1e-3:
                cand = negll(mu, sigma + dsig)
                if cand < best - tol:
                    sigma += dsig; best = cand; improved = True; break
        if not improved:
            step_mu *= 0.5
            step_sig *= 0.5
            if step_mu < 1e-6 and step_sig < 1e-6:
                break
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return None
    return {"mu": float(mu), "sigma": float(sigma), "loglik": float(-best),
            "n_sold": d, "n_params": 2}


def _weibull_survival(x, k, lam):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    with np.errstate(over="ignore"):
        return np.exp(-((x / lam) ** k))


def _lognormal_survival(x, mu, sigma):
    x = np.asarray(x, dtype=float)
    out = np.ones_like(x, dtype=float)
    m = x > 0
    z = (np.log(np.where(m, x, 1.0)) - mu) / sigma
    out[m] = np.clip(_norm_sf(z[m]), 0.0, 1.0)
    return out


def fit_parametric_tail(durations, event, family):
    """
    Fit a parametric survival model of the given family by censored MLE and return
    a dict with the fitted params, an S(x) callable, and AIC/BIC. family in
    {'weibull','lognormal'}. Returns None if not estimable.
    AIC = 2p - 2 loglik ; BIC = p ln(n_sold) - 2 loglik.
    """
    if family == "weibull":
        fit = weibull_mle(durations, event)
        if fit is None:
            return None
        Sfun = lambda x, f=fit: _weibull_survival(x, f["k"], f["lam"])
    elif family == "lognormal":
        fit = lognormal_mle(durations, event)
        if fit is None:
            return None
        Sfun = lambda x, f=fit: _lognormal_survival(x, f["mu"], f["sigma"])
    else:
        raise ValueError("family must be 'weibull' or 'lognormal'")
    p = fit["n_params"]
    ll = fit["loglik"]
    n = max(fit["n_sold"], 2)
    fit["aic"] = 2.0 * p - 2.0 * ll
    fit["bic"] = p * np.log(n) - 2.0 * ll
    fit["family"] = family
    fit["S"] = Sfun
    return fit


def _residual_survival_ratio(km, a, h, tail="km", weibull=None, param_fit=None):
    """
    Age-conditioned residual survival: P(D > a + h | D > a) = Shat(a+h)/Shat(a).

    Beyond the last KM event time the survival is extended by the chosen tail:
      tail='km'        : flat tail (last KM value; assumes no further depletion).
      tail='weibull'   : legacy Weibull tail from the (k,lam) OLS fit if provided,
                         else the censored-MLE parametric tail in param_fit.
      tail='lognormal' : censored-MLE log-normal parametric tail in param_fit.
      tail='param'     : use whatever family is in param_fit (dict with 'S').

    The parametric tail is scaled to match the KM value S_last at t_last so the
    extended survival is continuous at the KM support boundary (a standard
    KM-plus-parametric-tail splice). Within KM support the nonparametric KM is used.

    Parameters
    ----------
    km        : KM dict.
    a         : ndarray of ages already survived (days).
    h         : scalar horizon (days).
    param_fit : dict from fit_parametric_tail with an 'S' callable (for param tails).

    Returns
    -------
    ndarray of residual survival ratios in [0,1].
    """
    a = np.asarray(a, dtype=float)
    t_last = km["t"][-1]
    S_last = km["S"][-1]

    # Resolve the parametric survival callable, if any.
    Spar = None
    if tail in ("weibull", "lognormal", "param"):
        if param_fit is not None and "S" in param_fit:
            Spar = param_fit["S"]
        elif tail == "weibull" and weibull is not None:
            k, lam = weibull
            Spar = lambda x, k=k, lam=lam: _weibull_survival(x, k, lam)

    def S_ext(x):
        x = np.asarray(x, dtype=float)
        base = km_survival_at(km, x)  # flat-tail (KM) within and beyond support
        if Spar is not None and S_last > 1e-9:
            beyond = x > t_last
            Sp = Spar(x)
            Sp_last = float(Spar(np.array([t_last]))[0])
            if Sp_last > 1e-12:
                scaled = S_last * (Sp / Sp_last)  # splice: continuous at t_last
                base = np.where(beyond, np.clip(scaled, 0.0, S_last), base)
        return base

    Sa = S_ext(a)
    Sah = S_ext(a + h)
    safe = Sa > 1e-12
    ratio = np.zeros_like(Sa, dtype=float)
    np.divide(Sah, Sa, out=ratio, where=safe)
    return np.clip(ratio, 0.0, 1.0)


def _unsupported_query_share(km, ages, horizons):
    """
    Fraction of (current-stock item, horizon) queries whose age a_i + h exceeds
    the KM support (last event time), i.e. requires tail extrapolation. Required
    diagnostic output for the misspecification study.
    """
    t_last = km["t"][-1]
    A = np.asarray(ages, float)[:, None] + np.asarray(horizons, float)[None, :]
    if A.size == 0:
        return 0.0
    return float(np.mean(A > t_last))


def select_tail(durations, event, holdout_frac=0.0, rng=None,
                aic_margin=2.0):
    """
    Diagnostic-driven tail SELECTION rule. Fits Weibull and log-normal by censored
    MLE on the OBSERVED exits; picks a parametric tail only when the diagnostics
    support it, else returns 'km' (flat, nonparametric fallback).

    Rule (pre-specified, uses only observed data -- no ground-truth leakage):
      1. Fit Weibull and log-normal by censored MLE.
      2. Compute AIC for each on the observed (sold+censored) sample.
      3. Choose the parametric family with the lower AIC ONLY IF it improves on the
         nonparametric baseline enough to be trusted: we require the better
         parametric model's AIC to beat the WORSE one by > aic_margin (a decisive
         separation) OR, if holdout_frac>0, that it passes a held-out calibration
         check (predicted vs realized exits among sold items) better than a flat
         extrapolation. If neither parametric fit is estimable or they are not
         decisively separated, fall back to 'km'.
    Returns (tail_name, info_dict). tail_name in {'km','weibull','lognormal'};
    info_dict carries the fits + aic/bic + chosen family.
    """
    fw = fit_parametric_tail(durations, event, "weibull")
    fl = fit_parametric_tail(durations, event, "lognormal")
    info = {"weibull": fw, "lognormal": fl}
    cands = [(f["family"], f) for f in (fw, fl) if f is not None]
    if not cands:
        info["chosen"] = "km"
        info["reason"] = "no parametric fit estimable"
        return "km", info
    # rank by AIC
    cands.sort(key=lambda kv: kv[1]["aic"])
    best_fam, best_fit = cands[0]

    # Held-out calibration (optional): compare predicted vs realized survival among
    # SOLD items on a held-out split; pick the family whose predicted S better
    # matches the KM on the held-out sold exits. This is a data-driven check that
    # does not use the future / ground truth.
    if holdout_frac and holdout_frac > 0.0 and rng is not None:
        cal = _holdout_calibration(durations, event, rng, holdout_frac)
        if cal is not None:
            # choose family with lower calibration error; require it beats km proxy
            fam_cal = min(cal, key=lambda k: cal[k]["err"])
            info["calibration"] = cal
            if cal[fam_cal]["err"] < cal["km"]["err"]:
                info["chosen"] = fam_cal
                info["reason"] = "holdout calibration favors %s" % fam_cal
                return fam_cal, info
            else:
                info["chosen"] = "km"
                info["reason"] = "holdout calibration favors flat km"
                return "km", info

    # AIC-decisiveness gate (used when no holdout): require a clear AIC separation
    # between the two parametric families; otherwise the data do not distinguish a
    # tail shape and we fall back to the safe nonparametric flat tail.
    if len(cands) == 2:
        gap = cands[1][1]["aic"] - cands[0][1]["aic"]
        if gap < aic_margin:
            info["chosen"] = "km"
            info["reason"] = "AIC gap %.1f < margin (indistinct) -> km" % gap
            return "km", info
    info["chosen"] = best_fam
    info["reason"] = "AIC-selected %s" % best_fam
    return best_fam, info


def _holdout_calibration(durations, event, rng, holdout_frac):
    """
    Split observed items into fit/holdout; fit Weibull, log-normal, and a KM on the
    FIT split; on the HOLDOUT split measure how well each predicted survival matches
    the empirical (KM) survival of the holdout, evaluated at a grid of in-support
    times. Returns {'weibull':{'err':..}, 'lognormal':{..}, 'km':{..}} or None.
    Uses only observed data (no ground truth).
    """
    n = durations.size
    if n < 50:
        return None
    idx = rng.permutation(n)
    n_hold = max(20, int(holdout_frac * n))
    hold = idx[:n_hold]
    fit = idx[n_hold:]
    if fit.size < 20 or hold.size < 20:
        return None
    df, ef = durations[fit], event[fit]
    dh, eh = durations[hold], event[hold]
    km_fit = km_dwell(df, ef)
    km_hold = km_dwell(dh, eh)
    fw = fit_parametric_tail(df, ef, "weibull")
    fl = fit_parametric_tail(df, ef, "lognormal")
    # grid of in-support holdout event times
    grid = km_hold["t"][1:]
    if grid.size < 3:
        return None
    S_hold = km_survival_at(km_hold, grid)

    def err_of(Sfun):
        Sp = Sfun(grid)
        return float(np.mean((Sp - S_hold) ** 2))

    out = {}
    out["km"] = {"err": err_of(lambda x: km_survival_at(km_fit, x))}
    if fw is not None:
        out["weibull"] = {"err": err_of(fw["S"])}
    if fl is not None:
        out["lognormal"] = {"err": err_of(fl["S"])}
    return out


def depletion_forecast(events, w_end, horizons, tail="km", select_kwargs=None):
    """
    HEADLINE (Step 3). Forecast how the CURRENT censored stock cohort depletes over
    future horizons using age-conditioned residual survival from the KM curve.

    Current stock = censored items (entry <= W_end, no observed sale by W_end). Each
    has age a_i = W_end - entry. Predicted in-stock count at horizon h is
        Nhat(h) = sum_i Shat(a_i + h) / Shat(a_i)
    (age-conditioned residual survival). Nhat(0) = N_current exactly.

    Tail beyond KM support:
      tail='km'         : flat tail (last KM value; assumes no further depletion).
      tail='weibull'    : censored-MLE Weibull parametric tail (spliced at t_last).
      tail='lognormal'  : censored-MLE log-normal parametric tail (spliced).
      tail='select'     : diagnostic-driven selection (select_tail); falls back to
                          flat km when the data do not support a parametric tail.

    Parameters
    ----------
    events        : analyst event log [entry_date, exit_date].
    w_end         : observation window end (Timestamp).
    horizons      : iterable of horizons in DAYS.
    tail          : 'km' | 'weibull' | 'lognormal' | 'select'.
    select_kwargs : dict passed to select_tail when tail=='select'
                    (e.g. {'holdout_frac':0.3, 'rng':np.random.default_rng(seed)}).

    Returns
    -------
    dict: horizons, n_current, forecast, ages, tail (RESOLVED tail actually used),
    km, unsupported_share, and (for 'select') select_info.
    """
    durations, event = observed_durations(events, w_end)
    km = km_dwell(durations, event)

    entry_ns = _to_ns(events["entry_date"])
    exit_dt = pd.DatetimeIndex(events["exit_date"])
    is_censored = np.asarray(exit_dt.isna())
    w_end_ns = float(pd.Timestamp(w_end).value)
    ages = (w_end_ns - entry_ns[is_censored].astype(float)) / float(DAY_NS)
    ages = np.maximum(ages, 0.0)
    n_current = int(is_censored.sum())
    horizons = np.asarray(list(horizons), dtype=float)

    resolved_tail = tail
    select_info = None
    param_fit = None
    if tail == "select":
        sk = select_kwargs or {}
        resolved_tail, select_info = select_tail(durations, event, **sk)
    if resolved_tail in ("weibull", "lognormal"):
        param_fit = fit_parametric_tail(durations, event, resolved_tail)
        if param_fit is None:
            resolved_tail = "km"  # not estimable -> safe fallback

    forecast = np.empty(horizons.size, dtype=float)
    for i, h in enumerate(horizons):
        ratio = _residual_survival_ratio(km, ages, h, tail=resolved_tail,
                                         param_fit=param_fit)
        forecast[i] = float(np.sum(ratio))
    return {
        "horizons": horizons,
        "n_current": n_current,
        "forecast": forecast,
        "ages": ages,
        "tail": resolved_tail,
        "requested_tail": tail,
        "km": km,
        "unsupported_share": _unsupported_query_share(km, ages, horizons),
        "select_info": select_info,
    }


def depletion_forecast_bycat(events, w_end, horizons, tail="km"):
    """
    CATEGORY-SPECIFIC age-conditioned depletion forecast: fit a separate KM per
    category and apply age-conditioned residual survival within each category, then
    sum. This is the like-for-like counterpart of age_naive_depletion_bycat (both
    stratify by category), used for the exponential near-tie test so the comparison
    isolates the effect of AGE-CONDITIONING alone (not category stratification).
    Under memoryless time-on-shelf the within-category residual survival S_g(a+h)/S_g(a)
    equals S_g(h), so this must NOT beat the category age-naive baseline.
    """
    horizons = np.asarray(list(horizons), dtype=float)
    exit_dt = pd.DatetimeIndex(events["exit_date"])
    is_censored = np.asarray(exit_dt.isna())
    cats = events["category"].values
    entry_ns = _to_ns(events["entry_date"])
    w_end_ns = float(pd.Timestamp(w_end).value)
    n_current = int(is_censored.sum())
    forecast = np.zeros(horizons.size, dtype=float)
    for g in np.unique(cats):
        mask_g = cats == g
        ev_g = events.loc[mask_g]
        dur_g, event_g = observed_durations(ev_g, w_end)
        if dur_g.size == 0:
            continue
        km_g = km_dwell(dur_g, event_g)
        wb_g = _weibull_tail_fit(km_g) if tail == "weibull" else None
        ages_g = (w_end_ns - entry_ns[mask_g & is_censored].astype(float)) / float(DAY_NS)
        ages_g = np.maximum(ages_g, 0.0)
        for i, h in enumerate(horizons):
            forecast[i] += float(np.sum(
                _residual_survival_ratio(km_g, ages_g, h, tail=tail, weibull=wb_g)))
    return {"horizons": horizons, "n_current": n_current, "forecast": forecast}


# ==============================================================================
# Baselines
# ==============================================================================

def sold_only_dwell(events, w_end):
    """
    (b1) Sold-only time-on-shelf: DROP censored items entirely, summarise only observed
    sales. Returns dict with mean and median time-on-shelf (days). This is the naive
    practice; it is biased DOWNWARD because long-lived unsold items are removed.
    """
    durations, event = observed_durations(events, w_end)
    sold = durations[event == 1]
    if sold.size == 0:
        return {"mean": np.nan, "median": np.nan, "n": 0}
    return {"mean": float(np.mean(sold)), "median": float(np.median(sold)),
            "n": int(sold.size)}


def censor_at_end_dwell(events, w_end):
    """
    (b2) Censor-at-end time-on-shelf: treat CENSORED items as if they SOLD at W_end, i.e.
    use the full observed duration for every item regardless of event status.
    Biased downward (assigns the truncated in-window duration as if it were the
    complete time-on-shelf).
    """
    durations, event = observed_durations(events, w_end)
    if durations.size == 0:
        return {"mean": np.nan, "median": np.nan, "n": 0}
    return {"mean": float(np.mean(durations)), "median": float(np.median(durations)),
            "n": int(durations.size)}


def zero_opening_net_flow(events, month_ends):
    """
    (b3) Zero-opening-stock net-flow reconstruction: cumulative arrivals minus
    cumulative sales, assuming zero opening stock at the first month-end.
        stock(t) = (# entries <= t) - (# sales <= t).
    Identical in FORM to Step 1 EXCEPT it silently assumes no item was already in
    stock before the window: for left-truncated data (items present at window
    open) this omits their arrival AND fails to count them, biasing stock low.

    Implementation note: this baseline is computed the SAME way as reconstruct_stock
    when the analyst has the full entry history. Its bias only appears when the
    opening stock is unknown / entries before the window are not in the log. To
    make the distinction operational, this function counts only entries that fall
    WITHIN the observed window (entry >= first month start), mimicking an analyst
    who never saw pre-window arrivals.
    """
    entry_ns = _to_ns(events["entry_date"])
    exit_ns = _exit_ns_float(events["exit_date"])  # NaT -> NaN (correctly)
    t_ns = _to_ns(month_ends).astype(float)
    window_start_ns = float(t_ns.min()) - float(DAY_NS) * 31  # ~month before first end

    # An analyst with only in-window records: keep entries at/after window start.
    in_window_entry = entry_ns.astype(float) >= window_start_ns
    entries_sorted = np.sort(entry_ns[in_window_entry].astype(float))
    sold_mask = ~np.isnan(exit_ns)
    sold_exits = np.sort(exit_ns[sold_mask & in_window_entry])

    n_entered = np.searchsorted(entries_sorted, t_ns, side="right")
    if sold_exits.size > 0:
        n_exited = np.searchsorted(sold_exits, t_ns, side="right")
    else:
        n_exited = np.zeros_like(t_ns)
    stock = (n_entered - n_exited).astype(int)
    return pd.Series(stock, index=pd.DatetimeIndex(month_ends), name="zero_opening_net_flow")


def age_naive_depletion(events, w_end, horizons):
    """
    (b4) Age-naive depletion: predict the current stock's depletion using the
    UNCONDITIONAL survival S(h), ignoring how long each item has ALREADY survived.
        Nhat(h) = N_current * S(h).
    Because current-stock items have already survived a_i > 0, applying the
    unconditional S(h) (which starts at age 0) OVER-depletes them: it double-counts
    the early hazard they have already passed. Expected to over-predict depletion.
    """
    durations, event = observed_durations(events, w_end)
    km = km_dwell(durations, event)
    exit_dt = pd.DatetimeIndex(events["exit_date"])
    n_current = int(np.asarray(exit_dt.isna()).sum())
    horizons = np.asarray(list(horizons), dtype=float)
    S_h = km_survival_at(km, horizons)
    forecast = n_current * S_h
    return {"horizons": horizons, "n_current": n_current, "forecast": forecast}


def age_naive_depletion_bycat(events, w_end, horizons):
    """
    (b4-cat) CATEGORY-SPECIFIC age-naive depletion -- the STRENGTHENED main
    comparator. Fit a separate KM curve S_g on each category g, then deplete that
    category's current stock by its own UNCONDITIONAL survival:
        Nhat(h) = sum_g N_current_g * S_g(h).
    This still ignores the already-survived age a_i (the residual-survival
    conditioning the protocol adds), but removes the category-mix confound, so it
    is a much stronger baseline than a single pooled S(h). The protocol's win over
    THIS baseline isolates the value of age-conditioning specifically.
    """
    horizons = np.asarray(list(horizons), dtype=float)
    exit_dt = pd.DatetimeIndex(events["exit_date"])
    is_censored = np.asarray(exit_dt.isna())
    cats = events["category"].values
    n_current = int(is_censored.sum())
    forecast = np.zeros(horizons.size, dtype=float)
    for g in np.unique(cats):
        mask_g = cats == g
        ev_g = events.loc[mask_g]
        dur_g, event_g = observed_durations(ev_g, w_end)
        if dur_g.size == 0:
            continue
        km_g = km_dwell(dur_g, event_g)
        n_cur_g = int((is_censored & mask_g).sum())
        forecast += n_cur_g * km_survival_at(km_g, horizons)
    return {"horizons": horizons, "n_current": n_current, "forecast": forecast}


def censor_at_end_depletion(events, w_end, horizons):
    """
    Depletion baseline built from the censor-at-end time-on-shelf view: treat censored
    durations as complete sales at W_end, refit KM on that (mis-specified) sample,
    then apply age-conditioned residual survival to the current stock. Because the
    censor-at-end sample understates true time-on-shelf, its survival curve decays too
    fast, over-predicting depletion. This is the depletion analogue of baseline b2.
    """
    durations, event = observed_durations(events, w_end)
    # censor-at-end: every item is treated as an event (sold) at its duration.
    event_all = np.ones_like(event)
    km = km_dwell(durations, event_all)

    entry_ns = _to_ns(events["entry_date"])
    exit_dt = pd.DatetimeIndex(events["exit_date"])
    is_censored = np.asarray(exit_dt.isna())
    w_end_ns = float(pd.Timestamp(w_end).value)
    ages = (w_end_ns - entry_ns[is_censored].astype(float)) / float(DAY_NS)
    ages = np.maximum(ages, 0.0)
    n_current = int(is_censored.sum())

    horizons = np.asarray(list(horizons), dtype=float)
    forecast = np.empty(horizons.size, dtype=float)
    for i, h in enumerate(horizons):
        ratio = _residual_survival_ratio(km, ages, h, tail="km")
        forecast[i] = float(np.sum(ratio))
    return {"horizons": horizons, "n_current": n_current, "forecast": forecast}


def accounting_only_depletion(events, w_end, horizons, hist_months=None):
    """
    (b5) Accounting-only depletion: the current stock depletes at the HISTORICAL
    average monthly sale COUNT (a flat linear drawdown), ignoring durations and
    ages entirely.
        Nhat(h) = max(N_current - avg_monthly_sales * (h / 30.4375), 0).
    A pure flow-accounting extrapolation with no survival model. Expected to be a
    poor shape (linear vs curved) and to hit zero too early or too late.

    hist_months : number of months over which to average historical sales; if
    None, uses the full observed window span.
    """
    durations, event = observed_durations(events, w_end)
    exit_dt = pd.DatetimeIndex(events["exit_date"])
    entry_ns = _to_ns(events["entry_date"])
    n_current = int(np.asarray(exit_dt.isna()).sum())

    # historical monthly sale rate over the observed window
    sold_mask = event == 1
    n_sold = int(sold_mask.sum())
    span_days = (float(pd.Timestamp(w_end).value) - float(entry_ns.min())) / float(DAY_NS)
    span_months = max(span_days / 30.4375, 1.0)
    if hist_months is not None:
        span_months = float(hist_months)
    avg_monthly_sales = n_sold / span_months

    horizons = np.asarray(list(horizons), dtype=float)
    forecast = np.maximum(n_current - avg_monthly_sales * (horizons / 30.4375), 0.0)
    return {"horizons": horizons, "n_current": n_current,
            "forecast": forecast, "avg_monthly_sales": avg_monthly_sales}


# ==============================================================================
# Ground-truth depletion (scoring target)
# ==============================================================================

def true_depletion(truth_items, w_end, horizons):
    """
    TRUE number of the CURRENT stock still in stock at (W_end + h): count of
    current-stock items whose TRUE sale date > W_end + h.

    Current stock is defined identically to the forecast target: items that are
    censored (unsold at W_end) in the OBSERVED sense. Because truth_items carries
    each item's censored flag and true_sale_date, we score directly against the
    simulated future.

    Parameters
    ----------
    truth_items : ground-truth DataFrame with [true_sale_date, censored].
    w_end       : Timestamp.
    horizons    : iterable of horizons (days).

    Returns
    -------
    dict with 'horizons' and 'true_count' (ndarray).
    """
    cur = truth_items[truth_items["censored"]].copy()
    true_sale_ns = _to_ns(cur["true_sale_date"]).astype(float)
    w_end_ns = float(pd.Timestamp(w_end).value)
    horizons = np.asarray(list(horizons), dtype=float)
    true_count = np.empty(horizons.size, dtype=float)
    for i, h in enumerate(horizons):
        thresh = w_end_ns + h * float(DAY_NS)
        true_count[i] = float(np.sum(true_sale_ns > thresh))
    return {"horizons": horizons, "true_count": true_count,
            "n_current": int(len(cur))}


# ==============================================================================
# True time-on-shelf summaries (for the time-on-shelf-bias experiment)
# ==============================================================================

def true_dwell_summary(truth_items, tau=365.0):
    """
    True median time-on-shelf and true RMST(tau) from the FULL true time-on-shelf
    distribution of all in-world items (uncensored ground truth). RMST(tau) uses the
    empirical survival of true time-on-shelf:
        RMST(tau) = integral_0^tau P(D > u) du = mean( min(D, tau) ).
    """
    D = np.asarray(truth_items["true_dwell_days"], dtype=float)
    med = float(np.median(D))
    rmst_true = float(np.mean(np.minimum(D, float(tau))))
    return {"median": med, "rmst": rmst_true, "n": int(D.size)}


if __name__ == "__main__":
    # Minimal self-check against a tiny hand-computed KM example (ASCII only).
    # Classic example: times [6,6,6,6+,7,9+,10,10+,11+,13,16,17+,19+,20+,22,23,
    #  25+,32+,32+,34+,35+] is large; use a small textbook set instead.
    # Data: durations with events.
    #  t: 2(d),3(c),4(d),5(d),5(c)
    dur = np.array([2, 3, 4, 5, 5], dtype=float)
    ev = np.array([1, 0, 1, 1, 0], dtype=int)
    km = km_dwell(dur, ev)
    print("KM self-check")
    print("t   :", np.round(km["t"], 3))
    print("S   :", np.round(km["S"], 4))
    print("var :", np.round(km["var"], 6))
    # Hand: at t=2 n=5 d=1 S=0.8; t=4 n=3 d=1 S=0.8*2/3=0.5333; t=5 n=2 d=1
    #        S=0.5333*1/2=0.2667
    print("expected S at events: 0.8, 0.5333, 0.2667")
    print("RMST(5) =", round(rmst(km, 5.0), 4))
    print("median =", km_median(km))
