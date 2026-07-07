"""
synth.py -- Synthetic generator for the censoring-aware inventory/time-on-shelf
reconstruction protocol (MethodsX paper #5).

Pure NumPy/Pandas. ASCII-only stdout. No confidential data is read; every item,
date, and time-on-shelf here is simulated, so the *future* (true sale dates for
items that are censored in the observation window) is known to the generator.

------------------------------------------------------------------------------
PRIMARY design -- WINDOW-INDUCED censoring (generate()):
  Time-on-shelf is held FIXED and realistic across ALL censoring cells:
    * category medians accessory/jewelry/bag/watch = 60/90/120/180 days
      (the 1:1.5:2:3 ratio), CONSTANT, NEVER inflated;
    * Weibull shape in [1.2,1.5] (or an Exponential regime for the near-tie test).
  Censoring is induced by the OBSERVATION-WINDOW LENGTH, not by distorting time-on-shelf:
  one long master timeline (fixed arrivals + fixed time-on-shelf) is simulated, then for a
  target censoring rate we choose the observation-window length L (months). Items
  entering within the window [W_start, W_start+L] that are unsold by W_end = the
  window end are right-censored. A SHORT window -> many recent, still-unsold items
  -> HIGH censoring; a LONG window -> most items already sold -> LOW censoring.
  Thus "70% censoring" means "you have observed only a short window of a business
  whose watches take ~180 days to sell", NOT "watches take 5 years".

STRESS-TEST design -- TIME-ON-SHELF-SCALE inflation (generate_scale_inflation()):
  The older mechanism, retained but DEMOTED to a labelled appendix: it holds the
  window fixed and multiplies all time-on-shelf values by a single factor solved to hit the
  target censoring. This makes category medians grow with censoring (unrealistic at
  high rates) and is reported only as a robustness/stress figure.

Both generators return BOTH:
  (a) GROUND TRUTH -- item-level true D_i and true_sale_date for ALL in-window
      items (including censored ones), plus true daily+monthly stock I_t; and
  (b) the OBSERVED event log the analyst sees (entry_date; observed exit_date =
      true_sale_date if <= W_end else NaN; censor flag).
"""

import numpy as np
import pandas as pd


# --- Fixed structural constants -------------------------------------------------

CATEGORIES = ["bag", "watch", "jewelry", "accessory"]
CATEGORY_MIX = np.array([0.45, 0.20, 0.20, 0.15])  # bag/watch/jewelry/accessory

# TRUE median time-on-shelf (days) by category. Held CONSTANT in the primary
# window-induced design (no inflation). Weibull median = scale*(ln 2)^(1/shape),
# so we back out the scale per drawn shape to place the median exactly on target.
CATEGORY_MEDIAN_DAYS = {
    "accessory": 60.0,
    "jewelry": 90.0,
    "bag": 120.0,
    "watch": 180.0,
}

DAYS_PER_MONTH = 30.4375  # mean Gregorian month


# ==============================================================================
# Unit-safe datetime helpers (pandas 3.0 defaults DatetimeIndex to microseconds;
# we force nanoseconds so all integer arithmetic is in one consistent unit that
# matches pd.Timestamp.value).
# ==============================================================================

def _to_ns(dt_like):
    """int64 NANOSECONDS-since-epoch at ns resolution (pandas 2.x/3.x safe)."""
    return pd.DatetimeIndex(dt_like).as_unit("ns").asi8.astype(np.int64)


def _month_starts(n_months, origin="2016-01-01"):
    return pd.date_range(start=origin, periods=n_months, freq="MS")


def _month_ends_from_starts(month_starts):
    return month_starts + pd.offsets.MonthEnd(0)


def _weibull_scale_from_median(median_days, shape):
    """Scale lambda so Weibull(shape, lambda) has the given median."""
    return median_days / (np.log(2.0) ** (1.0 / shape))


def _exp_scale_from_median(median_days):
    """Scale (mean) of an Exponential with the given median: mean = median/ln2."""
    return median_days / np.log(2.0)


# Log-normal and gamma shape parameters for the misspecification DGPs. All three
# DGP families (Weibull, log-normal, gamma) are median-matched to the SAME category
# medians; only the shape of the distribution (and hence the tail) differs.
LOGNORMAL_SIGMA = 0.8   # log-scale SD (realistic dispersion; CV ~ comparable to Weibull)
GAMMA_SHAPE = 2.0       # gamma shape k

# Standard-gamma(GAMMA_SHAPE) median, estimated once from a large fixed-seed pilot
# so gamma scale can be solved for a target median WITHOUT scipy (numpy-only).
_GAMMA_STD_MEDIAN = float(np.median(
    np.random.default_rng(999).gamma(GAMMA_SHAPE, 1.0, 500000)))


def _lognormal_mu_from_median(median_days):
    """Log-normal location mu with the given median: median = exp(mu)."""
    return np.log(median_days)


def _gamma_scale_from_median(median_days, shape=GAMMA_SHAPE):
    """
    Gamma scale theta so Gamma(shape, theta) has the given median. median of a
    gamma equals theta * (standard-gamma median); the standard-gamma median is
    taken from the fixed pilot above (scipy-free).
    """
    return median_days / _GAMMA_STD_MEDIAN


def _seasonal_intensity(t_index, lambda0, growth_per_year=0.0):
    """
    Monthly arrival intensity with seasonality and optional gentle growth:
        lambda_t = lambda0 * exp(growth*t/12)
                          * exp(0.25*sin(2*pi*t/12) + 0.10*[Q4]).
    t maps calendar month = (t mod 12)+1; Q4 = Oct/Nov/Dec.
    """
    t_arr = np.asarray(t_index, dtype=float)
    calendar_month = (np.asarray(t_index) % 12) + 1
    q4 = np.asarray((calendar_month >= 10) & (calendar_month <= 12), dtype=float)
    trend = np.exp(growth_per_year * t_arr / 12.0)
    seas = np.exp(0.25 * np.sin(2.0 * np.pi * t_arr / 12.0) + 0.10 * q4)
    return lambda0 * trend * seas


def _nb_sample(rng, mean, dispersion_r):
    """One Negative-Binomial count with given mean and overdispersion (Var = mean + mean^2/r)."""
    r = float(dispersion_r)
    p = r / (r + mean)
    return int(rng.negative_binomial(r, p))


# ==============================================================================
# Master timeline: fixed arrivals + fixed time-on-shelf (shared across censoring cells)
# ==============================================================================

def _build_master_timeline(
    rng,
    n_months,
    lambda0,
    dispersion_r,
    origin,
    dwell_family="weibull",
    shape=None,
    growth_per_year=0.0,
):
    """
    Simulate ONE long master timeline of item arrivals and FIXED true time-on-shelf
    values. Returns a DataFrame with columns [entry_date, category, true_dwell_days,
    true_sale_ns] plus the drawn shape. Time-on-shelf does NOT depend on the observation
    window, so every censoring cell (which only changes the window length) reuses
    the same latent process.

    dwell_family : 'weibull' (shape in [1.2,1.5]) or 'exponential' (memoryless).
                   For 'exponential' the per-category MEAN is set so the median
                   matches CATEGORY_MEDIAN_DAYS (mean = median/ln2); this is the
                   near-tie regime where age-conditioning must not help.
    """
    month_starts = _month_starts(n_months, origin=origin)
    entry_list, midx_list = [], []
    for t in range(n_months):
        mean_t = _seasonal_intensity(t, lambda0, growth_per_year)
        cnt = _nb_sample(rng, mean_t, dispersion_r)
        if cnt <= 0:
            continue
        m_start = month_starts[t]
        dim = (m_start + pd.offsets.MonthEnd(0)).day
        day_off = rng.integers(0, dim, size=cnt)
        entry_list.append((m_start + pd.to_timedelta(day_off, unit="D")).values)
        midx_list.append(np.full(cnt, t, dtype=int))
    entry_date = pd.DatetimeIndex(np.concatenate(entry_list))
    order = np.argsort(entry_date.values)
    entry_date = entry_date[order]
    n = len(entry_date)

    categories = np.array(CATEGORIES)[rng.choice(len(CATEGORIES), size=n, p=CATEGORY_MIX)]

    if dwell_family == "weibull":
        if shape is None:
            shape = float(rng.uniform(1.2, 1.5))
        dwell = np.empty(n, dtype=float)
        for cat in CATEGORIES:
            mask = categories == cat
            if not np.any(mask):
                continue
            scale = _weibull_scale_from_median(CATEGORY_MEDIAN_DAYS[cat], shape)
            dwell[mask] = scale * rng.weibull(shape, size=int(mask.sum()))
    elif dwell_family == "exponential":
        shape = 1.0  # exponential is Weibull(shape=1); memoryless
        dwell = np.empty(n, dtype=float)
        for cat in CATEGORIES:
            mask = categories == cat
            if not np.any(mask):
                continue
            mean = _exp_scale_from_median(CATEGORY_MEDIAN_DAYS[cat])
            dwell[mask] = rng.exponential(mean, size=int(mask.sum()))
    elif dwell_family == "lognormal":
        shape = LOGNORMAL_SIGMA  # report sigma in the 'shape' slot
        dwell = np.empty(n, dtype=float)
        for cat in CATEGORIES:
            mask = categories == cat
            if not np.any(mask):
                continue
            mu = _lognormal_mu_from_median(CATEGORY_MEDIAN_DAYS[cat])
            dwell[mask] = rng.lognormal(mu, LOGNORMAL_SIGMA, size=int(mask.sum()))
    elif dwell_family == "gamma":
        shape = GAMMA_SHAPE
        dwell = np.empty(n, dtype=float)
        for cat in CATEGORIES:
            mask = categories == cat
            if not np.any(mask):
                continue
            theta = _gamma_scale_from_median(CATEGORY_MEDIAN_DAYS[cat])
            dwell[mask] = rng.gamma(GAMMA_SHAPE, theta, size=int(mask.sum()))
    else:
        raise ValueError(
            "dwell_family must be 'weibull', 'exponential', 'lognormal' or 'gamma'")

    dwell = np.maximum(dwell, 1.0)
    # Cap at ~200 years so entry_ns + dwell stays within int64-ns range.
    dwell = np.minimum(dwell, 200.0 * 365.25)

    entry_ns = _to_ns(entry_date)
    true_sale_ns = entry_ns + np.round(dwell * 86400.0 * 1e9).astype(np.int64)

    master = pd.DataFrame({
        "entry_date": entry_date,
        "entry_ns": entry_ns,
        "category": categories,
        "true_dwell_days": dwell,
        "true_sale_ns": true_sale_ns,
    })
    return master, float(shape)


def _true_stock_at(entry_ns, sale_ns, grid_ns):
    """
    TRUE #in-stock at each grid time: entry<=t AND true_sale>t, over the given
    item subset. in_stock(t) = (#entries<=t) - (#true_sales<=t).
    """
    e = np.sort(entry_ns.astype(np.int64))
    s = np.sort(sale_ns.astype(np.int64))
    n_in = np.searchsorted(e, grid_ns, side="right")
    n_out = np.searchsorted(s, grid_ns, side="right")
    return n_in - n_out


def _realised_window_censoring(master, ws_ns, we_ns):
    """
    Censoring rate for a window [ws, we): items ENTERING within the window whose
    true sale is after we. Returns (rate, n_in_window).
    """
    ent = master["entry_ns"].values
    sale = master["true_sale_ns"].values
    in_win = (ent >= ws_ns) & (ent < we_ns)
    n = int(in_win.sum())
    if n == 0:
        return 0.0, 0
    unsold = sale[in_win] > we_ns
    return float(unsold.mean()), n


def _solve_window_length(master, ws_ns, month_starts, ws_month_idx, target_rate,
                         min_L=1, max_L=None):
    """
    Choose the window length L (in whole months) so the realised window-induced
    censoring rate is as close as possible to target_rate. Censoring DECREASES as
    L increases (longer window -> more items have sold), so we search L on a grid
    of month-end cutoffs and pick the L whose realised rate is nearest the target.

    Returns (L_months, we_ns, realised_rate, n_in_window).
    """
    n_months_total = len(month_starts)
    if max_L is None:
        max_L = n_months_total - ws_month_idx - 1
    best = None
    for L in range(min_L, max_L + 1):
        we_idx = ws_month_idx + L
        if we_idx >= n_months_total:
            break
        we_ns = int(pd.Timestamp(month_starts[we_idx]).value)
        rate, n_in = _realised_window_censoring(master, ws_ns, we_ns)
        if n_in < 300:  # keep samples usable
            continue
        gap = abs(rate - target_rate)
        cand = (gap, L, we_ns, rate, n_in)
        if best is None or cand[0] < best[0]:
            best = cand
    if best is None:
        # fall back to the longest window
        we_idx = n_months_total - 1
        we_ns = int(pd.Timestamp(month_starts[we_idx]).value)
        rate, n_in = _realised_window_censoring(master, ws_ns, we_ns)
        return we_idx - ws_month_idx, we_ns, rate, n_in
    _, L, we_ns, rate, n_in = best
    return L, we_ns, rate, n_in


# ==============================================================================
# Assembly of the observed log + ground truth for a chosen window
# ==============================================================================

def _assemble(master, ws_ns, we_ns, month_starts, ws_month_idx, L_months,
              seed, meta_extra, include_opening_stock=False):
    """
    Given a master timeline and a window [ws, we), build the analyst's observed
    event log and the item-level + monthly ground truth.

    include_opening_stock : if True, items that entered BEFORE ws but are still in
      stock at ws (true sale > ws) are ADDED to the observed log as opening stock
      with known entry date (age at ws). This is the left-truncation scenario.
    """
    ent = master["entry_ns"].values
    sale = master["true_sale_ns"].values
    cats = master["category"].values
    dwell = master["true_dwell_days"].values

    in_window = (ent >= ws_ns) & (ent < we_ns)
    opening = np.zeros(len(master), dtype=bool)
    if include_opening_stock:
        opening = (ent < ws_ns) & (sale > ws_ns)  # entered earlier, still in stock at ws
    keep = in_window | opening

    entry_ns = ent[keep]
    sale_ns = sale[keep]
    category = cats[keep]
    dwell_days = dwell[keep]
    left_trunc = opening[keep]

    # Observed exit = true sale if <= we, else NaT (censored / still in stock).
    sold_by_we = sale_ns <= we_ns
    obs_exit = np.full(keep.sum(), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    obs_exit[sold_by_we] = sale_ns[sold_by_we].astype("datetime64[ns]")
    censored = ~sold_by_we

    entry_ts = pd.DatetimeIndex(entry_ns.astype("datetime64[ns]"))
    opening_age = np.where(left_trunc,
                           (ws_ns - entry_ns) / (86400.0 * 1e9), np.nan)

    events = pd.DataFrame({
        "item_id": np.arange(keep.sum()),
        "category": category,
        "entry_date": entry_ts,
        "exit_date": pd.DatetimeIndex(obs_exit),
        "censored": censored,
        "left_truncated": left_trunc,
        "opening_age_days": opening_age,
    })

    age_at_we = (we_ns - entry_ns) / (86400.0 * 1e9)
    truth_items = pd.DataFrame({
        "item_id": np.arange(keep.sum()),
        "category": category,
        "entry_date": entry_ts,
        "true_dwell_days": dwell_days,
        "true_sale_date": pd.DatetimeIndex(sale_ns.astype("datetime64[ns]")),
        "in_window_stock_at_wend": censored,   # unsold by we == still in stock at we
        "age_at_wend_days": age_at_we,
        "censored": censored,
        "left_truncated": left_trunc,
    })

    # Monthly + daily TRUE stock over [ws, we] using the kept items.
    ws_ts = pd.Timestamp(month_starts[ws_month_idx])
    we_ts = pd.Timestamp(month_starts[ws_month_idx + L_months])
    # month-end grid strictly within the window
    window_month_starts = month_starts[ws_month_idx: ws_month_idx + L_months]
    window_month_ends = _month_ends_from_starts(window_month_starts)
    window_month_ends = pd.DatetimeIndex(
        [pd.Timestamp(d.normalize()) for d in window_month_ends])
    me_ns = _to_ns(window_month_ends)
    true_monthly = _true_stock_at(entry_ns, sale_ns, me_ns)
    true_stock_monthly = pd.Series(true_monthly, index=window_month_ends,
                                   name="true_stock")

    day_grid = pd.date_range(ws_ts.normalize(), we_ts.normalize(), freq="D")
    dg_ns = _to_ns(day_grid)
    true_daily = _true_stock_at(entry_ns, sale_ns, dg_ns)
    true_stock_daily = pd.DataFrame({"day": day_grid, "true_in_stock": true_daily})

    realised = float(censored[~left_trunc].mean()) if (~left_trunc).sum() > 0 else 0.0

    meta = {
        "seed": int(seed),
        "W_start": ws_ts.normalize(),
        "W_end": we_ts.normalize(),
        "window_length_months": L_months,
        "realised_censoring": realised,
        "n_items_in_world": int(keep.sum()),
        "n_in_window": int(in_window.sum()),
        "n_opening_stock": int(left_trunc.sum()),
        "n_censored": int(censored.sum()),
        "n_sold_in_window": int(sold_by_we.sum()),
        "window_month_ends": window_month_ends,
    }
    meta.update(meta_extra)
    return {
        "events": events,
        "truth_items": truth_items,
        "true_stock_monthly": true_stock_monthly,
        "true_stock_daily": true_stock_daily,
        "meta": meta,
    }


# ==============================================================================
# PRIMARY generator -- WINDOW-INDUCED censoring (fixed realistic time-on-shelf)
# ==============================================================================

def generate(
    seed,
    target_censoring=0.3,
    left_truncation=False,
    dwell_family="weibull",
    ws_month_idx=24,
    master_months=180,
    lambda0=200.0,
    dispersion_r=8.0,
    growth_per_year=0.0,
    origin="2016-01-01",
):
    """
    Window-induced censoring generator (PRIMARY design).

    Builds a fixed master timeline (arrivals + realistic fixed time-on-shelf), then
    chooses the observation-window length L so the realised right-censoring rate (items
    entering within the window that are unsold at the window end) is nearest the
    target. Time-on-shelf (category medians 60/90/120/180 d, shape in [1.2,1.5]) is held
    CONSTANT across all cells -- only the window length changes.

    Parameters
    ----------
    seed : int (deterministic; no system randomness).
    target_censoring : desired realised right-censoring rate in {0,.1,.3,.5,.7}.
    left_truncation : if True, pre-window items still in stock at W_start are added
                      as opening stock (tests opening-stock handling).
    dwell_family : time-on-shelf distribution family, all median-matched to the
                   same category medians. 'weibull' (default), 'exponential'
                   (memoryless near-tie regime), 'lognormal' or 'gamma' (the
                   alternative data-generating families for the misspecification
                   study). Only the distribution SHAPE / tail differs; the medians
                   are identical across families.
    ws_month_idx : month index at which the observation window starts (>=1 so a
                   warm-up exists before it for opening stock).
    master_months : length of the master timeline (long enough that a ~0% cell is
                    reachable with a long window).
    lambda0, dispersion_r, growth_per_year : arrival process parameters.

    Returns
    -------
    dict: {events, truth_items, true_stock_monthly, true_stock_daily, meta}.
    meta['realised_censoring'] reports the ACHIEVED rate (the window design cannot
    always hit a target exactly; the nearest feasible L is chosen).
    """
    rng = np.random.default_rng(int(seed))
    month_starts = _month_starts(master_months, origin=origin)
    master, shape = _build_master_timeline(
        rng, master_months, lambda0, dispersion_r, origin,
        dwell_family=dwell_family, growth_per_year=growth_per_year)

    ws_ns = int(pd.Timestamp(month_starts[ws_month_idx]).value)

    if target_censoring <= 0.0:
        # Longest feasible window -> lowest achievable censoring (the 0% anchor).
        we_idx = master_months - 1
        we_ns = int(pd.Timestamp(month_starts[we_idx]).value)
        L = we_idx - ws_month_idx
        realised, n_in = _realised_window_censoring(master, ws_ns, we_ns)
    else:
        L, we_ns, realised, n_in = _solve_window_length(
            master, ws_ns, month_starts, ws_month_idx, target_censoring)

    meta_extra = {
        "design": "window_induced",
        "target_censoring": target_censoring,
        "dwell_family": dwell_family,
        "weibull_shape": shape,
        "growth_per_year": growth_per_year,
        "dwell_scale_factor": 1.0,  # NO inflation in the primary design
    }
    return _assemble(master, ws_ns, we_ns, month_starts, ws_month_idx, L,
                     seed, meta_extra, include_opening_stock=left_truncation)


# ==============================================================================
# STRESS-TEST generator -- TIME-ON-SHELF-SCALE inflation (fixed window, demoted)
# ==============================================================================

def _solve_dwell_scale_for_target(entry_ns, base_dwell_days, we_ns, target_rate):
    """
    Bisection: single multiplicative time-on-shelf factor c so the realised censoring
    rate (over items entering on/before we) matches target_rate. Censoring is monotone
    non-decreasing in c. Retained only for the stress-test appendix.
    """
    entry_ns = entry_ns.astype(np.float64)
    we = float(we_ns)
    in_world = entry_ns <= we
    e_in = entry_ns[in_world]
    d_in = base_dwell_days[in_world]

    def rate_at(c):
        sale = e_in + (c * d_in * 86400.0 * 1e9)
        return float((sale > we).mean())

    c_lo, c_hi = 1e-3, 1e-3
    r_hi = rate_at(c_hi); n = 0
    while r_hi < target_rate and n < 60:
        c_hi *= 1.5; r_hi = rate_at(c_hi); n += 1
    r_lo = rate_at(c_lo)
    if target_rate <= r_lo + 1e-9:
        return c_lo, r_lo
    if r_hi < target_rate:
        return c_hi, r_hi
    for _ in range(80):
        c_mid = 0.5 * (c_lo + c_hi)
        if rate_at(c_mid) < target_rate:
            c_lo = c_mid
        else:
            c_hi = c_mid
    c = 0.5 * (c_lo + c_hi)
    return c, rate_at(c)


def generate_scale_inflation(
    seed,
    warmup_months=12,
    window_months=36,
    tail_months=24,
    target_censoring=0.3,
    left_truncation=False,
    left_truncation_frac=0.12,
    lambda0=200.0,
    dispersion_r=8.0,
    origin="2016-01-01",
):
    """
    STRESS-TEST / APPENDIX generator (demoted). Holds the observation window FIXED
    and inflates all time-on-shelf values by a single factor to hit the target censoring.
    This distorts category medians upward at high censoring (e.g. watches drift to
    multi-year medians), so it is reported ONLY as a robustness stress figure, not
    as the main design. Returns the same dict structure as generate().
    """
    rng = np.random.default_rng(int(seed))
    total_months = warmup_months + window_months + tail_months
    month_starts = _month_starts(total_months, origin=origin)
    ws_month_idx = warmup_months

    # arrivals across full horizon
    entry_list = []
    for t in range(total_months):
        mean_t = _seasonal_intensity(t, lambda0)
        cnt = _nb_sample(rng, mean_t, dispersion_r)
        if cnt <= 0:
            continue
        m_start = month_starts[t]
        dim = (m_start + pd.offsets.MonthEnd(0)).day
        day_off = rng.integers(0, dim, size=cnt)
        entry_list.append((m_start + pd.to_timedelta(day_off, unit="D")).values)
    entry_date = pd.DatetimeIndex(np.concatenate(entry_list))
    entry_date = entry_date[np.argsort(entry_date.values)]
    n = len(entry_date)

    # optional left-truncation: re-date a fraction to before window start
    left_flag = np.zeros(n, dtype=bool)
    if left_truncation:
        n_shift = int(round(left_truncation_frac * n))
        if n_shift > 0:
            idx = rng.choice(n, size=n_shift, replace=False)
            span = max((month_starts[ws_month_idx].normalize()
                        - month_starts[0].normalize()).days, 1)
            new_off = rng.integers(0, span, size=n_shift)
            vals = entry_date.values.copy()
            vals[idx] = (month_starts[0].normalize()
                         + pd.to_timedelta(new_off, unit="D")).values
            entry_date = pd.DatetimeIndex(vals)
            entry_date = entry_date[np.argsort(entry_date.values)]
        left_flag = (entry_date.values
                     < np.datetime64(month_starts[ws_month_idx].normalize()))

    categories = np.array(CATEGORIES)[rng.choice(len(CATEGORIES), size=n, p=CATEGORY_MIX)]
    shape = float(rng.uniform(1.2, 1.5))
    base_dwell = np.empty(n, dtype=float)
    for cat in CATEGORIES:
        mask = categories == cat
        if np.any(mask):
            scale = _weibull_scale_from_median(CATEGORY_MEDIAN_DAYS[cat], shape)
            base_dwell[mask] = scale * rng.weibull(shape, size=int(mask.sum()))
    base_dwell = np.maximum(base_dwell, 1.0)

    we_idx = warmup_months + window_months - 1
    we_ns = int(pd.Timestamp(_month_ends_from_starts(month_starts)[we_idx].normalize()).value)
    entry_ns = _to_ns(entry_date)

    c_star, realised = _solve_dwell_scale_for_target(
        entry_ns, base_dwell, we_ns, target_censoring)
    dwell = np.minimum(np.maximum(base_dwell * c_star, 1.0), 200.0 * 365.25)
    true_sale_ns = entry_ns + np.round(dwell * 86400.0 * 1e9).astype(np.int64)

    # zero-censoring exactness: de-censor boundary items (as in the old design)
    if target_censoring == 0.0:
        in_world = entry_ns <= we_ns
        over = in_world & (true_sale_ns > we_ns)
        if np.any(over):
            exp_ns = (we_ns - entry_ns[over]).astype(np.float64)
            frac = rng.uniform(0.0, 1.0, size=int(over.sum()))
            nd = np.clip(frac * exp_ns, 0.0, exp_ns).astype(np.int64)
            true_sale_ns[over] = entry_ns[over] + nd
            dwell[over] = np.maximum(nd / (86400.0 * 1e9), 0.0)

    master = pd.DataFrame({
        "entry_date": entry_date,
        "entry_ns": entry_ns,
        "category": categories,
        "true_dwell_days": dwell,
        "true_sale_ns": true_sale_ns,
    })
    ws_ns = int(pd.Timestamp(month_starts[ws_month_idx]).value)
    L = window_months
    meta_extra = {
        "design": "scale_inflation",
        "target_censoring": target_censoring,
        "dwell_family": "weibull",
        "weibull_shape": shape,
        "dwell_scale_factor": c_star,
    }
    out = _assemble(master, ws_ns, we_ns, month_starts, ws_month_idx, L,
                    seed, meta_extra, include_opening_stock=False)
    # scale-inflation keeps its own left_truncation handling via re-dated entries;
    # mark them in the events/truth frames.
    if left_truncation:
        lt_mask = np.asarray(pd.DatetimeIndex(out["events"]["entry_date"]).normalize()
                             < out["meta"]["W_start"])
        out["events"]["left_truncated"] = lt_mask
    out["meta"]["realised_censoring"] = realised
    return out


if __name__ == "__main__":
    print("synth.py smoke test -- WINDOW-INDUCED (primary), fixed realistic time-on-shelf")
    print("target  realised   Lmonths  n_world  n_cens  medians(acc/jew/bag/watch,d)")
    for tgt in [0.0, 0.1, 0.3, 0.5, 0.7]:
        out = generate(seed=20260707, target_censoring=tgt, left_truncation=False)
        m = out["meta"]; ti = out["truth_items"]
        meds = [int(round(ti[ti.category == c]["true_dwell_days"].median()))
                for c in ["accessory", "jewelry", "bag", "watch"]]
        print("{:5.2f}  {:8.4f}  {:7d}  {:7d}  {:6d}  {}".format(
            tgt, m["realised_censoring"], m["window_length_months"],
            m["n_items_in_world"], m["n_censored"], meds))
    print("\nMisspecification DGP families (median-matched; medians must all match):")
    for fam in ["weibull", "lognormal", "gamma", "exponential"]:
        out = generate(seed=20260707, target_censoring=0.5, dwell_family=fam)
        m = out["meta"]; ti = out["truth_items"]
        meds = [int(round(ti[ti.category == c]["true_dwell_days"].median()))
                for c in ["accessory", "jewelry", "bag", "watch"]]
        print("  {:11s} target 0.50 realised {:.4f} L={}mo n={} medians {}".format(
            fam, m["realised_censoring"], m["window_length_months"],
            m["n_items_in_world"], meds))
    print("\nSTRESS-TEST scale-inflation (demoted appendix):")
    for tgt in [0.3, 0.7]:
        out = generate_scale_inflation(seed=20260707, target_censoring=tgt)
        m = out["meta"]; ti = out["truth_items"]
        meds = [int(round(ti[ti.category == c]["true_dwell_days"].median()))
                for c in ["accessory", "jewelry", "bag", "watch"]]
        print("  target {:.2f} realised {:.4f} scale {:.2f} medians {}".format(
            tgt, m["realised_censoring"], m["dwell_scale_factor"], meds))
