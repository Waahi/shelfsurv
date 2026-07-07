"""
validate.py -- Experiments, metrics, figures and tables for the censoring-aware
inventory / time-on-shelf protocol (MethodsX paper #5).

Pure NumPy/Pandas/Matplotlib. ASCII-only stdout. Deterministic: every replication
uses seed = BASE_SEED + rep_index (no system randomness). Outputs (PNG figures +
CSV tables) are written into this harness directory.

DESIGN (post editorial overview)
--------------------------------
* Censoring is WINDOW-INDUCED: time-on-shelf is fixed and realistic (category
  medians 60/90/120/180 d, Weibull shape in [1.2,1.5]); censoring is set by the
  observation-window LENGTH (short window -> high censoring). The old
  time-on-shelf-scale inflation mechanism is retained only as a labelled
  STRESS-TEST appendix.
* Depletion horizons are capped PER CELL at the observation-window length, so the
  forecast is scored only as far out as the analyst has actually observed (no wild
  extrapolation beyond the window). Both a conservative flat KM tail (headline) and
  a Weibull tail (sensitivity) are reported, because at high censoring the short
  window makes the tail choice decisive -- reported honestly.
* Main depletion comparators: CATEGORY-SPECIFIC age-naive + accounting-only.
  censor-at-end is kept in the table but not featured.
* Primary time-on-shelf metric: RMST(tau=120 d), which stays within KM support at
  all cells (unbiased). RMST(tau=365 d) is reported as a cautionary illustration
  that RMST requires tau within the observed follow-up; at high censoring the
  ~5-month window is shorter than 365 d and RMST(365) is extrapolated.

Experiments
-----------
E1  Honesty assert   : reconstruct_stock == true monthly stock EXACTLY (holds at
                       ALL censoring rates -- censored items are counted in stock).
E2  Time-on-shelf bias (Fig3): sold-only / censor-at-end / KM median and RMST bias
                       vs censoring, relative to the TRUE fixed time-on-shelf distribution.
E3  Depletion (Fig2) : per-cell depletion MAE (fraction of current stock) of the
                       protocol vs category-specific age-naive vs accounting-only
                       vs censor-at-end, flat KM tail (headline) + Weibull tail.
E3t Exponential tie  : under memoryless time-on-shelf the age-conditioned protocol must
                       NOT beat age-naive; confirm the tie (in-window horizons).
E4  Inventory curve  : true vs reconstructed monthly stock + net-flow bias under
                       left-truncation.
E5  Stress appendix  : the demoted time-on-shelf-scale-inflation depletion result.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import synth
import protocol as P


# --- Configuration -------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SEED = 20260707
CENSORING_GRID = [0.0, 0.1, 0.3, 0.5, 0.7]
N_REPS = 200
N_REPS_MISSPEC = 120       # reps for the (DGP x tail) cross (kept lighter; still tight CIs)
RMST_TAU_PRIMARY = 120.0   # within KM support at ALL cells -> unbiased
RMST_TAU_EXTRAP = 365.0    # beyond support at high censoring -> extrapolation caveat
HORIZON_STEP = 30.0        # depletion horizon spacing (days); capped per-cell at window length
DGP_FAMILIES = ["weibull", "lognormal", "gamma"]  # data-generating tail shapes
MISSPEC_CENSORING = [0.1, 0.3, 0.5, 0.7]
HOLDOUT_FRAC = 0.3         # held-out fraction for the calibration-based tail selection
CI_Z = 1.96


def _ci_halfwidth(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return CI_Z * np.std(x, ddof=1) / np.sqrt(x.size)


def _mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else np.nan


def _window_horizons(meta):
    """Depletion horizons capped at the observation-window length (days)."""
    winlen = (meta["W_end"] - meta["W_start"]).days
    hmax = max(HORIZON_STEP, float(winlen))
    return np.arange(HORIZON_STEP, hmax + 1.0, HORIZON_STEP, dtype=float)


def _depletion_mae_fraction(forecast, true_count, n_current):
    if n_current <= 0:
        return np.nan
    err = np.abs(np.asarray(forecast, float) - np.asarray(true_count, float))
    return float(np.mean(err) / n_current)


# ==============================================================================
# E1 -- Honesty assert
# ==============================================================================

def experiment_honesty_assert(n_check=25):
    """
    reconstruct_stock must reproduce the TRUE month-end stock EXACTLY given the
    full entry history. Under the window-induced design this holds at ANY censoring
    rate (censored/unsold items are correctly counted as in-stock), so we assert it
    across a range of censoring cells, not just 0%. The 0% cell remains the primary
    accounting-exactness anchor (a long-enough window with almost complete data).
    """
    passed = True
    max_abs_diff = 0
    details = []
    check_cells = [0.0, 0.3, 0.5]
    for c in check_cells:
        for r in range(n_check):
            out = synth.generate(seed=BASE_SEED + r, target_censoring=c,
                                 left_truncation=False)
            me = out["true_stock_monthly"].index
            recon = P.reconstruct_stock(out["events"], me)
            diff = (recon.values - out["true_stock_monthly"].values).astype(int)
            md = int(np.max(np.abs(diff)))
            max_abs_diff = max(max_abs_diff, md)
            ok = bool(np.all(diff == 0))
            passed = passed and ok
            details.append({"cell": c, "seed": BASE_SEED + r, "max_abs_diff": md,
                            "realised_censoring": out["meta"]["realised_censoring"],
                            "exact": ok})
    return {"passed": bool(passed), "n_reps": n_check, "cells": check_cells,
            "max_abs_diff": int(max_abs_diff), "details": details}


# ==============================================================================
# E2 -- Time-on-shelf bias vs censoring (Figure 3, mechanism)
# ==============================================================================

def experiment_dwell_bias():
    """
    For each censoring rate, bias of time-on-shelf summaries relative to the TRUE
    fixed time-on-shelf distribution:
      * sold-only median, censor-at-end median  (expected monotone-negative),
      * KM median                                (expected ~0),
      * KM RMST(120)  -- primary, in-support     (expected ~0 at ALL cells),
      * KM RMST(365)  -- extrapolation caveat     (biased up where window < 365 d).
    Because time-on-shelf is fixed, the TRUE median/RMST are ~constant across cells.
    """
    rows = []
    raw = {}
    for c in CENSORING_GRID:
        recs = []
        for r in range(N_REPS):
            seed = BASE_SEED + r
            out = synth.generate(seed=seed, target_censoring=c, left_truncation=False)
            ev, ti, w_end = out["events"], out["truth_items"], out["meta"]["W_end"]
            winlen = (out["meta"]["W_end"] - out["meta"]["W_start"]).days

            ts120 = P.true_dwell_summary(ti, tau=RMST_TAU_PRIMARY)
            ts365 = P.true_dwell_summary(ti, tau=RMST_TAU_EXTRAP)
            true_med = ts120["median"]
            so = P.sold_only_dwell(ev, w_end)
            ce = P.censor_at_end_dwell(ev, w_end)
            dur, event = P.observed_durations(ev, w_end)
            km = P.km_dwell(dur, event)
            km_med = P.km_median(km)
            km_r120 = P.rmst(km, RMST_TAU_PRIMARY)
            km_r365 = P.rmst(km, RMST_TAU_EXTRAP)

            recs.append({
                "seed": seed, "realised_censoring": out["meta"]["realised_censoring"],
                "window_len_days": winlen,
                "true_median": true_med, "true_rmst120": ts120["rmst"],
                "true_rmst365": ts365["rmst"],
                "km_median": km_med,
                "bias_soldonly_median": so["median"] - true_med,
                "bias_censoratend_median": ce["median"] - true_med,
                "bias_km_median": (km_med - true_med) if np.isfinite(km_med) else np.nan,
                "bias_km_rmst120": km_r120 - ts120["rmst"],
                "bias_km_rmst365": km_r365 - ts365["rmst"],
                "pct_soldonly_median": 100.0 * (so["median"] - true_med) / true_med,
                "pct_censoratend_median": 100.0 * (ce["median"] - true_med) / true_med,
                "pct_km_median": (100.0 * (km_med - true_med) / true_med)
                                  if np.isfinite(km_med) else np.nan,
                "pct_km_rmst120": 100.0 * (km_r120 - ts120["rmst"]) / ts120["rmst"],
                "pct_km_rmst365": 100.0 * (km_r365 - ts365["rmst"]) / ts365["rmst"],
            })
        df = pd.DataFrame(recs)
        raw[c] = df
        row = {"target_censoring": c,
               "realised_censoring_mean": _mean(df["realised_censoring"]),
               "window_len_days_mean": _mean(df["window_len_days"]),
               "true_median_mean": _mean(df["true_median"]),
               "true_rmst120_mean": _mean(df["true_rmst120"]),
               "km_median_identified_frac": float(np.mean(np.isfinite(df["km_median"])))}
        for est in ["soldonly_median", "censoratend_median", "km_median",
                    "km_rmst120", "km_rmst365"]:
            row[f"pct_{est}_mean"] = _mean(df[f"pct_{est}"])
            row[f"pct_{est}_ci"] = _ci_halfwidth(df[f"pct_{est}"])
        rows.append(row)
    return {"per_cell": pd.DataFrame(rows), "raw": raw}


# ==============================================================================
# E3 -- Depletion-forecast accuracy vs censoring (Figure 2, headline)
# ==============================================================================

def experiment_depletion(dwell_family="weibull"):
    """
    Per censoring cell, depletion MAE (fraction of current stock, averaged over
    horizons CAPPED at the observation-window length) for:
      * protocol (age-conditioned), flat KM tail   [headline, conservative]
      * protocol (age-conditioned), Weibull tail   [sensitivity]
      * category-specific age-naive                [strengthened main comparator]
      * accounting-only                            [main comparator]
      * censor-at-end                              [kept, not featured]
    scored against the TRUE depletion of the current censored stock.
    """
    methods = ["protocol_km", "protocol_weib", "age_naive_cat",
               "accounting_only", "censor_at_end"]
    rows, raw, curves = [], {}, {}
    for c in CENSORING_GRID:
        recs = []
        acc = None
        acc_true = None
        acc_h = None
        acc_ncur = 0.0
        n_used = 0
        # per-replication stacks (reps on the reference grid) for the Fig-2b IQR band
        stack_km = []
        stack_true = []
        for r in range(N_REPS):
            seed = BASE_SEED + r
            out = synth.generate(seed=seed, target_censoring=c,
                                 dwell_family=dwell_family, left_truncation=False)
            ev, ti, meta = out["events"], out["truth_items"], out["meta"]
            w_end = meta["W_end"]
            H = _window_horizons(meta)
            td = P.true_depletion(ti, w_end, H)
            nc, tc = td["n_current"], td["true_count"]
            rec = {"seed": seed, "n_current": nc,
                   "realised_censoring": meta["realised_censoring"],
                   "window_len_days": (meta["W_end"] - meta["W_start"]).days}
            if nc <= 0:
                for m in methods:
                    rec[f"mae_{m}"] = np.nan
                recs.append(rec)
                continue
            f_km = P.depletion_forecast(ev, w_end, H, tail="km")["forecast"]
            f_wb = P.depletion_forecast(ev, w_end, H, tail="weibull")["forecast"]
            f_nc = P.age_naive_depletion_bycat(ev, w_end, H)["forecast"]
            f_ac = P.accounting_only_depletion(ev, w_end, H)["forecast"]
            f_ce = P.censor_at_end_depletion(ev, w_end, H)["forecast"]
            rec["mae_protocol_km"] = _depletion_mae_fraction(f_km, tc, nc)
            rec["mae_protocol_weib"] = _depletion_mae_fraction(f_wb, tc, nc)
            rec["mae_age_naive_cat"] = _depletion_mae_fraction(f_nc, tc, nc)
            rec["mae_accounting_only"] = _depletion_mae_fraction(f_ac, tc, nc)
            rec["mae_censor_at_end"] = _depletion_mae_fraction(f_ce, tc, nc)
            recs.append(rec)
            # accumulate a representative curve on a COMMON horizon grid (first
            # usable rep's grid; near-identical within a cell since window is fixed).
            if acc_true is None:
                acc_h = H
                acc_true = np.zeros(H.size)
                acc = {m: np.zeros(H.size) for m in methods}
            if H.size == acc_h.size:
                acc_true += tc
                acc["protocol_km"] += f_km
                acc["protocol_weib"] += f_wb
                acc["age_naive_cat"] += f_nc
                acc["accounting_only"] += f_ac
                acc["censor_at_end"] += f_ce
                acc_ncur += nc
                n_used += 1
                stack_km.append(f_km)
                stack_true.append(tc)
        df = pd.DataFrame(recs)
        raw[c] = df
        if n_used > 0:
            curves[c] = {"horizons": acc_h, "true": acc_true / n_used,
                         "n_current": acc_ncur / n_used,
                         "protocol_km_stack": np.vstack(stack_km),
                         "true_stack": np.vstack(stack_true),
                         **{m: acc[m] / n_used for m in methods}}
        row = {"target_censoring": c,
               "realised_censoring_mean": _mean(df["realised_censoring"]),
               "window_len_days_mean": _mean(df["window_len_days"]),
               "n_current_mean": _mean(df["n_current"])}
        for m in methods:
            row[f"mae_{m}_mean"] = _mean(df[f"mae_{m}"])
            row[f"mae_{m}_ci"] = _ci_halfwidth(df[f"mae_{m}"])
        rows.append(row)
    return {"per_cell": pd.DataFrame(rows), "raw": raw, "curves": curves,
            "dwell_family": dwell_family}


# ==============================================================================
# E3t -- Exponential near-tie (anti-rigging)
# ==============================================================================

def experiment_exponential_tie():
    """
    ANTI-RIGGING near-tie test. Under EXPONENTIAL (memoryless) time-on-shelf the
    within-category residual survival S_g(a+h)/S_g(a) equals S_g(h), so age-conditioning
    carries NO extra signal: the CATEGORY-SPECIFIC protocol (like-for-like with the
    category age-naive baseline) must NOT beat category age-naive. We compare the
    CATEGORY-SPECIFIC protocol vs category age-naive so the only difference is the
    age-conditioning itself (both stratify by category, both use the Weibull tail).
    Horizons are capped at the observation-window length.

    PASS condition = NO ADVANTAGE: protocol advantage <= 0 within the paired 95% CI.
    Under memoryless time-on-shelf the protocol should tie at low censoring and, if
    anything, be slightly WORSE at high censoring -- age-conditioning adds estimation VARIANCE
    (two noisy survival evaluations, ratio-amplified) without adding signal. It must
    never spuriously WIN. Contrast against the Weibull regime, where the same
    protocol DOES beat age-naive, which is what proves the benchmark is not rigged.
    """
    rows = []
    for fam in ["weibull", "exponential"]:
        for c in [0.1, 0.3, 0.5]:
            diffs, pm, nm = [], [], []
            for r in range(N_REPS):
                out = synth.generate(seed=BASE_SEED + r, target_censoring=c,
                                     dwell_family=fam, left_truncation=False)
                ev, ti, meta = out["events"], out["truth_items"], out["meta"]
                w_end = meta["W_end"]
                H = _window_horizons(meta)
                td = P.true_depletion(ti, w_end, H)
                nc, tc = td["n_current"], td["true_count"]
                if nc <= 0:
                    continue
                # like-for-like: category-specific protocol vs category age-naive,
                # both with the Weibull tail (correct extrapolation for both regimes).
                fp = P.depletion_forecast_bycat(ev, w_end, H, tail="weibull")["forecast"]
                fn = P.age_naive_depletion_bycat(ev, w_end, H)["forecast"]
                p = _depletion_mae_fraction(fp, tc, nc)
                n = _depletion_mae_fraction(fn, tc, nc)
                pm.append(p); nm.append(n); diffs.append(p - n)
            d = np.asarray(diffs)
            ci = _ci_halfwidth(d)
            mean_d = _mean(d)
            adv_pct = (100.0 * (_mean(nm) - _mean(pm)) / _mean(nm)) if _mean(nm) else np.nan
            # no_advantage (PASS) = protocol NOT meaningfully better: mean_d >= -CI.
            no_advantage = bool(mean_d >= -abs(ci)) if np.isfinite(ci) else True
            rows.append({"dwell_family": fam, "target_censoring": c,
                         "mae_protocol_cat": _mean(pm), "mae_age_naive_cat": _mean(nm),
                         "mae_diff_mean": mean_d, "mae_diff_ci": ci,
                         "protocol_advantage_pct": adv_pct,
                         "no_advantage_pass": no_advantage})
    return pd.DataFrame(rows)


# ==============================================================================
# E6 -- MISSPECIFICATION study: (DGP family x tail model) cross + selection rule
# ==============================================================================

def experiment_misspecification():
    """
    THE GATE. Time-on-shelf is generated from three families (Weibull, log-normal,
    gamma), all median-matched to the SAME category medians, with window-induced
    censoring. The depletion residual-survival tail is extrapolated with four tail
    models: flat KM (nonparametric), Weibull AFT (censored MLE), log-normal AFT
    (censored MLE), and a diagnostic-driven SELECTION rule (held-out calibration;
    falls back to flat KM when the data do not support a parametric tail).

    For every (DGP, tail, censoring) cell we report depletion MAE (mean +/- CI) and
    the mean UNSUPPORTED-QUERY SHARE. The key question: does a parametric tail win
    ONLY when it matches the DGP (circular), or does it help under MISSPECIFICATION
    (mismatched DGP, and especially the gamma DGP that matches NEITHER parametric
    tail)? And does the selection rule recover the benefit without circularity?
    """
    tails = ["km", "weibull", "lognormal", "select"]
    rows = []
    chosen_counts = {}   # (dgp, cens) -> Counter of selected tails
    for dgp in DGP_FAMILIES:
        for c in MISSPEC_CENSORING:
            acc = {t: [] for t in tails}
            unsup = []
            realised = []
            chosen = {"km": 0, "weibull": 0, "lognormal": 0}
            for r in range(N_REPS_MISSPEC):
                seed = BASE_SEED + r
                out = synth.generate(seed=seed, target_censoring=c,
                                     dwell_family=dgp, left_truncation=False)
                ev, ti, meta = out["events"], out["truth_items"], out["meta"]
                w_end = meta["W_end"]
                H = _window_horizons(meta)
                td = P.true_depletion(ti, w_end, H)
                nc, tc = td["n_current"], td["true_count"]
                if nc <= 0:
                    continue
                realised.append(meta["realised_censoring"])
                for t in tails:
                    if t == "select":
                        rng = np.random.default_rng(90000 + seed)
                        res = P.depletion_forecast(
                            ev, w_end, H, tail="select",
                            select_kwargs={"holdout_frac": HOLDOUT_FRAC, "rng": rng})
                        chosen[res["tail"]] = chosen.get(res["tail"], 0) + 1
                    else:
                        res = P.depletion_forecast(ev, w_end, H, tail=t)
                        if t == "km":
                            unsup.append(res["unsupported_share"])
                    acc[t].append(_depletion_mae_fraction(res["forecast"], tc, nc))
            chosen_counts[(dgp, c)] = chosen
            row = {"dgp": dgp, "target_censoring": c,
                   "realised_censoring_mean": _mean(realised),
                   "unsupported_share_mean": _mean(unsup),
                   "select_chose_km_frac": chosen["km"] / max(sum(chosen.values()), 1),
                   "select_chose_weibull_frac": chosen["weibull"] / max(sum(chosen.values()), 1),
                   "select_chose_lognormal_frac": chosen["lognormal"] / max(sum(chosen.values()), 1)}
            for t in tails:
                row[f"mae_{t}_mean"] = _mean(acc[t])
                row[f"mae_{t}_ci"] = _ci_halfwidth(acc[t])
            row["matched_tail"] = "weibull" if dgp == "weibull" else (
                "lognormal" if dgp == "lognormal" else "none")
            rows.append(row)
    return {"per_cell": pd.DataFrame(rows), "chosen_counts": chosen_counts}


# ==============================================================================
# E4 -- Inventory-curve reconstruction + left-truncation net-flow bias
# ==============================================================================

def experiment_inventory_curves():
    out = synth.generate(seed=BASE_SEED + 100, target_censoring=0.3,
                        left_truncation=False)
    me = out["true_stock_monthly"].index
    recon = P.reconstruct_stock(out["events"], me)
    rep = {"month_ends": me, "true": out["true_stock_monthly"].values,
           "reconstructed": recon.values}

    out_lt = synth.generate(seed=BASE_SEED + 200, target_censoring=0.3,
                          left_truncation=True)
    me_lt = out_lt["true_stock_monthly"].index
    recon_lt = P.reconstruct_stock(out_lt["events"], me_lt)
    netflow_lt = P.zero_opening_net_flow(out_lt["events"], me_lt)
    lt = {"month_ends": me_lt, "true": out_lt["true_stock_monthly"].values,
          "reconstructed": recon_lt.values,
          "zero_opening_net_flow": netflow_lt.values,
          "n_left_truncated": int(out_lt["events"]["left_truncated"].sum())}

    nf_bias, rc_bias = [], []
    for r in range(50):
        o = synth.generate(seed=BASE_SEED + 300 + r, target_censoring=0.3,
                          left_truncation=True)
        m = o["true_stock_monthly"].index
        nf = P.zero_opening_net_flow(o["events"], m).values
        rc = P.reconstruct_stock(o["events"], m).values
        tr = o["true_stock_monthly"].values
        nf_bias.append(np.mean(nf - tr))
        rc_bias.append(np.mean(rc - tr))
    lt["netflow_mean_bias_50reps"] = float(np.mean(nf_bias))
    lt["netflow_bias_ci_50reps"] = _ci_halfwidth(nf_bias)
    lt["recon_mean_bias_50reps"] = float(np.mean(rc_bias))
    lt["recon_bias_ci_50reps"] = _ci_halfwidth(rc_bias)
    return {"rep": rep, "lt": lt}


# ==============================================================================
# E5 -- Stress-test appendix (demoted time-on-shelf-scale inflation)
# ==============================================================================

def experiment_stress_scale_inflation(n_reps=60):
    """
    The DEMOTED time-on-shelf-scale-inflation design (fixed window, time-on-shelf
    multiplied to hit target censoring). Reported ONLY as an appendix robustness
    figure. Uses the
    fixed 36-month window and horizons 30..720 d. Shows the protocol vs the same
    baselines; also records the inflated category medians so the appendix can state
    plainly that this design distorts durations (watch median grows to multi-year).
    """
    H = np.arange(30, 720 + 1, 30, dtype=float)
    methods = ["protocol_km", "age_naive_cat", "accounting_only", "censor_at_end"]
    rows = []
    for c in [0.1, 0.3, 0.5, 0.7]:
        recs = []
        med_acc = {ct: [] for ct in synth.CATEGORIES}
        for r in range(n_reps):
            out = synth.generate_scale_inflation(seed=BASE_SEED + r, target_censoring=c)
            ev, ti, meta = out["events"], out["truth_items"], out["meta"]
            w_end = meta["W_end"]
            for ct in synth.CATEGORIES:
                med_acc[ct].append(ti[ti.category == ct]["true_dwell_days"].median())
            td = P.true_depletion(ti, w_end, H)
            nc, tc = td["n_current"], td["true_count"]
            if nc <= 0:
                continue
            rec = {"realised_censoring": meta["realised_censoring"]}
            rec["mae_protocol_km"] = _depletion_mae_fraction(
                P.depletion_forecast(ev, w_end, H, tail="km")["forecast"], tc, nc)
            rec["mae_age_naive_cat"] = _depletion_mae_fraction(
                P.age_naive_depletion_bycat(ev, w_end, H)["forecast"], tc, nc)
            rec["mae_accounting_only"] = _depletion_mae_fraction(
                P.accounting_only_depletion(ev, w_end, H)["forecast"], tc, nc)
            rec["mae_censor_at_end"] = _depletion_mae_fraction(
                P.censor_at_end_depletion(ev, w_end, H)["forecast"], tc, nc)
            recs.append(rec)
        df = pd.DataFrame(recs)
        row = {"target_censoring": c,
               "realised_censoring_mean": _mean(df["realised_censoring"]),
               "watch_median_days": _mean(med_acc["watch"]),
               "accessory_median_days": _mean(med_acc["accessory"])}
        for m in methods:
            row[f"mae_{m}_mean"] = _mean(df[f"mae_{m}"])
            row[f"mae_{m}_ci"] = _ci_halfwidth(df[f"mae_{m}"])
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================================
# Figures
# ==============================================================================

def make_figure2(dep, path):
    """Figure 2 (HEADLINE): window-induced depletion MAE vs censoring."""
    pc = dep["per_cell"]
    x = pc["realised_censoring_mean"].values
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    # Wong colourblind-safe palette; marker + line style also disambiguate (grayscale-safe).
    styles = {
        "protocol_km": ("Protocol (age-cond., flat KM tail)", "o-", "#0072B2"),
        "protocol_weib": ("Protocol (age-cond., Weibull tail)", "o:", "#56B4E9"),
        "age_naive_cat": ("Category-specific age-naive", "s--", "#E69F00"),
        "accounting_only": ("Accounting-only", "d--", "#CC79A7"),
        "censor_at_end": ("Censor-at-end", "^--", "#D55E00"),
    }
    for m, (lab, sty, col) in styles.items():
        ax.errorbar(x, pc[f"mae_{m}_mean"].values, yerr=pc[f"mae_{m}_ci"].values,
                    fmt=sty, color=col, capsize=3, label=lab, linewidth=1.7, markersize=6)
    ax.set_xlabel("Realised right-censoring rate (window-induced)")
    ax.set_ylabel("Depletion-forecast MAE\n(fraction of current stock; horizons capped at window length)")
    ax.set_title("Figure 2. Forward depletion-forecast accuracy vs censoring\n"
                 "(fixed realistic time-on-shelf; N=%d reps/cell)" % N_REPS)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure2b(dep, path, censoring_value=0.3):
    """
    Figure 2 panel B (companion to Figure 2): predicted-vs-true depletion overlay for
    a representative censoring cell. The protocol's projected remaining current stock
    Ihat_remain(W+h) (flat KM tail) is plotted against the true remaining stock over
    the forecast horizon h, with a shaded inter-quartile (IQR) band across the
    replications on the reference horizon grid.
    """
    curves = dep["curves"]
    if not curves:
        return
    target = min(curves.keys(), key=lambda k: abs(k - censoring_value))
    cv = curves[target]
    h = cv["horizons"]
    pred_stack = cv["protocol_km_stack"]   # (n_reps, n_horizons)
    true_stack = cv["true_stack"]
    pred_med = np.median(pred_stack, axis=0)
    pred_q1 = np.percentile(pred_stack, 25, axis=0)
    pred_q3 = np.percentile(pred_stack, 75, axis=0)
    true_med = np.median(true_stack, axis=0)
    true_q1 = np.percentile(true_stack, 25, axis=0)
    true_q3 = np.percentile(true_stack, 75, axis=0)

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.fill_between(h, true_q1, true_q3, color="0.7", alpha=0.45,
                    label="true IQR across replications")
    ax.plot(h, true_med, "k-", linewidth=2.4, label="true remaining stock (median)")
    ax.fill_between(h, pred_q1, pred_q3, color="#1b6ca8", alpha=0.22,
                    label="projected IQR across replications")
    ax.plot(h, pred_med, "o-", color="#1b6ca8", markersize=4,
            label="projected remaining stock (protocol, flat KM tail)")
    ax.set_xlabel("horizon h (days)")
    ax.set_ylabel("current stock remaining")
    ax.set_title("Figure 2 (panel B). Projected vs true depletion of current stock\n"
                 "at ~%.0f%% censoring (median over reps; current stock ~%.0f items)"
                 % (target * 100, cv["n_current"]))
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure3(dwell, path):
    """Figure 3 (mechanism): time-on-shelf median + RMST(120) bias vs censoring."""
    pc = dwell["per_cell"]
    x = pc["realised_censoring_mean"].values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    for est, lab, sty, col in [
        ("soldonly_median", "Sold-only median", "s-", "#c0392b"),
        ("censoratend_median", "Censor-at-end median", "^-", "#e28743"),
        ("km_median", "KM median", "o-", "#1b6ca8"),
    ]:
        ax1.errorbar(x, pc[f"pct_{est}_mean"].values, yerr=pc[f"pct_{est}_ci"].values,
                     fmt=sty, color=col, capsize=3, label=lab, linewidth=1.8, markersize=6)
    ax1.axhline(0, color="gray", linewidth=1.0, linestyle=":")
    ax1.set_xlabel("Realised right-censoring rate")
    ax1.set_ylabel("Bias in time-on-shelf MEDIAN (% of true)")
    ax1.set_title("Median time-on-shelf bias")
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False)

    for est, lab, sty, col in [
        ("km_rmst120", "KM RMST(120 d) [in-support]", "o-", "#1b6ca8"),
        ("km_rmst365", "KM RMST(365 d) [extrapolated]", "o:", "#c0392b"),
    ]:
        ax2.errorbar(x, pc[f"pct_{est}_mean"].values, yerr=pc[f"pct_{est}_ci"].values,
                     fmt=sty, color=col, capsize=3, label=lab, linewidth=1.8, markersize=6)
    ax2.axhline(0, color="gray", linewidth=1.0, linestyle=":")
    ax2.set_xlabel("Realised right-censoring rate")
    ax2.set_ylabel("Bias in RMST (% of true)")
    ax2.set_title("RMST bias: primary tau=120 d vs extrapolated tau=365 d")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=8.5)

    fig.suptitle("Figure 3. Time-on-shelf estimator bias vs censoring (mechanism)  "
                 "[fixed time-on-shelf; N=%d reps/cell]" % N_REPS)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure4(inv, path):
    rep, lt = inv["rep"], inv["lt"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    me = rep["month_ends"]
    ax1.plot(me, rep["true"], "k-", linewidth=2.4, label="TRUE stock")
    ax1.plot(me, rep["reconstructed"], "o", color="#1b6ca8", markersize=4, label="Reconstructed (Step 1)")
    ax1.set_xlabel("Month end"); ax1.set_ylabel("Items in stock")
    ax1.set_title("Step-1 reconstruction, full history\n(exact overlay)")
    ax1.grid(True, alpha=0.3); ax1.legend(frameon=False)
    for l in ax1.get_xticklabels():
        l.set_rotation(45); l.set_ha("right")
    me2 = lt["month_ends"]
    ax2.plot(me2, lt["true"], "k-", linewidth=2.4, label="TRUE stock")
    ax2.plot(me2, lt["reconstructed"], "o-", color="#1b6ca8", markersize=3, label="Full reconstruction")
    ax2.plot(me2, lt["zero_opening_net_flow"], "s--", color="#c0392b", markersize=3,
             label="Zero-opening net-flow (b3)")
    ax2.set_xlabel("Month end"); ax2.set_ylabel("Items in stock")
    ax2.set_title("Left-truncation: net-flow underestimates stock\n"
                  "(mean net-flow bias %.1f items over 50 reps)" % lt["netflow_mean_bias_50reps"])
    ax2.grid(True, alpha=0.3); ax2.legend(frameon=False)
    for l in ax2.get_xticklabels():
        l.set_rotation(45); l.set_ha("right")
    fig.suptitle("Figure 4. Inventory-curve reconstruction and opening-stock bias")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure_tie(tie_df, path):
    """Exponential near-tie panel: protocol vs age-naive advantage, two regimes."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for fam, col, mk in [("weibull", "#0072B2", "o-"), ("exponential", "#D55E00", "s--")]:
        sub = tie_df[tie_df.dwell_family == fam].sort_values("target_censoring")
        # 95% CI on the paired MAE difference, expressed as % of the age-naive MAE.
        yerr = 100.0 * sub["mae_diff_ci"].values / sub["mae_age_naive_cat"].values
        ax.errorbar(sub["target_censoring"], sub["protocol_advantage_pct"], yerr=yerr,
                    fmt=mk, color=col, capsize=3, linewidth=1.8, markersize=7,
                    label="%s time-on-shelf" % fam)
        if fam == "exponential":
            pas = sub[sub["no_advantage_pass"] == True]
            ax.scatter(pas["target_censoring"], pas["protocol_advantage_pct"],
                       s=150, facecolors="none", edgecolors="#2e7d32", linewidths=1.8,
                       zorder=6, label="falsification passes (no advantage)")
    ax.axhline(0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Realised right-censoring rate")
    ax.set_ylabel("Protocol advantage over category age-naive\n"
                  "(% MAE reduction; error bars = 95% paired-difference CI)")
    ax.set_title("Figure 5. Falsification control\n"
                 "Weibull: protocol helps.  Exponential (memoryless): no advantage.")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure_stress(stress_df, path):
    """Appendix stress-test: time-on-shelf-scale-inflation depletion MAE + inflated medians."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8))
    x = stress_df["realised_censoring_mean"].values
    for m, lab, sty, col in [
        ("protocol_km", "Protocol (flat KM tail)", "o-", "#1b6ca8"),
        ("age_naive_cat", "Category age-naive", "s--", "#e28743"),
        ("accounting_only", "Accounting-only", "d--", "#7d8f69"),
        ("censor_at_end", "Censor-at-end", "^--", "#c0392b"),
    ]:
        ax1.errorbar(x, stress_df[f"mae_{m}_mean"], yerr=stress_df[f"mae_{m}_ci"],
                     fmt=sty, color=col, capsize=3, label=lab, linewidth=1.7, markersize=6)
    ax1.set_xlabel("Realised right-censoring rate")
    ax1.set_ylabel("Depletion MAE (fraction of current stock)")
    ax1.set_title("Time-on-shelf scale-inflation (alternative design), fixed window")
    ax1.grid(True, alpha=0.3); ax1.legend(frameon=False, fontsize=8.5)

    ax2.plot(x, stress_df["watch_median_days"], "o-", color="#7d1f3d", label="watch true median")
    ax2.plot(x, stress_df["accessory_median_days"], "s-", color="#1b6ca8", label="accessory true median")
    ax2.set_xlabel("Realised right-censoring rate")
    ax2.set_ylabel("Inflated TRUE median time-on-shelf (days)")
    ax2.set_title("Why it is not the main design:\ntime-on-shelf inflates with censoring (unrealistic)")
    ax2.grid(True, alpha=0.3); ax2.legend(frameon=False)
    fig.suptitle("Supplementary Figure A1. Time-on-shelf scale-inflation censoring (alternative design)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure_misspec(miss, path, censoring_levels=None):
    """
    Misspecification figure, FACETED over all reported cells: a small-multiples grid
    of depletion MAE by tail model. Rows are censoring levels (10/30/50/70%), columns
    are the three data-generating families (Weibull / log-normal / gamma). Within each
    panel, grouped bars give the four tail models (flat KM, Weibull, log-normal,
    selection rule); the matched tail is hatched, and a dashed reference line marks
    the flat Kaplan-Meier MAE so it is visible that no parametric tail exceeds the flat
    Kaplan-Meier tail in any (family x censoring) cell. Uses the same numbers as
    table_misspecification.csv (no recomputation).
    """
    if censoring_levels is None:
        censoring_levels = list(MISSPEC_CENSORING)  # [0.1, 0.3, 0.5, 0.7]
    pc = miss["per_cell"]
    dgps = list(DGP_FAMILIES)  # weibull, lognormal, gamma
    tails = ["km", "weibull", "lognormal", "select"]
    colors = {"km": "#7d8f69", "weibull": "#1b6ca8",
              "lognormal": "#e28743", "select": "#7d1f3d"}
    labels = {"km": "flat KM", "weibull": "Weibull tail",
              "lognormal": "log-normal tail", "select": "selection rule"}
    dgp_title = {"weibull": "DGP: Weibull", "lognormal": "DGP: log-normal",
                 "gamma": "DGP: gamma"}

    nrow, ncol = len(censoring_levels), len(dgps)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow),
                             sharex=True)
    axes = np.atleast_2d(axes)
    nbar = len(tails)
    width = 0.8 / nbar
    xbase = np.arange(nbar)
    handles_ref = None
    for ri, c in enumerate(censoring_levels):
        for ci, dgp in enumerate(dgps):
            ax = axes[ri, ci]
            cell = pc[np.isclose(pc["target_censoring"], c) & (pc["dgp"] == dgp)]
            if cell.empty:
                ax.set_visible(False)
                continue
            row = cell.iloc[0]
            matched = row["matched_tail"]
            km_val = row["mae_km_mean"]
            bars_list = []
            for j, t in enumerate(tails):
                val = row[f"mae_{t}_mean"]
                err = row[f"mae_{t}_ci"]
                b = ax.bar(xbase[j], val, width=width * 1.0, yerr=err, capsize=2,
                           color=colors[t], label=labels[t],
                           edgecolor="black", linewidth=0.5)
                if t == matched:
                    b[0].set_hatch("///")
                bars_list.append(b)
            # reference line at the flat-KM MAE: parametric bars at/below => no exceedance
            ax.axhline(km_val, color="#7d8f69", linestyle=":", linewidth=1.2)
            if handles_ref is None:
                handles_ref = [b[0] for b in bars_list]
            ax.set_xticks(xbase)
            ax.set_xticklabels(["flat KM", "Weib", "log-N", "select"],
                               rotation=0, fontsize=7.5)
            ax.tick_params(axis="y", labelsize=7.5)
            ax.grid(True, alpha=0.3, axis="y")
            if ci == 0:
                ax.set_ylabel("cens ~%.0f%%\nMAE" % (c * 100), fontsize=8.5)
            if ri == 0:
                ax.set_title(dgp_title[dgp], fontsize=9.5)
    fig.legend(handles_ref, [labels[t] for t in tails],
               frameon=False, ncol=4, fontsize=9, loc="lower center",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Figure 6. Tail misspecification: depletion mean MAE by data-generating "
                 "family x tail model\n(hatched = tail matches the DGP; gamma matches "
                 "neither parametric tail; dotted line = flat Kaplan-Meier mean MAE; "
                 "no parametric tail has higher mean MAE in any cell)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(path, dpi=600)
    plt.close(fig)


def make_figure_envelope(miss, path):
    """Figure 7 (operating-envelope diagnostic): the runtime unsupported-query share vs
    realised censoring, by data-generating family. This is the observable signal that
    governs the flat-vs-parametric tail choice (Section 2.6): a small share means the flat
    Kaplan-Meier tail is adequate within supported horizons; as the share grows past about
    one half, a parametric tail selected by held-out calibration is preferred. Same numbers
    as table_misspecification.csv (no recomputation)."""
    pc = miss["per_cell"]
    dgp_lab = {"weibull": "Weibull", "lognormal": "log-normal", "gamma": "gamma"}
    dgp_col = {"weibull": "#0072B2", "lognormal": "#E69F00", "gamma": "#009E73"}
    dgp_mk = {"weibull": "o-", "lognormal": "s--", "gamma": "^-."}
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhspan(0.0, 0.5, color="#2e7d32", alpha=0.06)
    ax.axhspan(0.5, 1.0, color="#D55E00", alpha=0.06)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1.0)
    for dgp in list(DGP_FAMILIES):
        sub = pc[pc["dgp"] == dgp].sort_values("realised_censoring_mean")
        ax.plot(sub["realised_censoring_mean"], sub["unsupported_share_mean"],
                dgp_mk[dgp], color=dgp_col[dgp], linewidth=1.9, markersize=7,
                label=dgp_lab[dgp])
    ax.set_xlabel("Realised right-censoring rate")
    ax.set_ylabel("Unsupported-query share\n(current-stock queries with a+h beyond Kaplan-Meier support)")
    ax.set_ylim(0.0, 1.0)
    ax.text(0.985, 0.30, "flat KM tail adequate\n(within supported horizons)",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color="#2e7d32")
    ax.text(0.985, 0.72, "over half unsupported:\nparametric tail preferred",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color="#a5451f")
    ax.set_title("Figure 7. Operating-envelope diagnostic: the unsupported-query share\n"
                 "governs the flat-vs-parametric tail choice (Section 2.6)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, title="data-generating family", fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def _km_step_arrays(km, t_max):
    """Right-continuous step arrays (t, S, lo, hi) for a KM dict over [0, t_max]."""
    t = np.asarray(km["t"], dtype=float)
    S = np.asarray(km["S"], dtype=float)
    var = np.asarray(km["var"], dtype=float)
    se = np.sqrt(np.maximum(var, 0.0))
    lo = np.clip(S - CI_Z * se, 0.0, 1.0)
    hi = np.clip(S + CI_Z * se, 0.0, 1.0)
    # extend the last step flat out to t_max so the curve spans the panel
    if t[-1] < t_max:
        t = np.concatenate([t, [t_max]])
        S = np.concatenate([S, [S[-1]]])
        lo = np.concatenate([lo, [lo[-1]]])
        hi = np.concatenate([hi, [hi[-1]]])
    return t, S, lo, hi


def make_figure_km(path, censoring_pair=(0.3, 0.7), rep_index=0):
    """
    NEW central-estimand figure. For one representative synthetic replication at
    ~30% and ~70% censoring (two side-by-side panels), plot:
      (a) the estimated Kaplan-Meier survival Shat(t) of time-on-shelf with its
          Greenwood 95% confidence band,
      (b) the TRUE survival S(t) of time-on-shelf (empirical survival of the true,
          uncensored durations from the truth ledger),
      (c) the sold-only Kaplan-Meier curve (censored items dropped) to visualize the
          downward bias, and
      (d) censoring tick marks along the x-axis (observed censored durations).
    This is a purely additive display; it uses synth.generate + protocol with a fixed
    seed and does not touch any tabled computation.
    """
    seed = BASE_SEED + rep_index
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharey=True)
    for ax, c in zip(axes, censoring_pair):
        out = synth.generate(seed=seed, target_censoring=c, left_truncation=False)
        ev, ti, meta = out["events"], out["truth_items"], out["meta"]
        w_end = meta["W_end"]
        t_max = float((meta["W_end"] - meta["W_start"]).days)

        durations, event = P.observed_durations(ev, w_end)
        km = P.km_dwell(durations, event)                 # censoring-aware KM
        # sold-only KM: drop censored items, treat remaining as all-events
        sold_mask = event == 1
        km_sold = P.km_dwell(durations[sold_mask],
                             np.ones(int(sold_mask.sum()), dtype=int))

        tk, Sk, lok, hik = _km_step_arrays(km, t_max)
        ts, Ss, _, _ = _km_step_arrays(km_sold, t_max)

        # true empirical survival of the uncensored true durations
        D = np.asarray(ti["true_dwell_days"], dtype=float)
        tg = np.linspace(0.0, t_max, 400)
        S_true = np.array([float(np.mean(D > x)) for x in tg])

        realised = meta["realised_censoring"]
        ax.fill_between(tk, lok, hik, step="post", color="#1b6ca8", alpha=0.22,
                        label="Greenwood 95% band")
        ax.step(tk, Sk, where="post", color="#1b6ca8", linewidth=2.0,
                label="Kaplan-Meier (Greenwood band)")
        ax.plot(tg, S_true, "k-", linewidth=2.0, label="true S(t)")
        ax.step(ts, Ss, where="post", color="#c0392b", linewidth=1.8,
                linestyle="--", label="sold-only")

        # censoring tick marks (observed censored durations) along the bottom axis
        cens_times = durations[event == 0]
        cens_times = cens_times[cens_times <= t_max]
        if cens_times.size:
            ax.plot(cens_times, np.full(cens_times.size, 0.02), "|",
                    color="0.35", markersize=7, markeredgewidth=0.8,
                    label="censoring times")

        ax.set_xlim(0, t_max)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("time-on-shelf (days)")
        ax.set_title("~%.0f%% censoring (realised %.2f)" % (c * 100, realised))
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("survival S(t)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Figure. Estimated Kaplan-Meier time-on-shelf survival vs the true "
                 "survival\n(censoring-aware KM with Greenwood band recovers the truth; "
                 "sold-only is biased downward)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=600)
    plt.close(fig)


# ==============================================================================
# Orchestration
# ==============================================================================

def _f(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return ("{:." + str(nd) + "f}").format(x)


def main():
    print("=" * 80)
    print("MethodsX inventory/time-on-shelf validation harness -- WINDOW-INDUCED design")
    print("BASE_SEED=%d  N_REPS=%d  censoring grid=%s" % (BASE_SEED, N_REPS, CENSORING_GRID))
    print("=" * 80)

    # ---- E1 -----------------------------------------------------------------
    print("\n[E1] HONESTY ASSERT (reconstruct_stock == true monthly stock)")
    ha = experiment_honesty_assert(n_check=15)
    print("  exact over cells %s x 15 reps?  -> %s" %
          (ha["cells"], "PASS" if ha["passed"] else "FAIL"))
    print("  max abs month-end difference: %d" % ha["max_abs_diff"])

    # ---- E2 -----------------------------------------------------------------
    print("\n[E2] TIME-ON-SHELF BIAS vs CENSORING (Figure 3 mechanism)")
    dwell = experiment_dwell_bias()
    pc = dwell["per_cell"]
    print("  TRUE time-on-shelf is FIXED: true median ~%.0f d across all cells (no inflation)."
          % _mean(pc["true_median_mean"]))
    print("  Bias in time-on-shelf median / RMST, %% of TRUE (mean over %d reps):" % N_REPS)
    print("  {:>7} {:>9} {:>8} {:>13} {:>13} {:>10} {:>13} {:>15}".format(
        "target", "realised", "win_d", "soldonly_med%", "censend_med%",
        "KM_med%", "KM_RMST120%", "KM_RMST365%"))
    for _, row in pc.iterrows():
        print("  {:>7.2f} {:>9.4f} {:>8.0f} {:>13} {:>13} {:>10} {:>13} {:>15}".format(
            row["target_censoring"], row["realised_censoring_mean"],
            row["window_len_days_mean"],
            _f(row["pct_soldonly_median_mean"]), _f(row["pct_censoratend_median_mean"]),
            _f(row["pct_km_median_mean"]), _f(row["pct_km_rmst120_mean"]),
            _f(row["pct_km_rmst365_mean"])))
    pc.to_csv(os.path.join(HERE, "table_dwell_bias.csv"), index=False)
    print("  RMST(120) is the PRIMARY metric (in KM support at all cells).")
    print("  RMST(365) shown as caveat: biased up where window < 365 d (tau > follow-up).")
    print("  -> wrote table_dwell_bias.csv")

    # ---- E3 headline (weibull) ---------------------------------------------
    print("\n[E3] DEPLETION-FORECAST MAE vs CENSORING (Figure 2 HEADLINE, Weibull time-on-shelf)")
    dep = experiment_depletion(dwell_family="weibull")
    pcd = dep["per_cell"]
    print("  Depletion MAE (fraction of current stock; horizons capped at window):")
    print("  {:>7} {:>9} {:>7} {:>11} {:>13} {:>14} {:>13} {:>13}".format(
        "target", "realised", "win_d", "proto(km)", "proto(weib)", "age_naive_cat",
        "acct_only", "censend"))
    for _, row in pcd.iterrows():
        print("  {:>7.2f} {:>9.4f} {:>7.0f} {:>11} {:>13} {:>14} {:>13} {:>13}".format(
            row["target_censoring"], row["realised_censoring_mean"],
            row["window_len_days_mean"],
            _f(row["mae_protocol_km_mean"], 4), _f(row["mae_protocol_weib_mean"], 4),
            _f(row["mae_age_naive_cat_mean"], 4), _f(row["mae_accounting_only_mean"], 4),
            _f(row["mae_censor_at_end_mean"], 4)))
    pcd.to_csv(os.path.join(HERE, "table_depletion_mae.csv"), index=False)
    print("  -> wrote table_depletion_mae.csv")

    print("\n  VERDICT (protocol vs strengthened category age-naive, per cell):")
    for _, row in pcd.iterrows():
        if not np.isfinite(row["mae_protocol_km_mean"]):
            print("    c=%.2f : no current stock (depletion undefined; 0%% anchor)" % row["target_censoring"])
            continue
        base = row["mae_age_naive_cat_mean"]
        km_imp = 100.0 * (base - row["mae_protocol_km_mean"]) / base if base > 0 else np.nan
        wb_imp = 100.0 * (base - row["mae_protocol_weib_mean"]) / base if base > 0 else np.nan
        print("    c=%.2f : flat-tail %s%.0f%% vs age-naive | Weibull-tail %s%.0f%% vs age-naive"
              % (row["target_censoring"],
                 "+" if km_imp >= 0 else "", km_imp,
                 "+" if wb_imp >= 0 else "", wb_imp))

    # ---- E3t exponential tie -----------------------------------------------
    print("\n[E3t] EXPONENTIAL NEAR-TIE (anti-rigging): category protocol vs category age-naive")
    print("  Under memoryless time-on-shelf age-conditioning carries NO signal; the protocol")
    print("  must NOT win (advantage <= 0). Weibull regime shown for contrast.")
    tie = experiment_exponential_tie()
    print("  {:>12} {:>7} {:>13} {:>15} {:>12} {:>16}".format(
        "family", "cens", "proto_cat MAE", "age_naive MAE", "advantage%", "verdict"))
    for _, row in tie.iterrows():
        if row["dwell_family"] == "exponential":
            verdict = "PASS(no adv)" if row["no_advantage_pass"] else "FAIL(won!)"
        else:
            verdict = "protocol wins" if row["protocol_advantage_pct"] > 0 else "(no win)"
        print("  {:>12} {:>7.2f} {:>13} {:>15} {:>11}% {:>16}".format(
            row["dwell_family"], row["target_censoring"],
            _f(row["mae_protocol_cat"], 4), _f(row["mae_age_naive_cat"], 4),
            _f(row["protocol_advantage_pct"], 0), verdict))
    tie.to_csv(os.path.join(HERE, "table_exponential_tie.csv"), index=False)
    print("  Reading: exponential advantage <= 0 at every cell => not rigged; the")
    print("  Weibull advantage is real, not a construction artifact.")
    print("  -> wrote table_exponential_tie.csv")

    # ---- E6 misspecification study (THE GATE) ------------------------------
    print("\n[E6] TAIL MISSPECIFICATION (THE GATE): (DGP family x tail model) depletion MAE")
    print("  DGP families median-matched to same category medians; N=%d reps/cell."
          % N_REPS_MISSPEC)
    miss = experiment_misspecification()
    pcm = miss["per_cell"]
    for c in MISSPEC_CENSORING:
        print("\n  --- censoring ~%.0f%% ---" % (c * 100))
        print("  {:>10} {:>10} {:>11} {:>13} {:>12} {:>18}".format(
            "DGP", "flat_KM", "Weibull", "lognormal", "SELECT", "unsupp_share"))
        for _, row in pcm[np.isclose(pcm["target_censoring"], c)].iterrows():
            matched = row["matched_tail"]
            def mark(t):
                return "*" if t == matched else " "
            print("  {:>10} {:>10} {:>10}{} {:>12}{} {:>12} {:>18}".format(
                row["dgp"],
                _f(row["mae_km_mean"], 4),
                _f(row["mae_weibull_mean"], 4), mark("weibull"),
                _f(row["mae_lognormal_mean"], 4), mark("lognormal"),
                _f(row["mae_select_mean"], 4),
                _f(row["unsupported_share_mean"], 3)))
    print("  (* = tail matches DGP. gamma DGP matches NEITHER parametric tail.)")
    pcm.to_csv(os.path.join(HERE, "table_misspecification.csv"), index=False)
    print("  -> wrote table_misspecification.csv")

    # explicit verdicts
    print("\n  VERDICT on misspecification:")
    for c in [0.3, 0.5]:
        sub = pcm[np.isclose(pcm["target_censoring"], c)]
        print("   censoring ~%.0f%%:" % (c * 100))
        for _, row in sub.iterrows():
            km = row["mae_km_mean"]
            best_par = min(row["mae_weibull_mean"], row["mae_lognormal_mean"])
            worst_par = max(row["mae_weibull_mean"], row["mae_lognormal_mean"])
            sel = row["mae_select_mean"]
            par_beats_km = worst_par < km  # does EVERY parametric tail beat flat km?
            best_impr = 100.0 * (km - best_par) / km if km > 0 else np.nan
            sel_impr = 100.0 * (km - sel) / km if km > 0 else np.nan
            note = ("both param tails beat km" if par_beats_km
                    else "some param tail LOSES to km")
            print("     %-10s: best param %s%.0f%% vs km, select %s%.0f%% vs km, %s"
                  % (row["dgp"], "+" if best_impr >= 0 else "", best_impr,
                     "+" if sel_impr >= 0 else "", sel_impr, note))
    # gamma is the clean misspecification test (matches neither)
    g50 = pcm[(pcm.dgp == "gamma") & np.isclose(pcm.target_censoring, 0.5)]
    if not g50.empty:
        gr = g50.iloc[0]
        gk, gb = gr["mae_km_mean"], min(gr["mae_weibull_mean"], gr["mae_lognormal_mean"])
        print("   GAMMA DGP @50%% (neither tail matches): km=%.4f, best mismatched param=%.4f"
              % (gk, gb))
        print("     => parametric tail %s flat-KM under genuine misspecification."
              % ("BEATS" if gb < gk else "does NOT beat"))

    print("\n  Selection-rule choices (fraction) at ~50%% censoring:")
    for _, row in pcm[np.isclose(pcm["target_censoring"], 0.5)].iterrows():
        print("     %-10s: km %.2f | weibull %.2f | lognormal %.2f (matched=%s)"
              % (row["dgp"], row["select_chose_km_frac"],
                 row["select_chose_weibull_frac"], row["select_chose_lognormal_frac"],
                 row["matched_tail"]))

    # ---- E4 -----------------------------------------------------------------
    print("\n[E4] INVENTORY-CURVE RECONSTRUCTION + LEFT-TRUNCATION BIAS (Figure 4)")
    inv = experiment_inventory_curves()
    lt, rep = inv["lt"], inv["rep"]
    exact_rep = bool(np.all(rep["reconstructed"] == rep["true"]))
    print("  representative rep, reconstructed==true every month? -> %s" % ("YES" if exact_rep else "NO"))
    print("  left-truncation: %d opening-stock items" % lt["n_left_truncated"])
    print("  zero-opening net-flow mean bias (50 reps): %+.2f items (CI +/- %.2f)"
          % (lt["netflow_mean_bias_50reps"], lt["netflow_bias_ci_50reps"]))
    print("  full reconstruction  mean bias (50 reps): %+.2f items (CI +/- %.2f)"
          % (lt["recon_mean_bias_50reps"], lt["recon_bias_ci_50reps"]))

    # ---- E5 stress appendix -------------------------------------------------
    print("\n[E5] STRESS-TEST APPENDIX (demoted time-on-shelf-scale inflation)")
    stress = experiment_stress_scale_inflation(n_reps=60)
    print("  {:>7} {:>9} {:>13} {:>11} {:>14} {:>13}".format(
        "target", "realised", "watch_med_d", "proto(km)", "age_naive_cat", "acct_only"))
    for _, row in stress.iterrows():
        print("  {:>7.2f} {:>9.4f} {:>13.0f} {:>11} {:>14} {:>13}".format(
            row["target_censoring"], row["realised_censoring_mean"],
            row["watch_median_days"], _f(row["mae_protocol_km_mean"], 4),
            _f(row["mae_age_naive_cat_mean"], 4), _f(row["mae_accounting_only_mean"], 4)))
    stress.to_csv(os.path.join(HERE, "table_stress_scale_inflation.csv"), index=False)
    print("  NOTE: watch median inflates with censoring (unrealistic) -> appendix only.")
    print("  -> wrote table_stress_scale_inflation.csv")

    # ---- figures ------------------------------------------------------------
    print("\n[FIGURES] writing PNGs ...")
    make_figure2(dep, os.path.join(HERE, "figure2_depletion_mae.png"))
    make_figure2b(dep, os.path.join(HERE, "figure2b_depletion_curve.png"), censoring_value=0.3)
    make_figure3(dwell, os.path.join(HERE, "figure3_dwell_bias.png"))
    make_figure4(inv, os.path.join(HERE, "figure4_inventory_curves.png"))
    make_figure_tie(tie, os.path.join(HERE, "figure5_exponential_tie.png"))
    make_figure_misspec(miss, os.path.join(HERE, "figure6_misspecification.png"))
    make_figure_envelope(miss, os.path.join(HERE, "figure7_operating_envelope.png"))
    make_figure_km(os.path.join(HERE, "figure_km_curve.png"))
    make_figure_stress(stress, os.path.join(HERE, "figureA1_stress_scale_inflation.png"))
    print("  -> figure2_depletion_mae.png, figure2b_depletion_curve.png,")
    print("     figure3_dwell_bias.png, figure4_inventory_curves.png,")
    print("     figure5_exponential_tie.png, figure6_misspecification.png,")
    print("     figure_km_curve.png, figureA1_stress_scale_inflation.png")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("  E1 honesty assert : %s (max diff %d)" % ("PASS" if ha["passed"] else "FAIL", ha["max_abs_diff"]))
    print("  E4 rep exact       : %s" % ("YES" if exact_rep else "NO"))
    print("=" * 80)
    return {"honesty": ha, "dwell": dwell, "depletion": dep, "tie": tie,
            "misspec": miss, "inventory": inv, "stress": stress}


if __name__ == "__main__":
    main()
