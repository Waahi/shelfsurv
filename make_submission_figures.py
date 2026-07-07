"""
make_submission_figures.py -- Build the two COMBINED, submission-ready panel figures
that the MethodsX manuscript cites, each as one 600-dpi PNG with bold "A"/"B" panel
labels.

    Figure_2.png
      Panel A : depletion-forecast MAE vs censoring (read from table_depletion_mae.csv;
                protocol flat-KM and Weibull tails, category-specific age-naive,
                accounting-only, censor-at-end; 95% CI half-widths as error bars).
      Panel B : predicted-vs-true remaining current stock over the forecast horizon at
                ~30% censoring, with an inter-quartile band across replications
                (reproduces validate.make_figure2b over a few reps; deterministic).

    Figure_3.png
      Panel A : estimated Kaplan-Meier time-on-shelf survival with Greenwood band, the
                true S(t), and the sold-only curve, at ~30% and ~70% censoring
                (reproduces validate.make_figure_km; compact 1x2 inside panel A).
      Panel B : time-on-shelf bias vs censoring (read from table_dwell_bias.csv;
                sold-only median, censor-at-end median, KM median, RMST(120)).

The aggregate panels (2A, 3B) READ the tabled CSVs so they match the manuscript exactly
and run instantly; the overlay (2B) and KM (3A) reproduce the harness logic on a few
replications with the SAME deterministic seeds as validate.py. This script does NOT
re-run the 200-replication experiments, does NOT write or modify any CSV, and does NOT
overwrite the per-panel figure*.png files.

Pure NumPy/Pandas/Matplotlib. ASCII-only. Run with PYTHONUTF8=1.
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import synth
import protocol as P
import validate as V   # for BASE_SEED, CI_Z, _window_horizons, _km_step_arrays


HERE = os.path.dirname(os.path.abspath(__file__))
N_REPS_OVERLAY = 40          # fast reproduction of the Fig-2b IQR band (deterministic)
OVERLAY_CENSORING = 0.3
KM_CENSORING = (0.3, 0.7)

# Shared plot styling (kept consistent with validate.py).
DEP_STYLES = {
    "protocol_km": ("Protocol (age-cond., flat KM tail)", "o-", "#0072B2"),
    "protocol_weib": ("Protocol (age-cond., Weibull tail)", "o:", "#56B4E9"),
    "age_naive_cat": ("Category-specific age-naive", "s--", "#E69F00"),
    "accounting_only": ("Accounting-only", "d--", "#CC79A7"),
    "censor_at_end": ("Censor-at-end", "^--", "#D55E00"),
}


def _panel_label(ax, text):
    """Bold panel label ("A"/"B") at the top-left, outside the axes box."""
    ax.text(-0.12, 1.06, text, transform=ax.transAxes,
            fontsize=17, fontweight="bold", va="bottom", ha="right")


# ==============================================================================
# Panel 2A -- depletion MAE vs censoring, read from the CSV (exact, fast).
# ==============================================================================

def _plot_panel_depletion_mae(ax):
    df = pd.read_csv(os.path.join(HERE, "table_depletion_mae.csv"))
    x = df["realised_censoring_mean"].values
    for m, (lab, sty, col) in DEP_STYLES.items():
        ax.errorbar(x, df[f"mae_{m}_mean"].values, yerr=df[f"mae_{m}_ci"].values,
                    fmt=sty, color=col, capsize=3, label=lab,
                    linewidth=1.7, markersize=6)
    ax.set_xlabel("Realised right-censoring rate (window-induced)")
    ax.set_ylabel("Depletion-forecast MAE\n(fraction of current stock; "
                  "horizons capped at window length)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    return df


# ==============================================================================
# Panel 2B -- predicted-vs-true depletion overlay with IQR band (few reps).
# Reproduces validate.make_figure2b logic exactly, on N_REPS_OVERLAY reps.
# ==============================================================================

def _compute_overlay_stacks(censoring=OVERLAY_CENSORING, n_reps=N_REPS_OVERLAY):
    """
    Reproduce the depletion-overlay data on a FIXED common horizon grid so the IQR
    band pools all usable replications (not only those whose window length happens to
    match a reference rep). The grid is the reference rep's window-capped horizons; the
    SAME grid is passed to true_depletion and depletion_forecast for every rep, so each
    rep's current-stock cohort is projected and scored on the identical horizon axis.
    This is a display-only overlay (the tabled MAE uses per-cell capped horizons and is
    untouched); it uses the same protocol functions and fixed seeds, so it is exact and
    deterministic.
    """
    ref = synth.generate(seed=V.BASE_SEED, target_censoring=censoring,
                         dwell_family="weibull", left_truncation=False)
    H = V._window_horizons(ref["meta"])   # fixed common horizon grid
    stack_km = []
    stack_true = []
    ncur_sum = 0.0
    n_used = 0
    for r in range(n_reps):
        seed = V.BASE_SEED + r
        out = synth.generate(seed=seed, target_censoring=censoring,
                             dwell_family="weibull", left_truncation=False)
        ev, ti, meta = out["events"], out["truth_items"], out["meta"]
        w_end = meta["W_end"]
        td = P.true_depletion(ti, w_end, H)
        nc, tc = td["n_current"], td["true_count"]
        if nc <= 0:
            continue
        f_km = P.depletion_forecast(ev, w_end, H, tail="km")["forecast"]
        stack_km.append(f_km)
        stack_true.append(tc)
        ncur_sum += nc
        n_used += 1
    return {"horizons": H,
            "protocol_km_stack": np.vstack(stack_km),
            "true_stack": np.vstack(stack_true),
            "n_current": ncur_sum / max(n_used, 1),
            "n_used": n_used}


def _plot_panel_overlay(ax, ov):
    h = ov["horizons"]
    pred = ov["protocol_km_stack"]
    true = ov["true_stack"]
    pred_med = np.median(pred, axis=0)
    pred_q1 = np.percentile(pred, 25, axis=0)
    pred_q3 = np.percentile(pred, 75, axis=0)
    true_med = np.median(true, axis=0)
    true_q1 = np.percentile(true, 25, axis=0)
    true_q3 = np.percentile(true, 75, axis=0)

    ax.fill_between(h, true_q1, true_q3, color="0.7", alpha=0.45,
                    label="true IQR across replications")
    ax.plot(h, true_med, "k-", linewidth=2.4, label="true remaining stock (median)")
    ax.fill_between(h, pred_q1, pred_q3, color="#1b6ca8", alpha=0.22,
                    label="projected IQR across replications")
    ax.plot(h, pred_med, "o-", color="#1b6ca8", markersize=4,
            label="projected remaining stock (protocol, flat KM tail)")
    ax.set_xlabel("horizon h (days)")
    ax.set_ylabel("current stock remaining")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("~%.0f%% censoring (median over %d reps; current stock ~%.0f items)"
                 % (OVERLAY_CENSORING * 100, ov["n_used"], ov["n_current"]),
                 fontsize=9.5)


# ==============================================================================
# Panel 3A -- KM survival vs true vs sold-only (two censoring cells).
# Reproduces validate.make_figure_km content in a compact 1x2 inside panel A.
# ==============================================================================

def _plot_km_axis(ax, censoring, rep_index=0, show_legend=False):
    seed = V.BASE_SEED + rep_index
    out = synth.generate(seed=seed, target_censoring=censoring, left_truncation=False)
    ev, ti, meta = out["events"], out["truth_items"], out["meta"]
    w_end = meta["W_end"]
    t_max = float((meta["W_end"] - meta["W_start"]).days)

    durations, event = P.observed_durations(ev, w_end)
    km = P.km_dwell(durations, event)
    sold_mask = event == 1
    km_sold = P.km_dwell(durations[sold_mask],
                         np.ones(int(sold_mask.sum()), dtype=int))

    tk, Sk, lok, hik = V._km_step_arrays(km, t_max)
    ts, Ss, _, _ = V._km_step_arrays(km_sold, t_max)

    D = np.asarray(ti["true_dwell_days"], dtype=float)
    tg = np.linspace(0.0, t_max, 400)
    S_true = np.array([float(np.mean(D > x)) for x in tg])

    ax.fill_between(tk, lok, hik, step="post", color="#1b6ca8", alpha=0.22,
                    label="Greenwood 95% band")
    ax.step(tk, Sk, where="post", color="#1b6ca8", linewidth=2.0,
            label="Kaplan-Meier (Greenwood band)")
    ax.plot(tg, S_true, "k-", linewidth=2.0, label="true S(t)")
    ax.step(ts, Ss, where="post", color="#c0392b", linewidth=1.8,
            linestyle="--", label="sold-only")
    cens_times = durations[event == 0]
    cens_times = cens_times[cens_times <= t_max]
    if cens_times.size:
        ax.plot(cens_times, np.full(cens_times.size, 0.02), "|",
                color="0.35", markersize=7, markeredgewidth=0.8,
                label="censoring times")
    ax.set_xlim(0, t_max)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("time-on-shelf (days)")
    ax.set_title("~%.0f%% censoring (realised %.2f)"
                 % (censoring * 100, meta["realised_censoring"]), fontsize=9.5)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(frameon=False, fontsize=7.8, loc="upper right")


# ==============================================================================
# Panel 3B -- time-on-shelf bias vs censoring, read from the CSV (exact, fast).
# ==============================================================================

def _plot_panel_tos_bias(ax):
    df = pd.read_csv(os.path.join(HERE, "table_dwell_bias.csv"))
    x = df["realised_censoring_mean"].values
    series = [
        ("soldonly_median", "Sold-only median", "s-", "#c0392b"),
        ("censoratend_median", "Censor-at-end median", "^-", "#e28743"),
        ("km_median", "KM median", "o-", "#1b6ca8"),
        ("km_rmst120", "KM RMST(120 d)", "D:", "#0e3d5c"),
    ]
    for est, lab, sty, col in series:
        ax.errorbar(x, df[f"pct_{est}_mean"].values, yerr=df[f"pct_{est}_ci"].values,
                    fmt=sty, color=col, capsize=3, label=lab,
                    linewidth=1.8, markersize=6)
    ax.axhline(0, color="gray", linewidth=1.0, linestyle=":")
    ax.set_xlabel("Realised right-censoring rate")
    ax.set_ylabel("Bias in time-on-shelf summary (% of true)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8.5)
    return df


# ==============================================================================
# Figure builders
# ==============================================================================

def build_figure_2(path):
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.6, 5.4))
    dep_df = _plot_panel_depletion_mae(axA)
    ov = _compute_overlay_stacks()
    _plot_panel_overlay(axB, ov)
    _panel_label(axA, "A")
    _panel_label(axB, "B")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return dep_df, ov


def build_figure_3(path):
    fig = plt.figure(figsize=(13.6, 9.2))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.2)
    axA1 = fig.add_subplot(gs[0, 0])
    axA2 = fig.add_subplot(gs[0, 1], sharey=axA1)
    axB = fig.add_subplot(gs[1, :])
    _plot_km_axis(axA1, KM_CENSORING[0], show_legend=True)
    _plot_km_axis(axA2, KM_CENSORING[1], show_legend=False)
    axA1.set_ylabel("survival S(t)")
    bias_df = _plot_panel_tos_bias(axB)
    _panel_label(axA1, "A")
    _panel_label(axB, "B")
    fig.savefig(path, dpi=600)
    plt.close(fig)
    return bias_df


def _dpi_of(path):
    """Read back the stored DPI from a PNG's pHYs chunk (pixels/metre -> dpi)."""
    from PIL import Image
    with Image.open(path) as im:
        dpi = im.info.get("dpi")
    return dpi


def main():
    print("=" * 78)
    print("Building combined submission figures (Figure_2.png, Figure_3.png)")
    print("  reading CSV tables for aggregate panels; reproducing overlay/KM on")
    print("  %d reps / same seeds (no 200-rep re-run; no CSV or figure*.png change)."
          % N_REPS_OVERLAY)
    print("=" * 78)

    f2 = os.path.join(HERE, "Figure_2.png")
    f3 = os.path.join(HERE, "Figure_3.png")
    dep_df, ov = build_figure_2(f2)
    bias_df = build_figure_3(f3)

    # ---- report key plotted numbers so they can be checked against the manuscript
    def cell(df, col, target):
        row = df.iloc[(df["target_censoring"] - target).abs().idxmin()]
        return row[col]

    print("\n[Figure_2 panel A] depletion MAE (from table_depletion_mae.csv):")
    print("  50%% protocol flat-KM tail : %.4f (manuscript ~0.0969)"
          % cell(dep_df, "mae_protocol_km_mean", 0.5))
    print("  50%% protocol Weibull tail : %.4f" % cell(dep_df, "mae_protocol_weib_mean", 0.5))
    print("  50%% category age-naive    : %.4f" % cell(dep_df, "mae_age_naive_cat_mean", 0.5))
    print("  70%% protocol flat-KM tail : %.4f" % cell(dep_df, "mae_protocol_km_mean", 0.7))
    print("  30%% protocol flat-KM tail : %.4f" % cell(dep_df, "mae_protocol_km_mean", 0.3))

    print("\n[Figure_2 panel B] overlay at ~30%% (reproduced, %d reps):" % ov["n_used"])
    h = ov["horizons"]
    pred_med = np.median(ov["protocol_km_stack"], axis=0)
    true_med = np.median(ov["true_stack"], axis=0)
    print("  current stock (h=0 cohort) ~%.0f items" % ov["n_current"])
    print("  first horizon h=%.0f d : projected median %.1f vs true median %.1f"
          % (h[0], pred_med[0], true_med[0]))
    print("  last  horizon h=%.0f d : projected median %.1f vs true median %.1f"
          % (h[-1], pred_med[-1], true_med[-1]))

    print("\n[Figure_3 panel B] time-on-shelf bias (from table_dwell_bias.csv):")
    print("  70%% sold-only median bias     : %.2f%% (manuscript ~-62.1)"
          % cell(bias_df, "pct_soldonly_median_mean", 0.7))
    print("  50%% sold-only median bias     : %.2f%%" % cell(bias_df, "pct_soldonly_median_mean", 0.5))
    print("  70%% censor-at-end median bias : %.2f%%" % cell(bias_df, "pct_censoratend_median_mean", 0.7))
    print("  70%% KM median bias            : %.2f%%" % cell(bias_df, "pct_km_median_mean", 0.7))
    print("  50%% KM RMST(120) bias         : %.2f%%" % cell(bias_df, "pct_km_rmst120_mean", 0.5))

    print("\n[files]")
    for p in (f2, f3):
        try:
            dpi = _dpi_of(p)
        except Exception as e:
            dpi = "unknown (%s)" % e
        print("  %s  exists=%s  size=%d bytes  dpi=%s"
              % (os.path.basename(p), os.path.exists(p), os.path.getsize(p), dpi))
    print("\nDONE.")


if __name__ == "__main__":
    main()
