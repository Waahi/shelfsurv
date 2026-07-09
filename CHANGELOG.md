# Changelog

## v1.1.1 (2026-07-09)

Figure-numbering and packaging update to match the revised article. No changes to the data, the numbers, or the method; every figure and table regenerates from the same synthetic runs as v1.1.0.

- Renumbered the figures to the article's final consecutive-by-first-appearance order: the inventory-reconstruction figure is now Figure 2 and the depletion-forecast figure is now Figure 4 (previously Figure 2 = depletion, Figure 4 = inventory). Plot titles, stdout labels, and the README figure/table map were updated accordingly; the result tables map as depletion -> Table 5, time-on-shelf bias -> Table 4, misspecification -> Table 6.
- `figure5_exponential_tie.png` (Figure 5): capitalised "Weibull" in the legend.
- Removed non-release scratch and crop artifacts from the package.

## v1.1.0 (2026-07-08)

Figure additions and refinements. No changes to the data, the numbers, or the method: every figure regenerates from the same synthetic runs as v1.0.0.

- New `figure7_operating_envelope.png` (Figure 7): operating-envelope diagnostic, the unsupported-query share versus realised censoring by data-generating family.
- `figure5_exponential_tie.png` (Figure 5): added 95% paired-difference confidence intervals and falsification-pass markers.
- `figure6_misspecification.png` (Figure 6): caption reworded to "mean MAE" (the comparison is of mean MAE, not statistical dominance).
- `figure2_depletion_mae.png` and the combined depletion figure (now `Figure_4.png`): recoloured to a colourblind-safe (Wong) palette.

## v1.0.0 (2026)

MethodsX reproducibility release accompanying the method article *"shelfsurv: a censoring-aware protocol for reconstructing stock, time-on-shelf, and inventory depletion from unique-item resale event logs"* (Shao and Goto, 2026).

Initial public release:

- `synth.py` - synthetic event-ledger generator (window-induced censoring at fixed realistic time-on-shelf; a demoted dwell-scale-inflation stress-test mechanism).
- `protocol.py` - the protocol (accounting-identity stock reconstruction; hand-rolled Kaplan-Meier, Greenwood variance and RMST; age-conditioned depletion projection; Weibull and log-normal censored-MLE tails; held-out-calibration tail-selection rule) plus five practitioner baselines.
- `validate.py` - the validation harness: experiments E1-E6, figures and tables.

No external raw data. All validation runs on synthetic item-level data with known ground truth.
