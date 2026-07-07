# Changelog

## v1.1.0 (2026-07-08)

Figure additions and refinements. No changes to the data, the numbers, or the method: every figure regenerates from the same synthetic runs as v1.0.0.

- New `figure7_operating_envelope.png` (Figure 7): operating-envelope diagnostic, the unsupported-query share versus realised censoring by data-generating family.
- `figure5_exponential_tie.png` (Figure 5): added 95% paired-difference confidence intervals and falsification-pass markers.
- `figure6_misspecification.png` (Figure 6): caption reworded to "mean MAE" (the comparison is of mean MAE, not statistical dominance).
- `figure2_depletion_mae.png` and the combined `Figure_2.png`: recoloured to a colourblind-safe (Wong) palette.

## v1.0.0 (2026)

MethodsX reproducibility release accompanying the method article *"A censoring-aware protocol for reconstructing stock, time-on-shelf, and depletion from unique-item resale event logs"* (Shao and Goto, 2026).

Initial public release:

- `synth.py` - synthetic event-ledger generator (window-induced censoring at fixed realistic time-on-shelf; a demoted dwell-scale-inflation stress-test mechanism).
- `protocol.py` - the protocol (accounting-identity stock reconstruction; hand-rolled Kaplan-Meier, Greenwood variance and RMST; age-conditioned depletion projection; Weibull and log-normal censored-MLE tails; held-out-calibration tail-selection rule) plus five practitioner baselines.
- `validate.py` - the validation harness: experiments E1-E6, figures and tables.

No external raw data. All validation runs on synthetic item-level data with known ground truth.
