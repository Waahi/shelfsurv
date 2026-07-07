# Changelog

## v1.0.0 (2026)

MethodsX reproducibility release accompanying the method article *"A censoring-aware protocol for reconstructing stock, time-on-shelf, and depletion from unique-item resale event logs"* (Shao and Goto, 2026).

Initial public release:

- `synth.py` - synthetic event-ledger generator (window-induced censoring at fixed realistic time-on-shelf; a demoted dwell-scale-inflation stress-test mechanism).
- `protocol.py` - the protocol (accounting-identity stock reconstruction; hand-rolled Kaplan-Meier, Greenwood variance and RMST; age-conditioned depletion projection; Weibull and log-normal censored-MLE tails; held-out-calibration tail-selection rule) plus five practitioner baselines.
- `validate.py` - the validation harness: experiments E1-E6, figures and tables.

No external raw data. All validation runs on synthetic item-level data with known ground truth.
