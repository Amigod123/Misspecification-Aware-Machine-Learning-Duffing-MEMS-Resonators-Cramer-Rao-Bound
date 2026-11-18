# Misspecification-Aware Machine Learning for Parameter Estimation in Duffing MEMS Resonators, Benchmarked Against the Cramér–Rao Bound

MLEE course project (BITS Pilani, EEE). A controlled, reproducible simulation benchmark that asks **when** exact-model
fitting and machine learning approach the Cramér–Rao bound (CRB) for estimating a MEMS resonator's quality factor Q,
Duffing nonlinearity N and resonance shift (ppm) from up/down frequency sweeps — and what happens when the model is wrong.

> Everything here is **synthetic** (first-harmonic Duffing model, hardening only, single drive level, toy mismatch
> worlds, regular-part CRB). Nothing is claimed about real devices.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # exact versions (Python 3.13); older Python: requirements-min.txt
./reproduce.sh --smoke                                  # ~2 min end-to-end test on tiny data
./reproduce.sh                                          # full 5-seed run, ~70 min on 2 CPU cores
streamlit run app/demo_app.py                           # interactive demo
```

On Windows without bash, run the commands listed in `reproduce.sh` one by one (`python checks.py`, `python gen_data.py`, …).

## What each step does

| Step | Command | Output |
|---|---|---|
| 1. Validate simulator | `python checks.py` | `results/validation.json`, `results/tables/validation.csv` (21 checks; exits non-zero on failure) |
| 2. Generate data | `python gen_data.py --config configs/default.json` | `data/default/seed*/…npz`, `data/default/manifest.json` (ranges, seeds, versions, SHA-256) |
| 3. Experiments | `python run_exp.py --config configs/default.json` | `results/default/per_sample/seed*.pkl`, `results/default/logs/ls_fits_seed*.csv` (every fit: status, evaluations, runtime, residual) |
| 4. Tables | `python aggregate.py --config configs/default.json` | `results/default/results.json`, `results/default/tables/*.csv` with 95 % intervals |
| 5. Figures | `python figures.py` | `figures/*.png|pdf` (report figures) and `figures/poster/*` (large-text poster versions) |

## Design in one paragraph

Each sample is an up- and a down-sweep (201 points) of a normalised Duffing resonator computed by vectorised harmonic
balance (largest positive root = up-sweep, smallest = down-sweep; validated against a time-domain ODE). The measurement model
adds 0.5–2 % noise, an unknown gain (5 %), 12-bit quantisation and a window built from a coarse Q guess. Training uses
Q 800–8000 and N 0.2–2.5; held-out test blocks use Q 8000–20000 and N 2.5–4; test-only worlds add quintic stiffness,
amplitude-dependent damping, feedthrough or 3 % noise. Estimators: half-power bandwidth; bounded least squares on the exact
model (specified 3-start version, a robust version with a hysteresis-masked pre-fit, a background-term version and an
oracle-start diagnostic); gradient boosting and ridge on 47 handcrafted features; the regular-part CRB. A lack-of-fit statistic
and a leave-one-world-out ML classifier act as mismatch detectors, with split-conformal 90 % intervals and abstention.
Five seeds × 300 samples per condition, bootstrap / Wilson 95 % intervals.

## Interpreting the results

Read `results/default/tables/*.csv` (or Annex C of the report):
`accuracy.csv` (errors and failure rates), `crb_efficiency.csv` (error / CRB std; ~1 = efficient),
`optimizer_diagnostics.csv` (how often a fit ends at a worse cost than the oracle-start fit, i.e. an optimiser failure),
`error_budget.csv` (noise vs quantisation vs estimator floor), `up_vs_updown.csv`, `detectors.csv`,
`abstention_conformal.csv` (coverage and silent-failure rates), `learning_curve.csv`, `runtime.csv`.

## Repository map

See `docs/PLAN.md` (layout, tasks, acceptance criteria), `docs/literature_verification.md` (how each reference was checked).

## Credits

Python with NumPy, SciPy, scikit-learn, matplotlib and Streamlit. The reference prototype was adapted and extended for this
repository.
