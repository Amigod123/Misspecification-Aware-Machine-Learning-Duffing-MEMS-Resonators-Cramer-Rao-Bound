# Repository and implementation plan

## Repository layout

| Path | Purpose |
|---|---|
| `configs/default.json` | Every setting used for reported results: seeds, ranges, sizes, noise, worlds, optimiser, hyper-parameters, bootstrap |
| `configs/smoke.json` | Same pipeline at toy size for a fast end-to-end test (results not meaningful) |
| `sim.py` | Vectorised harmonic-balance Duffing simulator and measurement model |
| `lib.py` | Dataset generation, 47 features, half-power baseline, least-squares fitting with diagnostics, regular-part CRB, statistics |
| `checks.py` | Simulator validation → `results/validation.json`, `results/tables/validation.csv` |
| `gen_data.py` | Deterministic datasets per seed + `data/<config>/manifest.json` (ranges, seeds, versions, SHA-256) |
| `run_exp.py` | All experiments per seed → `results/<config>/per_sample/seed*.pkl`, optimiser logs |
| `aggregate.py` | Multi-seed tables with bootstrap / Wilson 95 % intervals → `results/<config>/tables/*.csv`, `results.json` |
| `figures.py` | All figures: illustrative, results, and large-text poster versions (PNG 300 dpi + PDF) |
| `app/demo_core.py`, `app/demo_app.py` | Streamlit demo and its UI-independent back end |
| `docs/` | This plan, validation record, literature verification log |
| `reproduce.sh` | One command: checks → data → experiments → tables → figures |

## Tasks and acceptance criteria

| # | Task | Acceptance criterion |
|---|---|---|
| 1 | Simulator | Vectorised; no time stepping in data generation; 2000 samples < 1 s |
| 2 | Validation | FWHM·Q = 1 ± 0.5 % (Q = 500, 3000, 15000); peak shift within 2 % of N for N = 0.1, 0.3, 0.6; hysteresis onset within 1 % of 8/(3√3); up = down below onset; ODE vs harmonic balance within 0.2 % on stable branches (Q = 200, N = 2) |
| 3 | Data | Fixed seeds per dataset; disjoint train / calibration / test streams; held-out Q and N blocks; manifest with ranges, versions, hashes |
| 4 | Baselines | Half-power (shift bias stated); least squares with three N-starts, up-only and up+down variants, background-term variant; convergence status, nfev, runtime, residuals and failures logged per fit |
| 5 | Optimiser vs model | Oracle-start fit (true parameters as start) separates optimiser failures from model error; improved ("robust") variant reported alongside the specified one |
| 6 | ML | 47 features, HistGradientBoosting per target on log Q, N, shift; ridge; hyper-parameters fixed in config before testing |
| 7 | CRB | Regular-part Fisher bound, gain as nuisance, jump points excluded, both sweeps and up-sweep only; robust spread as headline efficiency |
| 8 | Experiments | All 10 experiments on 5 seeds × 300 samples per condition, with bootstrap / Wilson 95 % intervals and per-seed ranges |
| 9 | Detectors, conformal | Lack-of-fit and leave-one-world-out ML detector AUROC; noise control; feedthrough with/without background term; conformal coverage before / after abstention; silent-failure rates |
| 10 | Figures | Labelled axes with units, legends, captions; generated only from result files |
| 11 | Demo | Streamlit app; shows abstention and range warnings; never states certainty under mismatch |
| 12 | Reproduction | Fresh virtual environment from `requirements.txt` runs `reproduce.sh --smoke` end to end; full run documented with timings |
