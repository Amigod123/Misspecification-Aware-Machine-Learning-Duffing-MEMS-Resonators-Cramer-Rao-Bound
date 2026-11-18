# Reproducibility record (18 Nov 2026)

| Check | Result |
|---|---|
| Simulator validation (`checks.py`) | 21 / 21 checks pass (`docs/validation.md`) |
| Full run (`run_exp.py`, 5 seeds × 9,900 least-squares fits + CRB + ML + detectors) | Completed in 3,736 s (≈ 62 min) on 2 CPU cores |
| Deterministic data | Regenerating all 75 datasets reproduces identical SHA-256 hashes, both in the original environment and in a fresh virtual environment |
| Clean environment | Fresh `python -m venv` + `pip install -r requirements.txt`, repository copied without data/results/figures: `./reproduce.sh --smoke` completed end to end (exit 0, ≈ 1 min 43 s) |
| Aggregation determinism | Re-running `aggregate.py` on the stored per-seed results in the clean environment gives a `results.json` identical to the reported one except for its timestamp |
| Bug found by the clean run | `checks.py` assumed `results/` existed; fixed (`os.makedirs`) before the final run of the test |

Not tested: Windows / macOS, Python versions other than 3.13 (`requirements-min.txt` is provided for older Python but is untested).
