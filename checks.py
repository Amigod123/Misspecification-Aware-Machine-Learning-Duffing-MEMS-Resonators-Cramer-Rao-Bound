"""Simulator validation.  Writes results/validation.json and results/tables/validation.csv.
Exit code 1 if any check fails.  Usage: python checks.py [--config configs/default.json]"""
import os, sys, json, time, argparse, csv
import numpy as np
from scipy.integrate import solve_ivp
from sim import response
from lib import load_config, generate, software_versions

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.json")
args = ap.parse_args()
cfg = load_config(args.config)
os.makedirs("results/tables", exist_ok=True)
rows = []
def add(name, value, expected, tol, ok, note=""):
    rows.append(dict(check=name, value=value, expected=expected, tolerance=tol, passed=bool(ok), note=note))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {value} (expected {expected}, tol {tol}) {note}")

# 1. linear limit: FWHM * Q = 1
for Q in (500, 3000, 15000):
    hw = 1 / (2 * Q)
    Om = 1 + np.linspace(-6, 6, 120001) * hw
    a = np.abs(response(Om, Q, 0.0)[0])
    m = a >= a.max() / np.sqrt(2)
    v = (Om[m][-1] - Om[m][0]) * Q
    add(f"Linear limit FWHM*Q, Q={Q}", round(v, 5), 1.0, "0.5 %", abs(v - 1) < 0.005)
    add(f"Linear limit peak amplitude, Q={Q}", round(a.max(), 5), 1.0, "0.5 %", abs(a.max() - 1) < 0.005)

# 2. small-N backbone shift ~ N half-widths
Q = 2000; hw = 1 / (2 * Q)
for N in (0.1, 0.3, 0.6):
    Om = 1 + np.linspace(-3, 3 + 2 * N + 1, 80001) * hw
    au = np.abs(response(Om, Q, N)[0])
    v = (Om[au.argmax()] - 1) / hw
    add(f"Peak shift / HWHM, N={N}", round(v, 4), N, "2 % (first-order theory)", abs(v - N) / N < 0.02)

# 3. hysteresis onset vs analytic N_c = 8/(3*sqrt(3))
def hyst(N, Q=2000):
    hw = 1 / (2 * Q)
    Om = 1 + np.linspace(-3, 3 + 2 * N + 2, 40001) * hw
    zu, zd = response(Om, Q, N)
    return np.max(np.abs(np.abs(zu) - np.abs(zd)))
lo, hi = 0.2, 3.0
for _ in range(30):
    mid = 0.5 * (lo + hi)
    lo, hi = (lo, mid) if hyst(mid) > 1e-3 else (mid, hi)
Nc = 8 / (3 * np.sqrt(3))
add("Hysteresis onset N_c (numerical, |up-down|>1e-3)", round(hi, 4), round(Nc, 4), "1 %", abs(hi - Nc) / Nc < 0.01)
v = hyst(1.4)
add("Up/down identical below onset, N=1.4", f"{v:.2e}", 0.0, "< 1e-6", v < 1e-6)
v = hyst(3.0)
add("Hysteresis present, N=3", round(v, 4), "> 0.1", "-", v > 0.1)

# 4. harmonic balance vs time-domain ODE (stable branches), Q = 200, N = 2
Q = 200; N = 2.0; beta = 4 * N / (3 * Q); c = 1 / Q; f = c; hw = 1 / (2 * Q)
errs = []
for xs in (0.0, 1.0, 2.0, 3.0):
    W = 1 + xs * hw
    zu, zd = response(np.array([W]), Q, N)
    for name, z in (("up", zu[0]), ("down", zd[0])):
        a_hb, phi = abs(z), np.angle(z)
        T = 2 * np.pi / W
        rhs = lambda t, y: [y[1], -c * y[1] - y[0] - beta * y[0] ** 3 + f * np.cos(W * t)]
        sol = solve_ivp(rhs, [0, 300 * T], [a_hb * np.cos(phi), -a_hb * W * np.sin(phi)], rtol=1e-9, atol=1e-11,
                        t_eval=np.linspace(280 * T, 300 * T, 4000))
        a_td = 0.5 * (sol.y[0].max() - sol.y[0].min())
        e = (a_td - a_hb) / a_hb * 100
        errs.append(e)
        add(f"ODE vs harmonic balance, x={xs:.0f} HWHM, {name}-branch (%)", round(e, 3), 0.0, "|err| < 0.2 %", abs(e) < 0.2,
            f"HB={a_hb:.4f} ODE={a_td:.4f}")

# 5. generation speed
t = time.perf_counter(); generate(2000, 1, cfg); dt = time.perf_counter() - t
add("Generate 2000 samples (s)", round(dt, 3), "< 1 s", "-", dt < 1.0)

ok = all(r["passed"] for r in rows)
json.dump(dict(checks=rows, all_passed=ok, ode_err_range_pct=[float(min(errs)), float(max(errs))],
               versions=software_versions(), date=time.strftime("%Y-%m-%d %H:%M:%S")),
          open("results/validation.json", "w"), indent=1)
with open("results/tables/validation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("ALL PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
