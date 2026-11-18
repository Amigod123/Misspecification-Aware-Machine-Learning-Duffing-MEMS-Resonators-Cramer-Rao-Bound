"""Demo back-end, independent of the UI (also used by the headless test).

analyse(...) simulates one up/down sweep pair with chosen physics and measurement settings, then runs
Baseline A (half-power), Baseline B (robust exact-model least squares; plus a background-term fit),
the gradient-boosting model with its split-conformal interval, the lack-of-fit check with the
abstention threshold from the pipeline, and the regular-part CRB at the true parameters."""
import os, sys, json, pickle, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from lib import (load_config, generate, dataset_seed, features, targets, half_power, ls_fit, crb, sigma_hat,
                 grid_limits, make_grid)
from sim import response, measure

CFG = load_config(os.path.join(ROOT, "configs", "default.json"))
MODEL_PATH = os.path.join(ROOT, "results", "default", "demo_model.pkl")


def get_model():
    """Gradient-boosting models trained exactly as in the pipeline (seed 0 training set), plus the
    conformal half-width and abstention threshold.  Cached on disk after the first call."""
    if os.path.exists(MODEL_PATH):
        return pickle.load(open(MODEL_PATH, "rb"))
    tr = generate(CFG["sizes"]["train"], dataset_seed(0, "train"), CFG, "train", "nominal")
    X, _ = features(tr); Y = targets(tr)
    ms = [HGR(**CFG["hgb"]).fit(X, Y[:, k]) for k in range(3)]
    ca = generate(CFG["sizes"]["calib"], dataset_seed(0, "calib"), CFG, "train", "nominal")
    P = ms[0].predict(features(ca)[0])
    res = np.sort(np.abs(P - np.log(ca["Q"]))); n = len(res)
    qhat = float(res[int(np.ceil((n + 1) * (1 - CFG["conformal_alpha"]))) - 1])
    thr = None
    rpath = os.path.join(ROOT, "results", "default", "results.json")
    if os.path.exists(rpath):
        thr = json.load(open(rpath)).get("abstain_threshold_per_seed", [None])[0]
    if thr is None:   # fall back: estimate threshold from 60 nominal fits
        ab = generate(60, dataset_seed(0, "abstain_calib"), CFG, "mismatch", "nominal")
        R = CFG["ls"]["robust"]; Qh, pk = half_power(ab); sh_ = sigma_hat(ab); T = []
        for i in range(60):
            r = ls_fit(ab["Om"][i], ab["yu"][i], ab["yd"][i], Qh[i], pk[i], CFG, starts=R["starts"], prefit_mask=True,
                       diff_step=R["diff_step"], max_nfev=R["max_nfev"])
            T.append(r["rss"] / (sh_[i] ** 2 * (r["n"] - r["npar"])))
        thr = float(np.quantile(T, CFG["abstain_quantile"]))
    obj = dict(models=ms, qhat=qhat, threshold=thr)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    pickle.dump(obj, open(MODEL_PATH, "wb"))
    return obj


def analyse(Q=3000.0, N=1.5, shift_ppm=20.0, noise=0.01, world="nominal", strength=0.5, seed=0, gain=1.0):
    """world: nominal | quintic (strength = gamma/beta) | damp (strength = eta) | feed (strength = |bg|)."""
    M = get_model()
    rng = np.random.default_rng(seed)
    rho = rng.uniform(*CFG["window"]["rho"])
    lo, hi = grid_limits(np.array([Q]), np.array([rho]), CFG["window"]["x_lo"], CFG["window"]["x_hi"], CFG["window"]["margin_ppm"])
    Om = make_grid(lo, hi, CFG["n_points"])
    beta = 4 * N / (3 * Q)
    gamma = strength * beta if world == "quintic" else 0.0
    eta = strength if world == "damp" else 0.0
    bg = strength * np.exp(1j * rng.uniform(0, 2 * np.pi)) if world == "feed" else 0j
    zu, zd = response(Om / (1 + shift_ppm * 1e-6), Q, N, gamma, eta)
    yu, yd = measure(zu, zd, gain, bg, noise, rng, CFG["adc_bits"], CFG["adc_fullscale"])
    d = dict(Om=Om, yu=yu, yd=yd, Q=np.array([Q]), N=np.array([N]), sh=np.array([shift_ppm]))
    Qh, pk = half_power(d)
    R = CFG["ls"]["robust"]
    t = time.perf_counter()
    ls = ls_fit(Om[0], yu[0], yd[0], Qh[0], pk[0], CFG, starts=R["starts"], prefit_mask=True, diff_step=R["diff_step"], max_nfev=R["max_nfev"])
    t_ls = time.perf_counter() - t
    lsbg = ls_fit(Om[0], yu[0], yd[0], Qh[0], pk[0], CFG, bg=True, starts=R["starts"], prefit_mask=True, diff_step=R["diff_step"], max_nfev=R["max_nfev"])
    t = time.perf_counter()
    X, _ = features(d); P = np.array([m.predict(X)[0] for m in M["models"]])
    t_ml = time.perf_counter() - t
    sig = float(sigma_hat(d)[0])
    lof = ls["rss"] / (sig ** 2 * (ls["n"] - ls["npar"]))
    abstain = bool(lof > M["threshold"])
    trQ, trN = CFG["ranges"]["train"]["Q"], CFG["ranges"]["train"]["N"]
    in_range = bool(trQ[0] <= ls["Q"] <= trQ[1] and trN[0] <= ls["N"] <= trN[1])
    sd, _ = crb(Om[0], Q, N, shift_ppm, gain, max(noise, 1e-6)) if world == "nominal" and noise > 0 else (np.full(3, np.nan), 0)
    Qml = float(np.exp(P[0]))
    return dict(Om=Om[0], yu=yu[0], yd=yd[0], truth=dict(Q=Q, N=N, shift=shift_ppm),
                halfpower=dict(Q=float(Qh[0]), shift=float(pk[0])),
                ls=dict(Q=ls["Q"], N=ls["N"], shift=ls["sh"], ms=t_ls * 1e3),
                ls_bg=dict(Q=lsbg["Q"], N=lsbg["N"], shift=lsbg["sh"]),
                ml=dict(Q=Qml, N=float(P[1]), shift=float(P[2]), Q_lo=Qml * np.exp(-M["qhat"]), Q_hi=Qml * np.exp(M["qhat"]), ms=t_ml * 1e3),
                lof=float(lof), threshold=float(M["threshold"]), abstain=abstain, in_train_range=in_range,
                crb_sd=dict(Q=float(sd[0]), N=float(sd[1]), shift=float(sd[2])), sigma_hat=sig)
