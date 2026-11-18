"""Full per-seed experiment pipeline.

    python run_exp.py --config configs/default.json [--seeds 0 1 ...] [--force]

For every seed: trains the ML models, runs every estimator on every test condition, computes the
regular-part CRB, detectors, conformal/abstention analysis, learning curve and an error-budget
ablation, and stores all per-sample results in results/<config>/per_sample/seed<s>.pkl plus a
per-fit optimiser log (CSV).  Aggregation into tables happens in aggregate.py.
Hyper-parameters come only from the config file and are never tuned on test data."""
import os, sys, time, json, pickle, argparse, csv
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import HistGradientBoostingRegressor as HGR, HistGradientBoostingClassifier as HGC
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from lib import (load_config, load_dataset, features, targets, half_power, ls_fit, true_p0, crb, sigma_hat,
                 dataset_seed, draw_params, simulate, software_versions)

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.json")
ap.add_argument("--seeds", type=int, nargs="*")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
CN = cfg["name"]
DATA = os.path.join("data", CN)
OUT = os.path.join("results", CN); os.makedirs(os.path.join(OUT, "per_sample"), exist_ok=True)
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
CONDS = list(cfg["test_conditions"])
T0 = time.time()
LOG = open(os.path.join(OUT, "logs", "run.log"), "a")


def log(*a):
    msg = f"[{time.time() - T0:7.0f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True); LOG.write(msg + "\n"); LOG.flush()


R = cfg["ls"]["robust"]
VARIANTS = {   # name -> keyword arguments for lib.ls_fit
    "ls":           dict(use="both"),
    "ls_robust":    dict(use="both", starts=R["starts"], prefit_mask=R["prefit_mask"], diff_step=R["diff_step"], max_nfev=R["max_nfev"]),
    "ls_oracle":    dict(use="both", oracle=True),
    "ls_up":        dict(use="up"),
    "ls_up_robust": dict(use="up", starts=R["starts"], diff_step=R["diff_step"], max_nfev=R["max_nfev"]),
    "ls_bg":        dict(use="both", bg=True, starts=R["starts"], prefit_mask=R["prefit_mask"], diff_step=R["diff_step"], max_nfev=R["max_nfev"]),
}
PLAN = {"ls": CONDS, "ls_robust": CONDS + ["abstain_calib"], "ls_oracle": CONDS,
        "ls_up": ["nominal", "N_extrap", "nom_N08"], "ls_up_robust": ["nominal", "N_extrap", "nom_N08"],
        "ls_bg": ["nom_N08", "feedthrough"]}


def _work(job):
    key, Om, yu, yd, Q0, sh0, kw, p0 = job
    kw = dict(kw); kw.pop("oracle", None)
    return key, ls_fit(Om, yu, yd, Q0, sh0, cfg, p0_list=p0, **kw)


def hgb():
    return HGR(**cfg["hgb"])


def fit3(X, Y):
    return [hgb().fit(X, Y[:, k]) for k in range(3)]


def pred3(ms, X):
    return np.column_stack([m.predict(X) for m in ms])


def run_seed(seed, pool):
    t_seed = time.time()
    D = {n: load_dataset(os.path.join(DATA, f"seed{seed}", f"{n}.npz"), cfg) for n in
         ["train", "calib", "abstain_calib"] + [f"test_{c}" for c in CONDS] + [f"det_{w}" for w in ("nominal", "feed", "quintic", "damp")]}
    T = {c: D[f"test_{c}"] for c in CONDS}
    out = dict(seed=seed, truth={}, est={}, crb={}, ls_diag={}, timing={}, lof={}, det={}, lc={}, eb={})

    # ---------------- ML training
    Xtr, names = features(D["train"]); Ytr = targets(D["train"])
    t = time.time(); M = fit3(Xtr, Ytr); out["timing"]["hgb_train_s"] = time.time() - t
    RID = make_pipeline(StandardScaler(), Ridge(**cfg["ridge"])).fit(Xtr, Ytr)
    Xtr_u, _ = features(D["train"], up_only=True); MU = fit3(Xtr_u, Ytr)
    Pc = pred3(M, features(D["calib"])[0])
    res = np.sort(np.abs(Pc[:, 0] - np.log(D["calib"]["Q"])))
    n = len(res); k = int(np.ceil((n + 1) * (1 - cfg["conformal_alpha"]))) - 1
    out["qhat"] = float(res[min(k, n - 1)])
    out["feature_names"] = names
    log(f"seed {seed}: ML trained ({out['timing']['hgb_train_s']:.1f}s), qhat={out['qhat']:.4f}")

    # ---------------- ML + half-power on every condition
    ml_times = []
    for c in CONDS + ["abstain_calib"]:
        d = T[c] if c in T else D[c]
        out["truth"][c] = dict(Q=d["Q"], N=d["N"], sh=d["sh"], G=d["G"], noise=d["noise"], sigma_hat=sigma_hat(d))
        t = time.perf_counter(); X, _ = features(d); P = pred3(M, X); ml_times.append((time.perf_counter() - t) / len(d["Q"]))
        Pr = RID.predict(X); Pu = pred3(MU, features(d, up_only=True)[0]); Qh, pk = half_power(d)
        out["est"][(c, "hgb")] = dict(Q=np.exp(P[:, 0]), N=P[:, 1], sh=P[:, 2])
        out["est"][(c, "ridge")] = dict(Q=np.exp(Pr[:, 0]), N=Pr[:, 1], sh=Pr[:, 2])
        out["est"][(c, "hgb_up")] = dict(Q=np.exp(Pu[:, 0]), N=Pu[:, 1], sh=Pu[:, 2])
        out["est"][(c, "halfpower")] = dict(Q=Qh, N=np.full(len(Qh), np.nan), sh=pk)
    out["timing"]["hgb_ms_per_estimate"] = 1e3 * float(np.median(ml_times))

    # ---------------- least squares (parallel)
    jobs = []
    for v, conds in PLAN.items():
        for c in conds:
            d = T[c] if c in T else D[c]
            Qh, pk = half_power(d)
            for i in range(len(d["Q"])):
                p0 = true_p0(d, i, bg=VARIANTS[v].get("bg", False)) if VARIANTS[v].get("oracle") else None
                jobs.append(((c, v, i), d["Om"][i], d["yu"][i], d["yd"][i], Qh[i], pk[i], VARIANTS[v], p0))
    log(f"seed {seed}: {len(jobs)} least-squares fits")
    t = time.time(); results = pool.map(_work, jobs, chunksize=8)
    log(f"seed {seed}: least squares done in {time.time() - t:.0f}s")
    by = {}
    for (c, v, i), r in results:
        by.setdefault((c, v), {})[i] = r
    with open(os.path.join(OUT, "logs", f"ls_fits_seed{seed}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "variant", "i", "Q_true", "N_true", "Q_hat", "N_hat", "sh_hat", "rss", "success", "status", "nfev", "time_s", "best_start", "Q_relerr"])
        for (c, v), rr in by.items():
            d = T[c] if c in T else D[c]
            arr = {k: np.array([rr[i][k] for i in range(len(rr))]) for k in rr[0]}
            out["est"][(c, v)] = dict(Q=arr["Q"], N=arr["N"], sh=arr["sh"])
            out["ls_diag"][(c, v)] = {k: arr[k] for k in ("rss", "n", "npar", "success", "status", "nfev", "time_s", "best_start")}
            for i in range(len(rr)):
                r = rr[i]
                w.writerow([c, v, i, f"{d['Q'][i]:.2f}", f"{d['N'][i]:.4f}", f"{r['Q']:.2f}", f"{r['N']:.4f}", f"{r['sh']:.3f}", f"{r['rss']:.6g}",
                            r["success"], r["status"], r["nfev"], f"{r['time_s']:.4f}", r["best_start"], f"{r['Q'] / d['Q'][i] - 1:.5f}"])

    # ---------------- lack-of-fit statistic
    for (c, v), dg in out["ls_diag"].items():
        if v in ("ls_robust", "ls_bg"):
            sh_ = out["truth"][c]["sigma_hat"]
            out["lof"][(c, v)] = dg["rss"] / (sh_ ** 2 * (dg["n"] - dg["npar"]))

    # ---------------- CRB (both sweeps and up-sweep only)
    for c in cfg["crb_conditions"]:
        d = T[c]; sdb, sdu, nb = [], [], []
        for i in range(len(d["Q"])):
            s, k = crb(d["Om"][i], d["Q"][i], d["N"][i], d["sh"][i], d["G"][i], d["noise"][i], "both", cfg["crb"]["jump_tol"])
            su, _ = crb(d["Om"][i], d["Q"][i], d["N"][i], d["sh"][i], d["G"][i], d["noise"][i], "up", cfg["crb"]["jump_tol"])
            sdb.append(s); sdu.append(su); nb.append(k)
        out["crb"][c] = dict(both=np.array(sdb), up=np.array(sdu), n_used=np.array(nb))
    log(f"seed {seed}: CRB done")

    # ---------------- ML mismatch detector, leave-one-world-out (+ noise control)
    Xdet = {w: features(D[f"det_{w}"])[0] for w in ("nominal", "feed", "quintic", "damp")}
    wmap = {"feedthrough": "feed", "quintic": "quintic", "damping": "damp"}
    Xn = features(T["nom_N08"])[0]
    for held in ("feedthrough", "quintic", "damping", "noise3"):
        tw = [w for w in ("feed", "quintic", "damp") if w != wmap.get(held)]
        Xd = np.vstack([Xdet["nominal"]] + [Xdet[w] for w in tw])
        yd = np.r_[np.zeros(len(Xdet["nominal"])), np.ones(sum(len(Xdet[w]) for w in tw))]
        clf = HGC(**cfg["detector_hgb"]).fit(Xd, yd)
        s0 = clf.predict_proba(Xn)[:, 1]; s1 = clf.predict_proba(features(T[held])[0])[:, 1]
        out["det"][held] = dict(ml_auc=float(roc_auc_score(np.r_[np.zeros(len(s0)), np.ones(len(s1))], np.r_[s0, s1])), trained_on=tw)
        l0 = out["lof"][("nom_N08", "ls_robust")]; l1 = out["lof"][(held, "ls_robust")]
        out["det"][held]["lof_auc"] = float(roc_auc_score(np.r_[np.zeros(len(l0)), np.ones(len(l1))], np.r_[l0, l1]))
    l0 = out["lof"][("nom_N08", "ls_bg")]; l1 = out["lof"][("feedthrough", "ls_bg")]
    out["det"]["feedthrough_bgmodel_lof_auc"] = float(roc_auc_score(np.r_[np.zeros(len(l0)), np.ones(len(l1))], np.r_[l0, l1]))

    # ---------------- learning curve
    for nlc in cfg["learning_curve"]:
        m_ = fit3(Xtr[:nlc], Ytr[:nlc]); row = {}
        for c in ("nominal", "Q_extrap", "N_extrap"):
            P = pred3(m_, features(T[c])[0])
            row[c] = dict(Q_med_pct=float(np.median(np.abs(np.exp(P[:, 0]) / T[c]["Q"] - 1)) * 100),
                          N_mae=float(np.mean(np.abs(P[:, 1] - T[c]["N"]))))
        out["lc"][nlc] = row
    log(f"seed {seed}: detectors + learning curve done")

    # ---------------- error budget: same parameters, four measurement settings
    rngp = np.random.default_rng(dataset_seed(seed, "eb_params"))
    p = draw_params(cfg["sizes"]["error_budget"], rngp, cfg, "train", "nominal")
    settings = {"noise-free, no ADC": (0.0, None), "ADC only": (0.0, "cfg"), "noise only": (None, None), "noise + ADC": (None, "cfg")}
    for name, (nz, bits) in settings.items():
        rngm = np.random.default_rng(dataset_seed(seed, "eb_meas"))
        Om, yu, yd = simulate(p, rngm, cfg, noise=nz, bits=bits)
        d = dict(p); d.update(Om=Om, yu=yu, yd=yd)
        Qh, pk = half_power(d)
        jobs = [((name, "ls_robust", i), Om[i], yu[i], yd[i], Qh[i], pk[i], VARIANTS["ls_robust"], None) for i in range(len(Qh))]
        jobs += [((name, "ls_oracle", i), Om[i], yu[i], yd[i], Qh[i], pk[i], VARIANTS["ls_oracle"], true_p0(d, i)) for i in range(len(Qh))]
        rr = dict(pool.map(_work, jobs, chunksize=4))
        P = pred3(M, features(d)[0])
        out["eb"][name] = dict(Q=p["Q"], N=p["N"], sh=p["sh"],
                               ls_robust=np.array([rr[(name, "ls_robust", i)]["Q"] for i in range(len(Qh))]),
                               ls_oracle=np.array([rr[(name, "ls_oracle", i)]["Q"] for i in range(len(Qh))]),
                               hgb=np.exp(P[:, 0]), halfpower=Qh)
    log(f"seed {seed}: error budget done; seed total {time.time() - t_seed:.0f}s")
    out["versions"] = software_versions()
    return out


if __name__ == "__main__":
    seeds = args.seeds if args.seeds else cfg["seeds"]
    with Pool(cfg["n_workers"]) as pool:
        for s in seeds:
            path = os.path.join(OUT, "per_sample", f"seed{s}.pkl")
            if os.path.exists(path) and not args.force:
                log(f"seed {s}: exists, skipping (use --force)"); continue
            o = run_seed(s, pool)
            pickle.dump(o, open(path, "wb"))
    log("ALL SEEDS DONE")
