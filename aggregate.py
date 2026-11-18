"""Aggregate per-seed results into CSV tables and results.json with uncertainty.

    python aggregate.py --config configs/default.json

Pooled estimates use all samples of all seeds; 95 % intervals are bootstrap (medians, means,
robust spreads) or Wilson (proportions).  Per-seed ranges are reported alongside so that seed-to-seed
variation is visible.  Every number used in the report and poster comes from these files."""
import os, json, glob, pickle, argparse, csv, time
import numpy as np
from lib import load_config, boot_ci, prop_ci

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/default.json")
args = ap.parse_args()
cfg = load_config(args.config)
OUT = os.path.join("results", cfg["name"]); TAB = os.path.join(OUT, "tables"); os.makedirs(TAB, exist_ok=True)
B = cfg["bootstrap"]
runs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(OUT, "per_sample", "seed*.pkl")))]
seeds = [r["seed"] for r in runs]
CONDS = list(cfg["test_conditions"])
FT = cfg["fail_threshold"]
print("seeds:", seeds)


def bci(x, stat=np.median):
    return boot_ci(x, stat, B["n"], B["seed"])


def errs(c, m):
    """Concatenate per-sample errors over seeds for condition c and method m (None if absent)."""
    eq, en, es, per = [], [], [], []
    for r in runs:
        if (c, m) not in r["est"]:
            return None
        t, e = r["truth"][c], r["est"][(c, m)]
        q = e["Q"] / t["Q"] - 1
        eq.append(q); en.append(e["N"] - t["N"]); es.append(e["sh"] - t["sh"])
        per.append(float(np.nanmedian(np.abs(q))) * 100)
    return np.concatenate(eq), np.concatenate(en), np.concatenate(es), per


def write(name, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(TAB, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)


RES = dict(config=cfg, seeds=seeds, generated=time.strftime("%Y-%m-%d %H:%M:%S"), versions=runs[0]["versions"],
           n_per_condition_per_seed=cfg["sizes"]["test"])

# ---------------------------------------------------------------- 1. accuracy
METHODS = ["halfpower", "ls", "ls_robust", "ls_oracle", "hgb", "ridge", "ls_up", "ls_up_robust", "hgb_up", "ls_bg"]
acc = []; RES["accuracy"] = {}
for c in CONDS:
    for m in METHODS:
        e = errs(c, m)
        if e is None:
            continue
        eq, en, es, per = e
        aq = np.abs(eq)
        q = bci(aq * 100); n_ = bci(np.abs(en), np.mean) if m != "halfpower" else (np.nan,) * 3
        s = bci(np.abs(es)); f = prop_ci(int(np.sum(~(aq <= FT))), len(aq))   # NaN (failed fit) counts as failure
        row = dict(condition=c, method=m, n=len(aq), Q_med_pct=r3(q[0]), Q_med_lo=r3(q[1]), Q_med_hi=r3(q[2]),
                   Q_p90_pct=r3(np.nanpercentile(aq, 90) * 100), fail_rate=r3(f[0]), fail_lo=r3(f[1]), fail_hi=r3(f[2]),
                   N_mae=r3(n_[0]), N_mae_lo=r3(n_[1]), N_mae_hi=r3(n_[2]),
                   sh_med_ppm=r3(s[0]), sh_med_lo=r3(s[1]), sh_med_hi=r3(s[2]),
                   Q_med_seed_min=r3(min(per)), Q_med_seed_max=r3(max(per)), n_nan=int(np.sum(~np.isfinite(eq))))
        acc.append(row); RES["accuracy"][f"{c}|{m}"] = row
write("accuracy.csv", acc)

# ---------------------------------------------------------------- 2. CRB efficiency
eff = []; RES["crb"] = {}
for c in cfg["crb_conditions"]:
    sd = np.concatenate([r["crb"][c]["both"] for r in runs])
    Q = np.concatenate([r["truth"][c]["Q"] for r in runs])
    RES["crb"][f"{c}|bound"] = dict(Q_pct_median=r3(np.median(sd[:, 0] / Q) * 100), N_median=r3(np.median(sd[:, 1])),
                                    sh_ppm_median=r3(np.median(sd[:, 2])),
                                    points_used_median=int(np.median(np.concatenate([r["crb"][c]["n_used"] for r in runs]))))
    for m in ("ls", "ls_robust", "ls_oracle", "hgb"):
        e = errs(c, m)
        if e is None:
            continue
        eq, en, es, _ = e
        row = dict(condition=c, method=m)
        for name, err, k in (("Q", eq * Q, 0), ("N", en, 1), ("shift", es, 2)):
            z = err / sd[:, k]
            a = bci(np.abs(z))
            row[f"{name}_robust"] = r3(a[0] / 0.6745); row[f"{name}_robust_lo"] = r3(a[1] / 0.6745); row[f"{name}_robust_hi"] = r3(a[2] / 0.6745)
            row[f"{name}_rms"] = r3(np.sqrt(np.nanmean(z ** 2)))
        eff.append(row); RES["crb"][f"{c}|{m}"] = row
write("crb_efficiency.csv", eff)

# ---------------------------------------------------------------- 3. up-sweep only vs up + down (RQ4)
ud = []; RES["updown"] = {}
for c in ("nominal", "nom_N08", "N_extrap"):
    sb = np.concatenate([r["crb"][c]["both"] for r in runs]); su = np.concatenate([r["crb"][c]["up"] for r in runs])
    ratio = su / sb
    row = dict(condition=c, crb_ratio_Q=r3(np.nanmedian(ratio[:, 0])), crb_ratio_N=r3(np.nanmedian(ratio[:, 1])), crb_ratio_shift=r3(np.nanmedian(ratio[:, 2])))
    for m in ("ls", "ls_up", "ls_robust", "ls_up_robust", "hgb", "hgb_up"):
        e = errs(c, m)
        if e is None:
            continue
        row[f"{m}_Q_med_pct"] = r3(np.nanmedian(np.abs(e[0])) * 100); row[f"{m}_N_mae"] = r3(np.nanmean(np.abs(e[1])))
        row[f"{m}_fail"] = r3(np.mean(~(np.abs(e[0]) <= FT)))
    ud.append(row); RES["updown"][c] = row
write("up_vs_updown.csv", ud)

# ---------------------------------------------------------------- 4. optimiser diagnostics
opt = []; RES["optimizer"] = {}
tol = cfg["ls"]["optimizer_failure_rel_cost"]
for c in CONDS:
    orc = np.concatenate([r["ls_diag"][(c, "ls_oracle")]["rss"] for r in runs])
    for v in ("ls", "ls_robust", "ls_oracle"):
        dg = {k: np.concatenate([r["ls_diag"][(c, v)][k] for r in runs]) for k in ("rss", "success", "nfev", "time_s", "status")}
        worse = dg["rss"] > orc * (1 + tol)
        e = errs(c, v)
        aq = np.abs(e[0])
        row = dict(condition=c, variant=v, n=len(aq), fail_rate=r3(np.mean(~(aq <= FT))),
                   rss_worse_than_oracle_start=r3(np.mean(worse)) if v != "ls_oracle" else 0.0,
                   fail_and_worse_than_oracle=r3(np.mean(worse & ~(aq <= FT))) if v != "ls_oracle" else 0.0,
                   scipy_success_rate=r3(np.mean(dg["success"])), nfev_median=int(np.median(dg["nfev"])),
                   time_ms_median=r3(np.median(dg["time_s"]) * 1e3))
        opt.append(row); RES["optimizer"][f"{c}|{v}"] = row
write("optimizer_diagnostics.csv", opt)

# ---------------------------------------------------------------- 5. error budget
eb = []; RES["error_budget"] = {}
for setting in runs[0]["eb"]:
    for m in ("halfpower", "ls_robust", "ls_oracle", "hgb"):
        aq = np.concatenate([np.abs(r["eb"][setting][m] / r["eb"][setting]["Q"] - 1) for r in runs])
        q = bci(aq * 100)
        row = dict(setting=setting, method=m, n=len(aq), Q_med_pct=r3(q[0]), Q_med_lo=r3(q[1]), Q_med_hi=r3(q[2]),
                   fail_rate=r3(np.mean(~(aq <= FT))))
        eb.append(row); RES["error_budget"][f"{setting}|{m}"] = row
write("error_budget.csv", eb)

# ---------------------------------------------------------------- 6. detectors
det = []; RES["detectors"] = {}
for w in ("feedthrough", "quintic", "damping", "noise3"):
    lof = np.array([r["det"][w]["lof_auc"] for r in runs]); ml = np.array([r["det"][w]["ml_auc"] for r in runs])
    row = dict(world=w, lof_auc_mean=r3(lof.mean()), lof_auc_sd=r3(lof.std(ddof=1) if len(lof) > 1 else 0), lof_auc_min=r3(lof.min()), lof_auc_max=r3(lof.max()),
               ml_auc_mean=r3(ml.mean()), ml_auc_sd=r3(ml.std(ddof=1) if len(ml) > 1 else 0), ml_auc_min=r3(ml.min()), ml_auc_max=r3(ml.max()),
               ml_trained_on="+".join(runs[0]["det"][w]["trained_on"]))
    det.append(row); RES["detectors"][w] = row
bgv = np.array([r["det"]["feedthrough_bgmodel_lof_auc"] for r in runs])
RES["detectors"]["feedthrough_with_bg_model"] = dict(lof_auc_mean=r3(bgv.mean()), lof_auc_min=r3(bgv.min()), lof_auc_max=r3(bgv.max()))
det.append(dict(world="feedthrough (fit includes background term)", lof_auc_mean=r3(bgv.mean()), lof_auc_min=r3(bgv.min()), lof_auc_max=r3(bgv.max())))
write("detectors.csv", det)

# ---------------------------------------------------------------- 7. conformal + abstention
ab = []; RES["abstention"] = {}; RES["qhat_per_seed"] = [r3(r["qhat"]) for r in runs]
RES["abstain_threshold_per_seed"] = [r3(np.quantile(r["lof"][("abstain_calib", "ls_robust")], cfg["abstain_quantile"])) for r in runs]
trQ, trN = cfg["ranges"]["train"]["Q"], cfg["ranges"]["train"]["N"]
for c in CONDS:
    acc_, cov_all, cov_acc, ls_e, hg_e, acc_mask, sil_ls, sil_hg, sil_hg_guard, acc_guard = ([] for _ in range(10))
    for r in runs:
        thr = np.quantile(r["lof"][("abstain_calib", "ls_robust")], cfg["abstain_quantile"])
        T = r["lof"][(c, "ls_robust")]; a = T <= thr
        t = r["truth"][c]; h = r["est"][(c, "hgb")]; L = r["est"][(c, "ls_robust")]
        cov = np.abs(np.log(h["Q"]) - np.log(t["Q"])) <= r["qhat"]
        eh = np.abs(h["Q"] / t["Q"] - 1); el = np.abs(L["Q"] / t["Q"] - 1)
        guard = a & (L["Q"] >= trQ[0]) & (L["Q"] <= trQ[1]) & (L["N"] >= trN[0]) & (L["N"] <= trN[1])
        acc_.append(a); cov_all.append(cov); ls_e.append(el); hg_e.append(eh); acc_guard.append(guard)
    a = np.concatenate(acc_); cov = np.concatenate(cov_all); el = np.concatenate(ls_e); eh = np.concatenate(hg_e); g = np.concatenate(acc_guard)
    n = len(a)
    pa = prop_ci(int(a.sum()), n); ca = prop_ci(int(cov.sum()), n); cc = prop_ci(int(cov[a].sum()), int(a.sum()))
    row = dict(condition=c, n=n, accept_rate=r3(pa[0]), accept_lo=r3(pa[1]), accept_hi=r3(pa[2]),
               coverage_all=r3(ca[0]), coverage_all_lo=r3(ca[1]), coverage_all_hi=r3(ca[2]),
               coverage_accepted=r3(cc[0]), coverage_accepted_lo=r3(cc[1]), coverage_accepted_hi=r3(cc[2]),
               ls_robust_Q_med_all=r3(np.nanmedian(el) * 100), ls_robust_Q_med_accepted=r3(np.nanmedian(el[a]) * 100) if a.any() else None,
               hgb_Q_med_all=r3(np.nanmedian(eh) * 100), hgb_Q_med_accepted=r3(np.nanmedian(eh[a]) * 100) if a.any() else None,
               silent_fail_ls_robust=r3(np.mean(a & ~(el <= FT))), silent_fail_hgb=r3(np.mean(a & ~(eh <= FT))),
               hgb_accept_with_range_guard=r3(g.mean()), silent_fail_hgb_with_range_guard=r3(np.mean(g & ~(eh <= FT))),
               coverage_with_range_guard=r3(cov[g].mean()) if g.any() else None)
    ab.append(row); RES["abstention"][c] = row
write("abstention_conformal.csv", ab)

# ---------------------------------------------------------------- 8. learning curve
lc = []; RES["learning_curve"] = {}
for nlc in cfg["learning_curve"]:
    row = dict(n_train=nlc)
    for c in ("nominal", "Q_extrap", "N_extrap"):
        v = np.array([r["lc"][nlc][c]["Q_med_pct"] for r in runs]); w = np.array([r["lc"][nlc][c]["N_mae"] for r in runs])
        row[f"{c}_Q_med_pct_mean"] = r3(v.mean()); row[f"{c}_Q_med_pct_sd"] = r3(v.std(ddof=1) if len(v) > 1 else 0)
        row[f"{c}_N_mae_mean"] = r3(w.mean())
    lc.append(row); RES["learning_curve"][nlc] = row
write("learning_curve.csv", lc)

# ---------------------------------------------------------------- 9. runtime
rt = []; RES["runtime"] = {}
hg = float(np.median([r["timing"]["hgb_ms_per_estimate"] for r in runs]))
for v in ("ls", "ls_robust", "ls_oracle", "ls_up", "ls_bg"):
    ts = np.concatenate([r["ls_diag"][k]["time_s"] for r in runs for k in r["ls_diag"] if k[1] == v and k[0] != "abstain_calib"])
    rt.append(dict(method=v, ms_per_estimate_median=r3(np.median(ts) * 1e3), ms_p90=r3(np.percentile(ts, 90) * 1e3), note="one CPU core per fit"))
rt.append(dict(method="hgb (features + 3 models)", ms_per_estimate_median=r3(hg), note="batch of test samples, per-sample average"))
rt.append(dict(method="hgb training (3 targets, 6000 samples)", ms_per_estimate_median=r3(np.median([r["timing"]["hgb_train_s"] for r in runs]) * 1e3)))
for row in rt:
    RES["runtime"][row["method"]] = row
write("runtime.csv", rt)

json.dump(RES, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("tables:", sorted(os.listdir(TAB)))
