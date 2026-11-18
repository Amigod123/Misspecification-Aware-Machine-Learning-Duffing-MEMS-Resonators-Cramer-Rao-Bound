"""All figures for the project (PNG 300 dpi + PDF).

    python figures.py                                   # illustrative + result + poster figures (default config)
    python figures.py --config configs/smoke.json       # illustrative + result figures for the smoke run
    python figures.py --only results                    # one group: illustrative | results | poster

Groups
  illustrative : example sweeps, mismatch worlds, pipeline diagram            -> figures/
  results      : accuracy, CRB efficiency, optimiser, detectors, learning curve -> figures/ (figures/<config>/ for non-default)
  poster       : simplified large-text versions for the A0 poster printed at A3 -> figures/poster/
Result and poster figures are drawn only from results/<config>/results.json.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sim import response

# ============================================================== shared style: one fixed colour per method
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dcdad5", "#ffffff"
COL = {"halfpower": "#8a8984", "ls": "#2a78d6", "hgb": "#eb6834", "ls_robust": "#1baf7a", "ls_bg": "#eda100",
       "ls_oracle": "#6250d6", "ridge": "#e87ba4", "up": "#2a78d6", "down": "#eb6834",
       "lof": "#2a78d6", "ml_det": "#eb6834", "accepted": "#1baf7a"}
NAME = {"halfpower": "Half-power bandwidth", "ls": "Least squares (3 starts)", "ls_robust": "Least squares (robust)",
        "ls_oracle": "Least squares (true start, diagnostic)", "hgb": "Gradient boosting (ML)", "ridge": "Ridge (ML)",
        "ls_bg": "Least squares + background term"}
COND = {"nominal": "Nominal", "Q_extrap": "Q extrap.", "N_extrap": "N extrap.", "nom_N08": "Nominal\n(N≥0.8)",
        "noise3": "Noise 3 %", "feedthrough": "Feed-\nthrough", "quintic": "Quintic\nstiffness", "damping": "Ampl.-dep.\ndamping"}


def setup():
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "lines.linewidth": 2, "savefig.facecolor": SURF})


def save(fig, path):
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


# ============================================================== 1. illustrative figures (no results needed)
def illustrative():
    plt.rcdefaults(); setup()
    FIGDIR = "figures"; os.makedirs(FIGDIR, exist_ok=True)
    # ---- 1. sweeps for N = 0, 1, 3 at Q = 2000
    Q = 2000; hw = 1 / (2 * Q)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, N in zip(axs, (0.0, 1.0, 3.0)):
        x = np.linspace(-4, 6 + 1.3 * N, 6000)
        zu, zd = response(1 + x * hw, Q, N)
        ax.plot(x, np.abs(zu), color=COL["up"], label="Up-sweep")
        ax.plot(x, np.abs(zd), color=COL["down"], ls="--", label="Down-sweep")
        tag = "linear" if N == 0 else ("no hysteresis" if N < 1.54 else "hysteresis")
        ax.set_title(f"N = {N:g} ({tag})", fontsize=9, color=INK)
        ax.set_xlabel("Detuning (half-widths)")
    axs[0].set_ylabel("Amplitude (normalised)"); axs[0].legend(loc="upper left", fontsize=8)
    save(fig, f"{FIGDIR}/fig_sweeps")

    # ---- 2. mismatch worlds vs nominal, same Q and N
    Q = 2000; N = 2.0; hw = 1 / (2 * Q); beta = 4 * N / (3 * Q)
    x = np.linspace(-6, 10, 4000); W = 1 + x * hw
    zu0, zd0 = response(W, Q, N)
    worlds = [("Quintic stiffness (γ = 0.6β)", response(W, Q, N, gamma=0.6 * beta), None),
              ("Ampl.-dep. damping (η = 0.6)", response(W, Q, N, eta=0.6), None),
              ("Feedthrough (15 %, phase 1 rad)", (zu0, zd0), 0.15 * np.exp(1j))]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, (title, (zu, zd), bg) in zip(axs, worlds):
        ax.plot(x, np.abs(zu0), color="#8a8984", lw=1.5, label="Nominal model")
        yu = np.abs(zu + bg) if bg is not None else np.abs(zu)
        ax.plot(x, yu, color=COL["ls"], label="Mismatch world (up)")
        ax.set_title(title, fontsize=8.5, color=INK); ax.set_xlabel("Detuning (half-widths)")
    axs[0].set_ylabel("Amplitude (normalised)")
    fig.subplots_adjust(bottom=0.3, wspace=0.12)
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    save(fig, f"{FIGDIR}/fig_worlds")

    # ---- 3. pipeline diagram
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=INK2, lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.25)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9, shrinkA=0, shrinkB=0))
    top = 46; h = 11
    box(1, top, 29, h, "1. Duffing simulator\nharmonic balance, validated\nagainst a time-domain ODE", "#e6f0fb")
    box(35.5, top, 29, h, "2. Measurement model\nnoise, unknown gain, 12-bit ADC\n+ test-only mismatch worlds", "#e6f0fb")
    box(70, top, 29, h, "3. Leakage-free datasets\nheld-out Q / N blocks,\n5 seeds, hashed manifests", "#e6f0fb")
    arrow(30.6, top + h / 2, 35, top + h / 2); arrow(65.1, top + h / 2, 69.5, top + h / 2)
    mid = 24; w = 22.6
    labels = [("4a. Half-power\nbandwidth", "#f1f0ec"), ("4b. Exact-model\nleast squares (3-start,\nrobust, +background)", "#e3f5ee"),
              ("4c. ML: 47 features\n→ gradient boosting,\nridge", "#fdebe3"), ("4d. Cramér–Rao\nbound (regular part)", "#ece9fb")]
    xs = [1 + i * (w + 2.5) for i in range(4)]
    for x, (t, fc) in zip(xs, labels):
        box(x, mid, w, 13, t, fc)
    ax.plot([84.5, 84.5], [top - 0.6, 41], color=INK2, lw=0.9); ax.plot([xs[0] + w / 2, xs[3] + w / 2], [41, 41], color=INK2, lw=0.9)
    for x in xs:
        arrow(x + w / 2, 41, x + w / 2, mid + 13.5)
    box(1, 3, 47, 13, "5. Model check and abstention\nlack-of-fit test vs ML detector (leave-one-world-out);\nsplit-conformal 90 % intervals", "#f1f0ec")
    box(52, 3, 47, 13, "6. Evaluation (5 seeds, bootstrap CIs)\nerror and failure rate, error / CRB efficiency,\nextrapolation, mismatch, runtime", "#f1f0ec")
    ax.plot([xs[1] + w / 2, xs[1] + w / 2], [mid - 0.6, 19], color=INK2, lw=0.9)
    arrow(xs[1] + w / 2, 19, xs[1] + w / 2, 16.5)
    ax.plot([xs[2] + w / 2, xs[2] + w / 2], [mid - 0.6, 19], color=INK2, lw=0.9); ax.plot([xs[3] + w / 2, xs[3] + w / 2], [mid - 0.6, 19], color=INK2, lw=0.9)
    ax.plot([xs[2] + w / 2, xs[3] + w / 2], [19, 19], color=INK2, lw=0.9); arrow(75.5, 19, 75.5, 16.5)
    arrow(48.6, 9.5, 51.5, 9.5)
    save(fig, f"{FIGDIR}/fig_pipeline")
    print("illustrative figures written to", FIGDIR)


# ============================================================== 2. result figures for the report
def results(config_path):
    plt.rcdefaults(); setup()
    cfg = json.load(open(config_path))
    R = json.load(open(os.path.join("results", cfg["name"], "results.json")))
    FIG = "figures" if cfg["name"] == "default" else os.path.join("figures", cfg["name"])
    os.makedirs(FIG, exist_ok=True)
    A = R["accuracy"]
    nseeds = len(R["seeds"])
    note = f"{nseeds} seeds × {R['n_per_condition_per_seed']} samples per condition; bars = pooled value, whiskers = 95 % bootstrap / Wilson CI"


    def bars(ax, conds, methods, key, lo, hi, scale=1.0, log=False):
        w = 0.8 / len(methods)
        for j, m in enumerate(methods):
            xs, v, e0, e1 = [], [], [], []
            for i, c in enumerate(conds):
                row = A.get(f"{c}|{m}")
                if row is None or row[key] is None:
                    continue
                xs.append(i + (j - (len(methods) - 1) / 2) * w)
                v.append(row[key] * scale); e0.append(row[lo] * scale); e1.append(row[hi] * scale)
            v, e0, e1 = map(np.array, (v, e0, e1))
            ax.bar(xs, v, w, color=COL[m], edgecolor="white", linewidth=1, label=NAME[m])
            ax.errorbar(xs, v, yerr=[np.clip(v - e0, 0, None), np.clip(e1 - v, 0, None)], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([COND[c] for c in conds], fontsize=8)
        if log:
            ax.set_yscale("log")


    # ---------------- accuracy + failure rate
    conds = ["nominal", "Q_extrap", "N_extrap", "noise3", "feedthrough", "quintic", "damping"]
    meths = ["halfpower", "ls", "ls_robust", "hgb", "ls_bg"]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    bars(axs[0], conds, meths, "Q_med_pct", "Q_med_lo", "Q_med_hi", log=True)
    axs[0].set_ylabel("Median |Q error| (%)"); axs[0].set_ylim(0.08, 150)
    axs[0].legend(ncol=3, fontsize=7.2, loc="upper left")
    bars(axs[1], conds, meths, "fail_rate", "fail_lo", "fail_hi", scale=100)
    axs[1].set_ylabel("Samples with |Q error| > 10 % (%)"); axs[1].set_ylim(0, 105)
    axs[1].text(0, -0.32, note, transform=axs[1].transAxes, fontsize=7, color=INK2)
    save(fig, f"{FIG}/fig_accuracy")

    # ---------------- CRB efficiency
    C = R["crb"]
    cc = ["nominal", "nom_N08", "noise3", "Q_extrap", "N_extrap"]
    mm = ["ls", "ls_robust", "hgb"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, (p, lab) in zip(axs, [("Q", "Q"), ("N", "N"), ("shift", "Shift")]):
        w = 0.8 / len(mm)
        for j, m in enumerate(mm):
            xs, v, lo, hi = [], [], [], []
            for i, c in enumerate(cc):
                row = C.get(f"{c}|{m}")
                if row is None:
                    continue
                xs.append(i + (j - 1) * w); v.append(row[f"{p}_robust"]); lo.append(row[f"{p}_robust_lo"]); hi.append(row[f"{p}_robust_hi"])
            v, lo, hi = map(np.array, (v, lo, hi))
            ax.bar(xs, v, w, color=COL[m], edgecolor="white", linewidth=1, label=NAME[m])
            ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
        ax.axhline(1, color=INK, lw=0.9, ls="--"); ax.set_yscale("log"); ax.set_title(lab, fontsize=9)
        ax.set_xticks(range(len(cc))); ax.set_xticklabels(["Nom.", "Nom.\nN≥0.8", "Noise\n3 %", "Q\nextr.", "N\nextr."], fontsize=7)
    axs[0].set_ylabel("Robust error / CRB std\n(1 = efficient)")
    h_, l_ = axs[0].get_legend_handles_labels(); fig.legend(h_, l_, ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    save(fig, f"{FIG}/fig_crb")

    # ---------------- optimiser vs model: failure rates of three LS variants
    O = R["optimizer"]
    cc = ["nominal", "Q_extrap", "N_extrap", "noise3", "feedthrough", "quintic", "damping"]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    vv = ["ls", "ls_robust", "ls_oracle"]
    w = 0.8 / 3
    for j, v in enumerate(vv):
        ax.bar([i + (j - 1) * w for i in range(len(cc))], [O[f"{c}|{v}"]["fail_rate"] * 100 for c in cc], w, color=COL[v], edgecolor="white", linewidth=1, label=NAME[v])
    ax.set_xticks(range(len(cc))); ax.set_xticklabels([COND[c] for c in cc], fontsize=8)
    ax.set_ylabel("|Q error| > 10 % (%)"); ax.legend(fontsize=7.2, ncol=3, loc="upper left"); ax.set_ylim(0, 112)
    save(fig, f"{FIG}/fig_optimizer")

    # ---------------- detectors + conformal coverage
    D = R["detectors"]; AB = R["abstention"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ws = ["feedthrough", "quintic", "damping", "noise3"]
    for j, (k, col, lab) in enumerate([("lof", COL["lof"], "Lack-of-fit test (classical)"), ("ml", COL["ml_det"], "ML classifier, world held out")]):
        m_ = np.array([D[w][f"{k}_auc_mean"] for w in ws]); lo = np.array([D[w][f"{k}_auc_min"] for w in ws]); hi = np.array([D[w][f"{k}_auc_max"] for w in ws])
        xs = np.arange(4) + (j - 0.5) * 0.38
        axs[0].bar(xs, m_, 0.38, color=col, edgecolor="white", linewidth=1, label=lab)
        axs[0].errorbar(xs, m_, yerr=[m_ - lo, hi - m_], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=1.5)
    axs[0].axhline(0.5, color=INK, lw=0.9, ls="--"); axs[0].set_ylim(0, 1.3)
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels([COND[w] for w in ws[:3]] + ["Noise 3 %\n(control)"], fontsize=7.5)
    axs[0].set_ylabel("AUROC vs nominal"); axs[0].legend(fontsize=7, loc="upper left"); axs[0].set_title("Detecting a wrong model", fontsize=9)
    cs = ["nom_N08", "Q_extrap", "N_extrap", "quintic", "feedthrough", "damping"]
    for j, (k, col, lab) in enumerate([("coverage_all", "#8a8984", "All samples"), ("coverage_accepted", COL["accepted"], "After abstaining (lack-of-fit)")]):
        v = np.array([AB[c][k] if AB[c][k] is not None else np.nan for c in cs]) * 100
        xs = np.arange(len(cs)) + (j - 0.5) * 0.38
        axs[1].bar(xs, v, 0.38, color=col, edgecolor="white", linewidth=1, label=lab)
        for x_, v_ in zip(xs, v):
            if np.isfinite(v_):
                axs[1].text(x_, v_ + 2, f"{v_:.0f}", ha="center", va="bottom", fontsize=6, color=INK2)
    axs[1].axhline(90, color=INK, lw=0.9, ls="--"); axs[1].set_ylim(0, 130)
    axs[1].set_xticks(range(len(cs))); axs[1].set_xticklabels(["Nominal\n(N≥0.8)", "Q\nextrap.", "N\nextrap.", "Quintic", "Feed-\nthrough", "Ampl.\ndamping"], fontsize=7)
    axs[1].set_ylabel("Coverage of 90 % interval (%)"); axs[1].legend(fontsize=7, loc="upper right"); axs[1].set_title("Conformal intervals for ML estimate of Q", fontsize=9)
    save(fig, f"{FIG}/fig_detect")

    # ---------------- learning curve
    L = R["learning_curve"]; ns = sorted(int(k) for k in L)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for c, col, lab in [("nominal", COL["ls"], "In distribution"), ("N_extrap", COL["ls_robust"], "N extrapolation"), ("Q_extrap", COL["hgb"], "Q extrapolation")]:
        m_ = np.array([L[str(n)][f"{c}_Q_med_pct_mean"] for n in ns]); s_ = np.array([L[str(n)][f"{c}_Q_med_pct_sd"] for n in ns])
        ax.plot(ns, m_, "o-", color=col, label=lab, ms=4); ax.fill_between(ns, np.clip(m_ - s_, 1e-3, None), m_ + s_, color=col, alpha=0.15, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Training samples"); ax.set_ylabel("Median |Q error| (%)")
    ax.legend(fontsize=7.5); ax.set_title("Gradient boosting: mean ± 1 s.d. over seeds", fontsize=8.5)
    save(fig, f"{FIG}/fig_learning")
    print("result figures written to", FIG)


# ============================================================== 3. poster figures (large text for A3 printing)
def poster(config_path):
    plt.rcdefaults(); setup()
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11})
    cfg = json.load(open(config_path))
    OUT = "figures/poster"; os.makedirs(OUT, exist_ok=True)
    R = json.load(open(os.path.join("results", cfg["name"], "results.json"))); A = R["accuracy"]
    SHORT = {"nominal": "Nom.", "Q_extrap": "Q ext.", "N_extrap": "N ext.", "noise3": "Noise", "feedthrough": "Feed.",
             "quintic": "Quint.", "damping": "Damp.", "nom_N08": "Nom.*"}
    SN = {"halfpower": "Half-power", "ls": "LS 3-start", "ls_robust": "LS robust", "hgb": "Grad. boosting", "ls_bg": "LS + background"}
    # ---- 1. accuracy (median |Q error|)
    conds = ["nominal", "Q_extrap", "N_extrap", "noise3", "feedthrough", "quintic", "damping"]
    meths = ["halfpower", "ls_robust", "hgb", "ls_bg"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9)); w = 0.8 / len(meths)
    for j, m in enumerate(meths):
        xs, v, lo, hi = [], [], [], []
        for i, c in enumerate(conds):
            r = A.get(f"{c}|{m}")
            if r is None:
                continue
            xs.append(i + (j - 1.5) * w); v.append(r["Q_med_pct"]); lo.append(r["Q_med_lo"]); hi.append(r["Q_med_hi"])
        v, lo, hi = map(np.array, (v, lo, hi))
        ax.bar(xs, v, w, color=COL[m], edgecolor="white", linewidth=1, label=SN[m])
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_yscale("log"); ax.set_ylim(0.1, 400); ax.set_xticks(range(len(conds))); ax.set_xticklabels([SHORT[c] for c in conds])
    ax.set_ylabel("Median |Q error| (%)"); ax.legend(ncol=2, loc="upper left", frameon=False)
    save(fig, f"{OUT}/fig_accuracy")

    # ---- 2. CRB efficiency for Q
    C = R["crb"]; cc = ["nominal", "nom_N08", "noise3", "Q_extrap", "N_extrap"]; mm = ["ls", "ls_robust", "hgb"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); w = 0.8 / 3
    for j, m in enumerate(mm):
        v = np.array([C[f"{c}|{m}"]["Q_robust"] for c in cc]); lo = np.array([C[f"{c}|{m}"]["Q_robust_lo"] for c in cc]); hi = np.array([C[f"{c}|{m}"]["Q_robust_hi"] for c in cc])
        xs = np.arange(len(cc)) + (j - 1) * w
        ax.bar(xs, v, w, color=COL[m], edgecolor="white", linewidth=1, label=SN[m])
        ax.errorbar(xs, v, yerr=[v - lo, hi - v], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
    ax.axhline(1, color=INK, lw=1, ls="--"); ax.set_yscale("log"); ax.set_ylim(0.5, 400)
    ax.set_xticks(range(len(cc))); ax.set_xticklabels([SHORT[c] for c in cc]); ax.set_ylabel("Q error / CRB std")
    ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=10.5)
    save(fig, f"{OUT}/fig_crb")

    # ---- 3. detectors + coverage
    D = R["detectors"]; AB = R["abstention"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[1, 1.35]))
    ws = ["feedthrough", "quintic", "damping", "noise3"]
    for j, (k, col, lab) in enumerate([("lof", COL["lof"], "Lack-of-fit"), ("ml", COL["ml_det"], "ML detector")]):
        m_ = np.array([D[w][f"{k}_auc_mean"] for w in ws])
        axs[0].bar(np.arange(4) + (j - 0.5) * 0.38, m_, 0.38, color=col, edgecolor="white", linewidth=1, label=lab)
    axs[0].axhline(0.5, color=INK, lw=1, ls="--"); axs[0].set_ylim(0, 1.45); axs[0].set_yticks([0, 0.5, 1])
    axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["Feed.", "Quint.", "Damp.", "Noise"]); axs[0].set_ylabel("AUROC")
    axs[0].legend(loc="upper left", frameon=False, fontsize=10.5, ncol=1)
    cs = ["nom_N08", "Q_extrap", "quintic", "feedthrough", "damping"]
    for j, (k, col, lab) in enumerate([("coverage_all", "#8a8984", "All"), ("coverage_accepted", COL["accepted"], "After abstaining")]):
        v = np.array([AB[c][k] if AB[c][k] is not None else np.nan for c in cs]) * 100
        axs[1].bar(np.arange(len(cs)) + (j - 0.5) * 0.38, v, 0.38, color=col, edgecolor="white", linewidth=1, label=lab)
    axs[1].axhline(90, color=INK, lw=1, ls="--"); axs[1].set_ylim(0, 145); axs[1].set_yticks([0, 50, 90])
    axs[1].set_xticks(range(len(cs))); axs[1].set_xticklabels(["Nom.", "Q ext.", "Quint.", "Feed.", "Damp."]); axs[1].set_ylabel("Coverage (%)")
    axs[1].legend(loc="upper right", frameon=False, fontsize=10.5, ncol=1)
    fig.tight_layout(); save(fig, f"{OUT}/fig_detect")

    # ---- 4. pipeline (short labels)
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    def box(x, y, w, h, t, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0", fc=fc, ec=INK2, lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=11, color=INK, linespacing=1.2)
    def arr(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=0, shrinkB=0))
    bw, bh = 29, 12
    for i, (t, fc) in enumerate([("Duffing\nsimulator", "#e6f0fb"), ("Measurement\n+ mismatch", "#e6f0fb"), ("Datasets\n5 seeds", "#e6f0fb")]):
        box(1 + i * 35, 32, bw, bh, t, fc)
        if i:
            arr(1 + (i - 1) * 35 + bw + 0.8, 38, 1 + i * 35 - 0.8, 38)
    for i, (t, fc) in enumerate([("Half-power", "#f1f0ec"), ("Least\nsquares", "#e3f5ee"), ("Gradient\nboosting", "#fdebe3"), ("Cramér–Rao\nbound", "#ece9fb")]):
        x = 1 + i * 25.3; box(x, 15, 22, 11, t, fc); arr(x + 11, 31, x + 11, 26.6)
    ax.plot([86.5, 86.5], [31.4, 31], color=INK2, lw=1.2); ax.plot([12, 87], [31, 31], color=INK2, lw=1.2)
    box(1, 0.5, 47, 9.5, "Lack-of-fit check → abstain", "#f1f0ec"); box(52, 0.5, 47, 9.5, "Error, failure rate, efficiency", "#f1f0ec")
    arr(37.3, 14.4, 37.3, 10.6); arr(75, 14.4, 75, 10.6)
    save(fig, f"{OUT}/fig_pipeline")
    print("poster figures written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--only", choices=["illustrative", "results", "poster"])
    a = ap.parse_args()
    name = json.load(open(a.config))["name"]
    if a.only in (None, "illustrative"):
        illustrative()
    if a.only in (None, "results"):
        results(a.config)
    if a.only == "poster" or (a.only is None and name == "default"):
        poster(a.config)
