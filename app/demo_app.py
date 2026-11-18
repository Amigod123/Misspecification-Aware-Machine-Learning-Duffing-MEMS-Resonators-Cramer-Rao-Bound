"""Interactive demo.   streamlit run app/demo_app.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from demo_core import analyse, get_model

st.set_page_config(page_title="Duffing MEMS estimator demo", layout="wide")
st.title("Duffing MEMS resonator: classical fit vs ML vs Cramér–Rao bound")
st.caption("Synthetic first-harmonic model, hardening nonlinearity, single drive level. "
           "Mismatch worlds are deliberate toy models, not measured hardware behaviour.")

with st.sidebar:
    st.header("Simulated device")
    Q = st.slider("Quality factor Q", 500, 20000, 3000, step=100)
    N = st.slider("Nonlinearity N (peak shift in half-widths)", 0.0, 4.0, 1.5, step=0.05)
    sh = st.slider("Resonance shift (ppm)", -100.0, 100.0, 20.0, step=1.0)
    noise = st.slider("Additive noise (fraction of peak)", 0.0, 0.05, 0.01, step=0.001, format="%.3f")
    world = st.selectbox("World", ["nominal", "quintic", "damp", "feed"],
                         format_func={"nominal": "Nominal (model is correct)", "quintic": "Quintic stiffness",
                                      "damp": "Amplitude-dependent damping", "feed": "Electrical feedthrough"}.get)
    strength = st.slider("Mismatch strength (γ/β, η or |background|)", 0.0, 1.0, 0.5, step=0.05, disabled=(world == "nominal"))
    seed = st.number_input("Random seed", 0, 10_000, 0)
    with st.spinner("Loading model (first run trains it, about 10 s)…"):
        get_model()

r = analyse(Q, N, sh, noise, world, strength, int(seed))
c1, c2 = st.columns([3, 2])
with c1:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = (r["Om"] - 1) * 1e6
    ax.plot(x, r["yu"], color="#2a78d6", lw=1.6, label="Up-sweep")
    ax.plot(x, r["yd"], color="#eb6834", lw=1.4, ls="--", label="Down-sweep")
    ax.set_xlabel("Frequency offset from nominal resonance (ppm)"); ax.set_ylabel("Measured amplitude (a.u.)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    st.pyplot(fig)
with c2:
    t = r["truth"]
    rows = [["True value", t["Q"], t["N"], t["shift"], ""],
            ["Half-power bandwidth", r["halfpower"]["Q"], np.nan, r["halfpower"]["shift"], "shift biased by ≈N half-widths"],
            ["Least squares (robust)", r["ls"]["Q"], r["ls"]["N"], r["ls"]["shift"], f"{r['ls']['ms']:.0f} ms"],
            ["Least squares + background", r["ls_bg"]["Q"], r["ls_bg"]["N"], r["ls_bg"]["shift"], ""],
            ["Gradient boosting (ML)", r["ml"]["Q"], r["ml"]["N"], r["ml"]["shift"], f"{r['ml']['ms']:.1f} ms; Q 90 % interval {r['ml']['Q_lo']:.0f}–{r['ml']['Q_hi']:.0f}"]]
    if np.isfinite(r["crb_sd"]["Q"]):
        rows.append(["CRB std (best possible)", r["crb_sd"]["Q"], r["crb_sd"]["N"], r["crb_sd"]["shift"], "regular-part bound at true parameters"])
    df = pd.DataFrame(rows, columns=["", "Q", "N", "Shift (ppm)", "Note"])
    st.dataframe(df.style.format({"Q": "{:.1f}", "N": "{:.3f}", "Shift (ppm)": "{:.2f}"}, na_rep="–"), hide_index=True)

st.subheader("Can these estimates be trusted?")
if r["abstain"]:
    st.error(f"Lack-of-fit statistic {r['lof']:.2f} exceeds the threshold {r['threshold']:.2f}: the Duffing model does not fit "
             "this sweep. Abstain: treat every estimate above (including the ML interval) as unreliable.")
else:
    st.success(f"Lack-of-fit statistic {r['lof']:.2f} ≤ threshold {r['threshold']:.2f}: no evidence against the model.")
    st.warning("Passing this check does not prove the estimates are right: amplitude-dependent damping usually passes it "
               "while biasing Q by tens of percent (a silent failure found in this project).")
if not r["in_train_range"]:
    st.warning("The fitted Q or N lies outside the ML training range (Q 800–8000, N 0.2–2.5). The ML estimate and its "
               "conformal interval are extrapolations and are not valid here.")
st.caption("The 90 % conformal interval is calibrated on nominal in-distribution data only; its coverage collapses "
           "under extrapolation or model mismatch (see the report).")
