"""Shared library: configuration, dataset generation, features, classical baselines,
least-squares fitting with diagnostics, regular-part Cramér–Rao bound, metrics."""
import json, time, hashlib, platform
import numpy as np
from scipy.optimize import least_squares
from sim import response, grid_limits, make_grid, measure

# ----------------------------------------------------------------------------- config
def load_config(path="configs/default.json"):
    with open(path) as f:
        return json.load(f)

# Fixed per-dataset codes so every dataset of every seed has its own RNG stream.
DATASET_CODES = {"train": 1, "calib": 2, "abstain_calib": 3,
                 "test_nominal": 10, "test_Q_extrap": 11, "test_N_extrap": 12, "test_nom_N08": 13,
                 "test_noise3": 14, "test_feedthrough": 15, "test_quintic": 16, "test_damping": 17,
                 "det_nominal": 20, "det_feed": 21, "det_quintic": 22, "det_damp": 23,
                 "eb_params": 40, "eb_meas": 41}


def dataset_seed(seed, name):
    return 100_000 * (seed + 1) + DATASET_CODES[name]


def software_versions():
    import scipy, sklearn, matplotlib
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()}

# ----------------------------------------------------------------------------- data generation
def draw_params(B, rng, cfg, ranges_key="train", world="nominal"):
    """Draw B independent parameter vectors (each sample is its own 'parameter block')."""
    r = cfg["ranges"][ranges_key]
    Q = np.exp(rng.uniform(np.log(r["Q"][0]), np.log(r["Q"][1]), B))
    N = rng.uniform(r["N"][0], r["N"][1], B)
    sh = rng.uniform(-cfg["shift_ppm"], cfg["shift_ppm"], B)
    rho = rng.uniform(*cfg["window"]["rho"], B)
    noise = rng.uniform(*cfg["noise_range"], B)
    G = np.exp(cfg["gain_sigma"] * rng.standard_normal(B))
    psi = rng.uniform(0, 2 * np.pi, B)
    feed = np.zeros(B); gamma = np.zeros(B); eta = np.zeros(B)
    w = cfg["worlds"]
    if world == "noise3":
        noise = np.full(B, cfg["noise_world_sigma"])
    elif world == "feed":
        feed = rng.uniform(*w["feed_mag"], B)
    elif world == "quintic":
        gamma = rng.uniform(*w["quintic_kappa"], B) * 4 * N / (3 * Q)
    elif world == "damp":
        eta = rng.uniform(*w["damping_eta"], B)
    elif world != "nominal":
        raise ValueError(world)
    lo, hi = grid_limits(Q, rho, cfg["window"]["x_lo"], cfg["window"]["x_hi"], cfg["window"]["margin_ppm"])
    return dict(Q=Q, N=N, sh=sh, rho=rho, noise=noise, G=G, bg=feed * np.exp(1j * psi), feed=feed,
                gamma=gamma, eta=eta, lo=lo, hi=hi, world=np.array(world), ranges=np.array(ranges_key))


def simulate(p, rng, cfg, noise=None, bits="cfg"):
    """Noise-free complex responses + measured sweeps for parameter dict p."""
    Om = make_grid(p["lo"], p["hi"], cfg["n_points"])
    Omn = Om / (1 + p["sh"][:, None] * 1e-6)                 # resonance moved by sh ppm
    zu, zd = response(Omn, p["Q"][:, None], p["N"][:, None], p["gamma"][:, None], p["eta"][:, None])
    sig = (p["noise"] if noise is None else np.broadcast_to(noise, p["Q"].shape))[:, None]
    b = cfg["adc_bits"] if bits == "cfg" else bits
    yu, yd = measure(zu, zd, p["G"][:, None], p["bg"][:, None], sig, rng, b, cfg["adc_fullscale"])
    return Om, yu, yd


def generate(B, seed, cfg, ranges_key="train", world="nominal"):
    rng = np.random.default_rng(seed)
    p = draw_params(B, rng, cfg, ranges_key, world)
    Om, yu, yd = simulate(p, rng, cfg)
    d = dict(p); d.update(Om=Om, yu=yu, yd=yd)
    return d


def save_dataset(d, path):
    keep = {k: v for k, v in d.items() if k != "Om"}
    keep["yu"] = d["yu"].astype(np.float32); keep["yd"] = d["yd"].astype(np.float32)   # exact for 12-bit codes
    np.savez_compressed(path, **keep)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_dataset(path, cfg):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["yu"] = d["yu"].astype(float); d["yd"] = d["yd"].astype(float)
    d["Om"] = make_grid(d["lo"], d["hi"], cfg["n_points"])
    return d


def subset(d, idx):
    B = len(d["Q"])
    return {k: (v[idx] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == B else v) for k, v in d.items()}

# ----------------------------------------------------------------------------- features
def _cross(xp, y, L, side):
    """Interpolated position where y crosses level L: first rising ('l') or last falling ('r')."""
    B, n = y.shape
    L = np.asarray(L, float)
    m = y >= (L[:, None] if L.ndim else L)
    rows = np.arange(B)
    if side == "l":
        i = np.argmax(m, axis=1); i0 = np.clip(i - 1, 0, n - 1)
    else:
        i = n - 1 - np.argmax(m[:, ::-1], axis=1); i0 = np.clip(i + 1, 0, n - 1)
    y1, y0 = y[rows, i], y[rows, i0]
    x1, x0 = xp[rows, i], xp[rows, i0]
    den = np.where(np.abs(y1 - y0) < 1e-12, 1.0, y1 - y0)
    t = np.clip((L - y0) / den, 0, 1)
    return x0 + t * (x1 - x0)


LEVELS = (0.3, 0.5, 0.707, 0.9)


def _sweep_feats(xp, y, ref):
    yn = y / ref[:, None]
    f = {}
    rows = np.arange(len(yn)); ip = np.argmax(yn, axis=1)
    f["pk"] = yn[rows, ip]
    f["pkx"] = xp[rows, ip]
    for L in LEVELS:
        l = _cross(xp, yn, L * f["pk"], "l"); r = _cross(xp, yn, L * f["pk"], "r")
        f[f"l{L}"] = l; f[f"r{L}"] = r; f[f"w{L}"] = r - l; f[f"c{L}"] = 0.5 * (l + r) - f["pkx"]
    w = yn ** 2; w = w / w.sum(1, keepdims=True)
    mu = (w * xp).sum(1); sd = np.sqrt((w * (xp - mu[:, None]) ** 2).sum(1))
    f["mu"] = mu; f["sd"] = sd
    f["skew"] = (w * (xp - mu[:, None]) ** 3).sum(1) / sd ** 3
    f["noise"] = np.std(np.diff(yn, 2, axis=1), axis=1) / np.sqrt(6)
    return f


def features(d, up_only=False):
    """47 handcrafted features (22 per sweep + 3 up/down-difference features); 22 if up_only.
    Frequencies are in ppm relative to the nominal resonance; amplitudes normalised by the up-sweep peak."""
    xp = (d["Om"] - 1.0) * 1e6
    ref = d["yu"].max(1)
    fu = _sweep_feats(xp, d["yu"], ref)
    cols = {f"up_{k}": v for k, v in fu.items()}
    if not up_only:
        fd = _sweep_feats(xp, d["yd"], ref)
        cols.update({f"dn_{k}": v for k, v in fd.items()})
        diff = np.abs(d["yu"] - d["yd"]) / ref[:, None]
        cols["diff_max"] = diff.max(1)
        cols["diff_area"] = diff.sum(1) * (xp[:, 1] - xp[:, 0])
        cols["diff_pos"] = xp[np.arange(len(xp)), diff.argmax(1)] - fu["pkx"]
    names = list(cols)
    return np.column_stack([cols[k] for k in names]), names


def targets(d):
    return np.column_stack([np.log(d["Q"]), d["N"], d["sh"]])


def sigma_hat(d):
    """Robust white-noise estimate from second differences, averaged over both sweeps (absolute units)."""
    s = [np.median(np.abs(np.diff(y, 2, axis=1)), 1) / 0.6745 / np.sqrt(6) for y in (d["yu"], d["yd"])]
    return 0.5 * (s[0] + s[1])

# ----------------------------------------------------------------------------- baseline A
def half_power(d):
    """Q from the 0.707-level width of the up-sweep; 'shift' = raw peak position in ppm
    (biased by ~N half-widths because the nonlinear peak leans)."""
    xp = (d["Om"] - 1.0) * 1e6
    yn = d["yu"] / d["yu"].max(1, keepdims=True)
    l = _cross(xp, yn, 0.7071, "l"); r = _cross(xp, yn, 0.7071, "r")
    Qh = 1e6 / np.maximum(r - l, 1e-3)
    return Qh, xp[np.arange(len(xp)), np.argmax(yn, 1)]

# ----------------------------------------------------------------------------- baseline B
def model_amp(Om, Q, N, sh, G, bg=0j, use="both"):
    zu, zd = response(Om / (1 + sh * 1e-6), Q, N)
    yu = np.abs(G * zu + bg)
    return yu if use == "up" else np.concatenate([yu, np.abs(G * zd + bg)])


def ls_fit(Om, yu, yd, Q0, sh0, cfg, use="both", bg=False, starts=None, p0_list=None, prefit_mask=False,
           diff_step=None, max_nfev=None):
    """Bounded multi-start nonlinear least squares on the exact Duffing model.

    Parameters p = (log Q, N, shift_ppm, log G[, Re bg, Im bg]).  Starting points: N0 in `starts`
    with Q0 and a nonlinearity-corrected shift from Baseline A, or explicit `p0_list` (e.g. the true
    parameters for the oracle-start diagnostic).  prefit_mask=True first fits only grid points where
    the measured up- and down-sweeps agree (outside the hysteresis loop, where the model is continuous
    in its parameters), then refines on all points.  Returns estimates and full diagnostics."""
    L = cfg["ls"]
    ds = L["diff_step"] if diff_step is None else diff_step
    mx = L["max_nfev"] if max_nfev is None else max_nfev
    t0 = time.perf_counter()
    obs_full = yu if use == "up" else np.concatenate([yu, yd])
    lb = [L["logQ_bounds"][0], L["N_bounds"][0], L["shift_bounds_ppm"][0], L["logG_bounds"][0]] + ([L["bg_bounds"][0]] * 2 if bg else [])
    ub = [L["logQ_bounds"][1], L["N_bounds"][1], L["shift_bounds_ppm"][1], L["logG_bounds"][1]] + ([L["bg_bounds"][1]] * 2 if bg else [])
    xs = [0.2, 0.5, 20.0, 0.05] + ([0.05, 0.05] if bg else [])

    def make_res(mask):
        obs = obs_full[mask]
        def res(p):
            b = (p[4] + 1j * p[5]) if bg else 0j
            try:
                m = model_amp(Om, np.exp(p[0]), p[1], p[2], np.exp(p[3]), b, use)[mask]
            except Exception:
                return np.full(obs.shape, 10.0)
            return np.nan_to_num(m - obs, nan=10.0)
        return res

    full_mask = np.ones(len(obs_full), bool)
    if p0_list is None:
        starts = L["starts"] if starts is None else starts
        Q0c = float(np.clip(Q0, 250, 5e4))
        p0_list = [[np.log(Q0c), N0, float(np.clip(sh0 - N0 * 1e6 / (2 * Q0c), -300, 300)), 0.0] + ([0.0, 0.0] if bg else [])
                   for N0 in starts]
    p0_list = [np.clip(np.asarray(p, float), np.array(lb) + 1e-9, np.array(ub) - 1e-9) for p in p0_list]

    nfev = 0; best = None; best_i = -1
    stage1_mask = full_mask
    if prefit_mask and use == "both":
        n = len(yu)
        sig = np.median(np.abs(np.diff(np.r_[yu, yd], 2))) / 0.6745 / np.sqrt(6)
        agree = np.abs(yu - yd) <= 4 * max(sig, 1e-4)
        if agree.sum() > 40:
            stage1_mask = np.r_[agree, agree]
    res1 = make_res(stage1_mask)
    for i, p0 in enumerate(p0_list):
        try:
            s = least_squares(res1, p0, bounds=(lb, ub), diff_step=ds, max_nfev=mx, x_scale=xs)
        except Exception:
            continue
        nfev += s.nfev
        if best is None or s.cost < best.cost:
            best, best_i = s, i
    if best is None:
        return dict(Q=np.nan, N=np.nan, sh=np.nan, G=np.nan, rss=np.nan, n=len(obs_full), npar=len(lb),
                    success=False, status=-99, nfev=nfev, time_s=time.perf_counter() - t0, best_start=-1)
    if stage1_mask is not full_mask:   # refine on all points
        s = least_squares(make_res(full_mask), best.x, bounds=(lb, ub), diff_step=ds, max_nfev=mx, x_scale=xs)
        nfev += s.nfev
        best = s
    p = best.x
    rss = float(np.sum(make_res(full_mask)(p) ** 2))
    return dict(Q=float(np.exp(p[0])), N=float(p[1]), sh=float(p[2]), G=float(np.exp(p[3])),
                bg_re=float(p[4]) if bg else 0.0, bg_im=float(p[5]) if bg else 0.0,
                rss=rss, n=len(obs_full), npar=len(p), success=bool(best.success), status=int(best.status),
                nfev=int(nfev), time_s=time.perf_counter() - t0, best_start=best_i)


def true_p0(d, i, bg=False):
    p = [np.log(d["Q"][i]), d["N"][i], d["sh"][i], np.log(d["G"][i])]
    if bg:
        p += [float(np.real(d["bg"][i])), float(np.imag(d["bg"][i]))]
    return [p]

# ----------------------------------------------------------------------------- Cramér–Rao bound
def crb(Om, Q, N, sh, G, sigma, use="both", jump_tol=0.03):
    """Regular-part CRB standard deviations for (Q, N, shift_ppm), gain as nuisance.

    Fisher information F = J^T J / sigma^2 with J the central finite-difference Jacobian of the exact
    model w.r.t. (Q, N, shift, G), assuming white Gaussian noise of known sigma on every point.
    Grid points whose value changes by more than jump_tol under any +-h perturbation (a branch jump
    moving across the point) are excluded, so jump-location information is ignored; ADC quantisation
    is ignored.  Returns (sd[3], number_of_points_used)."""
    th = np.array([Q, N, sh, G], float)
    steps = np.array([1e-3 * Q, 1e-3 * max(N, 1.0), 0.02, 1e-3 * G])
    cols = []; flags = None
    for k in range(4):
        tp, tm = th.copy(), th.copy(); tp[k] += steps[k]; tm[k] -= steps[k]
        mp = model_amp(Om, tp[0], max(tp[1], 0.0), tp[2], tp[3], use=use)
        mm = model_amp(Om, tm[0], max(tm[1], 0.0), tm[2], tm[3], use=use)
        dif = mp - mm
        flags = (np.abs(dif) > jump_tol) if flags is None else (flags | (np.abs(dif) > jump_tol))
        cols.append(dif / (tp[k] - tm[k]))
    J = np.column_stack(cols)[~flags]
    try:
        C = np.linalg.inv(J.T @ J / sigma ** 2)
        sd = np.sqrt(np.clip(np.diag(C), 0, None))[:3]
    except np.linalg.LinAlgError:
        sd = np.full(3, np.nan)
    return sd, int((~flags).sum())

# ----------------------------------------------------------------------------- statistics
def boot_ci(x, stat=np.median, n=2000, seed=12345, alpha=0.05):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    bs = np.array([stat(x[i]) for i in idx]) if stat not in (np.median, np.mean) else stat(x[idx], axis=1)
    return (float(stat(x)), float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))))


def prop_ci(k, n, z=1.96):
    """Wilson score interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (float(p), float(max(0.0, c - h)), float(min(1.0, c + h)))


def robust_spread(z):
    z = np.asarray(z, float); z = z[np.isfinite(z)]
    return float(np.median(np.abs(z)) / 0.6745)
