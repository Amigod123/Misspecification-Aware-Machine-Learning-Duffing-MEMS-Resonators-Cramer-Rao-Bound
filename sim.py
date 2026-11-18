"""Vectorised Duffing-resonator frequency-sweep simulator (first-harmonic balance).

Normalised units: omega_0 = 1, linear damping c = 1/Q, drive f = c, so the
linear peak amplitude is 1.  With u = a^2 the steady-state amplitude obeys

    u * [ (c*W*(1 + eta*u))^2 + (1 - W^2 + 3/4*beta*u + 5/8*gamma*u^2)^2 ] = f^2

(gamma = quintic stiffness, eta = amplitude-dependent damping; both are 0 in the
nominal model, giving the cubic of the standard Duffing oscillator).

Hardening only (beta >= 0, gamma >= 0): when several positive real roots exist
the outer ones are the stable branches, so an up-sweep follows the LARGEST root
and a down-sweep the SMALLEST.  No time stepping or branch tracking is needed.

Reference implementation adapted from the course-project prototype; validated
in checks.py against analytic results and a time-domain ODE solution.
"""
import numpy as np


def _poly_coeffs(W, Q, N, gamma, eta):
    """Coefficients (highest power first) of the degree-5 polynomial in u."""
    c = 1.0 / Q
    f = c
    beta = 4.0 * N / (3.0 * Q)
    a1, a0 = c * W * eta, c * W                      # damping term  A = a1*u + a0
    b2, b1, b0 = 0.625 * gamma, 0.75 * beta, 1.0 - W ** 2   # stiffness term B = b2*u^2 + b1*u + b0
    shp = np.broadcast(W, Q, N, gamma, eta).shape
    S = np.zeros(shp + (5,))                         # A^2 + B^2, degree 4
    S[..., 0] = b2 ** 2
    S[..., 1] = 2 * b2 * b1
    S[..., 2] = b1 ** 2 + 2 * b2 * b0 + a1 ** 2
    S[..., 3] = 2 * b1 * b0 + 2 * a1 * a0
    S[..., 4] = b0 ** 2 + a0 ** 2
    full = np.concatenate([S, np.zeros(shp + (1,))], axis=-1)   # multiply by u
    full[..., 5] -= np.broadcast_to(f ** 2, shp)
    return full


def _batched_roots(coefs):
    """Roots of many polynomials at once via companion-matrix eigenvalues.
    Leading coefficients that are zero for the whole batch are stripped
    (nominal model -> cubic)."""
    flat = coefs.reshape(-1, coefs.shape[-1])
    while flat.shape[1] > 2 and np.all(np.abs(flat[:, 0]) < 1e-16):
        flat = flat[:, 1:]
    n = flat.shape[1] - 1
    mon = flat[:, 1:] / flat[:, :1]
    comp = np.zeros((flat.shape[0], n, n))
    comp[:, 0, :] = -mon
    idx = np.arange(1, n)
    comp[:, idx, idx - 1] = 1.0
    return np.linalg.eigvals(comp).reshape(coefs.shape[:-1] + (n,))


def response(Omega, Q, N, gamma=0.0, eta=0.0, imag_tol=1e-6):
    """Complex steady-state response (z_up, z_down) at drive frequencies Omega.

    Shapes broadcast: Omega (n,) or (B, n); Q, N, gamma, eta scalar or (B, 1).
    The complex response is z = f / (B + i*A) with |z| = a.
    """
    Omega = np.asarray(Omega, float)
    Q, N = np.asarray(Q, float), np.asarray(N, float)
    gamma, eta = np.asarray(gamma, float), np.asarray(eta, float)
    r = _batched_roots(_poly_coeffs(Omega, Q, N, gamma, eta))
    ok = (np.abs(r.imag) < imag_tol * (1 + np.abs(r.real))) & (r.real > 0)
    re = np.where(ok, r.real, np.nan)
    u_up, u_dn = np.nanmax(re, axis=-1), np.nanmin(re, axis=-1)
    c = 1.0 / Q
    beta = 4.0 * N / (3.0 * Q)
    out = []
    for u in (u_up, u_dn):
        B = (1.0 - Omega ** 2) + 0.75 * beta * u + 0.625 * gamma * u ** 2
        A = c * Omega * (1.0 + eta * u)
        out.append(c / (B + 1j * A))
    return out[0], out[1]


def grid_limits(Q, rho, x_lo=-6.0, x_hi=12.0, margin_ppm=120.0):
    """Sweep-window limits from a coarse Q guess Q/rho... (half-width hw' = rho/(2Q))."""
    Q = np.asarray(Q, float)
    hw = np.asarray(rho, float) / (2.0 * Q)
    lo = 1.0 + x_lo * hw - margin_ppm * 1e-6
    hi = 1.0 + x_hi * hw + margin_ppm * 1e-6
    return lo, hi


def make_grid(lo, hi, n=201):
    lo = np.asarray(lo, float).reshape(-1, 1)
    hi = np.asarray(hi, float).reshape(-1, 1)
    t = np.linspace(0.0, 1.0, n)[None, :]
    return lo + (hi - lo) * t


def measure(z_up, z_dn, G, bg, noise, rng, bits=12, fullscale=2.0):
    """Measured amplitudes: |G*z + bg| + white Gaussian noise, then ADC quantisation.

    G, bg (complex), noise: (B, 1) or scalars.  bits=None disables quantisation.
    The same G and bg apply to both sweeps of a sample; noise is independent.
    """
    ys = []
    for z in (z_up, z_dn):
        y = np.abs(G * z + bg) + np.asarray(noise) * rng.standard_normal(z.shape)
        if bits is not None:
            lv = 2 ** bits
            y = np.round(np.clip(y, 0.0, fullscale) / fullscale * lv) / lv * fullscale
        ys.append(y)
    return ys[0], ys[1]
