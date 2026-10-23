#!/usr/bin/env python3
"""
chicago_nearshore_waves.py
==========================
Simulate wave conditions in the nearshore zone off Chicago (southwest Lake Michigan).

Pipeline
--------
1. Offshore waves
   - Hindcast from wind speed/direction/duration using the USACE Coastal Engineering
     Manual (CEM, EM 1110-2-1100, Part II-2) fetch- and duration-limited growth laws,
     with direction-dependent fetch lengths measured from the Chicago lakefront; or
   - Pull the latest wind (and optionally waves) from an NOAA NDBC station.
2. Spectrum
   - JONSWAP spectrum and a random-phase surface-elevation time series offshore.
3. Nearshore transformation along a cross-shore profile (1-D, straight parallel contours)
   - Linear wave theory: shoaling + Snell's-law refraction.
   - Random-wave breaking with the Battjes & Janssen (1978) dissipation model.
   - Wave setup from the radiation-stress gradient.
   - Lake-level anomaly (Lake Michigan levels swing roughly 1-2 m over years).

Usage examples
--------------
    python chicago_nearshore_waves.py                          # NNE storm demo
    python chicago_nearshore_waves.py --wind-speed 12 --wind-dir 45 --duration 8
    python chicago_nearshore_waves.py --ndbc 45007             # latest buoy wind
    python chicago_nearshore_waves.py --ndbc 45007 --use-buoy-waves
    python chicago_nearshore_waves.py --profile my_profile.csv # columns: x_m,depth_m
    python chicago_nearshore_waves.py --sweep                  # Hs vs wind dir/speed map

Outputs: chicago_waves.png, chicago_waves_profile.csv (and chicago_waves_sweep.png).

Caveats: screening-level model. Fetches are approximate radial distances; nearshore
structures (breakwaters, revetments, harbor walls) and ice cover are not modeled.
For design work, use a calibrated spectral model (e.g. SWAN/STWAVE) and measured data.
"""

import argparse
import math
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
RHO = 1000.0  # fresh water, kg/m^3

# ---------------------------------------------------------------------------
# Site description: downtown Chicago lakefront
# ---------------------------------------------------------------------------
# Approximate radial fetch (km) over open water for wind blowing FROM each direction
# (degrees true). Westerly winds blow offshore -> no fetch. Edit for your exact site.
FETCH_TABLE_KM = [
    (0.0, 280), (22.5, 330), (45.0, 160), (67.5, 120),
    (90.0, 80), (112.5, 65), (135.0, 35), (157.5, 15),
    (180.0, 2), (202.5, 0), (225.0, 0), (247.5, 0),
    (270.0, 0), (292.5, 0), (315.0, 0), (337.5, 20),
]
SHORE_NORMAL_DEG = 90.0  # direction waves come FROM when they hit the shore head-on


def fetch_km(wind_from_deg):
    """Interpolate fetch around the compass (wraps at 360)."""
    d = np.array([p[0] for p in FETCH_TABLE_KM] + [360.0])
    f = np.array([p[1] for p in FETCH_TABLE_KM] + [FETCH_TABLE_KM[0][1]])
    return float(np.interp(wind_from_deg % 360.0, d, f))


# ---------------------------------------------------------------------------
# 1. Offshore wave growth (CEM)
# ---------------------------------------------------------------------------
def cem_wave_growth(u10, fetch_m, duration_hr=None):
    """Fetch/duration-limited Hm0 (m) and Tp (s) from U10 (m/s), CEM Part II-2."""
    if u10 <= 0 or fetch_m <= 0:
        return 0.0, 0.0, "no fetch (offshore wind)" if fetch_m <= 0 else "calm"
    cd = 0.001 * (1.1 + 0.035 * u10)
    us = u10 * math.sqrt(cd)  # friction velocity
    xhat = G * fetch_m / us**2
    regime = "fetch-limited"
    if duration_hr is not None:
        # equivalent fetch reached in the given duration: g t/u* = 77.23 (gX/u*^2)^0.67
        xhat_t = (G * duration_hr * 3600.0 / (77.23 * us)) ** (1.0 / 0.67)
        if xhat_t < xhat:
            xhat, regime = xhat_t, "duration-limited"
    hm0 = 4.13e-2 * us**2 / G * math.sqrt(xhat)
    tp = 0.651 * us / G * xhat ** (1.0 / 3.0)
    hm0_fd, tp_fd = 211.5 * us**2 / G, 239.8 * us / G  # fully developed limits
    if hm0 >= hm0_fd:
        hm0, tp, regime = hm0_fd, tp_fd, "fully developed"
    return hm0, tp, regime


# ---------------------------------------------------------------------------
# NDBC real-time data
# ---------------------------------------------------------------------------
def fetch_ndbc(station, anemometer_height=4.1):
    """Return latest dict(wspd10, wdir, wvht, dpd, mwd, time) from NDBC realtime2."""
    import urllib.request
    url = f"https://www.ndbc.noaa.gov/data/realtime2/{station.upper()}.txt"
    with urllib.request.urlopen(url, timeout=20) as r:
        lines = r.read().decode().splitlines()
    head = lines[0].lstrip("#").split()
    idx = {k: i for i, k in enumerate(head)}

    def val(row, key):
        try:
            v = row[idx[key]]
            return None if v == "MM" else float(v)
        except (KeyError, IndexError, ValueError):
            return None

    for line in lines[2:]:
        row = line.split()
        wspd, wdir = val(row, "WSPD"), val(row, "WDIR")
        if wspd is not None and wdir is not None:
            # adjust anemometer height to 10 m with a 1/7 power law
            u10 = wspd * (10.0 / anemometer_height) ** (1.0 / 7.0)
            return dict(time=" ".join(row[:5]), wspd10=u10, wdir=wdir,
                        wvht=val(row, "WVHT"), dpd=val(row, "DPD"), mwd=val(row, "MWD"))
    raise RuntimeError(f"No valid wind observations in {url}")


# ---------------------------------------------------------------------------
# Linear wave theory helpers
# ---------------------------------------------------------------------------
def wavenumber(T, h):
    """Solve w^2 = g k tanh(k h) by Newton iteration (vectorized over h)."""
    h = np.maximum(np.asarray(h, dtype=float), 1e-3)
    w = 2 * np.pi / T
    k = w**2 / G / np.sqrt(np.tanh(w**2 / G * h))  # good initial guess
    for _ in range(30):
        f = G * k * np.tanh(k * h) - w**2
        df = G * np.tanh(k * h) + G * k * h / np.cosh(k * h) ** 2
        k -= f / df
    return k


def celerities(T, h):
    k = wavenumber(T, h)
    c = 2 * np.pi / T / k
    kh = k * np.asarray(h)
    n = 0.5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 700)))
    return k, c, n * c, n


def jonswap(f, hm0, tp, gamma=3.3):
    """JONSWAP spectrum S(f) [m^2/Hz], scaled so 4*sqrt(m0) = Hm0."""
    fp = 1.0 / tp
    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-((f - fp) ** 2) / (2 * sigma**2 * fp**2))
    s = f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * gamma**r
    trapz = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz
    m0 = trapz(s, f)
    return s * (hm0 / 4.0) ** 2 / m0


def surface_timeseries(f, S, duration_s=600, dt=0.25, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, dt)
    df = np.gradient(f)
    amp = np.sqrt(2 * S * df)
    phase = rng.uniform(0, 2 * np.pi, f.size)
    eta = (amp[:, None] * np.cos(2 * np.pi * f[:, None] * t[None, :] + phase[:, None])).sum(0)
    return t, eta


# ---------------------------------------------------------------------------
# Bathymetry
# ---------------------------------------------------------------------------
def dean_profile(depth_offshore=12.0, A=0.10, dx=2.0):
    """Dean equilibrium profile h = A x^(2/3); x measured offshore from the still-water line.
    A ~ 0.1 m^(1/3) for fine sand (d50 ~ 0.2-0.3 mm), typical of Chicago beaches."""
    x_max = (depth_offshore / A) ** 1.5
    x = np.arange(0, x_max + dx, dx)
    return x, A * x ** (2.0 / 3.0)


def load_profile(path, dx=2.0):
    data = np.genfromtxt(path, delimiter=",", names=True)
    x, h = data[data.dtype.names[0]], data[data.dtype.names[1]]
    order = np.argsort(x)
    xg = np.arange(x.min(), x.max() + dx, dx)
    return xg, np.interp(xg, x[order], h[order])


# ---------------------------------------------------------------------------
# 2. Nearshore transformation: Battjes-Janssen energy balance + setup
# ---------------------------------------------------------------------------
def qb_fraction(b):
    """Fraction of breaking waves: solve (1-Q)/ln(Q) = -b^2, b = Hrms/Hmax."""
    if b >= 1.0:
        return 1.0
    if b < 0.2:
        return 0.0
    lo, hi = 1e-12, 1.0 - 1e-12
    for _ in range(80):  # bisection
        q = 0.5 * (lo + hi)
        # f(q) = (1-q)/ln(q) + b^2 decreases from b^2 (q->0) to b^2-1 (q->1)
        if (1 - q) / math.log(q) + b**2 > 0:
            lo = q
        else:
            hi = q
    return 0.5 * (lo + hi)


def transform(x, h_still, hm0_0, tp, theta0_deg, gamma=0.78, alpha=1.0, n_iter=3):
    """March wave energy flux from offshore (max x) to shore (x=0).
    theta0_deg: wave angle relative to shore normal at the offshore boundary."""
    order = np.argsort(-x)  # offshore -> onshore
    xs, hs = x[order], h_still[order]
    dx = np.abs(np.diff(xs))
    fp = 1.0 / tp
    setup = np.zeros_like(hs)
    for _ in range(n_iter):
        h = np.maximum(hs + setup, 0.01)
        k, c, cg, n = celerities(tp, h)
        sin_ratio = math.sin(math.radians(theta0_deg)) / c[0]
        sin_t = np.clip(c * sin_ratio, -1, 1)
        cos_t = np.sqrt(1 - sin_t**2)
        hrms = np.zeros_like(h)
        qb = np.zeros_like(h)
        hrms[0] = hm0_0 / math.sqrt(2)
        E = RHO * G * hrms[0] ** 2 / 8.0
        F = E * cg[0] * cos_t[0]
        for i in range(len(h) - 1):
            hmax = gamma * h[i]
            qb[i] = qb_fraction(hrms[i] / hmax) if hmax > 0 else 1.0
            D = 0.25 * alpha * RHO * G * fp * qb[i] * hmax**2
            # exponential update: same as F - D*dx for small steps, never overshoots below 0
            F = F * math.exp(-D * dx[i] / F) if F > 0 else 0.0
            E = F / (cg[i + 1] * cos_t[i + 1])
            hrms[i + 1] = min(math.sqrt(8 * E / (RHO * G)), gamma * h[i + 1])
        qb[-1] = 1.0
        E = RHO * G * hrms**2 / 8.0
        sxx = E * (n * (cos_t**2 + 1) - 0.5)
        # d(eta)/ds = -dSxx/ds / (rho g (h+eta)), s = shoreward distance
        new_setup = np.zeros_like(hs)
        for i in range(len(h) - 1):
            new_setup[i + 1] = new_setup[i] - (sxx[i + 1] - sxx[i]) / (RHO * G * h[i])
        setup = new_setup
    out = dict(x=xs, h_still=hs, depth=h, hs=np.sqrt(2) * hrms, hrms=hrms, qb=qb,
               setup=setup, theta=np.degrees(np.arcsin(sin_t)), sxx=sxx)
    return {kk: vv[::-1] for kk, vv in out.items()}  # return ordered shore -> offshore


# ---------------------------------------------------------------------------
# Plotting / reporting
# ---------------------------------------------------------------------------
def report(res, meta):
    print("\n=== Chicago nearshore wave simulation ===")
    for k, v in meta.items():
        print(f"  {k:<26}: {v}")
    print("\n  depth(m)  dist(m)   Hs(m)  angle(deg)  break%  setup(cm)")
    targets = [res["h_still"].max(), 8, 6, 4, 3, 2, 1, 0.5]
    for d in targets:
        if d > res["h_still"].max():
            continue
        i = int(np.argmin(np.abs(res["h_still"] - d)))
        print(f"  {res['h_still'][i]:7.2f}  {res['x'][i]:7.0f}  {res['hs'][i]:6.2f}"
              f"  {res['theta'][i]:9.1f}  {100 * res['qb'][i]:6.1f}  {100 * res['setup'][i]:8.1f}")
    brk = np.where(res["qb"] > 0.05)[0]
    if brk.size:
        ib = brk.max()  # outermost point with >5% of waves breaking
        print(f"\n  Breaking starts ~{res['x'][ib]:.0f} m offshore in {res['h_still'][ib]:.1f} m depth;"
              f" max Hs {res['hs'].max():.2f} m; shoreline setup {100 * res['setup'][0]:.0f} cm")


def plot(res, f, S, t, eta, meta, fname="chicago_waves.png"):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    a.fill_between(res["x"], -res["h_still"], -res["h_still"].max() * 1.05, color="tan")
    a.plot(res["x"], res["setup"], "b", lw=1.5, label="mean water level (setup)")
    a.plot(res["x"], res["setup"] + res["hs"] / 2, "c--", lw=1, label="± Hs/2")
    a.plot(res["x"], res["setup"] - res["hs"] / 2, "c--", lw=1)
    a.invert_xaxis(); a.set_xlabel("distance offshore (m)"); a.set_ylabel("elevation (m)")
    a.set_title("Cross-shore profile"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(res["x"], res["hs"], "k", label="Hs")
    a2 = a.twinx()
    a2.plot(res["x"], 100 * res["qb"], "r", alpha=0.6, label="% breaking")
    a2.set_ylabel("% waves breaking", color="r")
    a.invert_xaxis(); a.set_xlabel("distance offshore (m)"); a.set_ylabel("Hs (m)")
    a.set_title("Wave height transformation")

    a = ax[1, 0]
    a.plot(f, S, "navy")
    a.set_xlim(0, min(1.0, 4 / meta["_tp"])); a.set_xlabel("frequency (Hz)")
    a.set_ylabel("S(f) (m²/Hz)"); a.set_title("Offshore JONSWAP spectrum")

    a = ax[1, 1]
    a.plot(t, eta, lw=0.7)
    a.set_xlabel("time (s)"); a.set_ylabel("η (m)")
    a.set_title("Offshore surface elevation (random-phase realization)")

    title = (f"Chicago nearshore waves  |  {meta['Offshore Hm0']}, Tp {meta['Peak period Tp']}, "
             f"from {meta['Wave direction (from)']}")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"\n  Figure saved: {fname}")


def sweep(duration_hr, fname="chicago_waves_sweep.png"):
    dirs = np.arange(0, 360, 5.0)
    speeds = np.arange(2, 26, 1.0)
    H = np.array([[cem_wave_growth(u, fetch_km(d) * 1e3, duration_hr)[0] for d in dirs]
                  for u in speeds])
    fig, a = plt.subplots(figsize=(10, 5))
    m = a.pcolormesh(dirs, speeds, H, shading="auto", cmap="viridis")
    cs = a.contour(dirs, speeds, H, levels=[0.5, 1, 2, 3, 4, 5], colors="w", linewidths=0.7)
    a.clabel(cs, fmt="%.1f m", fontsize=8)
    fig.colorbar(m, label="offshore Hm0 (m)")
    a.set_xticks(range(0, 361, 45))
    a.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"])
    a.set_xlabel("wind from"); a.set_ylabel("U10 (m/s)")
    lim = f"{duration_hr:g} h duration" if duration_hr else "fetch-limited (unlimited duration)"
    a.set_title(f"Offshore wave height off Chicago vs. wind ({lim})")
    fig.tight_layout(); fig.savefig(fname, dpi=130)
    print(f"Sweep figure saved: {fname}")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wind-speed", type=float, default=15.0, help="U10 wind speed, m/s (default 15)")
    p.add_argument("--wind-dir", type=float, default=20.0, help="wind FROM direction, deg true (default 20 = NNE)")
    p.add_argument("--duration", type=float, default=12.0, help="storm duration, hours (0 = unlimited)")
    p.add_argument("--ndbc", type=str, help="NDBC station ID for live wind, e.g. 45007")
    p.add_argument("--use-buoy-waves", action="store_true", help="use buoy WVHT/DPD/MWD as offshore waves")
    p.add_argument("--profile", type=str, help="CSV with header, columns x_m (offshore distance), depth_m")
    p.add_argument("--depth-offshore", type=float, default=12.0, help="offshore boundary depth for Dean profile, m")
    p.add_argument("--dean-a", type=float, default=0.10, help="Dean profile parameter A, m^(1/3)")
    p.add_argument("--water-level", type=float, default=0.0, help="lake level anomaly above datum, m")
    p.add_argument("--shore-normal", type=float, default=SHORE_NORMAL_DEG, help="shore-normal direction, deg")
    p.add_argument("--gamma", type=float, default=0.78, help="breaker index Hmax/h")
    p.add_argument("--sweep", action="store_true", help="also plot offshore Hs vs wind dir/speed")
    args = p.parse_args()
    duration = args.duration if args.duration > 0 else None

    u10, wdir, source = args.wind_speed, args.wind_dir, "user scenario"
    hm0 = tp = None
    wave_dir = wdir
    if args.ndbc:
        try:
            obs = fetch_ndbc(args.ndbc)
            u10, wdir = obs["wspd10"], obs["wdir"]
            wave_dir = wdir
            source = f"NDBC {args.ndbc.upper()} @ {obs['time']} UTC"
            if args.use_buoy_waves and obs["wvht"] and obs["dpd"]:
                hm0, tp = obs["wvht"], obs["dpd"]
                wave_dir = obs["mwd"] if obs["mwd"] is not None else wdir
                source += " (measured waves)"
        except Exception as e:
            print(f"Could not read NDBC {args.ndbc}: {e}\nFalling back to scenario wind.", file=sys.stderr)

    fetch = fetch_km(wdir)
    regime = "measured"
    if hm0 is None:
        hm0, tp, regime = cem_wave_growth(u10, fetch * 1e3, duration)

    if args.sweep:
        sweep(duration)

    theta0 = (wave_dir - args.shore_normal + 180) % 360 - 180
    if hm0 < 0.05 or abs(theta0) >= 89:
        print(f"Wind {u10:.1f} m/s from {wdir:.0f}° blows offshore or along shore off Chicago "
              f"(fetch {fetch:.0f} km): nearshore is essentially calm (Hm0 {hm0:.2f} m).")
        return

    x, h = load_profile(args.profile) if args.profile else dean_profile(args.depth_offshore, args.dean_a)
    h = h + args.water_level
    res = transform(x, h, hm0, tp, theta0, gamma=args.gamma)

    f = np.linspace(0.02, 1.0, 400)
    S = jonswap(f, hm0, tp)
    t, eta = surface_timeseries(f, S)

    meta = {
        "Source": source,
        "Wind U10": f"{u10:.1f} m/s from {wdir:.0f}°",
        "Fetch": f"{fetch:.0f} km",
        "Growth regime": regime,
        "Offshore Hm0": f"{hm0:.2f} m",
        "Peak period Tp": f"{tp:.1f} s",
        "Wave direction (from)": f"{wave_dir:.0f}°",
        "Angle to shore normal": f"{theta0:.0f}°",
        "Lake level anomaly": f"{args.water_level:+.2f} m",
        "_tp": tp,
    }
    report(res, {k: v for k, v in meta.items() if not k.startswith("_")})
    plot(res, f, S, t, eta, meta)

    np.savetxt("chicago_waves_profile.csv",
               np.column_stack([res["x"], res["h_still"], res["hs"], res["qb"], res["setup"], res["theta"]]),
               delimiter=",", header="x_offshore_m,depth_m,Hs_m,frac_breaking,setup_m,angle_deg",
               comments="", fmt="%.4f")
    print("  Profile data saved: chicago_waves_profile.csv")


if __name__ == "__main__":
    main()
