"""
features.py
-----------
Turn a light curve into a small set of numbers (features) the classifier can
learn from.

The classifier cannot look at thousands of flux points directly in a way that
generalises well, so we summarise each light curve with features that capture
the *shape* differences between the four classes. Each feature below is chosen
because it separates the classes for a physical reason:

  bls_snr        - how significant the detected dip is. Real signals score high,
                   pure noise scores low.
  depth          - planets are shallow, eclipsing binaries are deep, blends are
                   shallow-but-wrong. Depth alone carries a lot of information.
  duration_ratio - transit duration divided by period. Eclipses tend to be a
                   larger fraction of the orbit than planets.
  odd_even_diff  - difference in depth between odd and even dips. An eclipsing
                   binary's secondary often makes alternate dips differ; a real
                   planet's dips are all the same.
  v_shape        - how V-shaped vs U-shaped the dip is. Planets are flat-
                   bottomed (U), grazing eclipses are pointy (V).
  secondary      - strength of any dip half a period after the main one. A
                   secondary eclipse points to a binary, not a planet.
  scatter        - overall point-to-point scatter (noisiness of the star).
  skew, kurt     - distribution shape of the flux, which differs between a
                   clean transit and starspot wobble.

Everything is wrapped so a single bad light curve yields a NaN-free feature
row (filled with safe defaults) rather than crashing the batch.
"""

import numpy as np

try:
    from scipy.stats import skew, kurtosis
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


FEATURE_NAMES = [
    "bls_snr", "depth", "duration_ratio", "odd_even_diff",
    "v_shape", "secondary", "scatter", "flux_skew", "flux_kurt",
]


def _safe(x, default=0.0):
    """Return x if finite, else a default. Keeps NaNs out of the feature row."""
    return float(x) if (x is not None and np.isfinite(x)) else default


def _fold(time, flux, period, t0):
    """Phase-fold onto [-0.5, 0.5] centred on the transit."""
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def _odd_even_depth(time, flux, period, t0):
    """
    Compare the average depth of odd-numbered dips vs even-numbered dips.
    Big difference -> likely an eclipsing binary masquerading via its secondary.
    """
    cycle = np.floor((time - t0) / period + 0.5).astype(int)
    in_dip = np.abs(((time - t0 + 0.5 * period) % period) - 0.5 * period) \
        < 0.05 * period
    if in_dip.sum() < 4:
        return 0.0
    odd = in_dip & (cycle % 2 == 1)
    even = in_dip & (cycle % 2 == 0)
    if odd.sum() < 2 or even.sum() < 2:
        return 0.0
    d_odd = 1.0 - np.median(flux[odd])
    d_even = 1.0 - np.median(flux[even])
    denom = max(abs(d_odd) + abs(d_even), 1e-6)
    return abs(d_odd - d_even) / denom


def _dip_halfwidth(phase, flux):
    """
    Estimate the dip's half-width in phase from the data itself, so the
    shape/secondary windows adapt to the actual transit instead of guessing a
    fixed width. We find where the folded flux first climbs back to half its
    minimum depth on either side of phase 0.
    """
    # bin the folded curve so noise does not dominate
    nb = 80
    edges = np.linspace(-0.5, 0.5, nb + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nb - 1)
    binned = np.full(nb, np.nan)
    for k in range(nb):
        sel = idx == k
        if sel.any():
            binned[k] = np.median(flux[sel])
    centers = 0.5 * (edges[:-1] + edges[1:])
    if np.all(np.isnan(binned)):
        return 0.05, centers, binned
    base = np.nanmedian(binned)
    depth = base - np.nanmin(binned)
    if depth <= 0:
        return 0.05, centers, binned
    half_level = base - 0.5 * depth
    below = centers[np.where(binned < half_level)[0]] \
        if np.any(binned < half_level) else np.array([0.0])
    hw = float(np.max(np.abs(below))) if below.size else 0.05
    return max(hw, 0.01), centers, binned


def _v_shape(phase, flux):
    """
    How pointy the dip is. Compare the flux at the very centre to the flux at
    the edges of the dip (measured at the data-driven half-width). A flat
    bottom (planet, U-shape) keeps the centre much lower than the edges; a
    pointy V (grazing eclipse) makes them similar.

    Returns ~0 for U-shaped (planet), ~1 for V-shaped (eclipse).
    """
    hw, _, _ = _dip_halfwidth(phase, flux)
    core = np.abs(phase) < 0.3 * hw
    edge = (np.abs(phase) >= 0.7 * hw) & (np.abs(phase) < 1.1 * hw)
    if core.sum() < 2 or edge.sum() < 2:
        return 0.0
    base = np.median(flux[np.abs(phase) > 0.25])
    core_depth = base - np.median(flux[core])
    edge_depth = base - np.median(flux[edge])
    if core_depth <= 1e-6:
        return 0.0
    return float(np.clip(edge_depth / core_depth, 0.0, 1.0))


def _secondary_strength(phase, flux):
    """
    Look for a dip at phase ~0.5 (half a period after the main dip). A
    secondary eclipse there is a strong sign of an eclipsing binary rather
    than a planet. We measure it relative to the dip depth so it is comparable
    across stars.
    """
    hw, _, _ = _dip_halfwidth(phase, flux)
    w = max(hw, 0.03)
    sec = np.abs(np.abs(phase) - 0.5) < w
    base_mask = (np.abs(phase) > 0.2) & (np.abs(np.abs(phase) - 0.5) > 2 * w)
    primary = np.abs(phase) < w
    if sec.sum() < 2 or base_mask.sum() < 2 or primary.sum() < 2:
        return 0.0
    base = np.median(flux[base_mask])
    sec_depth = max(0.0, base - np.median(flux[sec]))
    pri_depth = max(1e-6, base - np.median(flux[primary]))
    # secondary depth as a fraction of primary depth
    return float(np.clip(sec_depth / pri_depth, 0.0, 2.0))


def extract(time, flux, bls_result):
    """
    Build one feature row from a cleaned light curve and its BLS result.

    Returns a dict {feature_name: value}. Always returns every feature, filled
    with safe defaults when something cannot be computed.
    """
    feats = {name: 0.0 for name in FEATURE_NAMES}

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if time.size < 20:
        return feats

    feats["bls_snr"] = _safe(bls_result.get("snr"))
    feats["depth"] = _safe(bls_result.get("depth"))
    feats["scatter"] = _safe(np.median(np.abs(np.diff(flux))))

    if _HAVE_SCIPY:
        feats["flux_skew"] = _safe(skew(flux))
        feats["flux_kurt"] = _safe(kurtosis(flux))
    else:
        m = np.mean(flux)
        s = np.std(flux) + 1e-9
        feats["flux_skew"] = _safe(np.mean(((flux - m) / s) ** 3))
        feats["flux_kurt"] = _safe(np.mean(((flux - m) / s) ** 4) - 3.0)

    period = bls_result.get("period")
    t0 = bls_result.get("t0")
    dur = bls_result.get("duration")

    if period and period > 0 and t0 is not None:
        feats["duration_ratio"] = _safe((dur or 0.0) / period)
        feats["odd_even_diff"] = _safe(
            _odd_even_depth(time, flux, period, t0))
        phase, fflux = _fold(time, flux, period, t0)
        feats["v_shape"] = _safe(_v_shape(phase, fflux))
        feats["secondary"] = _safe(_secondary_strength(phase, fflux))

    return feats


def to_row(feats):
    """Feature dict -> ordered list, matching FEATURE_NAMES."""
    return [feats[name] for name in FEATURE_NAMES]


if __name__ == "__main__":
    import synth
    import preprocess
    import bls

    rng = np.random.default_rng(3)
    print("class      | snr   depth   durR   o/e    vshape sec")
    for label in synth.LABELS:
        t, f, p = synth.make_one(label, rng)
        t, f = preprocess.preprocess(t, f)
        b = bls.run_bls(t, f, n_periods=3000)
        ft = extract(t, f, b)
        print("{:10s} | {:5.1f} {:.4f} {:.4f} {:.3f}  {:.3f}  {:.4f}"
              .format(label, ft["bls_snr"], ft["depth"],
                      ft["duration_ratio"], ft["odd_even_diff"],
                      ft["v_shape"], ft["secondary"]))
