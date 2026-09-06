"""
preprocess.py
-------------
Clean a raw light curve so the transit search has a fair chance.

Raw TESS light curves come with three nuisances we have to deal with before
searching for a tiny dip:

  1. Different stars have different absolute brightness. We only care about the
     *relative* change, so we normalise every light curve to hover around 1.0.

  2. Cosmic rays, momentum dumps and other glitches throw in single crazy
     points (huge spikes up or down). A sigma-clip removes them so they do not
     fool the dip search.

  3. Slow trends -- the star slowly brightens/dims over the sector from
     instrumental drift or stellar rotation. If we leave this in, the search
     locks onto the trend instead of the transit. We flatten it out with a
     running-median filter, being careful not to flatten the transit itself.

Everything here is deliberately simple and explainable: median, MAD, and a
sliding median window. No black boxes.
"""

import numpy as np


def normalize(flux):
    """Divide by the median so the baseline sits at ~1.0."""
    med = np.nanmedian(flux)
    if med == 0 or not np.isfinite(med):
        return flux.copy()
    return flux / med


def sigma_clip(time, flux, sigma=5.0, iters=3):
    """
    Remove outlier points using the median and MAD (median absolute
    deviation). MAD is used instead of the standard deviation because it is
    robust -- a few wild outliers do not inflate it, so we do not accidentally
    keep them.

    We clip a few times because removing the worst points changes the MAD and
    can reveal slightly-less-wild outliers underneath.
    """
    t = np.asarray(time, dtype=float)
    f = np.asarray(flux, dtype=float)
    keep = np.isfinite(f) & np.isfinite(t)

    for _ in range(iters):
        med = np.median(f[keep])
        mad = np.median(np.abs(f[keep] - med))
        # 1.4826 makes MAD comparable to a standard deviation for Gaussian data
        std = 1.4826 * mad if mad > 0 else np.std(f[keep])
        if std == 0:
            break
        within = np.abs(f - med) < sigma * std
        new_keep = keep & within
        if new_keep.sum() == keep.sum():
            break  # nothing more to remove
        keep = new_keep

    return t[keep], f[keep]


def running_median(flux, window):
    """
    Median in a sliding window. Used to estimate the slow trend. Window is in
    number of points and is forced odd so each point has a symmetric window.
    """
    n = flux.size
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2

    trend = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        trend[i] = np.median(flux[lo:hi])
    return trend


def flatten(time, flux, window_days=0.5, cadence_days=(2.0 / 1440.0)):
    """
    Remove the slow trend by dividing out a running median.

    window_days sets how long the smoothing window is. It must be comfortably
    LONGER than a transit (hours) so the filter treats the transit as a feature
    to keep, not a trend to remove. Half a day is a good default for TESS.
    """
    window_pts = int(window_days / cadence_days)
    trend = running_median(flux, window_pts)
    # guard against divide-by-zero
    trend[trend == 0] = np.nanmedian(flux)
    flat = flux / trend
    return time, flat


def preprocess(time, flux, sigma=5.0, window_days=0.5,
               cadence_days=(2.0 / 1440.0)):
    """
    Full cleaning pipeline: normalise -> sigma clip -> flatten.

    Returns cleaned (time, flux). Designed to never raise on weird input --
    empty or all-NaN light curves come back as empty arrays rather than
    crashing the whole run.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if time.size == 0 or flux.size == 0:
        return np.array([]), np.array([])

    if not np.any(np.isfinite(flux)):
        return np.array([]), np.array([])

    flux = normalize(flux)
    time, flux = sigma_clip(time, flux, sigma=sigma)

    if time.size < 10:
        # too few points left to flatten meaningfully; return what we have
        return time, flux

    time, flux = flatten(time, flux, window_days=window_days,
                         cadence_days=cadence_days)
    return time, flux


if __name__ == "__main__":
    # Verify on a synthetic transit: after preprocessing, baseline should be
    # ~1.0 and the transit dip should still be there.
    import synth
    rng = np.random.default_rng(0)
    t, f, params = synth.make_transit(rng)

    # inject a slow trend and a couple of spikes to make it realistic
    f = f * (1.0 + 0.01 * np.sin(2 * np.pi * t / 20.0))
    f[100] += 0.05
    f[2000] -= 0.04

    t2, f2 = preprocess(t, f)
    print("before: n={}, median={:.4f}, min={:.4f}"
          .format(t.size, np.median(f), f.min()))
    print("after:  n={}, median={:.4f}, min={:.4f}"
          .format(t2.size, np.median(f2), f2.min()))
    print("injected transit depth ~ {:.4f}".format(params["depth"]))
    print("deepest dip still present after cleaning: {:.4f}"
          .format(1.0 - f2.min()))
