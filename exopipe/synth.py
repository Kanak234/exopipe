"""
synth.py
--------
Generate realistic synthetic TESS-style light curves.

We need labelled training data for the classifier, and we also want a way to
test every stage of the pipeline without depending on a live download. So we
build light curves from simple physical models and add TESS-like noise.

Four classes, matching what the problem statement asks us to separate:

  transit : a planet crossing the star. Shallow (<~3%), flat-bottomed dip,
            U-shaped, same depth every period.
  eclipse : an eclipsing binary. Deeper, often V-shaped, and frequently shows
            a secondary eclipse half a period later.
  blend   : a real eclipse/transit diluted by light from a nearby star in the
            aperture (crowded field). Looks like a transit but much shallower
            and the depth does not match the stellar radius.
  noise   : no real periodic dip -- just correlated stellar noise / starspots.
            This is the "other / false alarm" bucket.

The numbers here (cadence, noise levels, transit shapes) are picked to look
like TESS 2-minute data so the classifier learns features that transfer to the
real light curves.
"""

import numpy as np


# TESS short-cadence is 2 minutes. We work in days.
CADENCE_DAYS = 2.0 / (60.0 * 24.0)


def _time_grid(days=27.0, cadence=CADENCE_DAYS):
    """A TESS-sector-length time axis (~27 days) at the given cadence."""
    n = int(days / cadence)
    return np.arange(n) * cadence


def _add_noise(flux, rng, white=1.2e-3, red_amp=5e-4):
    """
    Add TESS-like noise: white (per-point) plus a slow red/correlated wander
    that mimics instrumental drift and stellar variability.
    """
    white_noise = rng.normal(0.0, white, size=flux.size)

    # red noise: smooth random walk, low frequency
    steps = rng.normal(0.0, 1.0, size=flux.size)
    walk = np.cumsum(steps)
    walk = walk / (np.max(np.abs(walk)) + 1e-9) * red_amp
    return flux + white_noise + walk


def _trapezoid_dip(t, period, t0, duration, depth, ingress_frac=0.15):
    """
    A trapezoidal dip repeated every `period`. ingress_frac controls how
    rounded the shoulders are: small -> flat-bottomed (planet-like U),
    large -> pointy (V-shaped, more eclipse-like).
    """
    flux = np.ones_like(t)
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    half = 0.5 * duration
    ingress = ingress_frac * duration

    inside = np.abs(phase) < half
    # full depth in the flat core
    core = np.abs(phase) < (half - ingress)
    flux[core] = 1.0 - depth

    # linear ramps on ingress/egress
    ramp = inside & ~core
    if np.any(ramp):
        d = (half - np.abs(phase[ramp])) / max(ingress, 1e-9)
        flux[ramp] = 1.0 - depth * np.clip(d, 0.0, 1.0)
    return flux


def make_transit(rng, days=27.0):
    """Planet transit: shallow, flat-bottomed, no secondary."""
    t = _time_grid(days)
    period = rng.uniform(2.0, 12.0)
    t0 = rng.uniform(0, period)
    duration = rng.uniform(0.05, 0.2)         # days
    depth = rng.uniform(0.0008, 0.03)         # 0.08% - 3%
    flux = _trapezoid_dip(t, period, t0, duration, depth, ingress_frac=0.12)
    flux = _add_noise(flux, rng)
    params = dict(period=period, t0=t0, duration=duration, depth=depth)
    return t, flux, params


def make_eclipse(rng, days=27.0):
    """Eclipsing binary: deep, V-ish, with a secondary eclipse."""
    t = _time_grid(days)
    period = rng.uniform(1.5, 10.0)
    t0 = rng.uniform(0, period)
    duration = rng.uniform(0.08, 0.3)
    depth = rng.uniform(0.04, 0.25)           # much deeper than a planet
    flux = _trapezoid_dip(t, period, t0, duration, depth, ingress_frac=0.4)
    # secondary eclipse, shallower, half a period later
    sec_depth = depth * rng.uniform(0.1, 0.5)
    flux *= _trapezoid_dip(t, period, t0 + 0.5 * period, duration * 0.9,
                           sec_depth, ingress_frac=0.4)
    flux = _add_noise(flux, rng)
    params = dict(period=period, t0=t0, duration=duration, depth=depth)
    return t, flux, params


def make_blend(rng, days=27.0):
    """
    Diluted eclipse in a crowded field: a real deep eclipse whose depth is
    washed out by a brighter neighbour's light in the aperture. Looks shallow
    like a transit but the shape is off.
    """
    t = _time_grid(days)
    period = rng.uniform(1.5, 10.0)
    t0 = rng.uniform(0, period)
    duration = rng.uniform(0.08, 0.3)
    true_depth = rng.uniform(0.05, 0.25)
    dilution = rng.uniform(0.7, 0.95)          # fraction of light from blend
    observed_depth = true_depth * (1.0 - dilution)
    flux = _trapezoid_dip(t, period, t0, duration, observed_depth,
                          ingress_frac=0.35)
    flux = _add_noise(flux, rng, white=1.5e-3)
    params = dict(period=period, t0=t0, duration=duration,
                  depth=observed_depth)
    return t, flux, params


def make_noise(rng, days=27.0):
    """No real transit: starspot modulation + noise, no clean periodic dip."""
    t = _time_grid(days)
    # slow sinusoidal starspot signal at a random rotation period
    rot = rng.uniform(3.0, 15.0)
    amp = rng.uniform(2e-4, 2e-3)
    flux = 1.0 + amp * np.sin(2 * np.pi * t / rot + rng.uniform(0, 2 * np.pi))
    flux = _add_noise(flux, rng, white=1.3e-3, red_amp=8e-4)
    params = dict(period=np.nan, t0=np.nan, duration=np.nan, depth=0.0)
    return t, flux, params


LABELS = ["transit", "eclipse", "blend", "noise"]
_MAKERS = {
    "transit": make_transit,
    "eclipse": make_eclipse,
    "blend": make_blend,
    "noise": make_noise,
}


def make_one(label, rng, days=27.0):
    """Make a single labelled light curve of the requested class."""
    return _MAKERS[label](rng, days=days)


def make_dataset(n_per_class=150, seed=42, days=27.0):
    """
    Build a balanced labelled dataset.

    Returns a list of dicts: {time, flux, label, params}. We keep it as a list
    of raw light curves (not yet featurised) so the same data can feed both the
    feature extractor and any visual checks.
    """
    rng = np.random.default_rng(seed)
    data = []
    for label in LABELS:
        for _ in range(n_per_class):
            t, flux, params = make_one(label, rng, days=days)
            data.append(dict(time=t, flux=flux, label=label, params=params))
    rng.shuffle(data)
    return data


if __name__ == "__main__":
    ds = make_dataset(n_per_class=5)
    print("made {} light curves".format(len(ds)))
    for label in LABELS:
        ex = next(d for d in ds if d["label"] == label)
        f = ex["flux"]
        print("  {:8s} n={:5d}  flux range [{:.4f}, {:.4f}]  depth~{}"
              .format(label, f.size, f.min(), f.max(),
                      round(ex["params"]["depth"], 4)))
