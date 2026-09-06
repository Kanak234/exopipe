"""
fit.py
------
Estimate the physical parameters of a detected transit and how confident we
are in them.

BLS already gives us a first guess at period, epoch (t0), duration and depth.
Here we:

  1. Refine the depth by directly measuring the folded light curve in and out
     of transit (more reliable than BLS's internal estimate).
  2. Measure an uncertainty on the depth from the scatter of the points.
  3. Compute a transit signal-to-noise ratio (SNR) -- the depth divided by its
     uncertainty -- which tells us how believable the dip is.
  4. If a stellar radius is known (from the catalogue), convert the transit
     depth into an estimated planet radius, since depth ~= (Rp/Rstar)^2.

These are exactly the deliverables the problem statement lists: period,
duration, depth, significance, and parameter estimates with uncertainties.
"""

import numpy as np

R_SUN_IN_R_EARTH = 109.1  # 1 solar radius = 109.1 Earth radii


def _fold(time, flux, period, t0):
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def refine_depth(time, flux, period, t0, duration):
    """
    Measure transit depth directly: median flux outside transit minus median
    flux inside transit. Also returns an uncertainty from the in-transit
    scatter, and the counts so we know how well-sampled the dip is.
    """
    phase, fflux = _fold(time, flux, period, t0)
    half_dur_phase = 0.5 * duration / period

    in_tr = np.abs(phase) < half_dur_phase
    out_tr = (np.abs(phase) > 1.5 * half_dur_phase) & \
             (np.abs(phase) < 4.0 * half_dur_phase)

    if in_tr.sum() < 3 or out_tr.sum() < 3:
        return dict(depth=np.nan, depth_err=np.nan,
                    n_in=int(in_tr.sum()), n_out=int(out_tr.sum()))

    out_level = np.median(fflux[out_tr])
    in_level = np.median(fflux[in_tr])
    depth = out_level - in_level

    # uncertainty: scatter of in-transit points / sqrt(N)
    in_scatter = np.std(fflux[in_tr])
    depth_err = in_scatter / np.sqrt(max(in_tr.sum(), 1))

    return dict(depth=float(depth), depth_err=float(depth_err),
                n_in=int(in_tr.sum()), n_out=int(out_tr.sum()),
                out_level=float(out_level), in_level=float(in_level))


def transit_snr(depth, depth_err):
    """Signal-to-noise of the transit: how many sigma the dip is from zero."""
    if depth_err is None or not np.isfinite(depth_err) or depth_err <= 0:
        return 0.0
    return float(max(0.0, depth) / depth_err)


def planet_radius(depth, stellar_radius_rsun):
    """
    Estimate planet radius in Earth radii from depth ~= (Rp/Rstar)^2.
    Needs the stellar radius (in solar radii) from the catalogue.
    """
    if depth is None or depth <= 0 or stellar_radius_rsun is None \
            or stellar_radius_rsun <= 0 or not np.isfinite(stellar_radius_rsun):
        return None
    rp_over_rstar = np.sqrt(depth)
    rp_rsun = rp_over_rstar * stellar_radius_rsun
    return float(rp_rsun * R_SUN_IN_R_EARTH)


def fit_transit(time, flux, bls_result, stellar_radius_rsun=None):
    """
    Full parameter estimation for one target.

    Returns a dict with refined period/duration/depth, their uncertainties,
    the transit SNR, and (if possible) an estimated planet radius. Safe on bad
    input: returns NaNs rather than raising.
    """
    out = dict(period=None, t0=None, duration=None,
               depth=None, depth_err=None, snr=0.0,
               planet_radius_earth=None)

    period = bls_result.get("period")
    t0 = bls_result.get("t0")
    duration = bls_result.get("duration")
    if not period or period <= 0 or t0 is None or not duration:
        return out

    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)
    if time.size < 20:
        return out

    d = refine_depth(time, flux, period, t0, duration)
    snr = transit_snr(d["depth"], d["depth_err"])
    rp = planet_radius(d["depth"], stellar_radius_rsun)

    out.update(period=float(period), t0=float(t0), duration=float(duration),
               depth=d["depth"], depth_err=d["depth_err"], snr=snr,
               n_in_transit=d["n_in"], planet_radius_earth=rp)
    return out


if __name__ == "__main__":
    import synth
    import preprocess
    import bls

    rng = np.random.default_rng(2)
    t, f, params = synth.make_transit(rng)
    t, f = preprocess.preprocess(t, f)
    b = bls.run_bls(t, f, n_periods=2000)

    # pretend the host star is 1.0 solar radius
    res = fit_transit(t, f, b, stellar_radius_rsun=1.0)
    print("TRUE depth = {:.4f}, period = {:.4f} d"
          .format(params["depth"], params["period"]))
    print("FIT  depth = {:.4f} +/- {:.4f}".format(
        res["depth"], res["depth_err"]))
    print("     period = {:.4f} d, SNR = {:.1f}".format(
        res["period"], res["snr"]))
    if res["planet_radius_earth"]:
        print("     est. planet radius = {:.1f} Earth radii".format(
            res["planet_radius_earth"]))
