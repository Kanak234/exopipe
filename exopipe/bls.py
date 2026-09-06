"""
bls.py
------
Box Least Squares (BLS) period search.

This is the heart of transit detection. The idea is simple and worth being
able to explain out loud:

  A transiting planet makes the star's brightness drop by a fixed amount for a
  fixed slice of time, and it repeats on a fixed period. So we try lots of
  candidate periods. For each period we "fold" the light curve (wrap all the
  data onto one period) and slide a little box -- a dip of some width and some
  start phase -- across it. The box that best explains a dip (low inside, flat
  outside) wins.

  The period whose best box gives the strongest, cleanest dip is our detected
  period.

We use astropy's BoxLeastSquares for the heavy lifting (it is the standard,
well-tested implementation), but we drive it ourselves and pull out the
numbers the problem statement asks for: period, duration, depth, and a
signal strength we turn into an SNR-like number.
"""

import numpy as np

try:
    from astropy.timeseries import BoxLeastSquares
    _HAVE_ASTROPY = True
except Exception:
    _HAVE_ASTROPY = False


def run_bls(time, flux, min_period=0.5, max_period=15.0, n_periods=8000,
            durations=None):
    """
    Search for the best periodic box-shaped dip.

    Parameters
    ----------
    time, flux : arrays
        Cleaned light curve (flux ~ 1.0 baseline).
    min_period, max_period : float
        Period search range in days. TESS single-sector data (~27 d) can only
        confirm a period if it repeats at least twice, so going much past ~13 d
        is optimistic -- we cap at 15 by default.
    n_periods : int
        How finely to sample the period grid. More = finer but slower.
    durations : array or None
        Candidate transit durations (in days) to try. Defaults span a few hours
        to most of a day, which covers realistic transits.

    Returns
    -------
    dict with keys:
        period, t0, duration, depth, snr, power
        plus the full period/power arrays for plotting.
    Returns a dict with period=None if the search cannot run.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    result = dict(period=None, t0=None, duration=None, depth=None,
                  snr=0.0, power=0.0, periods=None, powers=None)

    if time.size < 50 or not np.any(np.isfinite(flux)):
        return result

    if durations is None:
        durations = np.array([0.05, 0.08, 0.12, 0.18, 0.25])

    # span sanity: cannot search periods longer than the data span / 2
    span = time.max() - time.min()
    max_period = min(max_period, span / 2.0)
    if max_period <= min_period:
        max_period = min_period * 1.5

    if not _HAVE_ASTROPY:
        return _run_bls_fallback(time, flux, min_period, max_period,
                                 n_periods, durations)

    try:
        model = BoxLeastSquares(time, flux)
        periods = np.linspace(min_period, max_period, n_periods)
        power = model.power(periods, durations)

        i_best = int(np.argmax(power.power))
        best_period = float(power.period[i_best])
        best_t0 = float(power.transit_time[i_best])
        best_dur = float(power.duration[i_best])
        best_depth = float(power.depth[i_best])

        # Turn the BLS power into an SNR-like number: how far above the typical
        # power level is the peak, measured in robust standard deviations.
        p = power.power
        med = np.median(p)
        mad = np.median(np.abs(p - med))
        std = 1.4826 * mad if mad > 0 else np.std(p)
        snr = float((p[i_best] - med) / std) if std > 0 else 0.0

        result.update(period=best_period, t0=best_t0, duration=best_dur,
                      depth=best_depth, snr=snr,
                      power=float(p[i_best]),
                      periods=np.asarray(power.period),
                      powers=np.asarray(p))
        return result
    except Exception as e:
        # never let a numerical hiccup kill the run
        result["error"] = "{}: {}".format(type(e).__name__, e)
        return result


def _run_bls_fallback(time, flux, min_period, max_period, n_periods,
                      durations):
    """
    A tiny pure-numpy BLS, used only if astropy is unavailable. Slower and
    coarser but keeps the pipeline running. Phase-folds on each trial period
    and looks for the deepest consistent dip.
    """
    result = dict(period=None, t0=None, duration=None, depth=None,
                  snr=0.0, power=0.0, periods=None, powers=None)
    periods = np.linspace(min_period, max_period, min(n_periods, 2000))
    powers = np.zeros_like(periods)
    best = dict(power=-np.inf)

    f = flux - np.median(flux)
    for k, P in enumerate(periods):
        phase = (time % P) / P
        order = np.argsort(phase)
        fp = f[order]
        # try a box covering ~5% of the phase, sliding in 20 steps
        width = max(3, int(0.05 * fp.size))
        best_dip = 0.0
        for start in range(0, fp.size - width, max(1, fp.size // 20)):
            inside = fp[start:start + width]
            dip = -np.mean(inside)  # deeper dip -> larger positive
            if dip > best_dip:
                best_dip = dip
        powers[k] = best_dip
        if best_dip > best["power"]:
            best = dict(power=best_dip, period=P)

    med = np.median(powers)
    mad = np.median(np.abs(powers - med))
    std = 1.4826 * mad if mad > 0 else np.std(powers)
    i_best = int(np.argmax(powers))
    snr = float((powers[i_best] - med) / std) if std > 0 else 0.0

    result.update(period=float(best["period"]), t0=0.0, duration=0.1,
                  depth=float(best["power"]), snr=snr,
                  power=float(powers[i_best]), periods=periods, powers=powers)
    return result


if __name__ == "__main__":
    import synth
    import preprocess

    rng = np.random.default_rng(7)
    t, f, params = synth.make_transit(rng)
    t, f = preprocess.preprocess(t, f)

    res = run_bls(t, f)
    print("TRUE  period={:.4f} d  duration={:.3f} d  depth={:.4f}"
          .format(params["period"], params["duration"], params["depth"]))
    if res["period"]:
        print("BLS   period={:.4f} d  duration={:.3f} d  depth={:.4f}  snr={:.1f}"
              .format(res["period"], res["duration"], res["depth"], res["snr"]))
        err = abs(res["period"] - params["period"]) / params["period"] * 100
        # BLS often locks onto a harmonic (2x or 1/2 period); report that too
        print("period error vs truth: {:.1f}%".format(err))
    else:
        print("BLS found nothing:", res.get("error"))
