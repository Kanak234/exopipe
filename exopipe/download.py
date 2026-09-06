"""
download.py
-----------
Fetch real TESS light curves for a list of TIC IDs.

This runs on YOUR machine (where the MAST archive is reachable), not in the
sandbox. It is built defensively because downloading is where the old pipeline
fell over: some TIC IDs simply have no light curve products, and a single
failure used to look like a crash.

Key ideas:

  * check_available() asks MAST whether a target has any light curve BEFORE we
    try to download it. This is how we avoid the "No light curves found" spam
    -- we filter the list down to targets that actually have data.
  * download_one() pulls the light curve, prefers the official SPOC PDCSAP
    flux (already corrected for instrument systematics), and returns clean
    time/flux arrays.
  * Every call is wrapped so one bad target never stops the batch. Failures
    are collected and reported, not raised.

Requires: lightkurve (pip install lightkurve).
"""

import numpy as np

try:
    from lightkurve import search_lightcurve
    _HAVE_LK = True
except Exception:
    _HAVE_LK = False


def _require_lk():
    if not _HAVE_LK:
        raise RuntimeError(
            "lightkurve is not installed. Run: pip install lightkurve")


def check_available(tic, mission="TESS", author="SPOC"):
    """
    Return True if this TIC has at least one downloadable light curve.

    Doing this check first is what keeps the run clean: we only attempt
    downloads for targets that have data, so the log is not full of errors.
    """
    _require_lk()
    try:
        res = search_lightcurve("TIC {}".format(tic), mission=mission,
                                author=author)
        if len(res) == 0:
            # fall back to any author, not just SPOC
            res = search_lightcurve("TIC {}".format(tic), mission=mission)
        return len(res) > 0
    except Exception:
        return False


def download_one(tic, mission="TESS", author="SPOC", flux_column="pdcsap_flux"):
    """
    Download and return (time, flux) for one TIC, or (None, None, reason) on
    failure.

    Returns
    -------
    time : ndarray or None
    flux : ndarray or None
    info : str
        "ok" on success, otherwise a short reason.
    """
    _require_lk()
    try:
        res = search_lightcurve("TIC {}".format(tic), mission=mission,
                                author=author)
        if len(res) == 0:
            res = search_lightcurve("TIC {}".format(tic), mission=mission)
        if len(res) == 0:
            return None, None, "no light curve products"

        lc = res[0].download()
        if lc is None:
            return None, None, "download returned nothing"

        # prefer PDCSAP (corrected) flux; fall back to default flux
        try:
            lc = lc.remove_nans()
        except Exception:
            pass

        time = np.asarray(lc.time.value, dtype=float)
        try:
            flux = np.asarray(lc[flux_column].value, dtype=float)
        except Exception:
            flux = np.asarray(lc.flux.value, dtype=float)

        good = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[good], flux[good]
        if time.size < 50:
            return None, None, "too few valid points"

        return time, flux, "ok"
    except Exception as e:
        return None, None, "{}: {}".format(type(e).__name__, str(e)[:80])


def download_batch(tics, mission="TESS", author="SPOC", prefilter=True,
                   verbose=True):
    """
    Download a list of TICs, skipping ones with no data.

    Returns
    -------
    results : dict
        tic -> (time, flux) for every successful download.
    report : dict
        bookkeeping: which succeeded, which were skipped, which failed and why.
    """
    _require_lk()
    results = {}
    report = dict(ok=[], skipped_no_data=[], failed=[])

    for i, tic in enumerate(tics):
        if verbose:
            print("  [{}/{}] TIC {} ...".format(i + 1, len(tics), tic))

        if prefilter and not check_available(tic, mission=mission,
                                             author=author):
            report["skipped_no_data"].append(tic)
            if verbose:
                print("      skip: no light curve in archive")
            continue

        time, flux, info = download_one(tic, mission=mission, author=author)
        if info == "ok":
            results[tic] = (time, flux)
            report["ok"].append(tic)
            if verbose:
                print("      ok: {} points".format(time.size))
        else:
            report["failed"].append((tic, info))
            if verbose:
                print("      fail: {}".format(info))

    if verbose:
        print("\nsummary: {} ok, {} skipped (no data), {} failed".format(
            len(report["ok"]), len(report["skipped_no_data"]),
            len(report["failed"])))
    return results, report


if __name__ == "__main__":
    # This needs network access to MAST, so it only really runs on your laptop.
    if not _HAVE_LK:
        print("lightkurve not installed here; this module is meant to run on "
              "your machine where MAST is reachable.")
    else:
        # a known-good TIC with TESS data
        t, f, info = download_one(307210830)
        print("download_one TIC 307210830 ->", info,
              "points:" if t is not None else "", t.size if t is not None else "")
