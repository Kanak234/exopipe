"""
run_pipeline.py
---------------
The main entry point. Ties everything together:

    catalogue -> pick targets -> download -> preprocess -> BLS ->
    features -> classify -> fit parameters -> visualise -> save results

Two ways to run it:

  1. On real data (your laptop, MAST reachable):
       python run_pipeline.py --catalog /path/to/exo_CTL_...csv --n 20
     Picks bright targets from the catalogue, downloads their TESS light
     curves, and runs the full analysis on each.

  2. By explicit TIC list:
       python run_pipeline.py --tic 307210830 231663901
     Skips the catalogue and just analyses the TICs you name.

  3. Demo / offline (no network needed, good for testing):
       python run_pipeline.py --demo 10
     Generates synthetic light curves and runs the full pipeline on them, so
     you can see everything work without downloading anything.

Results are written to outputs/results.json and a detection plot (and GIF for
the first few) per target.
"""

import os
import sys
import json
import argparse

import numpy as np

# allow running from anywhere -- modules live in ../exopipe relative to scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exopipe"))

import preprocess
import bls
import features as feat_mod
import fit as fit_mod
import classify
import visualize


HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "outputs")
MODEL_PATH = os.path.join(OUTDIR, "classifier.joblib")


def analyse_one(time, flux, model, tic=None, stellar_radius=None,
                make_plot=True, make_animation=False):
    """
    Run the full analysis chain on a single light curve.

    Returns a result dict with the class, confidence, fitted parameters and
    paths to any figures. Never raises on a single bad target -- problems are
    recorded in the result instead.
    """
    result = dict(tic=tic, status="ok", label=None, confidence=None,
                  period=None, duration=None, depth=None, depth_err=None,
                  snr=None, planet_radius_earth=None, plot=None, gif=None)

    try:
        raw = np.asarray(flux, dtype=float).copy()
        t, f = preprocess.preprocess(time, flux)
        if t.size < 20:
            result["status"] = "too few points after cleaning"
            return result

        b = bls.run_bls(t, f, n_periods=3000)
        ft = feat_mod.extract(t, f, b)
        row = feat_mod.to_row(ft)

        if model is not None:
            label, conf, _ = classify.predict(model, row)
        else:
            # no model available: fall back to a simple SNR rule so we still
            # return something sensible
            label = "transit" if ft["bls_snr"] > 10 else "noise"
            conf = float(min(1.0, ft["bls_snr"] / 30.0))

        fr = fit_mod.fit_transit(t, f, b, stellar_radius_rsun=stellar_radius)

        result.update(
            label=label, confidence=conf,
            period=fr["period"], duration=fr["duration"],
            depth=fr["depth"], depth_err=fr["depth_err"],
            snr=fr["snr"], planet_radius_earth=fr["planet_radius_earth"])

        if make_plot:
            png = os.path.join(OUTDIR, "detection_{}.png".format(tic or "x"))
            visualize.plot_detection(t, f, b, fr, label, conf or 0.0,
                                     tic=tic, outpath=png)
            result["plot"] = png

        if make_animation:
            # resample raw to cleaned length for a tidy gif
            if raw.size != f.size:
                raw_g = np.interp(np.linspace(0, 1, f.size),
                                  np.linspace(0, 1, raw.size), raw)
            else:
                raw_g = raw
            gif = os.path.join(OUTDIR, "detection_{}.gif".format(tic or "x"))
            visualize.make_gif(t, f, raw_g, b, fr, label, tic=tic,
                               outpath=gif)
            result["gif"] = gif

    except Exception as e:
        result["status"] = "error: {}: {}".format(type(e).__name__,
                                                  str(e)[:120])
    return result


def run_demo(n, model, n_gifs=2):
    """Run the pipeline on synthetic light curves -- no network needed."""
    import synth
    rng = np.random.default_rng(123)
    labels_cycle = (synth.LABELS * (n // 4 + 1))[:n]
    results = []
    for i, lab in enumerate(labels_cycle):
        t, f, params = synth.make_one(lab, rng)
        res = analyse_one(t, f, model, tic="demo{}".format(i),
                          stellar_radius=1.0, make_plot=True,
                          make_animation=(i < n_gifs))
        res["true_label"] = lab
        results.append(res)
        print("  demo {:2d}: true={:8s} pred={:8s} conf={} P={}".format(
            i, lab, str(res["label"]),
            round(res["confidence"], 2) if res["confidence"] else None,
            round(res["period"], 3) if res["period"] else None))
    return results


def run_real(catalog, tics, n, model, n_gifs=2):
    """Run on real TESS data via download (needs MAST access)."""
    import download

    radii = {}
    if not tics:
        import catalog as catmod
        print("[catalog] selecting targets from", catalog)
        sel = catmod.select_targets(catalog, max_targets=n * 3)
        tics = [int(x) for x in sel["ID"].tolist()][: n * 3]
        radii = {int(r.ID): float(r.rad) for r in sel.itertuples()
                 if np.isfinite(r.rad)}
        print("[catalog] {} candidate TICs selected".format(len(tics)))

    print("[download] fetching light curves (skipping targets with no data)")
    lcs, dlreport = download.download_batch(tics, prefilter=True)

    results = []
    done = 0
    for tic, (t, f) in lcs.items():
        res = analyse_one(t, f, model, tic=tic,
                          stellar_radius=radii.get(tic),
                          make_plot=True, make_animation=(done < n_gifs))
        results.append(res)
        done += 1
        print("  TIC {}: class={} conf={} P={}".format(
            tic, res["label"],
            round(res["confidence"], 2) if res["confidence"] else None,
            round(res["period"], 3) if res["period"] else None))
        if done >= n:
            break

    results.append({"_download_report": dlreport})
    return results


def main():
    ap = argparse.ArgumentParser(description="Exoplanet transit detection pipeline")
    ap.add_argument("--catalog", help="path to TIC CTL catalogue csv")
    ap.add_argument("--tic", nargs="+", type=int, help="explicit TIC IDs")
    ap.add_argument("--n", type=int, default=10, help="number of targets")
    ap.add_argument("--demo", type=int, metavar="N",
                    help="run on N synthetic light curves (no network)")
    ap.add_argument("--gifs", type=int, default=2,
                    help="how many animated GIFs to make")
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)

    model = classify.load(MODEL_PATH)
    if model is None:
        print("[warn] no trained model at {}.".format(MODEL_PATH))
        print("       run train_model.py first for full classification.")
        print("       continuing with a simple SNR-based fallback.")
    else:
        print("[ok] loaded classifier from", MODEL_PATH)

    if args.demo:
        results = run_demo(args.demo, model, n_gifs=args.gifs)
    else:
        results = run_real(args.catalog, args.tic, args.n, model,
                           n_gifs=args.gifs)

    outpath = os.path.join(OUTDIR, "results.json")
    with open(outpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("\nsaved results ->", outpath)


if __name__ == "__main__":
    main()
