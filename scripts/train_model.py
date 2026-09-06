"""
train_model.py
--------------
Build the training set and train + save the classifier.

Run this ONCE before running the pipeline (or whenever you want to retrain,
e.g. after adding the curated real labelled set from ISRO).

    python train_model.py

What it does:
  1. Generates a balanced set of synthetic light curves (transit / eclipse /
     blend / noise) using the same physics models the pipeline understands.
  2. (Optional) appends any real labelled light curves you provide -- see
     load_real_labelled() below for the hook.
  3. Runs each through preprocess -> BLS -> feature extraction.
  4. Trains a HistGradientBoosting classifier and reports held-out accuracy.
  5. Saves the model to outputs/classifier.joblib so the pipeline can load it.

This is the step whose absence broke the old pipeline -- there, no model was
ever saved, so classification silently did nothing. Here it is explicit.
"""

import os
import sys
import json
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "exopipe"))
# also handle the layout where exopipe/ is a sibling of scripts/
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exopipe"))

import classify  # always needed (lightweight)


HERE = os.path.dirname(os.path.abspath(__file__))
# outputs lives at project root (one level up from scripts/)
OUTDIR = os.path.join(os.path.dirname(HERE), "outputs")
MODEL_PATH = os.path.join(OUTDIR, "classifier.joblib")
METRICS_PATH = os.path.join(OUTDIR, "train_metrics.json")


def load_real_labelled():
    """
    Hook for the curated real labelled dataset (known planets, false positives,
    eclipsing binaries) that the problem statement says will be provided.

    Return a list of dicts {time, flux, label} to mix into training, or an
    empty list if you have none yet. Labels must be among synth.LABELS
    ('transit', 'eclipse', 'blend', 'noise'); map the provided categories onto
    these as appropriate (e.g. 'EB' -> 'eclipse', 'PC'/'KP'/'CP' -> 'transit',
    'FP' -> 'noise' or 'blend').
    """
    return []


def build_training_set(n_per_class=70, seed=5, n_periods=800, verbose=True):
    """Generate synthetic + real labelled light curves and featurise them."""
    import synth
    import preprocess
    import bls
    import features as feat_mod

    rng_seed = seed
    ds = synth.make_dataset(n_per_class=n_per_class, seed=rng_seed)
    ds = ds + load_real_labelled()

    X, y = [], []
    t0 = time.time()
    for i, item in enumerate(ds):
        t, f = preprocess.preprocess(item["time"], item["flux"])
        if t.size < 20:
            continue
        b = bls.run_bls(t, f, n_periods=n_periods)
        ft = feat_mod.extract(t, f, b)
        X.append(feat_mod.to_row(ft))
        y.append(item["label"])
        if verbose and (i + 1) % 40 == 0:
            print("  featurised {}/{}  ({:.0f}s)".format(
                i + 1, len(ds), time.time() - t0))
    return np.array(X, dtype=float), np.array(y)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # reuse a cached feature table if present (saves time on re-runs)
    cache = os.path.join(OUTDIR, "features_cache.npz")
    if os.path.exists(cache):
        print("[train] loading cached features from", cache)
        d = np.load(cache, allow_pickle=True)
        X, y = d["X"], d["y"]
    else:
        print("[train] building training set (this takes a few minutes) ...")
        X, y = build_training_set()
        np.savez(cache, X=X, y=y)

    print("[train] feature table: X={}, classes={}".format(
        X.shape, sorted(set(y))))

    model, info = classify.train(X, y)
    classify.save(model, MODEL_PATH)
    with open(METRICS_PATH, "w") as fh:
        json.dump(info, fh, indent=2, default=str)

    print("\n[train] accuracy: {:.3f}".format(info["report"]["accuracy"]))
    print("[train] model saved ->", MODEL_PATH)
    print("[train] metrics saved ->", METRICS_PATH)


if __name__ == "__main__":
    main()
