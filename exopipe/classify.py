"""
classify.py
-----------
The classifier that decides whether a detected dip is a transit, an eclipse,
a blend, or just noise.

We use a Gradient Boosting tree (scikit-learn's HistGradientBoostingClassifier).
Why a tree and not a deep net:

  * It trains in seconds on our feature table, on a laptop, with no GPU.
  * It works well with a handful of hand-built features like ours.
  * It is easy to explain to judges: a series of decision trees, each fixing
    the mistakes of the last, voting on the class.
  * It gives calibrated-ish class probabilities, which we report as a
    confidence level -- exactly what the problem statement asks for.

The model is trained on labelled light curves (synthetic for now, plus any
curated real set later) and SAVED to disk. The pipeline loads it back. This is
the part that was broken before -- the old run never had a saved model, so it
silently fell back to nothing. Here, training and saving are one explicit step.
"""

import os
import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False

import features as feat_mod


def build_feature_table(dataset, preprocess_mod, bls_mod, verbose=True):
    """
    Turn a list of labelled light curves into an (X, y) feature table.

    For each light curve: preprocess -> BLS -> extract features. This is the
    same path the real pipeline uses, so the features the model trains on match
    the features it will see at prediction time.
    """
    X, y = [], []
    n = len(dataset)
    for i, item in enumerate(dataset):
        t, f = preprocess_mod.preprocess(item["time"], item["flux"])
        if t.size < 20:
            continue
        b = bls_mod.run_bls(t, f, n_periods=2000)
        ft = feat_mod.extract(t, f, b)
        X.append(feat_mod.to_row(ft))
        y.append(item["label"])
        if verbose and (i + 1) % 50 == 0:
            print("  [features] {}/{} light curves processed"
                  .format(i + 1, n), end="\r")
    if verbose:
        print()
    return np.array(X, dtype=float), np.array(y)


def train(X, y, seed=42, verbose=True):
    """
    Train the classifier and report how well it does on a held-out test split.

    Returns (model, report_dict). The report is kept so we can print it and
    drop the numbers straight into the methodology report.
    """
    if not _HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required to train the classifier")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=None,
        l2_regularization=1.0,
        random_state=seed,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True,
                                   zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=sorted(set(y)))

    if verbose:
        print(classification_report(y_te, y_pred, zero_division=0))
        print("confusion matrix (rows=true, cols=pred), labels={}:"
              .format(sorted(set(y))))
        print(cm)

    return model, dict(report=report, confusion=cm.tolist(),
                       labels=sorted(set(y)),
                       n_train=len(y_tr), n_test=len(y_te))


def save(model, path):
    """Persist the trained model so the pipeline can load it later."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load(path):
    """Load a saved model, or return None if it is not there."""
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


def predict(model, feature_row):
    """
    Classify one feature row.

    Returns (label, confidence, all_probabilities). Confidence is the
    probability of the winning class -- this is the "confidence level" the
    problem statement asks us to report.
    """
    x = np.asarray(feature_row, dtype=float).reshape(1, -1)
    label = model.predict(x)[0]
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    conf = float(proba[classes.index(label)])
    prob_map = {c: float(p) for c, p in zip(classes, proba)}
    return label, conf, prob_map


if __name__ == "__main__":
    # Full self-contained train + evaluate on synthetic data.
    import synth
    import preprocess
    import bls

    print("building synthetic dataset ...")
    ds = synth.make_dataset(n_per_class=120, seed=1)
    print("extracting features from {} light curves ...".format(len(ds)))
    X, y = build_feature_table(ds, preprocess, bls)
    print("feature table: X={}, classes={}".format(X.shape, sorted(set(y))))

    model, info = train(X, y)
    print("\ntrained on {} / tested on {}".format(
        info["n_train"], info["n_test"]))

    out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "outputs", "classifier.joblib")
    save(model, out)
    print("saved model ->", out, "exists:", os.path.exists(out))
