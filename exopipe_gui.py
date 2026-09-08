"""
exopipe_gui.py
--------------
Web GUI for the exoplanet detection pipeline.

A point-and-click front end so you can run the whole pipeline without touching
the command line: pick a data source, press Run, and watch the light curve get
cleaned, searched, classified, and fitted -- with the plots and parameters
shown right there in the browser.

Run it with:

    streamlit run exopipe_gui.py

Then open the link it prints (usually http://localhost:8501).

Requires (in addition to the pipeline's own deps):
    pip install streamlit

The GUI imports the same exopipe modules the command-line pipeline uses, so
whatever works here works there and vice versa.
"""

import os
import sys
import io
import json
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import streamlit as st
    _HAVE_STREAMLIT = True
except Exception:
    _HAVE_STREAMLIT = False
    st = None

if not _HAVE_STREAMLIT:
    print("[warn] streamlit is not installed. Web GUI skipped.")
    if __name__ == "__main__":
        sys.exit(0)

# --- make the exopipe package importable -------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "exopipe"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import synth
import preprocess
import bls
import features as feat_mod
import fit as fit_mod
import classify

# shared modules + optional interactive plotting
try:
    import store as _store
except Exception:
    _store = None
try:
    import exporter as _exporter
except Exception:
    _exporter = None
try:
    import plotly.graph_objects as go
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False

MODEL_PATH = os.path.join(HERE, "outputs", "classifier.joblib")


# -----------------------------------------------------------------------------
# small helpers
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model():
    """Load the trained classifier once and keep it around."""
    return classify.load(MODEL_PATH)


def fold(time, flux, period, t0):
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def analyse(time, flux, model, stellar_radius=None):
    """Run the full chain on one light curve and return everything we need."""
    raw = np.asarray(flux, dtype=float).copy()
    t, f = preprocess.preprocess(time, flux)
    if t.size < 20:
        return None
    b = bls.run_bls(t, f, n_periods=3000)
    ft = feat_mod.extract(t, f, b)
    row = feat_mod.to_row(ft)

    if model is not None:
        label, conf, probs = classify.predict(model, row)
    else:
        label = "transit" if ft["bls_snr"] > 10 else "noise"
        conf = float(min(1.0, ft["bls_snr"] / 30.0))
        probs = {label: conf}

    fr = fit_mod.fit_transit(t, f, b, stellar_radius_rsun=stellar_radius)
    return dict(t=t, f=f, raw=raw, bls=b, feats=ft, label=label,
                conf=conf, probs=probs, fit=fr)


def make_figure(res, title="target"):
    """Three-panel detection figure as a matplotlib Figure."""
    t, f, b, fr = res["t"], res["f"], res["bls"], res["fit"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    ax = axes[0]
    ax.plot(t, f, ".", ms=1.5, color="#1b2a4a", alpha=0.5)
    period, t0 = b.get("period"), b.get("t0")
    if period and t0 is not None:
        n0 = int((t.min() - t0) / period)
        n1 = int((t.max() - t0) / period) + 1
        for n in range(n0, n1 + 1):
            tc = t0 + n * period
            if t.min() <= tc <= t.max():
                ax.axvline(tc, color="#e8743b", lw=0.8, alpha=0.6)
    ax.set_title("cleaned light curve", fontsize=10)
    ax.set_xlabel("time (days)"); ax.set_ylabel("flux")

    ax = axes[1]
    if b.get("periods") is not None:
        ax.plot(b["periods"], b["powers"], lw=0.8, color="#1b2a4a")
        if period:
            ax.axvline(period, color="#e8743b", lw=1.2)
    ax.set_title("period search", fontsize=10)
    ax.set_xlabel("period (days)"); ax.set_ylabel("BLS power")

    ax = axes[2]
    if period and t0 is not None:
        phase, fflux = fold(t, f, period, t0)
        ax.plot(phase, fflux, ".", ms=1.5, color="#1b2a4a", alpha=0.4)
        depth, dur = fr.get("depth"), fr.get("duration")
        if depth and dur:
            hw = 0.5 * dur / period
            ax.plot([-0.5, -hw, -hw, hw, hw, 0.5],
                    [1, 1, 1 - depth, 1 - depth, 1, 1],
                    color="#e8743b", lw=2)
        ax.set_xlim(-0.5, 0.5)
    ax.set_title("phase-folded", fontsize=10)
    ax.set_xlabel("phase"); ax.set_ylabel("flux")

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="exopipe", page_icon="*", layout="wide")

st.title("exopipe")
st.caption("AI-enabled detection of exoplanets from noisy TESS light curves "
           "- BAH 2026, Problem 07")

model = get_model()
if model is None:
    st.warning("No trained model found at outputs/classifier.joblib. "
               "Run `python scripts/train_model.py` first. The app still runs "
               "with a simple SNR-based fallback in the meantime.")
else:
    st.success("Classifier loaded. Ready to analyse light curves.")

st.divider()

# --- input source ------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Where should the light curve come from?",
        ["Synthetic demo", "Upload CSV (time,flux)", "TESS download by TIC"],
        help="Synthetic demo needs no internet. TESS download needs "
             "lightkurve and a network connection.")

    stellar_radius = st.number_input(
        "Stellar radius (solar radii, optional)",
        min_value=0.0, value=0.0, step=0.1,
        help="If known, lets the pipeline estimate the planet radius.")
    stellar_radius = stellar_radius if stellar_radius > 0 else None

    run = st.button("Run detection", type="primary", use_container_width=True)


# --- gather the light curve(s) ----------------------------------------------
def get_lightcurves():
    """Return a list of (name, time, flux) based on the chosen source."""
    out = []

    if source == "Synthetic demo":
        kind = st.session_state.get("demo_kind", "transit")
        rng = np.random.default_rng(
            int(st.session_state.get("demo_seed", 0)))
        t, f, _ = synth.make_one(kind, rng)
        out.append(("synthetic {}".format(kind), t, f))

    elif source == "Upload CSV (time,flux)":
        up = st.session_state.get("uploaded")
        if up is not None:
            data = np.genfromtxt(io.StringIO(up), delimiter=",")
            if data.ndim == 2 and data.shape[1] >= 2:
                out.append(("uploaded", data[:, 0], data[:, 1]))

    elif source == "TESS download by TIC":
        tic = st.session_state.get("tic_id", "").strip()
        if tic:
            try:
                import download as dl
                t, f, info = dl.download_one(int(tic))
                if info == "ok":
                    out.append(("TIC {}".format(tic), t, f))
                else:
                    st.error("Could not download TIC {}: {}".format(tic, info))
            except Exception as e:
                st.error("Download failed: {}".format(e))
    return out


# source-specific extra controls (shown in main area)
if source == "Synthetic demo":
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["demo_kind"] = st.selectbox(
            "Signal type", synth.LABELS, index=0)
    with c2:
        st.session_state["demo_seed"] = st.number_input(
            "Random seed", min_value=0, value=0, step=1)

elif source == "Upload CSV (time,flux)":
    f = st.file_uploader("CSV with two columns: time, flux", type=["csv"])
    if f is not None:
        st.session_state["uploaded"] = f.getvalue().decode("utf-8",
                                                           errors="ignore")

elif source == "TESS download by TIC":
    st.session_state["tic_id"] = st.text_input(
        "TIC ID", value="307210830",
        help="A TESS Input Catalog ID, e.g. 307210830")


# --- run ---------------------------------------------------------------------
if run:
    lcs = get_lightcurves()
    if not lcs:
        st.info("No light curve to analyse yet. Choose a source and try again.")
    for name, t, f in lcs:
        st.subheader(name)
        with st.spinner("Cleaning, searching, classifying ..."):
            res = analyse(t, f, model, stellar_radius=stellar_radius)

        if res is None:
            st.error("Light curve had too few usable points after cleaning.")
            continue

        # headline result
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Class", res["label"])
        m1.caption("confidence {:.0%}".format(res["conf"]))
        fr = res["fit"]
        m2.metric("Period (d)",
                  "{:.4f}".format(fr["period"]) if fr["period"] else "-")
        m3.metric("Depth",
                  "{:.4f}".format(fr["depth"]) if fr["depth"] else "-")
        m3.caption("+/- {:.4f}".format(fr["depth_err"])
                   if fr.get("depth_err") else "")
        m4.metric("Transit SNR",
                  "{:.1f}".format(fr["snr"]) if fr["snr"] else "-")

        if fr.get("planet_radius_earth"):
            st.info("Estimated planet radius: {:.1f} Earth radii"
                    .format(fr["planet_radius_earth"]))

        # figure (static, three-panel)
        st.pyplot(make_figure(res, name))

        # interactive phase-folded plot (Plotly) if available
        if _HAVE_PLOTLY:
            period, t0 = res["bls"].get("period"), res["bls"].get("t0")
            if period and t0 is not None:
                phase, fflux = fold(res["t"], res["f"], period, t0)
                fig = go.Figure()
                fig.add_trace(go.Scattergl(
                    x=phase, y=fflux, mode="markers",
                    marker=dict(size=3, color="#1b2a4a", opacity=0.5),
                    name="folded flux"))
                fig.update_layout(
                    title="Interactive phase-folded light curve",
                    xaxis_title="phase", yaxis_title="normalised flux",
                    height=380, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)

        # save to the shared store so the desktop app sees this too
        if _store is not None:
            try:
                _store.add_result(dict(
                    name=name, label=res["label"],
                    confidence=round(res["conf"], 3),
                    period=fr.get("period"), duration=fr.get("duration"),
                    depth=fr.get("depth"), depth_err=fr.get("depth_err"),
                    snr=fr.get("snr"),
                    planet_radius_earth=fr.get("planet_radius_earth")))
            except Exception:
                pass

        # class probabilities
        if len(res["probs"]) > 1:
            st.write("**Class probabilities**")
            st.bar_chart(res["probs"])

        # raw numbers, foldable
        with st.expander("Detailed numbers"):
            st.json({
                "label": res["label"],
                "confidence": round(res["conf"], 4),
                "period_days": fr["period"],
                "duration_days": fr["duration"],
                "depth": fr["depth"],
                "depth_err": fr.get("depth_err"),
                "snr": fr["snr"],
                "planet_radius_earth": fr.get("planet_radius_earth"),
                "features": {k: round(v, 5)
                             for k, v in res["feats"].items()},
            })

st.divider()

# ---------------------------------------------------------------------------
# History dashboard -- shows ALL results from the shared store, so anything
# analysed in the desktop app appears here too.
# ---------------------------------------------------------------------------
st.header("History & comparison")

if _store is None:
    st.info("Shared store module not found; history is unavailable.")
else:
    colA, colB = st.columns([3, 1])
    with colB:
        if st.button("Refresh", use_container_width=True):
            st.rerun()
        if st.button("Clear history", use_container_width=True):
            _store.clear()
            st.rerun()

    records = _store.load_all()
    if not records:
        st.caption("No results yet. Run a detection above, or use the desktop "
                   "app -- results from both show up here.")
    else:
        import pandas as pd
        df = pd.DataFrame(records)
        show_cols = [c for c in ["timestamp", "name", "label", "confidence",
                                 "period", "depth", "snr",
                                 "planet_radius_earth"] if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, height=260)

        # quick summary counts by class
        if "label" in df.columns:
            st.caption("Class counts")
            st.bar_chart(df["label"].value_counts())

        # downloads in every available format
        st.subheader("Download all results")
        if _exporter is not None:
            import tempfile
            recs = records
            d = tempfile.gettempdir()
            fmts = _exporter.available_formats()
            cols = st.columns(len([k for k, v in fmts.items() if v]))
            i = 0
            for fmt, ok in fmts.items():
                if not ok:
                    continue
                ext = {"CSV": "csv", "JSON": "json", "Excel": "xlsx",
                       "Word": "docx", "PDF": "pdf"}[fmt]
                fn = {"CSV": _exporter.to_csv, "JSON": _exporter.to_json,
                      "Excel": _exporter.to_excel, "Word": _exporter.to_word,
                      "PDF": _exporter.to_pdf}[fmt]
                try:
                    path = fn(recs, os.path.join(d, "exopipe_all." + ext))
                    with open(path, "rb") as fh:
                        cols[i].download_button(
                            fmt, fh.read(),
                            file_name="exopipe_results." + ext,
                            use_container_width=True, key="dl_" + fmt)
                except Exception as e:
                    cols[i].caption("{}: {}".format(fmt, e))
                i += 1

st.divider()
st.caption("exopipe - modular pipeline: preprocess -> BLS -> features -> "
           "classify -> fit. Desktop and web apps share one results store.")
