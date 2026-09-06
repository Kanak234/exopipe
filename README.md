# exopipe — AI-Enabled Detection of Exoplanets from Noisy Light Curves

A data-analysis pipeline that finds transiting-planet signals in noisy TESS
light curves, classifies each periodic dip (transit / eclipse / blend / noise),
estimates the transit parameters, and reports a confidence level.

Built for BAH 2026, Problem 07.

---

## What it does

1. **Select targets** from the TIC CTL catalogue (the big CSV).
2. **Download** TESS light curves for those targets (skips ones with no data).
3. **Preprocess** — normalise, remove outliers, flatten slow trends.
4. **Search** for periodic dips with Box Least Squares (BLS).
5. **Extract features** that describe the dip shape.
6. **Classify** the dip: transit, eclipse, blend, or noise (with confidence).
7. **Fit** the transit: period, duration, depth (+ uncertainty), SNR, and an
   estimated planet radius if the stellar radius is known.
8. **Visualise** — a 3-panel summary plot per target, plus an animated GIF.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10–3.12 recommended.

---

## Usage

### 1. Train the classifier (run once)

```bash
cd scripts
python train_model.py
```

This builds a labelled training set, trains the model, and saves it to
`outputs/classifier.joblib`. **Do this before running the pipeline.**

### 2a. Run on real TESS data (needs internet / MAST access)

Pick bright targets straight from the catalogue and analyse them:

```bash
python run_pipeline.py --catalog /path/to/exo_CTL_08.01xTIC_v8.1.csv --n 20
```

Or analyse specific TIC IDs:

```bash
python run_pipeline.py --tic 307210830 231663901 --n 2
```

### 2b. Run offline (no download, good for a quick demo)

```bash
python run_pipeline.py --demo 10
```

Results go to `outputs/results.json`, with `detection_*.png` plots and a few
`detection_*.gif` animations.

### 3. Web GUI (point-and-click)

For a browser-based interface instead of the command line:

```bash
pip install streamlit
streamlit run exopipe_gui.py
```

Open the link it prints (usually http://localhost:8501). From there you can:

- run a **synthetic demo** (no internet needed) for any of the four classes,
- **upload a CSV** with two columns (time, flux) and analyse it,
- **download a TESS light curve by TIC ID** and analyse it live,

and see the class, confidence, fitted period/depth/SNR, the estimated planet
radius, and the three-panel detection plot, all in the browser. The GUI calls
the exact same pipeline modules as the command-line runner.

### 4. Two integrated apps (desktop + web, shared results)

Two front ends that share one results store, so anything analysed in one shows
up in the other:

**Desktop app** (plain Python window, good for clean text/CSV/PDF reports and
batch runs):
```bash
python3 exopipe_app.py
```
- three modes: synthetic demo, TESS download by TIC ID (one or many, comma or
  space separated for batch), or from the catalogue CSV,
- saves reports as PDF / Word / Excel / CSV / JSON,
- every result is written to the shared store.

**Web app** (browser, good for interactive plots and the history dashboard):
```bash
streamlit run exopipe_gui.py
```
- same three input modes,
- static three-panel plot plus an interactive Plotly phase-folded view,
- a History & comparison dashboard that lists every result from both apps,
  with class-count charts and one-click download of all results in any format.

Both read and write `outputs/results_store.json`, which is how they stay in
sync. Run a TIC in the desktop app, then open the web app and hit Refresh to
see it with richer visuals.

---

## Project layout

```
exopipe/
  exopipe/            the library modules
    catalog.py        chunked reader for the big TIC catalogue
    synth.py          synthetic light-curve generator (for training/testing)
    preprocess.py     normalise / sigma-clip / detrend
    bls.py            Box Least Squares period search
    features.py       feature extraction
    classify.py       gradient-boosted classifier (train/save/load/predict)
    fit.py            transit parameter fitting + SNR + planet radius
    download.py       TESS light-curve download (runs on your machine)
    visualize.py      detection plots and animated GIF
  scripts/
    train_model.py    build training set, train + save the model
    run_pipeline.py   end-to-end runner
  exopipe_gui.py      browser GUI (streamlit run exopipe_gui.py)
  outputs/            results, models, figures
  requirements.txt
  README.md
```

---

## Notes on accuracy and honesty

- Trained on physics-based synthetic light curves, since the curated real
  labelled set is provided separately; `train_model.py` has a hook
  (`load_real_labelled`) to mix in real labelled data when you have it.
- Transit vs eclipse is the hardest pair: an eclipsing binary with a weak
  secondary eclipse genuinely looks like a planet transit. This is a known
  astrophysical degeneracy, not a code bug. We tackle it with secondary-eclipse
  strength, odd/even depth difference, and dip-shape (V vs U) features, but
  some overlap remains and is reported honestly.
- Every stage is wrapped so a single bad target never crashes a batch —
  failures are recorded and the run continues.
