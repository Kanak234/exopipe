"""
exopipe_app.py
==============
Desktop app for exoplanet detection. One file, plain tkinter (ships with
Python, nothing extra to install for the window itself).

Run:
    python3 exopipe_app.py

What it does:
  - point it at your TIC catalogue CSV (the big 10 GB file) and it reads it in
    chunks (never loads the whole thing into memory),
  - or just type a TIC ID, or run a synthetic demo (no internet needed),
  - it preprocesses, runs a BLS period search, classifies the dip, fits the
    transit parameters, and shows the result,
  - "Save report" writes the result to a text + CSV file you can keep.

It reuses the exopipe modules if they are next to this file (folder exopipe/),
but also works on its own using lightkurve + astropy + scikit-learn.
"""

import os
import sys
import csv
import threading
import traceback
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "exopipe"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# shared modules (results store + multi-format exporter), optional
try:
    import store as _store
except Exception:
    _store = None
try:
    import exporter as _exporter
except Exception:
    _exporter = None


# ---------------------------------------------------------------------------
# Try to use the project modules; fall back to built-in copies if not present.
# ---------------------------------------------------------------------------
def _load_backend():
    backend = {}
    try:
        import preprocess
        import bls
        import features as feat_mod
        import fit as fit_mod
        import classify
        backend["preprocess"] = preprocess.preprocess
        backend["bls"] = bls.run_bls
        backend["extract"] = feat_mod.extract
        backend["to_row"] = feat_mod.to_row
        backend["fit"] = fit_mod.fit_transit
        model_path = os.path.join(HERE, "outputs", "classifier.joblib")
        backend["model"] = classify.load(model_path)
        backend["predict"] = classify.predict
        backend["source"] = "project modules"
    except Exception:
        backend = _builtin_backend()
        backend["source"] = "built-in fallback"
    return backend


def _builtin_backend():
    """Minimal self-contained versions if the exopipe/ folder is missing."""
    def preprocess(time, flux):
        t = np.asarray(time, float)
        f = np.asarray(flux, float)
        if t.size < 20:
            return np.array([]), np.array([])
        med = np.nanmedian(f)
        if med:
            f = f / med
        keep = np.isfinite(f) & np.isfinite(t)
        for _ in range(2):
            m = np.median(f[keep]); mad = np.median(np.abs(f[keep] - m))
            s = 1.4826 * mad if mad else np.std(f[keep])
            if not s:
                break
            keep = keep & (np.abs(f - m) < 5 * s)
        t, f = t[keep], f[keep]
        if t.size < 20:
            return t, f
        w = max(3, int(0.5 / (2.0 / 1440.0)))
        if w % 2 == 0:
            w += 1
        trend = np.array([np.median(f[max(0, i - w // 2):i + w // 2 + 1])
                          for i in range(t.size)])
        trend[trend == 0] = np.nanmedian(f)
        return t, f / trend

    def run_bls(time, flux, n_periods=3000):
        from astropy.timeseries import BoxLeastSquares
        t = np.asarray(time, float); f = np.asarray(flux, float)
        res = dict(period=None, t0=None, duration=None, depth=None, snr=0.0,
                   periods=None, powers=None)
        if t.size < 50:
            return res
        span = t.max() - t.min()
        model = BoxLeastSquares(t, f)
        periods = np.linspace(0.5, min(15.0, span / 2), n_periods)
        power = model.power(periods, [0.05, 0.08, 0.12, 0.18, 0.25])
        i = int(np.argmax(power.power))
        p = power.power
        m = np.median(p); mad = np.median(np.abs(p - m))
        s = 1.4826 * mad if mad else np.std(p)
        res.update(period=float(power.period[i]),
                   t0=float(power.transit_time[i]),
                   duration=float(power.duration[i]),
                   depth=float(power.depth[i]),
                   snr=float((p[i] - m) / s) if s else 0.0,
                   periods=np.asarray(power.period), powers=np.asarray(p))
        return res

    def fit_transit(time, flux, b, stellar_radius_rsun=None):
        out = dict(period=b.get("period"), duration=b.get("duration"),
                   depth=None, depth_err=None, snr=0.0,
                   planet_radius_earth=None)
        p, t0, d = b.get("period"), b.get("t0"), b.get("duration")
        if not p or t0 is None or not d:
            return out
        t = np.asarray(time, float); f = np.asarray(flux, float)
        phase = ((t - t0 + 0.5 * p) % p) / p - 0.5
        hw = 0.5 * d / p
        ins = np.abs(phase) < hw
        out_m = (np.abs(phase) > 1.5 * hw) & (np.abs(phase) < 4 * hw)
        if ins.sum() < 3 or out_m.sum() < 3:
            return out
        depth = float(np.median(f[out_m]) - np.median(f[ins]))
        derr = float(np.std(f[ins]) / np.sqrt(max(ins.sum(), 1)))
        snr = depth / derr if derr > 0 else 0.0
        rp = None
        if stellar_radius_rsun and depth > 0:
            rp = float(np.sqrt(depth) * stellar_radius_rsun * 109.1)
        out.update(depth=depth, depth_err=derr, snr=snr,
                   planet_radius_earth=rp)
        return out

    def classify_rule(b, fr):
        snr = b.get("snr", 0) or 0
        depth = fr.get("depth") or 0
        p = b.get("period") or 1
        dur = b.get("duration") or 0
        dratio = dur / p if p else 0
        if snr < 5:
            return "noise", 0.6
        if depth > 0.08 or dratio > 0.2:
            return "eclipse", 0.6
        if 0.0005 < depth < 0.05:
            return "transit", 0.7
        if depth >= 0.05:
            return "blend", 0.6
        return "noise", 0.4

    return dict(preprocess=preprocess, bls=run_bls, fit=fit_transit,
                extract=None, to_row=None, model=None,
                predict=None, classify_rule=classify_rule)


# ---------------------------------------------------------------------------
# Catalogue reading (chunked) -- official TIC v8 CTL column order
# ---------------------------------------------------------------------------
TIC_COLUMNS = (
    "ID version HIP TYC UCAC TWOMASS SDSS ALLWISE GAIA APASS KIC objType "
    "typeSrc ra dec POSflag pmRA e_pmRA pmDEC e_pmDEC PMflag plx e_plx PARflag "
    "gallong gallat eclong eclat Bmag e_Bmag Vmag e_Vmag umag e_umag gmag "
    "e_gmag rmag e_rmag imag e_imag zmag e_zmag Jmag e_Jmag Hmag e_Hmag Kmag "
    "e_Kmag TWOMflag prox w1mag e_w1mag w2mag e_w2mag w3mag e_w3mag w4mag "
    "e_w4mag GAIAmag e_GAIAmag Tmag e_Tmag TESSflag SPFlag Teff e_Teff logg "
    "e_logg MH e_MH rad e_rad mass e_mass rho e_rho lumclass lum e_lum d e_d "
    "ebv e_ebv numcont contratio disposition duplicate_id priority eneg_EBV "
    "epos_EBV EBVflag eneg_Mass epos_Mass eneg_Rad epos_Rad eneg_rho epos_rho "
    "eneg_logg epos_logg eneg_lum epos_lum eneg_dist epos_dist distflag "
    "eneg_Teff epos_Teff TeffFlag gaiabp e_gaiabp gaiarp e_gaiarp gaiaqflag "
    "starchareFlag VmagFlag BmagFlag splists e_RA e_Dec RA_orig Dec_orig "
    "e_RA_orig e_Dec_orig raddflag wdflag objID").split()


def select_targets_from_catalog(path, max_targets=20, tmag_max=12.0,
                                progress=None):
    """Stream the big catalogue and pick bright stars with a known radius."""
    import pandas as pd
    idx_ID = TIC_COLUMNS.index("ID")
    idx_ra = TIC_COLUMNS.index("ra")
    idx_dec = TIC_COLUMNS.index("dec")
    idx_tmag = TIC_COLUMNS.index("Tmag")
    idx_rad = TIC_COLUMNS.index("rad")
    use = [idx_ID, idx_ra, idx_dec, idx_tmag, idx_rad]
    names = ["ID", "ra", "dec", "Tmag", "rad"]

    rows = []
    seen = 0
    reader = pd.read_csv(path, header=None, usecols=use, names=TIC_COLUMNS,
                         chunksize=100_000, low_memory=False,
                         on_bad_lines="skip")
    for chunk in reader:
        seen += len(chunk)
        c = chunk[names].copy()
        for col in names:
            c[col] = pd.to_numeric(c[col], errors="coerce")
        good = c[c["ra"].notna() & c["dec"].notna() & c["Tmag"].notna()
                 & (c["Tmag"] <= tmag_max) & c["rad"].notna() & (c["rad"] > 0)]
        if len(good):
            rows.append(good)
        have = sum(len(r) for r in rows)
        if progress:
            progress("scanned {:,} rows, kept {} targets".format(seen, have))
        if have >= max_targets:
            break
    if not rows:
        return []
    out = []
    for r in rows:
        for t in r.itertuples():
            out.append((int(t.ID), float(t.rad)))
    return out[:max_targets]


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.backend = _load_backend()
        self.results = []

        root.title("exopipe - Exoplanet Detection")
        root.geometry("860x680")

        tk.Label(root, text="exopipe", font=("Helvetica", 22, "bold"),
                 fg="#1a2a4a").pack(pady=(12, 0))
        tk.Label(root, text="AI-enabled exoplanet detection from TESS light "
                 "curves  -  BAH 2026, Problem 07",
                 font=("Helvetica", 9), fg="#666").pack()
        tk.Label(root, text="engine: " + self.backend["source"],
                 font=("Helvetica", 8), fg="#999").pack(pady=(0, 8))

        frm = tk.LabelFrame(root, text="Input", padx=10, pady=10,
                            font=("Helvetica", 10, "bold"))
        frm.pack(fill="x", padx=12)

        self.mode = tk.StringVar(value="demo")
        tk.Radiobutton(frm, text="Synthetic demo (no internet)",
                       variable=self.mode, value="demo",
                       command=self._sync).grid(row=0, column=0, sticky="w")
        tk.Radiobutton(frm, text="TESS download by TIC ID",
                       variable=self.mode, value="tic",
                       command=self._sync).grid(row=0, column=1, sticky="w")
        tk.Radiobutton(frm, text="From catalogue CSV",
                       variable=self.mode, value="catalog",
                       command=self._sync).grid(row=0, column=2, sticky="w")

        tk.Label(frm, text="TIC ID(s):").grid(row=1, column=0, sticky="e",
                                              pady=6)
        self.tic = tk.StringVar(value="307210830")
        self.tic_entry = tk.Entry(frm, textvariable=self.tic, width=24)
        self.tic_entry.grid(row=1, column=1, sticky="w")

        tk.Label(frm, text="Catalogue CSV:").grid(row=2, column=0, sticky="e")
        self.csv = tk.StringVar()
        self.csv_entry = tk.Entry(frm, textvariable=self.csv, width=46)
        self.csv_entry.grid(row=2, column=1, columnspan=2, sticky="w")
        self.browse = tk.Button(frm, text="Browse", command=self._browse)
        self.browse.grid(row=2, column=3, padx=4)

        tk.Label(frm, text="How many targets:").grid(row=3, column=0,
                                                     sticky="e", pady=6)
        self.ntargets = tk.StringVar(value="5")
        tk.Entry(frm, textvariable=self.ntargets, width=8).grid(
            row=3, column=1, sticky="w")

        tk.Label(frm, text="Stellar radius (solar, optional):").grid(
            row=4, column=0, sticky="e")
        self.radius = tk.StringVar(value="")
        tk.Entry(frm, textvariable=self.radius, width=12).grid(
            row=4, column=1, sticky="w")

        btns = tk.Frame(root)
        btns.pack(pady=10)
        self.run_btn = tk.Button(btns, text="Run detection", command=self._run,
                                 bg="#e8743b", fg="white",
                                 font=("Helvetica", 12, "bold"),
                                 padx=18, pady=8)
        self.run_btn.pack(side="left", padx=6)
        self.save_btn = tk.Button(btns, text="Save report",
                                  command=self._save, state="disabled")
        self.save_btn.pack(side="left", padx=6)

        out = tk.LabelFrame(root, text="Results", padx=8, pady=8,
                            font=("Helvetica", 10, "bold"))
        out.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log = scrolledtext.ScrolledText(out, height=18,
                                             font=("Courier", 9))
        self.log.pack(fill="both", expand=True)

        self._sync()

    # -- ui helpers ---------------------------------------------------------
    def _sync(self):
        m = self.mode.get()
        self.tic_entry.config(state="normal" if m == "tic" else "disabled")
        cs = "normal" if m == "catalog" else "disabled"
        self.csv_entry.config(state=cs)
        self.browse.config(state=cs)

    def _browse(self):
        p = filedialog.askopenfilename(title="Select catalogue CSV",
                                       filetypes=[("CSV", "*.csv"),
                                                  ("All", "*.*")])
        if p:
            self.csv.set(p)

    def _write(self, text):
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.root.update_idletasks()

    # -- run ----------------------------------------------------------------
    def _run(self):
        self.run_btn.config(state="disabled")
        self.save_btn.config(state="disabled")
        self.log.delete("1.0", "end")
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            self.results = []
            m = self.mode.get()
            if m == "demo":
                self._run_demo()
            elif m == "tic":
                self._run_tic()
            else:
                self._run_catalog()
            if self.results:
                self.save_btn.config(state="normal")
        except Exception as e:
            self._write("ERROR: {}".format(e))
            self._write(traceback.format_exc())
        finally:
            self.run_btn.config(state="normal")

    def _radius(self):
        try:
            v = float(self.radius.get())
            return v if v > 0 else None
        except Exception:
            return None

    def _analyse(self, time, flux, name, stellar_radius=None):
        b = self.backend
        t, f = b["preprocess"](time, flux)
        if t.size < 20:
            self._write("  {}: too few usable points".format(name))
            return
        res = b["bls"](t, f)
        fr = b["fit"](t, f, res, stellar_radius_rsun=stellar_radius)

        if b.get("model") is not None and b.get("extract"):
            row = b["to_row"](b["extract"](t, f, res))
            label, conf, _ = b["predict"](b["model"], row)
        elif b.get("classify_rule"):
            label, conf = b["classify_rule"](res, fr)
        else:
            label, conf = ("transit" if (res.get("snr") or 0) > 10
                           else "noise"), 0.5

        rec = dict(name=name, label=label, confidence=round(conf, 3),
                   period=fr.get("period"), duration=fr.get("duration"),
                   depth=fr.get("depth"), depth_err=fr.get("depth_err"),
                   snr=fr.get("snr"),
                   planet_radius_earth=fr.get("planet_radius_earth"))
        self.results.append(rec)

        # push to the shared store so the web app shows the same result
        if _store is not None:
            try:
                _store.add_result(rec)
            except Exception:
                pass

        self._write("  {}".format(name))
        self._write("     class      : {}  (confidence {:.0%})"
                    .format(label, conf))
        if fr.get("period"):
            self._write("     period     : {:.4f} d".format(fr["period"]))
        if fr.get("depth") is not None:
            self._write("     depth      : {:.4f} +/- {:.4f}".format(
                fr["depth"], fr.get("depth_err") or 0))
        if fr.get("snr"):
            self._write("     transit SNR: {:.1f}".format(fr["snr"]))
        if fr.get("planet_radius_earth"):
            self._write("     est. radius: {:.1f} Earth radii".format(
                fr["planet_radius_earth"]))
        self._write("")

    def _run_demo(self):
        try:
            import synth
        except Exception:
            self._write("synth module not found; demo needs the exopipe/ "
                        "folder next to this file.")
            return
        n = int(self.ntargets.get() or 4)
        rng = np.random.default_rng(123)
        labels = (synth.LABELS * (n // 4 + 1))[:n]
        self._write("Running {} synthetic light curves...\n".format(n))
        for i, lab in enumerate(labels):
            t, f, _ = synth.make_one(lab, rng)
            self._analyse(t, f, "demo {} (true: {})".format(i, lab),
                          stellar_radius=1.0)
        self._write("Done.")

    def _run_tic(self):
        raw = self.tic.get().strip()
        # accept multiple IDs separated by comma or space (batch mode)
        ids = [x for x in raw.replace(",", " ").split() if x.isdigit()]
        if not ids:
            self._write("Enter one or more numeric TIC IDs "
                        "(comma or space separated).")
            return
        try:
            import download as dl
        except Exception as e:
            self._write("Download module error: {}".format(e))
            self._write("Make sure lightkurve is installed and you are online.")
            return
        self._write("Batch: {} TIC ID(s) to process.\n".format(len(ids)))
        for i, tic in enumerate(ids, 1):
            self._write("[{}/{}] Downloading TIC {} ...".format(i, len(ids),
                                                               tic))
            try:
                t, f, info = dl.download_one(int(tic))
            except Exception as e:
                self._write("  error: {}".format(e))
                continue
            if info != "ok":
                self._write("  skip: {}".format(info))
                continue
            self._write("  got {} points, analysing...".format(t.size))
            self._analyse(t, f, "TIC {}".format(tic), self._radius())
        self._write("Batch done.")

    def _run_catalog(self):
        path = self.csv.get().strip()
        if not path or not os.path.exists(path):
            self._write("Pick a valid catalogue CSV first.")
            return
        try:
            import download as dl
        except Exception as e:
            self._write("download module needed for catalogue mode: {}"
                        .format(e))
            return
        n = int(self.ntargets.get() or 5)
        self._write("Reading catalogue (in chunks, this can take a minute)...")
        targets = select_targets_from_catalog(
            path, max_targets=n * 3, progress=lambda s: self._write("  " + s))
        if not targets:
            self._write("No suitable targets found in catalogue.")
            return
        self._write("\nSelected {} candidate stars. Checking for light "
                    "curves and downloading...\n".format(len(targets)))
        done = 0
        for tic, rad in targets:
            ok = dl.check_available(tic)
            if not ok:
                self._write("  TIC {}: no light curve in archive, skipping"
                            .format(tic))
                continue
            t, f, info = dl.download_one(tic)
            if info != "ok":
                self._write("  TIC {}: {}".format(tic, info))
                continue
            self._analyse(t, f, "TIC {}".format(tic), rad)
            done += 1
            if done >= n:
                break
        self._write("Done. Analysed {} targets.".format(done))

    # -- save ---------------------------------------------------------------
    def _save(self):
        if not self.results:
            return
        # offer every format the exporter supports
        if _exporter is not None:
            types = [("PDF", "*.pdf"), ("Word", "*.docx"),
                     ("Excel", "*.xlsx"), ("CSV", "*.csv"),
                     ("JSON", "*.json")]
        else:
            types = [("CSV", "*.csv"), ("Text", "*.txt")]
        path = filedialog.asksaveasfilename(
            defaultextension=".pdf" if _exporter else ".csv",
            filetypes=types, initialfile="exopipe_results")
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if _exporter is not None and ext in (".pdf", ".docx", ".xlsx",
                                                 ".json", ".csv"):
                fn = {".pdf": _exporter.to_pdf, ".docx": _exporter.to_word,
                      ".xlsx": _exporter.to_excel, ".json": _exporter.to_json,
                      ".csv": _exporter.to_csv}[ext]
                fn(self.results, path)
            elif ext == ".txt":
                with open(path, "w") as fh:
                    for r in self.results:
                        fh.write("{}\n".format(r["name"]))
                        for k in ("label", "confidence", "period", "duration",
                                  "depth", "depth_err", "snr",
                                  "planet_radius_earth"):
                            fh.write("    {}: {}\n".format(k, r.get(k)))
                        fh.write("\n")
            else:
                # fallback CSV
                keys = ["name", "label", "confidence", "period", "duration",
                        "depth", "depth_err", "snr", "planet_radius_earth"]
                with open(path, "w", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=keys)
                    w.writeheader()
                    for r in self.results:
                        w.writerow({k: r.get(k) for k in keys})
            messagebox.showinfo("Saved", "Results saved to:\n{}".format(path))
        except Exception as e:
            messagebox.showerror("Error", str(e))


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
