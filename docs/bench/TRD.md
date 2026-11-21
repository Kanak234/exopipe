# TRD — Transit Recovery Bench

## Stack
Python 3.11+, Lightkurve (FITS read and `flatten`), Astropy `BoxLeastSquares`, SciPy `median_filter`, NumPy.

## Algorithm
1. Load the FITS file and keep finite time and flux, sorted.
   - Require ≥ 101 points, a span of at least 10 days, a positive median and strictly increasing time.
   - Normalize by the median.
2. For each period in (2.17, 3.43, 4.37) d, draw one epoch (seed 42). For each depth in `DEPTHS`, multiply the flux by a box of width 0.12 d. The phase is centred, so the box wraps correctly at the epoch.
3. Detrend:
   - `raw`: no change.
   - `flattened`: `LightCurve.flatten(window_length=101)`, a Savitzky–Golay filter.
   - `robust`: split the series at gaps longer than 5 × the median cadence, then divide each segment by a median filter (size ≤ 101, nearest-edge).
4. BLS over 1200 periods (1–5 d) × durations (0.08, 0.12, 0.16) d. Record the best period, relative error, power, recovered (≤ 1%) and seconds.

## Key decisions
- **Same realization for all methods**, so differences come only from preprocessing.
- **Null rows (depth 0)** are kept in the CSV for a future false-alarm threshold.
- **Code vs data vs docs.** The package `exopipe/bench/` holds only code and tests. Sample data is in `data/bench/`, and docs and results are in `docs/bench/`. Before this change, LICENSE, .gitignore and docs sat inside the Python package.

## Limits
The 101-cadence filter window is close to the transit duration. That is part of what the benchmark measures, not a bug. Next: sweep the window length.
