"""Run from the repository root with python -m exopipe.bench.bench."""

import argparse
import csv
import hashlib
from pathlib import Path
from time import perf_counter
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares
from scipy.ndimage import median_filter


def inject(time, flux, period, depth, epoch, duration=0.12):
    # Center the phase before applying the transit box so it wraps correctly at epoch.
    phase = (time - epoch + period / 2) % period - period / 2
    return flux * (1 - depth * (np.abs(phase) < duration / 2))


def trials(time, flux, seed=42):
    valid = np.isfinite(time) & np.isfinite(flux)
    t, f = np.asarray(time)[valid], np.asarray(flux)[valid]
    order = np.argsort(t)
    t, f = t[order], f[order]
    if len(t) < 101 or np.ptp(t) < 10 or np.median(f) <= 0:
        raise ValueError(
            "Need 101 finite samples spanning at least ten days with positive median"
        )
    if np.any(np.diff(t) <= 0):
        raise ValueError("Duplicate timestamps")
    f = f / np.median(f)
    rng = np.random.default_rng(seed)
    periods = np.linspace(1, 5, 1200)
    # Use the same realization for every method so a change in phase cannot favor one.
    for period in (2.17, 3.43, 4.37):
        epoch = t[0] + rng.uniform(0, period)
        for depth in (0, 0.001, 0.003, 0.01):
            injected = inject(t, f, period, depth, epoch)
            for method in ("raw", "flattened", "robust"):
                start = perf_counter()
                corrected = injected.copy()
                if method == "flattened":
                    corrected = (
                        lk.LightCurve(time=t, flux=corrected)
                        .flatten(window_length=101)
                        .flux.value
                    )
                elif method == "robust":
                    # Work within continuous segments to avoid bridging downlink gaps.
                    cuts = np.where(np.diff(t) > 5 * np.median(np.diff(t)))[0] + 1
                    for ids in np.split(np.arange(len(t)), cuts):
                        corrected[ids] /= median_filter(
                            injected[ids], size=min(101, len(ids)), mode="nearest"
                        )
                result = BoxLeastSquares(t, corrected).power(
                    periods, [0.08, 0.12, 0.16]
                )
                best = int(np.argmax(result.power))
                found = float(result.period[best])
                yield dict(
                    method=method,
                    period=period,
                    depth=depth,
                    epoch=epoch,
                    found=found,
                    relative_error=abs(found - period) / period,
                    power=float(result.power[best]),
                    recovered=bool(depth and abs(found - period) / period <= 0.01),
                    seconds=perf_counter() - start,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fits", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    curve = lk.read(str(args.fits))
    checksum = hashlib.sha256(args.fits.read_bytes()).hexdigest()
    rows = list(trials(curve.time.value, curve.flux.value))
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sha256", *rows[0]])
        writer.writeheader()
        writer.writerows(dict(sha256=checksum, **r) for r in rows)
    for method in ("raw", "flattened", "robust"):
        selected = [r for r in rows if r["method"] == method and r["depth"]]
        print(method, sum(r["recovered"] for r in selected), "/", len(selected))


if __name__ == "__main__":
    main()
