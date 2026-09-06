"""
visualize.py
------------
Make the plots that show what the pipeline found.

Three things, matching the problem statement's "visualization" deliverable:

  plot_detection() : a 3-panel static summary for one target --
      (1) the cleaned light curve with detected transits marked,
      (2) the BLS periodogram with the winning period,
      (3) the phase-folded light curve with the fitted transit shape.

  make_gif() : an animated GIF that walks through the detection story --
      raw -> cleaned -> period found -> folded -> classified. Nice for the
      finale because it shows the pipeline "thinking" step by step.

We use matplotlib with the Agg backend so it works headless (no display
needed), and we save everything to the outputs folder.
"""

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def _fold(time, flux, period, t0):
    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5
    order = np.argsort(phase)
    return phase[order], flux[order]


def plot_detection(time, flux, bls_result, fit_result, label, confidence,
                   tic=None, outpath=None):
    """
    Three-panel detection summary for one star. Returns the saved path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    title = "TIC {}".format(tic) if tic else "target"
    fig.suptitle("{}  |  class: {}  (confidence {:.0%})"
                 .format(title, label, confidence), fontsize=13)

    # panel 1: light curve with transit marks
    ax = axes[0]
    ax.plot(time, flux, ".", ms=1.5, color="#222", alpha=0.5)
    ax.set_xlabel("time (days)")
    ax.set_ylabel("normalised flux")
    ax.set_title("cleaned light curve")
    period = bls_result.get("period")
    t0 = bls_result.get("t0")
    if period and t0 is not None:
        n0 = int((time.min() - t0) / period)
        n1 = int((time.max() - t0) / period) + 1
        for n in range(n0, n1 + 1):
            tc = t0 + n * period
            if time.min() <= tc <= time.max():
                ax.axvline(tc, color="#c0392b", lw=0.8, alpha=0.6)

    # panel 2: periodogram
    ax = axes[1]
    if bls_result.get("periods") is not None:
        ax.plot(bls_result["periods"], bls_result["powers"],
                lw=0.8, color="#2c3e50")
        if period:
            ax.axvline(period, color="#c0392b", lw=1.2,
                       label="P = {:.3f} d".format(period))
            ax.legend(fontsize=9)
    ax.set_xlabel("period (days)")
    ax.set_ylabel("BLS power")
    ax.set_title("period search")

    # panel 3: phase-folded with fit
    ax = axes[2]
    if period and t0 is not None:
        phase, fflux = _fold(time, flux, period, t0)
        ax.plot(phase, fflux, ".", ms=1.5, color="#222", alpha=0.4)
        depth = fit_result.get("depth")
        dur = fit_result.get("duration")
        if depth and dur:
            hw = 0.5 * dur / period
            ax.plot([-0.5, -hw, -hw, hw, hw, 0.5],
                    [1, 1, 1 - depth, 1 - depth, 1, 1],
                    color="#c0392b", lw=2, label="fit depth {:.3f}".format(depth))
            ax.legend(fontsize=9)
        ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("phase")
    ax.set_ylabel("normalised flux")
    ax.set_title("phase-folded")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if outpath is None:
        outpath = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "outputs", "detection.png")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=130)
    plt.close(fig)
    return outpath


def make_gif(time, flux, raw_flux, bls_result, fit_result, label,
             tic=None, outpath=None, n_frames=48):
    """
    Animated walkthrough of the detection. Saves a GIF and returns its path.

    The animation cycles through the story: raw data, cleaned data, the
    periodogram peak, and the phase-folded transit. It loops so it can play on
    a slide.
    """
    period = bls_result.get("period")
    t0 = bls_result.get("t0")
    has_fold = bool(period and t0 is not None)
    if has_fold:
        phase, fflux = _fold(time, flux, period, t0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    title = "TIC {}".format(tic) if tic else "target"

    def frame(i):
        ax.clear()
        stage = (i // (n_frames // 4)) % 4

        if stage == 0:
            ax.plot(time, raw_flux, ".", ms=1.5, color="#888", alpha=0.5)
            ax.set_title("{} | step 1: raw light curve".format(title))
            ax.set_xlabel("time (days)"); ax.set_ylabel("flux")
        elif stage == 1:
            ax.plot(time, flux, ".", ms=1.5, color="#222", alpha=0.5)
            ax.set_title("{} | step 2: cleaned & detrended".format(title))
            ax.set_xlabel("time (days)"); ax.set_ylabel("normalised flux")
        elif stage == 2 and bls_result.get("periods") is not None:
            ax.plot(bls_result["periods"], bls_result["powers"],
                    lw=0.9, color="#2c3e50")
            if period:
                ax.axvline(period, color="#c0392b", lw=1.5)
                ax.text(0.98, 0.95, "P = {:.3f} d".format(period),
                        transform=ax.transAxes, ha="right", va="top",
                        color="#c0392b")
            ax.set_title("{} | step 3: period search".format(title))
            ax.set_xlabel("period (days)"); ax.set_ylabel("BLS power")
        else:
            if has_fold:
                ax.plot(phase, fflux, ".", ms=1.5, color="#222", alpha=0.4)
                depth = fit_result.get("depth")
                dur = fit_result.get("duration")
                if depth and dur:
                    hw = 0.5 * dur / period
                    ax.plot([-0.5, -hw, -hw, hw, hw, 0.5],
                            [1, 1, 1 - depth, 1 - depth, 1, 1],
                            color="#c0392b", lw=2)
                ax.set_xlim(-0.5, 0.5)
                ax.text(0.02, 0.05, "class: {}".format(label),
                        transform=ax.transAxes, color="#c0392b")
            ax.set_title("{} | step 4: folded + classified".format(title))
            ax.set_xlabel("phase"); ax.set_ylabel("normalised flux")
        fig.tight_layout()

    anim = FuncAnimation(fig, frame, frames=n_frames, interval=120)

    if outpath is None:
        outpath = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "outputs", "detection.gif")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    anim.save(outpath, writer=PillowWriter(fps=8))
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    import synth
    import preprocess
    import bls
    import fit

    rng = np.random.default_rng(4)
    t, raw, params = synth.make_transit(rng)
    tc, fc = preprocess.preprocess(t, raw)
    b = bls.run_bls(tc, fc, n_periods=2000)
    fr = fit.fit_transit(tc, fc, b, stellar_radius_rsun=1.0)

    png = plot_detection(tc, fc, b, fr, "transit", 0.92, tic=12345)
    print("saved plot:", png, "exists:", os.path.exists(png))

    # raw needs to align with cleaned length for the gif; re-grid simply
    gif = make_gif(tc, fc, fc, b, fr, "transit", tic=12345)
    print("saved gif:", gif, "exists:", os.path.exists(gif))
