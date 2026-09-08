"""
Generate the 3-page methodology report PDF for BAH 2026 Problem 07.
Pulls the real trained metrics from outputs/train_metrics.json.
"""
import os
import json


def generate_report(metrics_path=None, out_path=None):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                        Table, TableStyle, PageBreak)
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_JUSTIFY
    except ImportError:
        print("[warn] reportlab is not installed. PDF report generation skipped.")
        return None

    root = os.path.dirname(os.path.abspath(__file__))
    metrics_path = metrics_path or os.path.join(root, "outputs", "train_metrics.json")
    out_path = out_path or os.path.join(root, "outputs", "BAH2026_Problem07_Report.pdf")

    if not os.path.exists(metrics_path):
        print(f"[warn] {metrics_path} does not exist. Run train_model.py first.")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        M = json.load(f)

    acc = M.get("report", {}).get("accuracy", 0.0)
    labels = M.get("labels", [])
    cm = M.get("confusion", [])
    rep = M.get("report", {})

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.2,
                          leading=12.5, alignment=TA_JUSTIFY, spaceAfter=5)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                        spaceAfter=4, textColor=colors.HexColor("#1a2a4a"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=10.8,
                        spaceBefore=6, spaceAfter=3,
                        textColor=colors.HexColor("#2c3e50"))
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8,
                           leading=10, textColor=colors.HexColor("#555"))
    title = ParagraphStyle("title", parent=styles["Title"], fontSize=16,
                           textColor=colors.HexColor("#1a2a4a"), spaceAfter=2)

    story = []

    # ---------------- PAGE 1 -----------------------------------------------------
    story.append(Paragraph(
        "AI-Enabled Detection of Exoplanets from Noisy Light Curves", title))
    story.append(Paragraph(
        "Bharatiya Antariksh Hackathon 2026 &mdash; Problem Statement 07 &mdash; "
        "Methodology Report", small))
    story.append(Spacer(1, 6))

    story.append(Paragraph("1. Objective &amp; Approach", h1))
    story.append(Paragraph(
        "We present <b>exopipe</b>, an end-to-end pipeline that detects transiting-"
        "planet signals in noisy TESS light curves, classifies each periodic dip "
        "as a planetary <i>transit</i>, an <i>eclipsing-binary eclipse</i>, a "
        "<i>blend</i> (a diluted eclipse in a crowded field), or <i>noise</i>, and "
        "estimates the transit parameters with an associated confidence level. The "
        "design follows the natural physics of the problem: a real planet produces "
        "a shallow, flat-bottomed dip that repeats on a fixed period, while the "
        "false positives each leave their own tell-tale signature.", body))

    story.append(Paragraph("2. Pipeline Stages", h2))
    story.append(Paragraph(
        "<b>(a) Target selection.</b> The TIC v8 CTL catalogue (~10 GB) is read in "
        "100k-row chunks so memory stays low. We keep bright stars (TESS magnitude "
        "&le; 12) with a known stellar radius, since brightness sets the signal-to-"
        "noise and radius is needed to convert depth into a planet size.", body))
    story.append(Paragraph(
        "<b>(b) Download.</b> TESS light curves are fetched from the MAST archive. "
        "Targets are first checked for data availability, so the common "
        "\"no light curve found\" case is skipped cleanly rather than treated as a "
        "failure.", body))
    story.append(Paragraph(
        "<b>(c) Preprocessing.</b> Each light curve is normalised to a baseline of "
        "1.0, cleaned of outliers with an iterative median/MAD sigma-clip, and "
        "flattened with a running-median filter whose window (~0.5 d) is kept much "
        "wider than a transit so the dip itself is preserved.", body))
    story.append(Paragraph(
        "<b>(d) Period search.</b> A Box Least Squares (BLS) search scans a grid of "
        "trial periods (0.5&ndash;15 d). For each period the folded light curve is "
        "scanned with a box of varying width and phase; the strongest, most "
        "consistent dip wins. On validation transits the recovered period matches "
        "the injected period to within a fraction of a percent.", body))
    story.append(Paragraph(
        "<b>(e) Feature extraction.</b> Each dip is summarised by nine physically "
        "motivated features: BLS significance, depth, duration/period ratio, "
        "odd&ndash;even depth difference, dip shape (V vs U), secondary-eclipse "
        "strength, point-to-point scatter, and flux skew/kurtosis.", body))

    # small detection figure
    fig = os.path.join(root, "outputs", "detection_demo4.png")
    if not os.path.exists(fig):
        fig = os.path.join(root, "outputs", "detection.png")
    if os.path.exists(fig):
        story.append(Spacer(1, 4))
        story.append(Image(fig, width=170 * mm, height=47 * mm))
        story.append(Paragraph(
            "Figure 1. Per-target output: cleaned light curve with marked transits "
            "(left), BLS periodogram with the detected period (centre), and the "
            "phase-folded transit with the fitted depth (right).", small))

    story.append(PageBreak())

    # ---------------- PAGE 2 -----------------------------------------------------
    story.append(Paragraph("3. Classification &amp; Parameter Estimation", h1))
    story.append(Paragraph(
        "<b>Classifier.</b> A gradient-boosted decision-tree ensemble "
        "(HistGradientBoosting) maps the nine features to one of the four classes "
        "and returns class probabilities. The probability of the winning class is "
        "reported as the detection's <b>confidence level</b>. A tree model was "
        "chosen deliberately: it trains in seconds on a laptop without a GPU, works "
        "well with a compact hand-built feature set, and its decisions are "
        "inspectable &mdash; each feature's contribution can be explained, which "
        "matters for a scientific result.", body))
    story.append(Paragraph(
        "<b>Parameter fitting.</b> For dips classified as transits, the depth is "
        "re-measured directly from the folded curve (median out-of-transit minus "
        "median in-transit), with an uncertainty from the in-transit scatter. The "
        "transit SNR is depth divided by its uncertainty. Where the stellar radius "
        "is known, the planet radius follows from depth &asymp; (R<sub>p</sub>/"
        "R<sub>star</sub>)<super>2</super>. The reported parameters are therefore "
        "orbital period, transit duration, transit depth (&plusmn; uncertainty), "
        "SNR, and an estimated planet radius.", body))

    story.append(Paragraph("4. Validation Results", h2))
    story.append(Paragraph(
        "The model was trained and tested on a balanced set of physics-based "
        "synthetic light curves spanning the four classes, processed through the "
        "identical pipeline used at prediction time. On a held-out test split "
        "(%d train / %d test) the overall accuracy is <b>%.0f%%</b>."
        % (M.get("n_train", 0), M.get("n_test", 0), acc * 100), body))

    # per-class table
    tbl_data = [["Class", "Precision", "Recall", "F1", "Support"]]
    for lab in labels:
        r = rep.get(lab, {})
        tbl_data.append([lab,
                         "%.2f" % r.get("precision", 0),
                         "%.2f" % r.get("recall", 0),
                         "%.2f" % r.get("f1-score", 0),
                         "%d" % r.get("support", 0)])
    t = Table(tbl_data, colWidths=[32 * mm, 28 * mm, 24 * mm, 24 * mm, 24 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a2a4a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#eef2f8")]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(Spacer(1, 3))
    story.append(t)
    story.append(Spacer(1, 4))

    # confusion matrix
    cm_data = [[""] + labels]
    for i, lab in enumerate(labels):
        row_vals = cm[i] if i < len(cm) else []
        cm_data.append([lab] + [str(v) for v in row_vals])
    ct = Table(cm_data, colWidths=[28 * mm] + [22 * mm] * len(labels))
    ct.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    story.append(Paragraph("Confusion matrix (rows = true, columns = predicted):",
                           small))
    story.append(ct)

    story.append(PageBreak())

    # ---------------- PAGE 3 -----------------------------------------------------
    story.append(Paragraph("5. Assumptions, Uncertainties &amp; Limitations", h1))
    story.append(Paragraph(
        "<b>Assumptions.</b> (i) Transits are approximately box/trapezoid shaped, "
        "adequate for detection and first-pass parameter estimation. (ii) A single "
        "TESS sector (~27 d) is searched, so only periods short enough to repeat "
        "at least twice are reliably recovered (we cap the search at ~half the "
        "data span). (iii) Planet-radius estimates assume the catalogue stellar "
        "radius is correct and that all transiting light comes from the target "
        "star.", body))
    story.append(Paragraph(
        "<b>How uncertainties are estimated.</b> The transit depth uncertainty is "
        "the standard deviation of the in-transit flux divided by the square root "
        "of the number of in-transit points. This propagates into the SNR and into "
        "the planet-radius estimate. The classifier's confidence is the posterior "
        "probability of the predicted class, giving a per-detection reliability.", body))
    story.append(Paragraph(
        "<b>Known limitation &mdash; transit/eclipse degeneracy.</b> The hardest "
        "case is an eclipsing binary whose secondary eclipse is weak or absent: "
        "such a system genuinely resembles a planetary transit, and this is a "
        "well-known astrophysical degeneracy rather than a software fault. The "
        "confusion matrix above reflects this honestly. We mitigate it with "
        "secondary-eclipse strength, odd&ndash;even depth differences, and "
        "dip-shape features, and on real data it can be reduced further using "
        "stellar density consistency and centroid/pixel-level vetting.", body))

    story.append(Paragraph("6. Tools &amp; Libraries", h2))
    story.append(Paragraph(
        "Python with NumPy and SciPy (numerics), pandas (chunked catalogue "
        "reading), Astropy <i>BoxLeastSquares</i> (period search), lightkurve "
        "(TESS data access via MAST), scikit-learn (gradient-boosted classifier), "
        "and Matplotlib (plots and animated GIFs). All are open-source; no "
        "specialised software is required.", body))

    story.append(Paragraph("7. Reproducibility", h2))
    story.append(Paragraph(
        "The pipeline is modular: catalogue reader, synthetic generator, "
        "preprocessing, BLS, feature extraction, classifier, parameter fitting, "
        "download, and visualisation, joined by a single runner. Training is one "
        "explicit step that saves the model to disk; the runner loads it back. An "
        "offline demo mode reproduces the full chain on synthetic data with no "
        "network. A hook is provided to fold in the curated real labelled dataset "
        "(known planets, false positives, eclipsing binaries) when available, which "
        "is expected to improve real-world performance beyond the synthetic "
        "baseline reported here.", body))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "exopipe &mdash; BAH 2026 Problem 07. Synthetic-data validation baseline; "
        "designed to run on real TESS light curves via MAST.", small))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm)
    doc.build(story)
    print("saved:", out_path, "exists:", os.path.exists(out_path))
    return out_path


def main():
    generate_report()


if __name__ == "__main__":
    main()
