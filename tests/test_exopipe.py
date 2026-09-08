import json
import os
import tempfile
import numpy as np
import pytest

import synth
import preprocess
import bls
import features as feat_mod
import fit as fit_mod
import classify
import store
import exporter
import make_report
import exopipe_app
from run_pipeline import analyse_one


def test_synthetic_transit_generation():
    rng = np.random.default_rng(42)
    t, flux, params = synth.make_transit(rng, days=15.0)
    assert len(t) == len(flux)
    assert len(t) > 200
    assert np.isclose(np.median(flux), 1.0, atol=0.005)
    assert "period" in params
    assert "depth" in params
    assert params["period"] > 0
    assert params["depth"] > 0


def test_synthetic_eclipse_and_blend():
    rng = np.random.default_rng(42)
    t_e, f_e, p_e = synth.make_eclipse(rng, days=10.0)
    assert len(t_e) > 100
    assert p_e["depth"] > 0.03

    t_b, f_b, p_b = synth.make_blend(rng, days=10.0)
    assert len(t_b) > 100
    assert p_b["depth"] > 0

    t_n, f_n, p_n = synth.make_noise(rng, days=10.0)
    assert len(t_n) > 100
    assert p_n["depth"] == 0.0

    dataset = synth.make_dataset(n_per_class=2, seed=123)
    assert len(dataset) == 8
    assert set(d["label"] for d in dataset) == {"transit", "eclipse", "blend", "noise"}


def test_preprocessing_steps():
    t = np.linspace(0, 10, 500)
    flux = 1.0 + 0.002 * np.sin(t)
    flux[20] = 3.0  # outlier

    t_clean, f_clean = preprocess.preprocess(t, flux, sigma=3.0, window_days=0.5)
    assert len(t_clean) < len(t)
    assert np.isclose(np.median(f_clean), 1.0, atol=0.002)


def test_bls_period_search():
    rng = np.random.default_rng(10)
    t, flux, params = synth.make_transit(rng, days=20.0)
    t_c, f_c = preprocess.preprocess(t, flux)

    res = bls.run_bls(t_c, f_c, min_period=1.5, max_period=10.0, n_periods=400)
    assert res["period"] is not None
    assert res["depth"] > 0.0005
    assert res["snr"] > 1.0


def test_feature_extraction():
    rng = np.random.default_rng(20)
    t, flux, params = synth.make_transit(rng, days=15.0)
    t_c, f_c = preprocess.preprocess(t, flux)
    res = bls.run_bls(t_c, f_c, min_period=1.5, max_period=10.0, n_periods=200)

    ft = feat_mod.extract(t_c, f_c, res)
    assert isinstance(ft, dict)
    assert "bls_snr" in ft
    assert "depth" in ft
    assert "odd_even_diff" in ft

    row = feat_mod.to_row(ft)
    assert len(row) == 9
    assert np.all(np.isfinite(row))


def test_transit_fitting():
    t = np.linspace(0, 10, 500)
    flux = np.ones_like(t)
    # inject transit dips every 3 days
    phase = (t % 3.0) / 3.0
    flux[phase < (0.2 / 3.0)] -= 0.02

    b = {"period": 3.0, "t0": 0.0, "duration": 0.2, "depth": 0.02, "snr": 25.0}
    fr = fit_mod.fit_transit(t, flux, b, stellar_radius_rsun=1.0)

    assert fr["period"] == 3.0
    assert fr["duration"] == 0.2
    assert fr["depth"] > 0.015
    assert fr["planet_radius_earth"] is not None
    assert fr["planet_radius_earth"] > 0


def test_classify_predict_fallback():
    features_dummy = np.array([12.0, 15.0, 0.05, 0.01, 0.1, 0.05, 0.001, 0.0, 0.0])
    label, conf, prob_map = classify.predict(None, features_dummy)

    assert label in ("transit", "noise")
    assert 0.0 <= conf <= 1.0
    assert isinstance(prob_map, dict)


def test_results_store_operations():
    # Save original store path
    orig_path = store.STORE_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test_store.json")
        store.STORE_PATH = test_path
        store.STORE_DIR = tmpdir

        try:
            store.clear()
            assert len(store.load_all()) == 0

            store.add_result({"name": "TIC 1", "label": "transit", "confidence": 0.95, "period": 4.5})
            store.add_result({"name": "TIC 2", "label": "eclipse", "confidence": 0.88, "period": 1.2})

            all_results = store.load_all()
            assert len(all_results) == 2

            matches = store.find_by_name("TIC 1")
            assert len(matches) == 1
            assert matches[0]["label"] == "transit"

            rec_latest = store.latest(1)
            assert len(rec_latest) == 1
            assert rec_latest[0]["name"] == "TIC 2"
        finally:
            store.STORE_PATH = orig_path
            store.STORE_DIR = os.path.dirname(orig_path)


def test_multi_format_exporter():
    results = [
        {
            "name": "TIC_1001",
            "label": "transit",
            "confidence": 0.92,
            "period": 3.45,
            "duration": 0.18,
            "depth": 0.012,
            "depth_err": 0.001,
            "snr": 18.5,
            "planet_radius_earth": 2.4
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        json_out = os.path.join(tmpdir, "report.json")
        csv_out = os.path.join(tmpdir, "report.csv")

        exporter.to_json(results, json_out)
        exporter.to_csv(results, csv_out)

        assert os.path.isfile(json_out)
        assert os.path.isfile(csv_out)

        with open(json_out, "r", encoding="utf-8") as fh:
            loaded_json = json.load(fh)
        assert len(loaded_json) == 1
        assert loaded_json[0]["name"] == "TIC_1001"


def test_pipeline_analyse_one():
    rng = np.random.default_rng(55)
    t, flux, params = synth.make_transit(rng, days=15.0)
    res = analyse_one(t, flux, model=None, tic=999999, make_plot=False)

    assert "label" in res
    assert "confidence" in res
    assert "period" in res
    assert res["period"] is not None


def test_make_report_safe_run():
    # Without train_metrics.json, generate_report should warn and return None cleanly
    ret = make_report.generate_report("/nonexistent/metrics.json")
    assert ret is None


def test_gui_standalone_execution():
    assert exopipe_app.main() == 0
