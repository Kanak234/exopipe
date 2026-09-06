"""
store.py
--------
A shared results store that BOTH apps use -- the desktop app (exopipe_app.py)
and the web app (exopipe_gui.py). Whatever one analyses, the other can see.

It is deliberately simple: results are appended to a single JSON file in
outputs/results_store.json. Each entry is one analysed target with its class,
confidence, fitted parameters, a timestamp, and (optionally) the path to a
saved plot image. Because it is just a file, the two apps stay in sync without
any server or database -- the web app re-reads the file and shows the latest.

This is what lets the desktop app give you the clean text/CSV report while the
web app shows the same results with nicer images.
"""

import os
import json
import time
import threading

_LOCK = threading.Lock()

HERE = os.path.dirname(os.path.abspath(__file__))
# store lives in the project outputs folder, next to the model
STORE_DIR = os.path.join(os.path.dirname(HERE), "outputs") \
    if os.path.basename(HERE) == "exopipe" \
    else os.path.join(HERE, "outputs")
STORE_PATH = os.path.join(STORE_DIR, "results_store.json")


def _ensure_dir():
    os.makedirs(STORE_DIR, exist_ok=True)


def load_all():
    """Return the full list of stored result records (newest last)."""
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def add_result(record):
    """
    Append one result record and return it (with a timestamp added).

    `record` is a plain dict, e.g.:
        {"name": "TIC 307210830", "label": "transit", "confidence": 0.82,
         "period": 4.61, "depth": 0.0035, "snr": 64.0, "plot": "/path.png"}
    """
    _ensure_dir()
    record = dict(record)
    record.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
    with _LOCK:
        data = load_all()
        data.append(record)
        # keep the file from growing without bound
        if len(data) > 500:
            data = data[-500:]
        tmp = STORE_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, STORE_PATH)
    return record


def clear():
    """Wipe the store (used by the 'clear history' button)."""
    with _LOCK:
        if os.path.exists(STORE_PATH):
            os.remove(STORE_PATH)


def latest(n=10):
    """Return the most recent n records, newest first."""
    return list(reversed(load_all()[-n:]))


def find_by_name(name):
    """Return all records whose name matches (e.g. a TIC ID)."""
    return [r for r in load_all() if str(r.get("name")) == str(name)]


if __name__ == "__main__":
    # quick self-test
    clear()
    add_result({"name": "TIC 111", "label": "transit", "confidence": 0.9,
                "period": 3.5, "depth": 0.004, "snr": 30})
    add_result({"name": "TIC 222", "label": "eclipse", "confidence": 0.8,
                "period": 1.8, "depth": 0.09, "snr": 50})
    print("stored:", len(load_all()))
    print("latest:", [r["name"] for r in latest()])
    print("find TIC 111:", find_by_name("TIC 111"))
