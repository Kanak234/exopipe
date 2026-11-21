import numpy as np
import pytest
from .bench import inject, trials


def test_wrap():
    assert np.allclose(
        inject(np.array([-0.02, 0.02, 0.5]), np.ones(3), 2, 0.01, 0), [0.99, 0.99, 1]
    )


def test_short():
    with pytest.raises(ValueError):
        list(trials(np.arange(3), np.ones(3)))


def test_strong_signal():
    t = np.linspace(0, 20, 1001)
    rows = list(trials(t, np.ones_like(t)))
    assert len(rows) == 36
    assert all(
        r["recovered"] for r in rows if r["method"] == "raw" and r["depth"] == 0.01
    )
