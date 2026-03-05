import numpy as np

from mabt import mabt_ci


def test_mabt_ci_runs():
    rng = np.random.default_rng(0)
    n = 200
    true_labels = rng.integers(0, 2, size=n)
    pred_labels = np.column_stack(
        [
            rng.integers(0, 2, size=n),
            rng.integers(0, 2, size=n),
        ]
    )

    bound, tau, t0 = mabt_ci(true_labels, pred_labels, alpha=0.05, B=1000, seed=1)
    assert 0.0 <= bound <= 1.0
    assert np.isfinite(tau)
    assert 0.0 <= t0 <= 1.0