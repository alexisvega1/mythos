"""Tests for the scaling-law engine.

The headline check is a synthetic round-trip: generate points from a KNOWN law,
fit, and confirm the engine recovers it and extrapolates (leave-one-out) within
tolerance. If the fitter can't recover a law it generated itself, no prediction
about a real model can be trusted.
"""
from __future__ import annotations

import numpy as np
import pytest

from mythos.scaling import (
    ScalePoint,
    fit_scaling_law,
    leave_one_out,
    max_loo_rel_error,
    predict_with_ci,
)

# Known ground-truth law (toy units).
TRUE = dict(E=0.90, A=180.0, B=120.0, alpha=0.32, beta=0.30)
N_VALUES = (5e5, 1.5e6, 5e6, 1.5e7, 5e7)      # 5 sizes, ~100x span
D_VALUES = (5e6, 1e7, 2e7, 4e7)               # 4 token budgets


def _true_loss(N: float, D: float) -> float:
    return TRUE["E"] + TRUE["A"] / N ** TRUE["alpha"] + TRUE["B"] / D ** TRUE["beta"]


def _make_points(noise: float = 0.0, seed: int = 0) -> list[ScalePoint]:
    """Full N x D grid (20 points) so alpha/beta are separately identifiable."""
    rng = np.random.default_rng(seed)
    pts = []
    for N in N_VALUES:
        for D in D_VALUES:
            loss = _true_loss(N, D)
            if noise:
                loss *= float(np.exp(rng.normal(0.0, noise)))  # multiplicative log-noise
            pts.append(ScalePoint(N=N, D=D, loss=loss))
    return pts


def test_recovers_known_law_noiseless():
    law = fit_scaling_law(_make_points(noise=0.0))
    # Predictions on the grid must match the generating law tightly.
    for N in N_VALUES:
        for D in D_VALUES:
            assert law.predict(N, D) == pytest.approx(_true_loss(N, D), rel=0.02)
    # Exponents are the interpretable quantities; recover them closely.
    assert law.alpha == pytest.approx(TRUE["alpha"], abs=0.03)
    assert law.beta == pytest.approx(TRUE["beta"], abs=0.03)


def test_recovers_known_law_with_noise():
    law = fit_scaling_law(_make_points(noise=0.005, seed=7))
    for N in N_VALUES:
        for D in D_VALUES:
            assert law.predict(N, D) == pytest.approx(_true_loss(N, D), rel=0.05)


def test_leave_one_out_extrapolates():
    # On clean data, dropping any point and refitting should predict it well.
    err = max_loo_rel_error(_make_points(noise=0.0))
    assert err < 0.05, f"max LOO relative error {err:.3f} exceeds 5%"


def test_predict_held_out_larger_model():
    """The killer-demo move: predict a model larger than any fitted point."""
    law = fit_scaling_law(_make_points(noise=0.0))
    N_big, D_big = 1.5e8, 8e7   # ~3x the largest fitted N
    pred = law.predict(N_big, D_big)
    assert pred == pytest.approx(_true_loss(N_big, D_big), rel=0.10)


def test_predict_with_ci_brackets_truth():
    pts = _make_points(noise=0.005, seed=3)
    N_big, D_big = 1.2e8, 6e7
    point, lo, hi = predict_with_ci(pts, N_big, D_big, n_boot=40, seed=1)
    assert lo <= point <= hi
    assert lo <= _true_loss(N_big, D_big) <= hi


def test_predictions_monotonic():
    law = fit_scaling_law(_make_points(noise=0.0))
    # More params (fixed tokens) and more tokens (fixed params) both lower loss.
    assert law.predict(1e7, 1e7) < law.predict(1e6, 1e7)
    assert law.predict(1e7, 4e7) < law.predict(1e7, 1e7)


def test_requires_minimum_points():
    with pytest.raises(ValueError):
        fit_scaling_law(_make_points()[:4])


def test_loss_must_be_positive():
    with pytest.raises(ValueError):
        fit_scaling_law([ScalePoint(N=1e6, D=1e7, loss=-1.0)] + _make_points()[:5])
