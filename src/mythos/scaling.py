"""Scaling-law fitting — the scientific core of Project Logos.

Fit the Chinchilla / Hoffmann parametric law

    L(N, D) = E + A / N**alpha + B / D**beta

to a set of (params ``N``, tokens ``D``, loss ``L``) observations, then PREDICT
the loss of an unseen ``(N, D)``. This is what turns mythos's existing
``val_bpb`` + ``count_parameters`` plumbing into a falsifiable instrument: fit on
small runs, commit a prediction for a larger model *before* training it, then
report predicted-vs-observed.

Method (the Besiroglu replication of Hoffmann et al.):
  * optimize in LOG space, parameterizing E=exp(e), A=exp(a), B=exp(b) so the
    three terms — and therefore the predicted loss — are positive by construction;
  * predict ``log L`` via ``logsumexp([e, a - alpha*logN, b - beta*logD])``;
  * minimize a SUMMED Huber loss over the log-residuals (robust to a few
    under-trained outliers);
  * sweep a grid of initial points and keep the best optimum (the objective is
    non-convex; single-start L-BFGS lands in local minima).

Loss units: any positive per-token loss works (bpb is a constant multiple of
nats/token, and the functional form is closed under that scaling — the constant
is absorbed into E, A, B). Feed the MINIMUM val loss over training, not the
final checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class ScalePoint:
    """One observed run: N params trained on D tokens reaching loss ``loss``."""

    N: float
    D: float
    loss: float


@dataclass(frozen=True)
class ScalingLaw:
    E: float
    A: float
    B: float
    alpha: float
    beta: float
    huber: float  # summed Huber objective at the optimum (fit quality; lower is better)
    n_points: int

    def predict(self, N: float, D: float) -> float:
        """Predicted loss for a model of ``N`` params trained on ``D`` tokens."""
        return float(self.E + self.A / N**self.alpha + self.B / D**self.beta)

    def as_dict(self) -> dict[str, float]:
        return {
            "E": self.E, "A": self.A, "B": self.B,
            "alpha": self.alpha, "beta": self.beta,
            "huber": self.huber, "n_points": self.n_points,
        }


# Grid of L-BFGS initializations (e, a, b, alpha, beta). Kept moderate so a full
# fit + leave-one-out + bootstrap stays interactive; widen for a final report.
_DEFAULT_GRID: dict[str, Sequence[float]] = {
    "e": (0.0, 1.0),
    "a": (0.0, 8.0),
    "b": (0.0, 8.0),
    "alpha": (0.2, 0.5),
    "beta": (0.2, 0.5),
}

_BOUNDS = [(None, None), (None, None), (None, None), (1e-3, 2.0), (1e-3, 2.0)]


def _obj_and_grad(theta, logN, logD, logL, delta):
    """Summed Huber loss over log-residuals and its analytic gradient.

    log L_hat = logsumexp([e, a - alpha*logN, b - beta*logD]); the gradient of
    log L_hat w.r.t. the parameters is the softmax over those three terms (times
    -logN / -logD for the exponents), so the whole gradient is closed-form.
    """
    e, a, b, alpha, beta = theta
    z = np.stack([np.full_like(logN, e), a - alpha * logN, b - beta * logD])  # (3, n)
    log_pred = logsumexp(z, axis=0)                                            # (n,)
    r = log_pred - logL
    absr = np.abs(r)
    huber = np.where(absr <= delta, 0.5 * r * r, delta * (absr - 0.5 * delta))

    psi = np.where(absr <= delta, r, delta * np.sign(r))   # huber'(r)
    w = np.exp(z - log_pred)                               # softmax weights, columns sum to 1
    grad = np.array([
        np.sum(psi * w[0]),
        np.sum(psi * w[1]),
        np.sum(psi * w[2]),
        np.sum(psi * (-logN) * w[1]),
        np.sum(psi * (-logD) * w[2]),
    ])
    return float(huber.sum()), grad


def _pack(points: Sequence[ScalePoint]):
    N = np.array([p.N for p in points], dtype=np.float64)
    D = np.array([p.D for p in points], dtype=np.float64)
    L = np.array([p.loss for p in points], dtype=np.float64)
    if np.any(L <= 0):
        raise ValueError("losses must be positive (the law is fit in log space)")
    return np.log(N), np.log(D), np.log(L)


def fit_scaling_law(
    points: Sequence[ScalePoint],
    delta: float = 1e-3,
    grid: dict[str, Sequence[float]] | None = None,
    min_points: int = 5,
) -> ScalingLaw:
    """Fit ``L(N,D)=E+A/N^alpha+B/D^beta`` to ``points`` via grid-restarted L-BFGS."""
    if len(points) < min_points:
        raise ValueError(
            f"need >= {min_points} points for a credible fit (got {len(points)}); "
            "single/dual-point scaling fits are not meaningful"
        )
    grid = grid or _DEFAULT_GRID
    logN, logD, logL = _pack(points)

    best = None
    for e in grid["e"]:
        for a in grid["a"]:
            for b in grid["b"]:
                for alpha in grid["alpha"]:
                    for beta in grid["beta"]:
                        res = minimize(
                            _obj_and_grad,
                            np.array([e, a, b, alpha, beta], dtype=np.float64),
                            args=(logN, logD, logL, delta),
                            method="L-BFGS-B",
                            jac=True,
                            bounds=_BOUNDS,
                        )
                        if res.success or best is None:
                            if best is None or res.fun < best.fun:
                                best = res
    e, a, b, alpha, beta = best.x
    return ScalingLaw(
        E=float(np.exp(e)), A=float(np.exp(a)), B=float(np.exp(b)),
        alpha=float(alpha), beta=float(beta),
        huber=float(best.fun), n_points=len(points),
    )


@dataclass(frozen=True)
class LOOResult:
    held_out: ScalePoint
    predicted: float
    rel_error: float


def leave_one_out(points: Sequence[ScalePoint], **fit_kwargs) -> list[LOOResult]:
    """Drop each point, refit on the rest, predict the held-out point.

    The honest validity check: low LOO error means the law generalizes to unseen
    ``(N, D)`` rather than merely interpolating its own training points.
    """
    pts = list(points)
    out: list[LOOResult] = []
    for i, held in enumerate(pts):
        rest = pts[:i] + pts[i + 1:]
        law = fit_scaling_law(rest, **fit_kwargs)
        pred = law.predict(held.N, held.D)
        out.append(LOOResult(held, pred, abs(pred - held.loss) / held.loss))
    return out


def max_loo_rel_error(points: Sequence[ScalePoint], **fit_kwargs) -> float:
    return max(r.rel_error for r in leave_one_out(points, **fit_kwargs))


def predict_with_ci(
    points: Sequence[ScalePoint],
    N: float,
    D: float,
    n_boot: int = 100,
    ci: float = 0.95,
    seed: int = 0,
    **fit_kwargs,
) -> tuple[float, float, float]:
    """Point prediction + bootstrap confidence interval for an unseen ``(N, D)``.

    Returns ``(point_estimate, lo, hi)``. The CI comes from refitting on
    resamples of ``points``; report it so a prediction never reads as more
    certain than the data supports.
    """
    point = fit_scaling_law(points, **fit_kwargs).predict(N, D)
    rng = np.random.default_rng(seed)
    pts = list(points)
    preds = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(pts), size=len(pts))
        sample = [pts[i] for i in idx]
        try:
            preds.append(fit_scaling_law(sample, **fit_kwargs).predict(N, D))
        except ValueError:
            continue
    if not preds:
        return point, point, point
    lo = float(np.quantile(preds, (1 - ci) / 2))
    hi = float(np.quantile(preds, 1 - (1 - ci) / 2))
    return point, lo, hi
