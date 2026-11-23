"""Small constrained Bayesian search on a declared, physically admissible pool.

The Gaussian-process uncertainty is uncertainty of the search surrogate.  It is
not a calibrated experimental uncertainty or a source-atom bootstrap interval.
Input coordinates must already be normalized using the declared search bounds;
they are never rescaled using the observed extrema (which change during search).
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import ndtr


def _unit_points(value, name, dimension=None, allow_empty=False):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] < 1:
        raise ValueError(name + " must be a two-dimensional array with at least one coordinate")
    if not allow_empty and len(array) == 0:
        raise ValueError(name + " must contain at least one point")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError(name + " dimension differs from the training coordinates")
    if not np.all(np.isfinite(array)) or np.any(array < -1e-10) or np.any(array > 1 + 1e-10):
        raise ValueError(name + " must contain finite coordinates normalized to [0,1]")
    return np.clip(array, 0, 1)


def matern52(first, second, length_scale):
    """Unit-variance Matern 5/2 covariance on normalized input coordinates."""
    delta = (np.asarray(first)[:, None, :] - np.asarray(second)[None, :, :]) / length_scale
    squared = np.einsum("ijk,ijk->ij", delta, delta)
    distance = np.sqrt(np.maximum(squared, 0))
    root5 = np.sqrt(5.0) * distance
    return (1 + root5 + 5 * squared / 3) * np.exp(-root5)


class GaussianProcess:
    """Normalized-output Matern GP with diagonal observation variance.

    ``noise`` is the observation variance in standardized output units, not a
    standard deviation.  The posterior standard deviation is latent variance
    and excludes a new observation's noise.  With no specified length scale,
    marginal likelihood selects among a small, fixed set of plausible scales.
    """

    def __init__(self, length_scale=None, noise=1e-6, jitter=1e-10):
        if not np.isfinite(noise) or noise < 0 or not np.isfinite(jitter) or jitter <= 0:
            raise ValueError("GP noise must be nonnegative and jitter must be positive")
        self.length_scale = length_scale
        self.noise = float(noise)
        self.jitter = float(jitter)

    def _factor(self, covariance):
        jitter = self.jitter
        identity = np.eye(len(covariance))
        for _ in range(12):
            try:
                return np.linalg.cholesky(covariance + (self.noise + jitter) * identity), jitter
            except np.linalg.LinAlgError:
                jitter *= 10
        raise ValueError("Could not stabilize the GP covariance")

    def fit(self, X, y):
        self.X_ = _unit_points(X, "X").copy()
        y = np.asarray(y, dtype=float)
        if y.shape != (len(self.X_),) or not np.all(np.isfinite(y)):
            raise ValueError("GP y must be a finite vector with one value per training point")
        self.output_mean_ = float(np.mean(y))
        deviation = float(np.std(y))
        self.output_scale_ = deviation if deviation > 1e-12 else max(1.0, abs(self.output_mean_) * 1e-3)
        normalized = (y - self.output_mean_) / self.output_scale_
        candidates = (.15, .3, .6, 1.2) if self.length_scale is None else (self.length_scale,)
        best = None
        for candidate in candidates:
            scale = np.asarray(candidate, dtype=float)
            if scale.shape not in ((), (self.X_.shape[1],)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError("GP length_scale must be a positive scalar or one value per coordinate")
            covariance = matern52(self.X_, self.X_, scale)
            factor, jitter = self._factor(covariance)
            intermediate = solve_triangular(factor, normalized, lower=True)
            alpha = solve_triangular(factor.T, intermediate, lower=False)
            likelihood = float(-.5 * normalized @ alpha - np.log(np.diag(factor)).sum()
                               - len(normalized) * .5 * np.log(2 * np.pi))
            if best is None or likelihood > best[0]:
                best = (likelihood, scale.copy(), factor, alpha, jitter)
        self.log_marginal_likelihood_, self.length_scale_, self.factor_, self.alpha_, self.effective_jitter_ = best
        return self

    def predict(self, X, return_std=True):
        if not hasattr(self, "X_"):
            raise ValueError("Fit the Gaussian process before predicting")
        points = _unit_points(X, "prediction points", self.X_.shape[1])
        covariance = matern52(points, self.X_, self.length_scale_)
        mean = self.output_mean_ + self.output_scale_ * (covariance @ self.alpha_)
        if not return_std:
            return mean
        projected = solve_triangular(self.factor_, covariance.T, lower=True)
        variance = np.maximum(1 - np.einsum("ij,ij->j", projected, projected), 0)
        return mean, self.output_scale_ * np.sqrt(variance)


def _normal_improvement(mean, std, incumbent):
    improvement = mean - incumbent
    positive_variance = std > 1e-14
    safe_std = np.where(positive_variance, std, 1)
    z = improvement / safe_std
    value = improvement * ndtr(z) + safe_std * np.exp(-.5 * z ** 2) / np.sqrt(2 * np.pi)
    return np.where(positive_variance, np.maximum(value, 0), np.maximum(improvement, 0))


def _tie_choice(values, eligible, rng, secondary=None):
    available = np.flatnonzero(eligible)
    maximum = np.max(values[available])
    ties = available[np.isclose(values[available], maximum, rtol=1e-12, atol=1e-15)]
    if secondary is not None and len(ties) > 1:
        maximum = np.max(secondary[ties])
        ties = ties[np.isclose(secondary[ties], maximum, rtol=1e-12, atol=1e-15)]
    return int(rng.choice(ties))


def constrained_expected_improvement(X, y, margins, pool, seed=0):
    """Choose an unobserved pool index using constraint-weighted expected gain.

    ``y`` is the *maximized* robust physical objective. ``margins[n,k]`` are
    observed feasibility margins, with nonnegative meaning allowed. Constraints
    and objective are fitted separately. Independent marginal probabilities are
    multiplied as an acquisition heuristic; they are not a validated joint
    chance-constraint guarantee. If no feasible observation exists, the search
    first maximizes feasibility probability, then the worst normalized predicted
    margin. Exact/near-exact observed points are excluded before selection.

    Returns ``(pool_index, diagnostics)``; diagnostics contain JSON-safe values.
    """
    X = _unit_points(X, "X")
    pool = _unit_points(pool, "pool", X.shape[1])
    y = np.asarray(y, dtype=float)
    margins = np.asarray(margins, dtype=float)
    if y.shape != (len(X),) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite vector with one value per training point")
    if margins.ndim != 2 or margins.shape[0] != len(X) or not np.all(np.isfinite(margins)):
        raise ValueError("margins must be a finite array of shape (observations,constraints)")
    distances = np.max(np.abs(pool[:, None, :] - X[None, :, :]), axis=2)
    eligible = ~np.any(distances <= 1e-9, axis=1)
    if not np.any(eligible):
        raise ValueError("Candidate pool exhausted: every point has already been observed")
    target_gp = GaussianProcess().fit(X, y)
    mean, std = target_gp.predict(pool)
    mean_margins = np.empty((len(pool), margins.shape[1]))
    std_margins = np.empty_like(mean_margins)
    margin_scales, length_scales = [], []
    probability = np.ones(len(pool))
    for j in range(margins.shape[1]):
        model = GaussianProcess().fit(X, margins[:, j])
        mu, sigma = model.predict(pool)
        mean_margins[:, j], std_margins[:, j] = mu, sigma
        margin_scales.append(model.output_scale_)
        length_scales.append(model.length_scale_.tolist())
        nonzero = sigma > 1e-14
        probability *= np.where(nonzero, ndtr(mu / np.where(nonzero, sigma, 1)), (mu >= 0).astype(float))
    feasible = np.all(margins >= 0, axis=1)
    rng = np.random.default_rng(seed)
    if np.any(feasible):
        incumbent = float(np.max(y[feasible]))
        improvement = _normal_improvement(mean, std, incumbent)
        acquisition = probability * improvement
        selected = _tie_choice(acquisition, eligible, rng, secondary=probability * std)
        mode = "constrained_expected_improvement"
    else:
        incumbent = None
        improvement = np.zeros(len(pool))
        acquisition = probability.copy()
        worst_margin = np.min(mean_margins / np.asarray(margin_scales)[None, :], axis=1)
        selected = _tie_choice(acquisition, eligible, rng, secondary=worst_margin)
        mode = "feasibility_first"
    diagnostic = {
        "mode": mode, "selected_index": selected, "eligible_candidates": int(eligible.sum()),
        "excluded_observed_indices": np.flatnonzero(~eligible).tolist(),
        "best_observed_feasible": incumbent, "posterior_mean": mean.tolist(), "posterior_std": std.tolist(),
        "probability_feasible": probability.tolist(), "expected_improvement": improvement.tolist(),
        "acquisition": acquisition.tolist(), "margin_posterior_mean": mean_margins.tolist(),
        "margin_posterior_std": std_margins.tolist(), "objective_length_scale": target_gp.length_scale_.tolist(),
        "margin_length_scales": length_scales,
        "uncertainty_note": "Surrogate GP uncertainty only; independent feasibility product is not calibrated. "
                            "Source-atom bootstrap intervals and experimental uncertainty are separate.",
    }
    return selected, diagnostic
