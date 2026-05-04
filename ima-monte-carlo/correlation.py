"""
Correlated sampling via Gaussian copula.

Given:
    - A dict of marginal distributions {input_name: Distribution}
    - A partial correlation specification {(a, b): rho, ...}

Produce N correlated draws that preserve each marginal exactly while inducing
the requested rank-correlation structure across the continuous inputs.

Approach (Gaussian copula):
    1. Build the full correlation matrix from the partial spec (unspecified
       pairs default to 0, diagonal to 1).
    2. Clip to nearest PSD via eigenvalue clipping if any eigenvalues are
       negative (inconsistent partial correlations can produce non-PSD
       matrices).
    3. Cholesky-decompose.
    4. Draw independent standard normals, multiply by L to correlate them.
    5. Map each column through Phi(.) to get uniform(0, 1) marginals.
    6. Push each column through the target distribution's ppf to get the
       desired marginal.

Discrete, Empirical, and PointEstimate inputs are sampled independently; they
are not meaningful in the Gaussian copula.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
from scipy import stats

from distributions import Distribution


def _partial_to_full(
    names: list[str],
    partial: dict[tuple[str, str], float],
) -> np.ndarray:
    """Assemble the full KxK correlation matrix from the partial spec."""
    k = len(names)
    idx = {n: i for i, n in enumerate(names)}
    mat = np.eye(k)
    for (a, b), rho in partial.items():
        if a not in idx or b not in idx:
            # Silently skip pairs referring to non-continuous inputs — the
            # caller already filters these, but be defensive.
            continue
        if not -1.0 <= rho <= 1.0:
            raise ValueError(
                f"Correlation for ({a!r}, {b!r}) must be in [-1, 1], got {rho}"
            )
        i, j = idx[a], idx[b]
        mat[i, j] = rho
        mat[j, i] = rho
    return mat


def _nearest_psd(mat: np.ndarray, min_eig: float = 1e-8) -> np.ndarray:
    """
    Project a symmetric matrix to the nearest PSD matrix by clipping negative
    eigenvalues. Rescales the diagonal back to 1 so the result is a valid
    correlation matrix.
    """
    # Symmetrise to counteract floating-point drift
    sym = 0.5 * (mat + mat.T)
    w, v = np.linalg.eigh(sym)
    if np.all(w >= min_eig):
        return sym

    w_clipped = np.clip(w, min_eig, None)
    reconstructed = (v * w_clipped) @ v.T
    # Re-normalise diagonal back to 1 (so it's a correlation, not covariance)
    d = np.sqrt(np.clip(np.diag(reconstructed), min_eig, None))
    reconstructed = reconstructed / d[:, None] / d[None, :]
    # Force exact symmetry and unit diagonal
    reconstructed = 0.5 * (reconstructed + reconstructed.T)
    np.fill_diagonal(reconstructed, 1.0)
    return reconstructed


class CorrelatedSampler:
    """
    Draws N correlated samples from a set of marginal distributions.

    Usage:
        sampler = CorrelatedSampler(distributions, correlations)
        df = sampler.sample(n=50_000, rng=np.random.default_rng(42))
    """

    def __init__(
        self,
        distributions: dict[str, Distribution],
        correlations: dict[tuple[str, str], float] | None = None,
    ):
        self.distributions = distributions
        self.correlations = correlations or {}

        # Split into correlated continuous inputs vs. everything else
        self.continuous_names = [
            name for name, d in distributions.items() if d.is_continuous
        ]
        self.other_names = [
            name for name, d in distributions.items() if not d.is_continuous
        ]

        # Drop correlation entries referencing non-continuous inputs, warning
        # the user (binary catalysts should go through scenario_overlay, not
        # here).
        self._effective_correlations: dict[tuple[str, str], float] = {}
        for (a, b), rho in self.correlations.items():
            if a not in self.distributions or b not in self.distributions:
                warnings.warn(
                    f"Correlation references unknown input: ({a!r}, {b!r}) — skipping",
                    stacklevel=2,
                )
                continue
            if a not in self.continuous_names or b not in self.continuous_names:
                warnings.warn(
                    f"Correlation ({a!r}, {b!r}) involves a non-continuous "
                    f"input — skipped. Use scenario_overlay for discrete catalysts.",
                    stacklevel=2,
                )
                continue
            if a == b:
                continue
            self._effective_correlations[(a, b)] = float(rho)

        if self.continuous_names:
            self._corr_matrix = _partial_to_full(
                self.continuous_names, self._effective_correlations
            )
            psd = _nearest_psd(self._corr_matrix)
            if not np.allclose(psd, self._corr_matrix, atol=1e-6):
                warnings.warn(
                    "Correlation matrix was not positive semi-definite; "
                    "clipped to nearest PSD via eigenvalue projection. "
                    "Effective correlations may differ slightly from those specified.",
                    stacklevel=2,
                )
            self._psd_corr = psd
            self._chol = np.linalg.cholesky(psd)
        else:
            self._corr_matrix = np.zeros((0, 0))
            self._psd_corr = np.zeros((0, 0))
            self._chol = np.zeros((0, 0))

    # ------------------------------------------------------------------ #
    #  Introspection
    # ------------------------------------------------------------------ #
    @property
    def correlation_matrix(self) -> np.ndarray:
        """The PSD-projected correlation matrix that will actually be used."""
        return self._psd_corr

    def correlation_summary(self) -> str:
        if not self.continuous_names:
            return "(no continuous inputs — no correlation structure)"
        lines = ["Effective correlation matrix (PSD-projected):"]
        header = "            " + " ".join(f"{n[:8]:>9}" for n in self.continuous_names)
        lines.append(header)
        for i, n in enumerate(self.continuous_names):
            row = f"{n[:10]:>10}  " + " ".join(
                f"{self._psd_corr[i, j]:>+9.3f}" for j in range(len(self.continuous_names))
            )
            lines.append(row)
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Sampling
    # ------------------------------------------------------------------ #
    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """
        Return a dict mapping input name → array of n samples.

        Continuous inputs are sampled via the Gaussian copula (correlated).
        Non-continuous inputs are sampled independently from their marginals.
        """
        out: dict[str, np.ndarray] = {}

        if self.continuous_names:
            k = len(self.continuous_names)
            # Uncorrelated standard normals
            z = rng.standard_normal(size=(n, k))
            # Apply correlation
            z_corr = z @ self._chol.T
            # Map to uniforms, then to each target marginal
            u = stats.norm.cdf(z_corr)
            # Avoid exact 0/1 which blow up heavy-tailed ppf
            u = np.clip(u, 1e-12, 1.0 - 1e-12)
            for i, name in enumerate(self.continuous_names):
                dist = self.distributions[name]
                out[name] = np.asarray(dist.ppf(u[:, i]), dtype=float)

        for name in self.other_names:
            out[name] = self.distributions[name].sample(n, rng)

        return out


__all__ = ["CorrelatedSampler"]
