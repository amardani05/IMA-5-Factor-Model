"""
Probability distributions for valuation inputs.

Every distribution supports a uniform interface:
    .sample(n, rng)          — draw n samples using a numpy Generator
    .describe()              — human-readable summary
    .mean(), .median(), .std()
    .pdf(x)                  — probability density at x (for plotting)
    .cdf(x), .ppf(q)         — CDF / inverse CDF (used by the Gaussian copula
                               for correlated sampling)

Distributions that do not have a continuous CDF (Discrete, Empirical, Point)
participate in simulation but are excluded from correlated sampling — see
correlation.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


class Distribution(ABC):
    """Base class for all distributions used as valuation inputs."""

    label: str = ""
    is_continuous: bool = True  # Overridden by Discrete/Empirical/Point

    @abstractmethod
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples using the provided generator."""

    @abstractmethod
    def describe(self) -> str:
        """One-line human-readable summary."""

    @abstractmethod
    def mean(self) -> float: ...

    @abstractmethod
    def median(self) -> float: ...

    @abstractmethod
    def std(self) -> float: ...

    def pdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """Probability density at x. Default: not implemented."""
        raise NotImplementedError(
            f"pdf() is not defined for {self.__class__.__name__}"
        )

    def cdf(self, x: np.ndarray | float) -> np.ndarray | float:
        """Cumulative distribution function. Used by the copula."""
        raise NotImplementedError(
            f"cdf() is not defined for {self.__class__.__name__}"
        )

    def ppf(self, q: np.ndarray | float) -> np.ndarray | float:
        """Inverse CDF (quantile function). Used by the copula."""
        raise NotImplementedError(
            f"ppf() is not defined for {self.__class__.__name__}"
        )


# --------------------------------------------------------------------------- #
#  Continuous distributions
# --------------------------------------------------------------------------- #

@dataclass
class Normal(Distribution):
    """Normal(mean, std)."""
    mean_: float
    std_: float
    label: str = ""

    def __post_init__(self):
        if self.std_ <= 0:
            raise ValueError(f"Normal requires std > 0, got {self.std_}")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mean_, self.std_, size=n)

    def describe(self) -> str:
        return f"Normal(mean={self.mean_:.4g}, std={self.std_:.4g})"

    def mean(self) -> float: return self.mean_
    def median(self) -> float: return self.mean_
    def std(self) -> float: return self.std_

    def pdf(self, x): return stats.norm.pdf(x, loc=self.mean_, scale=self.std_)
    def cdf(self, x): return stats.norm.cdf(x, loc=self.mean_, scale=self.std_)
    def ppf(self, q): return stats.norm.ppf(q, loc=self.mean_, scale=self.std_)


@dataclass
class Triangular(Distribution):
    """
    Triangular(low, mode, high).

    The default recommendation for analyst inputs because most analysts think
    in "worst / base / best" terms.
    """
    low: float
    mode: float
    high: float
    label: str = ""

    def __post_init__(self):
        if not (self.low <= self.mode <= self.high):
            raise ValueError(
                f"Triangular requires low <= mode <= high, got "
                f"({self.low}, {self.mode}, {self.high})"
            )
        if self.low == self.high:
            raise ValueError("Triangular requires low < high (degenerate range)")

    @property
    def _c(self) -> float:
        # scipy's shape parameter: location of mode within [0, 1]
        return (self.mode - self.low) / (self.high - self.low)

    @property
    def _loc(self) -> float:
        return self.low

    @property
    def _scale(self) -> float:
        return self.high - self.low

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.low, self.mode, self.high, size=n)

    def describe(self) -> str:
        return (
            f"Triangular(low={self.low:.4g}, mode={self.mode:.4g}, "
            f"high={self.high:.4g})"
        )

    def mean(self) -> float:
        return (self.low + self.mode + self.high) / 3.0

    def median(self) -> float:
        return float(stats.triang.median(self._c, loc=self._loc, scale=self._scale))

    def std(self) -> float:
        a, b, c = self.low, self.high, self.mode
        var = (a**2 + b**2 + c**2 - a*b - a*c - b*c) / 18.0
        return float(np.sqrt(var))

    def pdf(self, x): return stats.triang.pdf(x, self._c, loc=self._loc, scale=self._scale)
    def cdf(self, x): return stats.triang.cdf(x, self._c, loc=self._loc, scale=self._scale)
    def ppf(self, q): return stats.triang.ppf(q, self._c, loc=self._loc, scale=self._scale)


@dataclass
class Uniform(Distribution):
    """Uniform(low, high)."""
    low: float
    high: float
    label: str = ""

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"Uniform requires low < high, got ({self.low}, {self.high})")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=n)

    def describe(self) -> str:
        return f"Uniform(low={self.low:.4g}, high={self.high:.4g})"

    def mean(self) -> float: return 0.5 * (self.low + self.high)
    def median(self) -> float: return 0.5 * (self.low + self.high)
    def std(self) -> float: return (self.high - self.low) / np.sqrt(12.0)

    def pdf(self, x): return stats.uniform.pdf(x, loc=self.low, scale=self.high - self.low)
    def cdf(self, x): return stats.uniform.cdf(x, loc=self.low, scale=self.high - self.low)
    def ppf(self, q): return stats.uniform.ppf(q, loc=self.low, scale=self.high - self.low)


@dataclass
class Lognormal(Distribution):
    """
    Lognormal parameterised by the desired output mean and std (not the
    underlying normal parameters).

    For a lognormal with output mean m and std s:
        sigma^2 = ln(1 + (s/m)^2)
        mu      = ln(m) - sigma^2 / 2
    """
    mean_: float
    std_: float
    label: str = ""

    def __post_init__(self):
        if self.mean_ <= 0:
            raise ValueError(f"Lognormal requires mean > 0, got {self.mean_}")
        if self.std_ <= 0:
            raise ValueError(f"Lognormal requires std > 0, got {self.std_}")
        self._sigma2 = np.log1p((self.std_ / self.mean_) ** 2)
        self._sigma = float(np.sqrt(self._sigma2))
        self._mu = float(np.log(self.mean_) - 0.5 * self._sigma2)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(self._mu, self._sigma, size=n)

    def describe(self) -> str:
        return (
            f"Lognormal(out_mean={self.mean_:.4g}, out_std={self.std_:.4g}; "
            f"mu={self._mu:.4g}, sigma={self._sigma:.4g})"
        )

    def mean(self) -> float: return self.mean_
    def median(self) -> float: return float(np.exp(self._mu))

    def std(self) -> float: return self.std_

    def pdf(self, x): return stats.lognorm.pdf(x, self._sigma, scale=np.exp(self._mu))
    def cdf(self, x): return stats.lognorm.cdf(x, self._sigma, scale=np.exp(self._mu))
    def ppf(self, q): return stats.lognorm.ppf(q, self._sigma, scale=np.exp(self._mu))


@dataclass
class TruncatedNormal(Distribution):
    """Normal truncated to [low, high]."""
    mean_: float
    std_: float
    low: float
    high: float
    label: str = ""

    def __post_init__(self):
        if self.std_ <= 0:
            raise ValueError(f"TruncatedNormal requires std > 0, got {self.std_}")
        if self.low >= self.high:
            raise ValueError(
                f"TruncatedNormal requires low < high, got ({self.low}, {self.high})"
            )
        self._a = (self.low - self.mean_) / self.std_
        self._b = (self.high - self.mean_) / self.std_

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        # scipy.stats.truncnorm does not accept a numpy Generator directly,
        # so we drive it with uniform draws from the Generator and use ppf.
        u = rng.uniform(0.0, 1.0, size=n)
        return stats.truncnorm.ppf(u, self._a, self._b, loc=self.mean_, scale=self.std_)

    def describe(self) -> str:
        return (
            f"TruncatedNormal(mean={self.mean_:.4g}, std={self.std_:.4g}, "
            f"low={self.low:.4g}, high={self.high:.4g})"
        )

    def mean(self) -> float:
        return float(stats.truncnorm.mean(self._a, self._b, loc=self.mean_, scale=self.std_))

    def median(self) -> float:
        return float(stats.truncnorm.median(self._a, self._b, loc=self.mean_, scale=self.std_))

    def std(self) -> float:
        return float(stats.truncnorm.std(self._a, self._b, loc=self.mean_, scale=self.std_))

    def pdf(self, x): return stats.truncnorm.pdf(x, self._a, self._b, loc=self.mean_, scale=self.std_)
    def cdf(self, x): return stats.truncnorm.cdf(x, self._a, self._b, loc=self.mean_, scale=self.std_)
    def ppf(self, q): return stats.truncnorm.ppf(q, self._a, self._b, loc=self.mean_, scale=self.std_)


# --------------------------------------------------------------------------- #
#  Degenerate / discrete / empirical
# --------------------------------------------------------------------------- #

@dataclass
class PointEstimate(Distribution):
    """A fixed value. Degenerate distribution, std=0."""
    value: float
    label: str = ""
    is_continuous: bool = False  # Skipped by the copula

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value, dtype=float)

    def describe(self) -> str:
        return f"PointEstimate(value={self.value:.4g})"

    def mean(self) -> float: return self.value
    def median(self) -> float: return self.value
    def std(self) -> float: return 0.0

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x == self.value, np.inf, 0.0)

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x < self.value, 0.0, 1.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        return np.full_like(q, self.value, dtype=float)


@dataclass
class Discrete(Distribution):
    """
    Discrete distribution over labelled outcomes.

    outcomes: {label: (probability, value)}
    Probabilities must sum to 1.0 (tolerance: 1e-6).
    """
    outcomes: dict[str, tuple[float, float]]
    label: str = ""
    is_continuous: bool = False

    def __post_init__(self):
        probs = [p for p, _ in self.outcomes.values()]
        total = sum(probs)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Discrete probabilities must sum to 1.0, got {total:.6f} "
                f"for outcomes: {list(self.outcomes.keys())}"
            )
        for name, (p, _) in self.outcomes.items():
            if p < 0:
                raise ValueError(f"Discrete probability for '{name}' is negative: {p}")

        self._labels = list(self.outcomes.keys())
        self._probs = np.array([p for p, _ in self.outcomes.values()])
        self._values = np.array([v for _, v in self.outcomes.values()])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self._values), size=n, p=self._probs)
        return self._values[idx]

    def sample_labels(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample outcome LABELS (useful for scenario conditioning)."""
        idx = rng.choice(len(self._values), size=n, p=self._probs)
        return np.array(self._labels)[idx]

    def describe(self) -> str:
        parts = [f"{lbl}: {p:.1%}→{v:.3g}" for lbl, (p, v) in self.outcomes.items()]
        return "Discrete(" + ", ".join(parts) + ")"

    def mean(self) -> float: return float(np.sum(self._probs * self._values))
    def median(self) -> float:
        order = np.argsort(self._values)
        cum = np.cumsum(self._probs[order])
        return float(self._values[order][np.searchsorted(cum, 0.5)])
    def std(self) -> float:
        m = self.mean()
        return float(np.sqrt(np.sum(self._probs * (self._values - m) ** 2)))


@dataclass
class Empirical(Distribution):
    """Bootstrap resampling from historical data."""
    historical_values: list[float]
    label: str = ""
    is_continuous: bool = False

    def __post_init__(self):
        self._arr = np.asarray(self.historical_values, dtype=float)
        if self._arr.size == 0:
            raise ValueError("Empirical requires at least one historical value")

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self._arr, size=n, replace=True)

    def describe(self) -> str:
        return (
            f"Empirical(n={len(self._arr)}, "
            f"mean={self._arr.mean():.4g}, std={self._arr.std():.4g})"
        )

    def mean(self) -> float: return float(self._arr.mean())
    def median(self) -> float: return float(np.median(self._arr))
    def std(self) -> float: return float(self._arr.std())


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #

_DIST_TYPES = {
    "normal": Normal,
    "triangular": Triangular,
    "uniform": Uniform,
    "lognormal": Lognormal,
    "truncated_normal": TruncatedNormal,
    "truncnorm": TruncatedNormal,
    "point": PointEstimate,
    "fixed": PointEstimate,
    "discrete": Discrete,
    "empirical": Empirical,
}


def make_distribution(spec: dict[str, Any] | Distribution) -> Distribution:
    """
    Hydrate a distribution from a dict specification.

    Pitch configs declare distributions as plain dicts so they can be stored as
    Python modules, YAML, or JSON without needing to import the classes.

    Example specs:
        {"type": "normal", "mean": 0.08, "std": 0.03}
        {"type": "triangular", "low": 0.05, "mode": 0.08, "high": 0.12}
        {"type": "point", "value": 100}
        {"type": "discrete",
         "outcomes": {"bull": [0.3, 50], "base": [0.5, 35], "bear": [0.2, 20]}}

    Pass-through: if given a Distribution instance, returns it unchanged.
    """
    if isinstance(spec, Distribution):
        return spec

    if not isinstance(spec, dict):
        raise TypeError(f"Expected dict or Distribution, got {type(spec).__name__}")

    spec = dict(spec)  # don't mutate caller's dict
    dtype = spec.pop("type", None)
    if dtype is None:
        raise ValueError(f"Distribution spec missing 'type' field: {spec}")
    dtype = dtype.lower()

    # Strip metadata fields consumed by other systems (e.g. calibration).
    # These are documented as belonging on the spec but are not constructor args.
    for meta_key in ("historical_metric",):
        spec.pop(meta_key, None)

    if dtype not in _DIST_TYPES:
        raise ValueError(
            f"Unknown distribution type '{dtype}'. "
            f"Valid types: {sorted(_DIST_TYPES)}"
        )

    cls = _DIST_TYPES[dtype]
    label = spec.pop("label", "")

    # Normalise parameter names. Users can say "mean" instead of "mean_".
    if dtype in ("normal", "lognormal"):
        spec["mean_"] = spec.pop("mean", spec.pop("mean_", None))
        spec["std_"] = spec.pop("std", spec.pop("std_", None))
    elif dtype in ("truncated_normal", "truncnorm"):
        spec["mean_"] = spec.pop("mean", spec.pop("mean_", None))
        spec["std_"] = spec.pop("std", spec.pop("std_", None))
    elif dtype == "triangular":
        # Accept "peak" as synonym for "high" (spec in prompt uses "peak")
        if "peak" in spec and "high" not in spec:
            spec["high"] = spec.pop("peak")
    elif dtype == "discrete":
        # Normalise outcomes: dict may use {"label": [p, v]} or {"label": {"probability": p, ...}}
        out = spec["outcomes"]
        normalised: dict[str, tuple[float, float]] = {}
        for name, val in out.items():
            if isinstance(val, (list, tuple)) and len(val) == 2:
                normalised[name] = (float(val[0]), float(val[1]))
            elif isinstance(val, dict):
                p = float(val["probability"])
                v = float(val.get("value", val.get("value_impact", 0.0)))
                normalised[name] = (p, v)
            else:
                raise ValueError(f"Unrecognised outcome format for '{name}': {val!r}")
        spec["outcomes"] = normalised

    try:
        obj = cls(**spec)
    except TypeError as e:
        raise ValueError(
            f"Cannot construct {cls.__name__} from {spec}: {e}"
        ) from e

    if label:
        obj.label = label
    return obj


__all__ = [
    "Distribution",
    "Normal",
    "Triangular",
    "Uniform",
    "Lognormal",
    "TruncatedNormal",
    "PointEstimate",
    "Discrete",
    "Empirical",
    "make_distribution",
]
