"""
Pluggable valuation model templates.

A valuation model maps a dict of sampled inputs → fair value per share.

Built-in models:
    MultiplesModel  — forward metric × multiple, equity or enterprise variants
    DCFModel        — N-year projection + terminal value (perpetuity or multiple)
    SOTPModel       — sum of independent segment valuations + holdco adjustments
    CustomModel     — user-supplied Python callable

Each model exposes:
    .inputs         — dict of input_name → Distribution (the full input set the
                      simulation engine needs to sample)
    .compute_fair_value(sampled_inputs)  — per-draw fair value
    .describe_inputs()                    — human-readable summary
"""
from __future__ import annotations

import warnings
from typing import Callable

import numpy as np

from config import TERMINAL_VALUE_CAP_MULTIPLE
from distributions import Distribution, make_distribution, PointEstimate


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def _hydrate_inputs(
    raw: dict[str, dict | Distribution],
) -> dict[str, Distribution]:
    """Convert a dict of specs/Distributions into a dict of Distributions."""
    return {name: make_distribution(spec) for name, spec in raw.items()}


def _prefixed_inputs(prefix: str, inputs: dict[str, Distribution]) -> dict[str, Distribution]:
    """Prefix every input name with '{prefix}/'. Used for SOTP namespacing."""
    return {f"{prefix}/{k}": v for k, v in inputs.items()}


def _unprefix(prefix: str, sampled: dict[str, float]) -> dict[str, float]:
    """Strip '{prefix}/' from keys so inner models see their native names."""
    plen = len(prefix) + 1
    return {k[plen:]: v for k, v in sampled.items() if k.startswith(f"{prefix}/")}


# --------------------------------------------------------------------------- #
#  Base class
# --------------------------------------------------------------------------- #

class ValuationModel:
    """Base class for valuation models."""

    def __init__(
        self,
        name: str,
        inputs: dict[str, dict | Distribution],
        current_price: float,
        shares_outstanding: float = 0.0,
    ):
        self.name = name
        self.inputs: dict[str, Distribution] = _hydrate_inputs(inputs)
        self.current_price = float(current_price)
        self.shares_outstanding = float(shares_outstanding)

    # To be overridden
    def compute_fair_value(self, sampled_inputs: dict[str, float]) -> float:
        raise NotImplementedError

    # Convenience: describe inputs
    def describe_inputs(self) -> str:
        lines = [f"Inputs for {self.name}:"]
        for name, dist in self.inputs.items():
            label = getattr(dist, "label", "") or name
            lines.append(f"  {name:<28} [{label}]  {dist.describe()}")
        return "\n".join(lines)

    # Used by visualization to label inputs prettily
    def input_label(self, name: str) -> str:
        dist = self.inputs.get(name)
        if dist is None:
            return name
        return getattr(dist, "label", "") or name


# --------------------------------------------------------------------------- #
#  Multiples model
# --------------------------------------------------------------------------- #

class MultiplesModel(ValuationModel):
    """
    Fair Value = (forward_metric * multiple) adjusted for capital structure.

    Two variants:
      - "equity":     fair_value = forward_metric * multiple       (P/E, P/FCF)
      - "enterprise": EV = forward_metric * multiple
                      equity = EV - net_debt + cash
                      fair_value = equity / shares
    """

    def __init__(
        self,
        name: str,
        inputs: dict,
        current_price: float,
        variant: str = "equity",
        shares_outstanding: float = 0.0,
    ):
        super().__init__(name, inputs, current_price, shares_outstanding)
        if variant not in ("equity", "enterprise"):
            raise ValueError(f"variant must be 'equity' or 'enterprise', got {variant!r}")
        self.variant = variant

        if "forward_metric" not in self.inputs:
            raise ValueError("MultiplesModel requires a 'forward_metric' input")
        if "multiple" not in self.inputs:
            raise ValueError("MultiplesModel requires a 'multiple' input")

        if variant == "enterprise":
            if self.shares_outstanding <= 0 and "shares" not in self.inputs:
                raise ValueError(
                    "Enterprise variant needs either shares_outstanding > 0 "
                    "or a 'shares' input."
                )
            # Ensure optional net_debt/cash exist as at least a point estimate
            self.inputs.setdefault("net_debt", PointEstimate(0.0, label="Net Debt ($)"))
            self.inputs.setdefault("cash", PointEstimate(0.0, label="Cash ($)"))

    def compute_fair_value(self, sampled: dict[str, float]) -> float:
        m = sampled["forward_metric"]
        mult = sampled["multiple"]
        if self.variant == "equity":
            return m * mult

        ev = m * mult
        net_debt = sampled.get("net_debt", 0.0)
        cash = sampled.get("cash", 0.0)
        equity = ev - net_debt + cash
        shares = sampled.get("shares", self.shares_outstanding)
        if shares <= 0:
            return float("nan")
        return equity / shares


# --------------------------------------------------------------------------- #
#  DCF model
# --------------------------------------------------------------------------- #

class DCFModel(ValuationModel):
    """
    Discounted cash flow: project FCF for N years, discount at WACC,
    add terminal value (perpetuity growth or exit multiple).

    Required inputs:
        base_fcf
        wacc
        one of: growth_rate  OR  growth_year_1, ..., growth_year_N
        one of: terminal_growth  OR  terminal_multiple

    Optional:
        net_debt, cash, shares

    Edge cases:
        - If wacc <= terminal_growth, cap terminal value at
          TERMINAL_VALUE_CAP_MULTIPLE × terminal FCF (a warning fires once).
        - Negative equity floored at zero (long-only).
    """

    def __init__(
        self,
        name: str,
        inputs: dict,
        current_price: float,
        projection_years: int = 5,
        shares_outstanding: float = 0.0,
    ):
        super().__init__(name, inputs, current_price, shares_outstanding)
        self.projection_years = int(projection_years)

        if "base_fcf" not in self.inputs:
            raise ValueError("DCFModel requires a 'base_fcf' input")
        if "wacc" not in self.inputs:
            raise ValueError("DCFModel requires a 'wacc' input")

        # Growth structure
        has_single = "growth_rate" in self.inputs
        has_per_year = all(
            f"growth_year_{y}" in self.inputs
            for y in range(1, self.projection_years + 1)
        )
        if not (has_single or has_per_year):
            raise ValueError(
                "DCFModel requires either a single 'growth_rate' input or one "
                f"'growth_year_y' input per projection year (1..{self.projection_years})."
            )

        # Terminal structure
        if "terminal_growth" not in self.inputs and "terminal_multiple" not in self.inputs:
            raise ValueError(
                "DCFModel requires either 'terminal_growth' or 'terminal_multiple'."
            )

        self.inputs.setdefault("net_debt", PointEstimate(0.0, label="Net Debt ($)"))
        self.inputs.setdefault("cash", PointEstimate(0.0, label="Cash ($)"))

        self._warned_tv_cap = False

    def compute_fair_value(self, sampled: dict[str, float]) -> float:
        base = sampled["base_fcf"]
        wacc = sampled["wacc"]

        if wacc <= 0:
            # Non-sensical WACC — will produce negative discount factors.
            return float("nan")

        # Project FCFs
        projected: list[float] = []
        current = base
        for year in range(1, self.projection_years + 1):
            key = f"growth_year_{year}"
            g = sampled[key] if key in sampled else sampled["growth_rate"]
            current = current * (1.0 + g)
            projected.append(current)

        # PV of explicit period
        pv_fcfs = sum(
            fcf / (1.0 + wacc) ** year
            for year, fcf in enumerate(projected, 1)
        )

        # Terminal value
        terminal_fcf = projected[-1]
        if "terminal_multiple" in sampled:
            terminal_value = terminal_fcf * sampled["terminal_multiple"]
        else:
            tg = sampled["terminal_growth"]
            if wacc <= tg:
                # Gordon Growth breaks. Cap at TERMINAL_VALUE_CAP_MULTIPLE * terminal FCF.
                terminal_value = terminal_fcf * TERMINAL_VALUE_CAP_MULTIPLE
                if not self._warned_tv_cap:
                    warnings.warn(
                        f"[{self.name}] WACC ({wacc:.3f}) <= terminal growth ({tg:.3f}) "
                        f"on at least one draw — capping terminal value at "
                        f"{TERMINAL_VALUE_CAP_MULTIPLE}× terminal FCF. This is "
                        f"expected for some draws when distributions overlap.",
                        stacklevel=2,
                    )
                    self._warned_tv_cap = True
            else:
                terminal_value = terminal_fcf * (1.0 + tg) / (wacc - tg)

        pv_terminal = terminal_value / (1.0 + wacc) ** self.projection_years

        # Enterprise → equity → per-share
        ev = pv_fcfs + pv_terminal
        equity = ev - sampled.get("net_debt", 0.0) + sampled.get("cash", 0.0)
        equity = max(equity, 0.0)  # Long-only floor
        shares = sampled.get("shares", self.shares_outstanding)
        if shares <= 0:
            return float("nan")
        return equity / shares


# --------------------------------------------------------------------------- #
#  SOTP model
# --------------------------------------------------------------------------- #

class SOTPModel(ValuationModel):
    """
    Sum of the Parts.

    segments: list of dicts like
        {
            "name": "Royalty Portfolio",
            "method": "multiples",     # or "dcf"
            "variant": "equity",       # for multiples; defaults to "equity"
            "projection_years": 5,     # for dcf
            "inputs": {input_name: dist_spec, ...},
        }

    holdco_adjustments:
        {
            "net_debt": dist_spec,                    # subtracted
            "cash":    dist_spec,                     # added (optional)
            "corporate_costs": dist_spec,             # annual overhead
            "corporate_costs_multiple": dist_spec,    # capitalisation multiple
            "shares": dist_spec,                      # shares outstanding
        }
    """

    def __init__(
        self,
        name: str,
        segments: list[dict],
        holdco_adjustments: dict,
        current_price: float,
        shares_outstanding: float = 0.0,
    ):
        # Build a combined input dict from all segments + holdco, prefixed to
        # avoid name collisions across segments.
        combined: dict[str, Distribution] = {}
        segment_specs: list[dict] = []
        for seg in segments:
            seg_name = seg["name"]
            method = seg["method"].lower()
            seg_inputs = _hydrate_inputs(seg.get("inputs", {}))
            # Track segment-local correlations if provided (applied by the
            # simulation engine via the top-level correlation spec — analysts
            # can pass them inside the segment dict for readability, then we
            # re-namespace them).
            seg_corr_raw = seg.get("correlations", {})
            seg_corr_prefixed: dict[tuple[str, str], float] = {}
            for (a, b), rho in seg_corr_raw.items():
                seg_corr_prefixed[(f"{seg_name}/{a}", f"{seg_name}/{b}")] = float(rho)

            seg_proj_years = int(seg.get("projection_years", 5))
            seg_variant = seg.get("variant", "equity")

            combined.update(_prefixed_inputs(seg_name, seg_inputs))
            segment_specs.append({
                "name": seg_name,
                "method": method,
                "variant": seg_variant,
                "projection_years": seg_proj_years,
                "input_names": list(seg_inputs.keys()),
                "correlations": seg_corr_prefixed,
            })

        hold_inputs = _hydrate_inputs(holdco_adjustments)
        combined.update(_prefixed_inputs("holdco", hold_inputs))

        super().__init__(name, combined, current_price, shares_outstanding)

        self.segments = segment_specs
        self.holdco_input_names = list(hold_inputs.keys())

        # Collect extra correlations that should be merged at the engine level.
        self._segment_correlations: dict[tuple[str, str], float] = {}
        for s in segment_specs:
            self._segment_correlations.update(s["correlations"])

        # Inputs is already hydrated; keep a flat view of the raw spec names
        # so callers (e.g. the simulation engine) can fetch them easily.

    @property
    def segment_correlations(self) -> dict[tuple[str, str], float]:
        """Correlations that the analyst specified inside segment dicts."""
        return self._segment_correlations

    # -------------------------------------------------------- #
    #  Fair value
    # -------------------------------------------------------- #
    def compute_fair_value(self, sampled: dict[str, float]) -> float:
        total_equity = 0.0

        for seg in self.segments:
            seg_name = seg["name"]
            seg_sampled = _unprefix(seg_name, sampled)
            method = seg["method"]

            if method == "multiples":
                m = seg_sampled["forward_metric"]
                mult = seg_sampled["multiple"]
                if seg["variant"] == "equity":
                    seg_equity = m * mult
                else:
                    seg_nd = seg_sampled.get("net_debt", 0.0)
                    seg_cash = seg_sampled.get("cash", 0.0)
                    seg_equity = m * mult - seg_nd + seg_cash
            elif method == "dcf":
                seg_equity = self._dcf_segment(seg, seg_sampled)
            else:
                raise ValueError(f"Unknown SOTP segment method: {method!r}")

            total_equity += seg_equity

        # Holdco adjustments
        holdco = _unprefix("holdco", sampled)
        net_debt = holdco.get("net_debt", 0.0)
        cash = holdco.get("cash", 0.0)
        corp_costs = holdco.get("corporate_costs", 0.0)
        corp_mult = holdco.get("corporate_costs_multiple", 0.0)

        total_equity = total_equity - net_debt + cash - (corp_costs * corp_mult)
        total_equity = max(total_equity, 0.0)

        shares = holdco.get("shares", self.shares_outstanding)
        if shares <= 0:
            return float("nan")
        return total_equity / shares

    def _dcf_segment(self, seg: dict, sampled: dict[str, float]) -> float:
        wacc = sampled["wacc"]
        if wacc <= 0:
            return float("nan")

        base = sampled["base_fcf"]
        years = seg["projection_years"]
        projected: list[float] = []
        current = base
        for y in range(1, years + 1):
            key = f"growth_year_{y}"
            g = sampled[key] if key in sampled else sampled["growth_rate"]
            current = current * (1.0 + g)
            projected.append(current)

        pv_fcfs = sum(fcf / (1.0 + wacc) ** y for y, fcf in enumerate(projected, 1))

        terminal_fcf = projected[-1]
        if "terminal_multiple" in sampled:
            terminal_value = terminal_fcf * sampled["terminal_multiple"]
        else:
            tg = sampled["terminal_growth"]
            if wacc <= tg:
                terminal_value = terminal_fcf * TERMINAL_VALUE_CAP_MULTIPLE
            else:
                terminal_value = terminal_fcf * (1.0 + tg) / (wacc - tg)

        pv_terminal = terminal_value / (1.0 + wacc) ** years
        ev = pv_fcfs + pv_terminal
        equity = ev - sampled.get("net_debt", 0.0) + sampled.get("cash", 0.0)
        return max(equity, 0.0)


# --------------------------------------------------------------------------- #
#  Custom model
# --------------------------------------------------------------------------- #

class CustomModel(ValuationModel):
    """
    Escape hatch: the analyst supplies a callable that maps sampled_inputs →
    fair value per share.
    """

    def __init__(
        self,
        name: str,
        fair_value_func: Callable[[dict[str, float]], float],
        inputs: dict,
        current_price: float,
        shares_outstanding: float = 0.0,
    ):
        super().__init__(name, inputs, current_price, shares_outstanding)
        if not callable(fair_value_func):
            raise TypeError("fair_value_func must be callable")
        self._fv = fair_value_func

    def compute_fair_value(self, sampled: dict[str, float]) -> float:
        return float(self._fv(sampled))


__all__ = [
    "ValuationModel",
    "MultiplesModel",
    "DCFModel",
    "SOTPModel",
    "CustomModel",
]
