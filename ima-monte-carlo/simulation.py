"""
Core Monte Carlo simulation engine.

Ties together:
    - input distributions (distributions.py)
    - correlated sampling (correlation.py)
    - valuation model (valuation_models.py)
    - discrete catalyst overlays (scenario_overlay.py)

Produces a SimulationResults object with fair-value distribution, summary
statistics, and diagnostic information.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CONFIDENCE_LEVELS,
    MAX_EXCLUDED_DRAW_FRACTION,
    N_SIMULATIONS,
    RANDOM_SEED,
)
from correlation import CorrelatedSampler
from scenario_overlay import CatalystOverlay
from valuation_models import SOTPModel, ValuationModel


@dataclass
class SimulationResults:
    # --- Identity ---
    ticker: str
    model_name: str
    n_simulations: int
    seed: int | None
    current_price: float

    # --- Core output ---
    fair_values: np.ndarray              # N-length array of fair values (post-catalyst)
    base_fair_values: np.ndarray         # Pre-catalyst fair values (for scenario analysis)
    sampled_inputs: pd.DataFrame         # N × K input matrix
    sampled_catalysts: dict[str, np.ndarray] = field(default_factory=dict)

    # --- Summary stats ---
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    percentiles: dict[float, float] = field(default_factory=dict)

    # --- Derived metrics ---
    expected_return: float = 0.0
    probability_upside: float = 0.0
    probability_20_upside: float = 0.0
    probability_20_downside: float = 0.0
    probability_above_cost: float | None = None
    upside_capture: float = 0.0
    downside_capture: float = 0.0
    risk_reward_ratio: float = float("nan")

    # --- VaR / CVaR ---
    var_5: float = 0.0
    var_10: float = 0.0
    cvar_5: float = 0.0

    # --- Tail metric ---
    max_loss_pct: float = 0.0

    # --- Quality ---
    n_excluded: int = 0

    # --- Sensitivity (populated by sensitivity.py) ---
    input_contributions: dict[str, float] = field(default_factory=dict)
    tornado_data: list[dict[str, Any]] = field(default_factory=list)
    scenario_conditional_means: dict[str, dict[str, float]] = field(default_factory=dict)


class MonteCarloSimulation:
    """
    Orchestrates a Monte Carlo valuation run.

    Parameters
    ----------
    model : ValuationModel
        The valuation model whose `inputs` dict defines all stochastic inputs.
    correlations : dict[tuple[str, str], float], optional
        Partial correlation specification. See correlation.py.
    catalysts : list[dict | Catalyst], optional
        Discrete catalyst overlays applied to the base fair value.
    n_simulations : int
    seed : int | None
    ticker : str
        Display ticker — used only for the results object + charts.
    cost_basis : float | None
        If provided, computes P(fair value > cost_basis).
    """

    def __init__(
        self,
        model: ValuationModel,
        correlations: dict[tuple[str, str], float] | None = None,
        catalysts: list[dict] | None = None,
        n_simulations: int = N_SIMULATIONS,
        seed: int | None = RANDOM_SEED,
        ticker: str = "",
        cost_basis: float | None = None,
    ):
        self.model = model
        self.ticker = ticker or model.name
        self.n_simulations = int(n_simulations)
        self.seed = seed
        self.cost_basis = cost_basis

        # Merge SOTP segment correlations with top-level correlations (if any)
        merged_corr = dict(correlations or {})
        if isinstance(model, SOTPModel):
            for pair, rho in model.segment_correlations.items():
                if pair not in merged_corr and tuple(reversed(pair)) not in merged_corr:
                    merged_corr[pair] = rho
        self.correlations = merged_corr

        self.sampler = CorrelatedSampler(model.inputs, self.correlations)
        self.overlay = CatalystOverlay(catalysts)

    # ------------------------------------------------------------------ #
    #  Execution
    # ------------------------------------------------------------------ #
    def run(self) -> SimulationResults:
        rng = np.random.default_rng(self.seed)

        # 1. Sample all inputs
        sampled: dict[str, np.ndarray] = self.sampler.sample(self.n_simulations, rng)
        input_df = pd.DataFrame(sampled)

        # 2. Sample catalyst outcomes
        catalyst_outcomes = self.overlay.sample_outcomes(self.n_simulations, rng)

        # 3. Compute per-draw base fair value (vectorisation is awkward with
        #    dict-based interface; loop at Python speed is fine — <30s for 50k
        #    draws on any of the example models)
        base_fvs = np.empty(self.n_simulations, dtype=float)
        input_names = list(input_df.columns)
        # Pre-extract columns as numpy arrays for speed
        col_arrays = {n: input_df[n].to_numpy() for n in input_names}

        for i in range(self.n_simulations):
            row = {n: col_arrays[n][i] for n in input_names}
            try:
                v = self.model.compute_fair_value(row)
            except Exception:
                v = float("nan")
            base_fvs[i] = v

        # 4. Apply catalyst overlay
        if self.overlay:
            fvs = self.overlay.apply(base_fvs, catalyst_outcomes)
        else:
            fvs = base_fvs

        # 5. Drop non-finite draws
        finite_mask = np.isfinite(fvs) & np.isfinite(base_fvs)
        n_excluded = int((~finite_mask).sum())
        if n_excluded > 0:
            frac = n_excluded / self.n_simulations
            msg = (
                f"{n_excluded} of {self.n_simulations} draws ({frac:.1%}) produced "
                f"NaN/inf fair values and were excluded."
            )
            if frac > MAX_EXCLUDED_DRAW_FRACTION:
                warnings.warn(msg + " Input distributions may be misconfigured.", stacklevel=2)
            else:
                warnings.warn(msg, stacklevel=2)

        fvs_clean = fvs[finite_mask]
        base_clean = base_fvs[finite_mask]
        input_df_clean = input_df.loc[finite_mask].reset_index(drop=True)
        catalyst_outcomes_clean = {
            n: arr[finite_mask] for n, arr in catalyst_outcomes.items()
        }

        # 6. Build results
        return self._build_results(
            fair_values=fvs_clean,
            base_fair_values=base_clean,
            sampled_inputs=input_df_clean,
            sampled_catalysts=catalyst_outcomes_clean,
            n_excluded=n_excluded,
        )

    # ------------------------------------------------------------------ #
    #  Result assembly
    # ------------------------------------------------------------------ #
    def _build_results(
        self,
        fair_values: np.ndarray,
        base_fair_values: np.ndarray,
        sampled_inputs: pd.DataFrame,
        sampled_catalysts: dict[str, np.ndarray],
        n_excluded: int,
    ) -> SimulationResults:
        price = float(self.model.current_price)
        fvs = fair_values

        mean = float(np.mean(fvs))
        median = float(np.median(fvs))
        std = float(np.std(fvs))

        pct = {q: float(np.quantile(fvs, q)) for q in CONFIDENCE_LEVELS}

        expected_return = (mean - price) / price if price > 0 else 0.0
        prob_up = float(np.mean(fvs > price))

        # Upside / downside capture
        upside_mask = fvs > price
        downside_mask = fvs < price
        up_cap = float(np.mean(fvs[upside_mask] - price)) if upside_mask.any() else 0.0
        dn_cap = float(np.mean(price - fvs[downside_mask])) if downside_mask.any() else 0.0
        rr = float(up_cap / dn_cap) if dn_cap > 0 else float("inf")

        var_5 = float(np.quantile(fvs, 0.05))
        var_10 = float(np.quantile(fvs, 0.10))
        tail_mask = fvs <= var_5
        cvar_5 = float(np.mean(fvs[tail_mask])) if tail_mask.any() else var_5

        max_loss_pct = (float(np.min(fvs)) - price) / price if price > 0 else 0.0

        prob_20up = float(np.mean(fvs >= 1.2 * price)) if price > 0 else 0.0
        prob_20dn = float(np.mean(fvs <= 0.8 * price)) if price > 0 else 0.0

        prob_above_cost = None
        if self.cost_basis is not None and self.cost_basis > 0:
            prob_above_cost = float(np.mean(fvs > self.cost_basis))

        return SimulationResults(
            ticker=self.ticker,
            model_name=self.model.name,
            n_simulations=len(fvs),
            seed=self.seed,
            current_price=price,
            fair_values=fvs,
            base_fair_values=base_fair_values,
            sampled_inputs=sampled_inputs,
            sampled_catalysts=sampled_catalysts,
            mean=mean,
            median=median,
            std=std,
            percentiles=pct,
            expected_return=expected_return,
            probability_upside=prob_up,
            probability_20_upside=prob_20up,
            probability_20_downside=prob_20dn,
            probability_above_cost=prob_above_cost,
            upside_capture=up_cap,
            downside_capture=dn_cap,
            risk_reward_ratio=rr,
            var_5=var_5,
            var_10=var_10,
            cvar_5=cvar_5,
            max_loss_pct=max_loss_pct,
            n_excluded=n_excluded,
        )


def check_driver_concentration(
    results: SimulationResults,
    threshold: float = 0.80,
    top_k: int = 2,
) -> dict:
    """
    Check whether the top-k inputs explain most of the output variance.

    Uses `results.input_contributions` (raw R²-style scores) normalised so
    they sum to 1 across positive contributors. This makes the
    concentration measure a proper share of explained variance.

    Returns
    -------
    dict with keys:
        is_concentrated : bool   — top_k share >= threshold
        top_drivers     : list of (input_name, normalised_share)
        concentration_pct : float (sum of top_k shares)
        message         : str    — analyst-facing summary
    """
    contribs = results.input_contributions or {}
    if not contribs:
        return {
            "is_concentrated": False,
            "top_drivers": [],
            "concentration_pct": 0.0,
            "message": "No input contributions available — sensitivity not yet run.",
        }

    positive = {k: v for k, v in contribs.items() if v > 0}
    total = sum(positive.values())
    if total <= 0:
        return {
            "is_concentrated": False,
            "top_drivers": [],
            "concentration_pct": 0.0,
            "message": "All inputs explain ~0% of variance (output is nearly constant).",
        }

    sorted_drivers = sorted(positive.items(), key=lambda kv: kv[1], reverse=True)
    normalised = [(name, val / total) for name, val in sorted_drivers]

    top_share = sum(s for _, s in normalised[:top_k])
    is_concentrated = top_share >= threshold

    if is_concentrated:
        top_str = ", ".join(
            f"{name.replace('catalyst:', 'Catalyst · ')} ({s*100:.0f}%)"
            for name, s in normalised[:top_k]
        )
        rest = max(0.0, 1.0 - top_share)
        n_rest = max(0, len(normalised) - top_k)
        message = (
            f"This pitch is effectively a {top_k}-driver bet. "
            f"Top {top_k} inputs explain {top_share*100:.0f}% of output "
            f"variance: {top_str}. The remaining {n_rest} input(s) collectively "
            f"contribute ~{rest*100:.0f}% and could be set as point estimates "
            f"without changing the conclusion."
        )
    else:
        message = (
            f"No single input dominates. Top {top_k} explain "
            f"{top_share*100:.0f}% of variance. This is a multi-driver thesis."
        )

    return {
        "is_concentrated": is_concentrated,
        "top_drivers": normalised[:5],
        "concentration_pct": top_share,
        "message": message,
    }


__all__ = ["MonteCarloSimulation", "SimulationResults", "check_driver_concentration"]
