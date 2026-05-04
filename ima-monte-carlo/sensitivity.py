"""
Sensitivity analysis on a completed simulation.

Two methods:

1. Tornado chart (one-at-a-time):
    Hold all inputs at their sampled median, vary one input from its P10 to
    its P90, and record the resulting fair values. The span (P90-case FV minus
    P10-case FV) is the tornado bar for that input.

2. Variance contribution (Sobol-like approximation):
    For each input, compute the squared correlation between the sampled input
    and the output fair value. This is a rough proxy for the fraction of
    output variance the input explains. It captures nonlinear effects
    reasonably well (via Pearson on non-transformed data), better than the
    tornado chart which only probes two extremes.

Also: scenario contribution — mean fair value conditional on each catalyst
outcome, and how much of output variance the catalyst accounts for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_overlay import CatalystOverlay
from simulation import SimulationResults
from valuation_models import ValuationModel


def tornado_analysis(
    model: ValuationModel,
    results: SimulationResults,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> list[dict]:
    """
    Return tornado-chart data, one entry per input:
        {
            "input": name,
            "label": display label,
            "low_value": input value at low_q,
            "high_value": input value at high_q,
            "fv_low": fair value when input is at low_q (others at median),
            "fv_high": fair value when input is at high_q (others at median),
            "span": abs(fv_high - fv_low),
        }

    Sorted descending by span.
    """
    df = results.sampled_inputs
    if df.empty:
        return []

    medians = df.median(numeric_only=True).to_dict()
    # Fair value with all inputs at median (used as a visual center on charts)
    try:
        fv_center = float(model.compute_fair_value(medians))
    except Exception:
        fv_center = float("nan")

    out: list[dict] = []
    for col in df.columns:
        vals = df[col].to_numpy()
        if np.std(vals) == 0:
            # Point-estimate / degenerate input — skip
            continue
        lo = float(np.quantile(vals, low_q))
        hi = float(np.quantile(vals, high_q))

        row_lo = dict(medians); row_lo[col] = lo
        row_hi = dict(medians); row_hi[col] = hi
        try:
            fv_lo = float(model.compute_fair_value(row_lo))
            fv_hi = float(model.compute_fair_value(row_hi))
        except Exception:
            fv_lo = fv_hi = float("nan")

        if not (np.isfinite(fv_lo) and np.isfinite(fv_hi)):
            continue

        out.append({
            "input": col,
            "label": model.input_label(col),
            "low_value": lo,
            "high_value": hi,
            "fv_low": fv_lo,
            "fv_high": fv_hi,
            "fv_center": fv_center,
            "span": abs(fv_hi - fv_lo),
        })

    out.sort(key=lambda d: d["span"], reverse=True)
    return out


def variance_contribution(
    results: SimulationResults,
    overlay: CatalystOverlay | None = None,
) -> dict[str, float]:
    """
    Variance contribution per input.

    For each continuous input, contribution = corr(input, fair_value) ** 2.
    For each catalyst, contribution is computed as the fraction of output
    variance explained by the categorical outcome (between-group variance /
    total variance), then mixed in.

    Contributions are NOT rescaled to sum to 1 (they're individual R² values
    and correlations across inputs mean they can over- or under-sum). The
    report rescales them for display.
    """
    fvs = results.fair_values
    df = results.sampled_inputs
    total_var = float(np.var(fvs))
    if total_var <= 0:
        return {name: 0.0 for name in df.columns}

    contribs: dict[str, float] = {}
    for col in df.columns:
        x = df[col].to_numpy()
        if np.std(x) == 0:
            contribs[col] = 0.0
            continue
        r = np.corrcoef(x, fvs)[0, 1]
        contribs[col] = float(r ** 2) if np.isfinite(r) else 0.0

    if overlay:
        for cat in overlay.catalysts:
            outcome_idx = results.sampled_catalysts.get(cat.name)
            if outcome_idx is None:
                continue
            # Between-group variance for a categorical predictor = eta²
            groups = np.unique(outcome_idx)
            if len(groups) < 2:
                contribs[f"catalyst:{cat.name}"] = 0.0
                continue
            overall_mean = fvs.mean()
            between = 0.0
            for g in groups:
                mask = outcome_idx == g
                n_g = int(mask.sum())
                if n_g == 0:
                    continue
                between += n_g * (fvs[mask].mean() - overall_mean) ** 2
            between /= len(fvs)
            eta2 = between / total_var
            contribs[f"catalyst:{cat.name}"] = float(eta2)

    return contribs


def scenario_conditional_means(
    results: SimulationResults,
    overlay: CatalystOverlay | None = None,
) -> dict[str, dict[str, float]]:
    """
    For each catalyst, return {outcome_label: mean_fair_value | outcome}.
    """
    if not overlay:
        return {}
    fvs = results.fair_values
    out: dict[str, dict[str, float]] = {}
    for cat in overlay.catalysts:
        idx = results.sampled_catalysts.get(cat.name)
        if idx is None:
            continue
        labels = list(cat.outcomes.keys())
        per_outcome: dict[str, float] = {}
        for oi, label in enumerate(labels):
            mask = idx == oi
            if mask.any():
                per_outcome[label] = float(fvs[mask].mean())
            else:
                per_outcome[label] = float("nan")
        out[cat.name] = per_outcome
    return out


def enrich_results(
    results: SimulationResults,
    model: ValuationModel,
    overlay: CatalystOverlay | None = None,
) -> SimulationResults:
    """
    Populate the sensitivity fields on a SimulationResults object in place,
    and return it.
    """
    results.tornado_data = tornado_analysis(model, results)
    results.input_contributions = variance_contribution(results, overlay)
    results.scenario_conditional_means = scenario_conditional_means(results, overlay)
    return results


__all__ = [
    "tornado_analysis",
    "variance_contribution",
    "scenario_conditional_means",
    "enrich_results",
]
