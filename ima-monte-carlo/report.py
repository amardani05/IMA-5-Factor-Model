"""
Formatted terminal output and CSV export for simulation results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from historical_calibration import CalibrationReport, METRIC_REGISTRY
from scenario_overlay import CatalystOverlay
from simulation import SimulationResults
from thesis_validator import wrap_thesis


# --------------------------------------------------------------------------- #
#  Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_price(x: float) -> str:
    if not np.isfinite(x):
        return "    n/a"
    return f"${x:,.2f}"


def _fmt_pct(x: float) -> str:
    if not np.isfinite(x):
        return "   n/a"
    return f"{x:+.1%}"


def _fmt_pct_abs(x: float) -> str:
    if not np.isfinite(x):
        return "   n/a"
    return f"{x:.1%}"


# --------------------------------------------------------------------------- #
#  Terminal report
# --------------------------------------------------------------------------- #
def print_report(
    results: SimulationResults,
    overlay: CatalystOverlay | None = None,
) -> None:
    price = results.current_price
    mean_ret = (results.mean - price) / price if price > 0 else 0.0
    med_ret = (results.median - price) / price if price > 0 else 0.0
    p5 = results.percentiles.get(0.05, float(np.quantile(results.fair_values, 0.05)))
    p10 = results.percentiles.get(0.10, float(np.quantile(results.fair_values, 0.10)))
    p25 = results.percentiles.get(0.25, float(np.quantile(results.fair_values, 0.25)))
    p75 = results.percentiles.get(0.75, float(np.quantile(results.fair_values, 0.75)))
    p90 = results.percentiles.get(0.90, float(np.quantile(results.fair_values, 0.90)))
    p95 = results.percentiles.get(0.95, float(np.quantile(results.fair_values, 0.95)))
    p5_ret = (p5 - price) / price if price > 0 else 0.0
    p10_ret = (p10 - price) / price if price > 0 else 0.0
    p90_ret = (p90 - price) / price if price > 0 else 0.0

    rr = results.risk_reward_ratio
    rr_str = f"{rr:.2f}x" if np.isfinite(rr) else "  inf"

    bar = "=" * 72
    sep = "  " + "─" * 38

    lines = []
    lines.append(bar)
    lines.append(f"  MONTE CARLO VALUATION — {results.ticker}")
    lines.append(
        f"  {results.model_name} | {results.n_simulations:,} simulations | "
        f"Seed: {results.seed}"
    )
    lines.append(bar)
    lines.append("")
    lines.append(f"  Current Price:         {_fmt_price(price)}")
    if results.n_excluded:
        lines.append(
            f"  Excluded draws:        {results.n_excluded:,} "
            f"(non-finite fair value)"
        )
    lines.append("")

    lines.append("  FAIR VALUE DISTRIBUTION")
    lines.append(sep)
    lines.append(f"  Mean:                  {_fmt_price(results.mean):>12}   ({_fmt_pct(mean_ret)})")
    lines.append(f"  Median:                {_fmt_price(results.median):>12}   ({_fmt_pct(med_ret)})")
    lines.append(f"  Std Dev:               {_fmt_price(results.std):>12}")
    lines.append("")
    lines.append(f"   5th percentile:       {_fmt_price(p5):>12}   ({_fmt_pct(p5_ret)})")
    lines.append(f"  10th percentile:       {_fmt_price(p10):>12}   ({_fmt_pct(p10_ret)})")
    lines.append(f"  25th percentile:       {_fmt_price(p25):>12}")
    lines.append(f"  75th percentile:       {_fmt_price(p75):>12}")
    lines.append(f"  90th percentile:       {_fmt_price(p90):>12}   ({_fmt_pct(p90_ret)})")
    lines.append(f"  95th percentile:       {_fmt_price(p95):>12}")
    lines.append("")

    lines.append("  PROBABILITY ANALYSIS")
    lines.append(sep)
    lines.append(f"  P(undervalued):        {_fmt_pct_abs(results.probability_upside)}")
    lines.append(f"  P(>20% upside):        {_fmt_pct_abs(results.probability_20_upside)}")
    lines.append(f"  P(>20% downside):      {_fmt_pct_abs(results.probability_20_downside)}")
    if results.probability_above_cost is not None:
        lines.append(f"  P(>cost basis):        {_fmt_pct_abs(results.probability_above_cost)}")
    lines.append(f"  Risk/Reward ratio:     {rr_str}")
    lines.append("")
    lines.append(f"  Upside capture:        {_fmt_price(results.upside_capture)}  /share (avg win)")
    lines.append(f"  Downside capture:      {_fmt_price(results.downside_capture)}  /share (avg loss)")
    lines.append(f"  CVaR (5%):             {_fmt_price(results.cvar_5)}   (avg of worst 5% outcomes)")
    lines.append(f"  Max loss (worst draw): {_fmt_pct(results.max_loss_pct)}")
    lines.append("")

    # Sensitivity
    if results.input_contributions:
        total = sum(v for v in results.input_contributions.values() if v > 0)
        if total > 0:
            ordered = sorted(
                results.input_contributions.items(),
                key=lambda kv: kv[1], reverse=True
            )
            lines.append("  SENSITIVITY (top inputs by variance contribution)")
            lines.append(sep)
            for i, (name, val) in enumerate(ordered[:6], 1):
                if val <= 0:
                    continue
                pct = val / total
                display_name = name.replace("catalyst:", "Catalyst · ")
                lines.append(f"  {i}. {display_name:<32} {pct:>6.1%} of variance")
            lines.append("")

    # Catalysts
    if overlay and overlay.catalysts and results.scenario_conditional_means:
        lines.append("  CATALYSTS")
        lines.append(sep)
        for cat in overlay.catalysts:
            means = results.scenario_conditional_means.get(cat.name, {})
            lines.append(f"  {cat.name}:")
            for label, (probability, value_impact) in [
                (lbl, (out.probability, out.value_impact)) for lbl, out in cat.outcomes.items()
            ]:
                mfv = means.get(label, float("nan"))
                mfv_s = _fmt_price(mfv) if np.isfinite(mfv) else "     n/a"
                lines.append(
                    f"      {label:<20}  p={probability:>5.0%}  →  mean FV = {mfv_s}"
                )
        lines.append("")

    lines.append(bar)
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
#  CSV exports
# --------------------------------------------------------------------------- #
def write_csvs(
    results: SimulationResults,
    output_dir: str | Path = "output",
    overlay: CatalystOverlay | None = None,
    ticker: str | None = None,
) -> dict[str, Path]:
    ticker = (ticker or results.ticker).lower().replace(" ", "_")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # 1. Full simulation results: fair values + sampled inputs + catalysts
    sim_df = results.sampled_inputs.copy()
    sim_df.insert(0, "fair_value", results.fair_values)
    sim_df.insert(1, "base_fair_value", results.base_fair_values)
    if overlay:
        label_arrays = overlay.outcome_label_arrays(results.sampled_catalysts)
        for cat in overlay.catalysts:
            col = f"catalyst__{cat.name.replace(' ', '_')}"
            sim_df[col] = label_arrays[cat.name]

    sim_path = out / f"{ticker}_simulation_results.csv"
    sim_df.to_csv(sim_path, index=False)
    written["simulation_results"] = sim_path

    # 2. Summary stats key-value file
    summary_rows = [
        ("ticker", results.ticker),
        ("model", results.model_name),
        ("n_simulations", results.n_simulations),
        ("seed", results.seed),
        ("current_price", results.current_price),
        ("mean", results.mean),
        ("median", results.median),
        ("std", results.std),
        ("expected_return", results.expected_return),
        ("probability_upside", results.probability_upside),
        ("probability_20_upside", results.probability_20_upside),
        ("probability_20_downside", results.probability_20_downside),
        ("risk_reward_ratio", results.risk_reward_ratio),
        ("upside_capture", results.upside_capture),
        ("downside_capture", results.downside_capture),
        ("var_5", results.var_5),
        ("var_10", results.var_10),
        ("cvar_5", results.cvar_5),
        ("max_loss_pct", results.max_loss_pct),
        ("n_excluded", results.n_excluded),
    ]
    for q, v in results.percentiles.items():
        summary_rows.append((f"percentile_{int(q*100):02d}", v))
    if results.probability_above_cost is not None:
        summary_rows.append(("probability_above_cost", results.probability_above_cost))

    summary_df = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary_path = out / f"{ticker}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    written["summary"] = summary_path

    # 3. Sensitivity data
    if results.tornado_data or results.input_contributions:
        tor_rows = []
        for d in results.tornado_data:
            tor_rows.append({
                "input": d["input"],
                "label": d["label"],
                "input_p10": d["low_value"],
                "input_p90": d["high_value"],
                "fv_at_p10": d["fv_low"],
                "fv_at_p90": d["fv_high"],
                "span": d["span"],
            })
        tor_df = pd.DataFrame(tor_rows)
        contrib_total = sum(v for v in results.input_contributions.values() if v > 0)
        contrib_df = pd.DataFrame([
            {
                "input": name,
                "raw_r2": v,
                "normalized_share": (v / contrib_total) if contrib_total > 0 else 0.0,
            }
            for name, v in results.input_contributions.items()
        ])
        sens_path = out / f"{ticker}_sensitivity.csv"
        with open(sens_path, "w") as f:
            f.write("# TORNADO\n")
            tor_df.to_csv(f, index=False)
            f.write("\n# VARIANCE CONTRIBUTION\n")
            contrib_df.to_csv(f, index=False)
        written["sensitivity"] = sens_path

    return written


# --------------------------------------------------------------------------- #
#  Pre-simulation calibration report
# --------------------------------------------------------------------------- #
_BAR = "=" * 72
_SEP = "  " + "─" * 38

_SEVERITY_GLYPH = {"warning": "⚠", "caution": "•", "info": "ℹ"}


def _fmt_metric(value: float, unit: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    if unit == "ratio":
        return f"{value:.1%}"
    return f"{value:.2f}x"


def _fmt_metric_range(lo: float, hi: float, unit: str) -> str:
    return f"{_fmt_metric(lo, unit)} to {_fmt_metric(hi, unit)}"


def print_calibration_report(
    thesis_text: str,
    thesis_validation: dict,
    calibration_reports: list[CalibrationReport],
) -> None:
    """
    Print the pre-simulation calibration block: thesis statement, per-input
    historical context with severity-tagged warnings, and a heads-up about
    the post-sim driver concentration check.
    """
    lines: list[str] = []
    lines.append(_BAR)
    lines.append("  PITCH CALIBRATION CHECK")
    lines.append(_BAR)
    lines.append("")

    # Thesis statement
    lines.append("  THESIS STATEMENT")
    lines.append(_SEP)
    for wl in wrap_thesis(thesis_text, width=64):
        lines.append(f"  {wl}")
    wc = thesis_validation.get("word_count", 0)
    drivers = thesis_validation.get("driver_count_estimate", 0)
    if thesis_validation.get("warnings"):
        status = "⚠ See thesis warnings below"
    else:
        status = "✓ Within target"
    lines.append("")
    lines.append(
        f"  [Word count: {wc}. Estimated drivers: {drivers}. {status}]"
    )
    for w in thesis_validation.get("warnings", []):
        lines.append(f"    ⚠ {w}")
    lines.append("")

    # Calibration per input
    lines.append("  HISTORICAL CALIBRATION")
    lines.append(_SEP)

    if not calibration_reports:
        lines.append("")
        lines.append(
            "  [No inputs declared `historical_metric` — calibration skipped. "
            "Add a `historical_metric` field to any input you want calibrated.]"
        )
        lines.append("")
    else:
        for report in calibration_reports:
            lines.append("")
            lines.append(f"  {report.input_name}  ({report.label})")
            spec = METRIC_REGISTRY.get(report.metric, {})
            unit = spec.get("unit", "ratio")

            if report.company_context is None and report.sector_context is None:
                if spec.get("extractor") is None:
                    note = spec.get(
                        "comment",
                        "This metric is forward-looking — no historical analog.",
                    )
                    lines.append(f"    [No historical data — {note}]")
                else:
                    lines.append(
                        "    [Insufficient historical data for both company and "
                        "sector — calibration unavailable.]"
                    )
                continue

            # Company context line
            cc = report.company_context
            if cc is not None:
                lines.append(
                    f"    Company history ({cc.lookback_years:.1f}y, n={cc.n_observations}): "
                    f"median {_fmt_metric(cc.median, unit)}, "
                    f"range {_fmt_metric_range(cc.min, cc.max, unit)}"
                )
            else:
                lines.append("    Company history: insufficient data")

            sc = report.sector_context
            if sc is not None:
                lines.append(
                    f"    {sc.entity}: "
                    f"median {_fmt_metric(sc.median, unit)}, "
                    f"range {_fmt_metric_range(sc.min, sc.max, unit)}"
                )

            sm = report.input_distribution_summary
            lines.append(
                f"    Analyst distribution:  bear {_fmt_metric(sm['p10'], unit)} / "
                f"base {_fmt_metric(sm['mean'], unit)} / "
                f"bull {_fmt_metric(sm['p90'], unit)}"
            )

            # Percentile placement
            def _pct_line(prefix: str, vals: tuple[Optional[float], Optional[float], Optional[float]]):
                lo, mid, hi = vals
                if any(v is None for v in (lo, mid, hi)):
                    return None
                return (
                    f"    {prefix} percentiles: "
                    f"bear {lo:.0f}th, base {mid:.0f}th, bull {hi:.0f}th"
                )

            line = _pct_line(
                "Company",
                (report.analyst_p10_company_percentile,
                 report.analyst_mean_company_percentile,
                 report.analyst_p90_company_percentile),
            )
            if line:
                lines.append(line)
            line = _pct_line(
                "Sector ",
                (report.analyst_p10_sector_percentile,
                 report.analyst_mean_sector_percentile,
                 report.analyst_p90_sector_percentile),
            )
            if line:
                lines.append(line)

            # Mean reversion note
            if report.reversion_note:
                lines.append("")
                # Wrap long note
                for wl in wrap_thesis(report.reversion_note, width=64):
                    lines.append(f"    {wl}")

            # Warnings
            if not report.warnings:
                lines.append("    ✓ Within historical norms")
            else:
                lines.append("")
                for w in report.warnings:
                    sev = w.get("severity", "info")
                    glyph = _SEVERITY_GLYPH.get(sev, "•")
                    msg = w.get("message", "")
                    lines.append(f"    {glyph} {msg}")
                    sug = w.get("suggestion")
                    if sug:
                        lines.append(f"       → {sug}")

    lines.append("")
    lines.append("  DRIVER CONCENTRATION (Pre-Simulation)")
    lines.append(_SEP)
    lines.append("  Sensitivity analysis runs after the simulation. If 1-2 inputs")
    lines.append("  explain >80% of output variance, the model is effectively")
    lines.append("  that few drivers — focused theses are GOOD.")
    lines.append(_BAR)
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
#  Post-simulation concentration verdict
# --------------------------------------------------------------------------- #
def print_concentration_verdict(concentration: dict) -> None:
    """Print the post-simulation driver-concentration verdict."""
    if not concentration:
        return

    lines: list[str] = []
    lines.append("")
    lines.append("  DRIVER CONCENTRATION (Post-Simulation)")
    lines.append(_SEP)
    drivers = concentration.get("top_drivers", [])
    if drivers:
        lines.append("  Top inputs by variance contribution:")
        for name, share in drivers:
            display = name.replace("catalyst:", "Catalyst · ")
            lines.append(f"    {display:<32} {share*100:>5.0f}%  of variance")
    lines.append("")
    glyph = "✓" if concentration.get("is_concentrated") else "⚠"
    msg = concentration.get("message", "")
    # Wrap message
    import textwrap
    wrapped = textwrap.wrap(msg, width=66)
    for i, wl in enumerate(wrapped):
        prefix = f"  {glyph} " if i == 0 else "    "
        lines.append(f"{prefix}{wl}")
    lines.append("")
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
#  Calibration CSV export
# --------------------------------------------------------------------------- #
def write_calibration_csv(
    calibration_reports: list[CalibrationReport],
    output_dir: str | Path = "output",
    ticker: str = "pitch",
) -> Optional[Path]:
    """Write a per-input calibration summary CSV. Skips if there's nothing to write."""
    if not calibration_reports:
        return None
    rows: list[dict] = []
    for r in calibration_reports:
        sm = r.input_distribution_summary
        cc = r.company_context
        sc = r.sector_context
        rows.append({
            "input": r.input_name,
            "metric": r.metric,
            "label": r.label,
            "analyst_mean": sm.get("mean"),
            "analyst_p10": sm.get("p10"),
            "analyst_p90": sm.get("p90"),
            "company_n": cc.n_observations if cc else None,
            "company_lookback_y": cc.lookback_years if cc else None,
            "company_median": cc.median if cc else None,
            "company_p10": cc.p10 if cc else None,
            "company_p90": cc.p90 if cc else None,
            "company_min": cc.min if cc else None,
            "company_max": cc.max if cc else None,
            "company_long_run_mean": cc.long_run_mean if cc else None,
            "company_recent_mean": cc.recent_mean if cc else None,
            "mean_reversion_implied": cc.mean_reversion_implied if cc else None,
            "sector_n": sc.n_observations if sc else None,
            "sector_median": sc.median if sc else None,
            "sector_p10": sc.p10 if sc else None,
            "sector_p90": sc.p90 if sc else None,
            "co_pct_mean": r.analyst_mean_company_percentile,
            "co_pct_p10": r.analyst_p10_company_percentile,
            "co_pct_p90": r.analyst_p90_company_percentile,
            "sec_pct_mean": r.analyst_mean_sector_percentile,
            "sec_pct_p10": r.analyst_p10_sector_percentile,
            "sec_pct_p90": r.analyst_p90_sector_percentile,
            "max_severity": r.max_severity,
            "n_warnings": len(r.warnings),
            "warnings_summary": " | ".join(
                f"[{w.get('severity', 'info')}] {w.get('message', '')}"
                for w in r.warnings
            ),
            "reversion_note": r.reversion_note or "",
        })
    df = pd.DataFrame(rows)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.lower().replace(" ", "_")
    path = out / f"{safe_ticker}_calibration.csv"
    df.to_csv(path, index=False)
    return path


__all__ = [
    "print_report",
    "write_csvs",
    "print_calibration_report",
    "print_concentration_verdict",
    "write_calibration_csv",
]
