"""
Publication-quality charts for a completed Monte Carlo simulation.

All charts are saved as PNG under an output directory (default: ./output).
The module exposes one `render_all()` entry point used by main.py, plus
individual chart functions for analysts who want to call them directly.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from config import HIST_BINS
from historical_calibration import CalibrationReport, METRIC_REGISTRY
from scenario_overlay import CatalystOverlay
from simulation import SimulationResults
from valuation_models import ValuationModel


# --------------------------------------------------------------------------- #
#  Styling
# --------------------------------------------------------------------------- #
GREEN = "#2E7D32"
RED = "#C62828"
GRAY = "#4A4A4A"
LIGHT_GRAY = "#CCCCCC"
BLUE = "#1565C0"
AMBER = "#F9A825"


def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
        "legend.frameon": False,
        "legend.fontsize": 10,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
    })


def _currency(x: float) -> str:
    if abs(x) >= 1e9: return f"${x/1e9:.1f}B"
    if abs(x) >= 1e6: return f"${x/1e6:.1f}M"
    if abs(x) >= 1e3: return f"${x/1e3:.1f}K"
    return f"${x:.2f}"


def _dollars(x, _pos=None) -> str:
    return _currency(x)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
#  1. Fair value distribution
# --------------------------------------------------------------------------- #
def plot_fair_value_distribution(
    results: SimulationResults, out_path: str | Path, bins: int = HIST_BINS
) -> Path:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fvs = results.fair_values
    price = results.current_price

    counts, edges, _ = ax.hist(
        fvs, bins=bins, color=LIGHT_GRAY, edgecolor="white", linewidth=0.4
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    for c, v in zip(counts, centers):
        color = GREEN if v >= price else RED
        ax.bar(v, c, width=edges[1] - edges[0], color=color, alpha=0.55,
               edgecolor="white", linewidth=0.3)

    p10 = results.percentiles.get(0.10, float(np.quantile(fvs, 0.10)))
    p50 = results.percentiles.get(0.50, float(np.quantile(fvs, 0.50)))
    p90 = results.percentiles.get(0.90, float(np.quantile(fvs, 0.90)))

    ax.axvline(price, color="black", linewidth=2.0, label=f"Current: {_currency(price)}")
    ax.axvline(p10, color=GRAY, linestyle="--", linewidth=1.2, label=f"P10: {_currency(p10)}")
    ax.axvline(p50, color=BLUE, linestyle="--", linewidth=1.2, label=f"Median: {_currency(p50)}")
    ax.axvline(p90, color=GRAY, linestyle="--", linewidth=1.2, label=f"P90: {_currency(p90)}")

    ax.set_xlabel("Fair Value per Share")
    ax.set_ylabel("Simulated Frequency")
    ax.set_title(
        f"{results.ticker} — Monte Carlo Fair Value Distribution  "
        f"(N={results.n_simulations:,})"
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.grid(axis="y", alpha=0.4)
    ax.legend(loc="upper right")

    ann = (
        f"P(undervalued) = {results.probability_upside:.1%}\n"
        f"Expected return = {results.expected_return:+.1%}\n"
        f"Mean FV = {_currency(results.mean)}   Std = {_currency(results.std)}"
    )
    ax.text(
        0.02, 0.97, ann, transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#BBBBBB"),
    )

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  2. Tornado chart
# --------------------------------------------------------------------------- #
def plot_tornado(
    results: SimulationResults, out_path: str | Path, max_inputs: int = 12
) -> Path | None:
    data = results.tornado_data
    if not data:
        return None

    _apply_style()
    data = data[:max_inputs]
    labels = [d["label"] for d in data]
    fv_lo = np.array([d["fv_low"] for d in data])
    fv_hi = np.array([d["fv_high"] for d in data])
    center = data[0].get("fv_center", float(np.median(results.fair_values)))

    # Sort so biggest span is at the top (matplotlib barh plots bottom-up)
    y_pos = np.arange(len(data))[::-1]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(data) + 2)))

    for y, d in zip(y_pos, data):
        lo, hi = d["fv_low"], d["fv_high"]
        left = min(lo, hi, center)
        # Left bar (below center)
        if min(lo, hi) < center:
            ax.barh(y, center - min(lo, hi), left=min(lo, hi),
                    color=RED, alpha=0.8, edgecolor="white", height=0.7)
        if max(lo, hi) > center:
            ax.barh(y, max(lo, hi) - center, left=center,
                    color=GREEN, alpha=0.8, edgecolor="white", height=0.7)

        # Annotate input range
        iv_lo, iv_hi = d["low_value"], d["high_value"]
        ax.text(
            max(lo, hi), y,
            f"  {iv_lo:.3g} → {iv_hi:.3g}",
            va="center", fontsize=9, color="#444444"
        )

    ax.axvline(center, color="black", linewidth=1.6, label=f"Median FV: {_currency(center)}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Fair Value per Share (P10 → P90 of input, all others at median)")
    ax.set_title(f"{results.ticker} — Tornado Sensitivity")
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.grid(axis="x", alpha=0.4)
    ax.legend(loc="lower right")

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  3. Variance contribution
# --------------------------------------------------------------------------- #
def plot_variance_contribution(
    results: SimulationResults, out_path: str | Path, min_pct: float = 0.03
) -> Path | None:
    contribs = dict(results.input_contributions)
    if not contribs:
        return None

    # Rescale to sum to 1 for display; tiny entries collapse into "Other"
    total = sum(v for v in contribs.values() if v > 0)
    if total <= 0:
        return None

    items = []
    other = 0.0
    for name, v in sorted(contribs.items(), key=lambda kv: kv[1], reverse=True):
        if v <= 0:
            continue
        share = v / total
        if share >= min_pct:
            items.append((name, share))
        else:
            other += share
    if other > 0:
        items.append(("Other", other))

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(items) + 2)))
    y = np.arange(len(items))[::-1]

    # Color catalysts differently
    colors = [AMBER if n.startswith("catalyst:") else BLUE for n, _ in items]
    names = [n.replace("catalyst:", "Catalyst · ") for n, _ in items]
    shares = [s for _, s in items]

    ax.barh(y, shares, color=colors, alpha=0.85, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for yi, s in zip(y, shares):
        ax.text(s + 0.005, yi, f"{s:.1%}", va="center", fontsize=10)

    ax.set_xlabel("Share of Output Variance (normalised)")
    ax.set_title(f"{results.ticker} — Variance Contribution by Input")
    ax.set_xlim(0, max(1.0, max(shares) * 1.15))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0%}"))
    ax.grid(axis="x", alpha=0.4)

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  4. Input distribution panel
# --------------------------------------------------------------------------- #
def plot_input_distributions(
    results: SimulationResults,
    model: ValuationModel,
    out_path: str | Path,
) -> Path:
    _apply_style()
    df = results.sampled_inputs
    cols = list(df.columns)
    n = len(cols)
    if n == 0:
        return Path(out_path)

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for i, col in enumerate(cols):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        vals = df[col].to_numpy()
        if np.std(vals) == 0:
            # Degenerate — show vertical line
            ax.axvline(vals[0], color=BLUE, linewidth=2.0)
            ax.set_xlim(vals[0] - 1, vals[0] + 1)
            ax.text(
                0.5, 0.5, f"Point\n{vals[0]:.3g}",
                transform=ax.transAxes, ha="center", va="center", fontsize=11
            )
        else:
            ax.hist(vals, bins=50, color=BLUE, alpha=0.55, edgecolor="white")
            p10, p50, p90 = np.quantile(vals, [0.10, 0.50, 0.90])
            ax.axvline(p10, color=GRAY, linestyle="--", linewidth=1.0)
            ax.axvline(p50, color="black", linewidth=1.2)
            ax.axvline(p90, color=GRAY, linestyle="--", linewidth=1.0)

            ax.text(
                0.97, 0.95,
                f"μ={vals.mean():.3g}\nP10={p10:.3g}\nP90={p90:.3g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"),
            )

        ax.set_title(model.input_label(col), fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle(f"{results.ticker} — Sampled Input Distributions", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  5. Cumulative probability chart
# --------------------------------------------------------------------------- #
def plot_cumulative_probability(
    results: SimulationResults, out_path: str | Path
) -> Path:
    _apply_style()
    fvs = np.sort(results.fair_values)
    cum = np.linspace(0, 1, len(fvs), endpoint=False) + 1 / len(fvs)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fvs, cum, color=BLUE, linewidth=2.0)
    ax.fill_between(fvs, 0, cum, color=BLUE, alpha=0.12)

    price = results.current_price
    # P(fair > price) at the crossing of the CDF with price
    prob_below = float(np.mean(results.fair_values <= price))
    prob_up = 1.0 - prob_below

    ax.axvline(price, color="black", linewidth=1.8, label=f"Current: {_currency(price)}")
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1.0, label="Median")

    for h in (0.10, 0.25, 0.50, 0.75, 0.90):
        ax.axhline(h, color=LIGHT_GRAY, linewidth=0.6)

    # Annotate the crossing
    ax.scatter([price], [prob_below], color=RED, zorder=5, s=45)
    ax.annotate(
        f"P(fair value > current) = {prob_up:.1%}",
        xy=(price, prob_below),
        xytext=(0.55, 0.20), textcoords="axes fraction",
        fontsize=10, color="#333333",
        arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#BBBBBB"),
    )

    ax.set_xlabel("Fair Value per Share")
    ax.set_ylabel("Cumulative Probability")
    ax.set_ylim(0, 1)
    ax.set_title(f"{results.ticker} — Cumulative Fair Value Distribution")
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0%}"))
    ax.grid(alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  6. Scenario comparison
# --------------------------------------------------------------------------- #
def plot_scenario_comparison(
    results: SimulationResults,
    overlay: CatalystOverlay,
    out_path: str | Path,
) -> Path | None:
    if not overlay:
        return None

    # Pick the catalyst with the largest variance contribution, if any
    cats = list(overlay.catalysts)
    if not cats:
        return None
    cat_scores = {
        cat.name: results.input_contributions.get(f"catalyst:{cat.name}", 0.0)
        for cat in cats
    }
    focal = max(cats, key=lambda c: cat_scores.get(c.name, 0.0))
    outcome_idx = results.sampled_catalysts.get(focal.name)
    if outcome_idx is None:
        return None

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    fvs = results.fair_values

    colors = [GREEN, RED, AMBER, BLUE, "#6A1B9A", "#00838F"]
    labels = list(focal.outcomes.keys())
    for i, label in enumerate(labels):
        mask = outcome_idx == i
        if not mask.any():
            continue
        sub = fvs[mask]
        ax.hist(
            sub, bins=60, alpha=0.45,
            color=colors[i % len(colors)],
            label=f"{label} (N={mask.sum():,}, μ={_currency(sub.mean())})",
            edgecolor="white",
        )

    ax.axvline(results.current_price, color="black", linewidth=1.8,
               label=f"Current: {_currency(results.current_price)}")
    ax.set_xlabel("Fair Value per Share")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{results.ticker} — Fair Value by Catalyst Outcome: {focal.name}")
    ax.xaxis.set_major_formatter(FuncFormatter(_dollars))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  7. Risk-reward scatter (multi-pitch)
# --------------------------------------------------------------------------- #
def plot_risk_reward(
    results_list: list[SimulationResults], out_path: str | Path
) -> Path:
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    xs, ys, labels = [], [], []
    for r in results_list:
        if r.current_price <= 0:
            continue
        p10 = r.percentiles.get(0.10, float(np.quantile(r.fair_values, 0.10)))
        downside = (p10 - r.current_price) / r.current_price * 100  # %
        upside = r.expected_return * 100
        xs.append(downside)
        ys.append(upside)
        labels.append(r.ticker)

    xs = np.array(xs); ys = np.array(ys)
    colors = np.where(np.abs(ys) > np.abs(xs), GREEN, RED)
    ax.scatter(xs, ys, c=colors, s=120, edgecolor="black", linewidth=0.6, alpha=0.8)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(
            lbl, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=10
        )

    lim = max(abs(np.nanmin(xs)) if len(xs) else 10,
              abs(np.nanmax(ys)) if len(ys) else 10, 10)
    ax.plot([-lim, 0], [lim, 0], color=GRAY, linestyle="--", linewidth=1.0,
            label="1:1 risk/reward")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Downside: (P10 fair value − current) / current  (%)")
    ax.set_ylabel("Upside: Expected return  (%)")
    ax.set_title("Pitch Comparison — Risk vs. Reward")
    ax.grid(alpha=0.4)
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0f}%"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _p: f"{x:.0f}%"))

    fig.tight_layout()
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  8. Historical calibration — single-input chart
# --------------------------------------------------------------------------- #
def _ratio_or_multiple_formatter(unit: str):
    if unit == "ratio":
        return FuncFormatter(lambda x, _p: f"{x:.0%}")
    return FuncFormatter(lambda x, _p: f"{x:.1f}x")


def _severity_color(severity: str) -> str:
    return {"warning": RED, "caution": AMBER, "info": BLUE}.get(severity, GRAY)


def plot_calibration_chart(
    report: CalibrationReport, out_path: str | Path,
) -> Path | None:
    """
    Side-by-side comparison of company history, sector peer history, and the
    analyst's distribution for a single calibrated input. Skips silently if
    there is no historical context at all.
    """
    if report.company_context is None and report.sector_context is None:
        return None

    spec = METRIC_REGISTRY.get(report.metric, {})
    unit = spec.get("unit", "ratio")
    label = report.label
    summary = report.input_distribution_summary

    panels = [
        ("Company history", report.company_context),
        ("Sector peers", report.sector_context),
    ]
    panels = [p for p in panels if p[1] is not None]

    _apply_style()
    n_panels = len(panels) + 1  # +1 for analyst distribution
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(10, 2.4 * n_panels + 1.2), sharex=True
    )
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(
        f"{report.input_name} — {label} — Calibration",
        fontsize=13, fontweight="bold",
    )

    # 1. Historical panels (one per available context)
    analyst_mean = summary.get("mean", float("nan"))
    analyst_p10 = summary.get("p10", float("nan"))
    analyst_p90 = summary.get("p90", float("nan"))

    fmt = (lambda x: f"{x:.1%}") if unit == "ratio" else (lambda x: f"{x:.2f}x")

    for ax, (title, ctx) in zip(axes[: len(panels)], panels):
        ax.hist(
            ctx.raw_values, bins=30, color=LIGHT_GRAY, alpha=0.85,
            edgecolor="white",
        )
        ax.axvline(ctx.median, color="black", linewidth=1.3,
                   label=f"median {fmt(ctx.median)}")
        ax.axvline(ctx.p10, color=GRAY, linestyle="--", linewidth=0.9)
        ax.axvline(ctx.p90, color=GRAY, linestyle="--", linewidth=0.9)
        if np.isfinite(analyst_mean):
            ax.axvline(analyst_mean, color=BLUE, linewidth=2.0,
                       label=f"analyst base {fmt(analyst_mean)}")
        if np.isfinite(analyst_p90):
            ax.axvline(analyst_p90, color=GREEN, linestyle=":", linewidth=1.6,
                       label=f"analyst bull {fmt(analyst_p90)}")
        if np.isfinite(analyst_p10):
            ax.axvline(analyst_p10, color=RED, linestyle=":", linewidth=1.6,
                       label=f"analyst bear {fmt(analyst_p10)}")
        ax.set_title(
            f"{title} — {ctx.entity}  "
            f"(n={ctx.n_observations}, lookback {ctx.lookback_years:.1f}y)",
            fontsize=11,
        )
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
        ax.xaxis.set_major_formatter(_ratio_or_multiple_formatter(unit))

    # 2. Analyst distribution panel — show as a bar (P10 — mean — P90)
    ax = axes[-1]
    ax.set_title("Analyst distribution (P10 / mean / P90)", fontsize=11)
    if all(np.isfinite([analyst_p10, analyst_mean, analyst_p90])):
        ax.hlines(
            y=0.5, xmin=analyst_p10, xmax=analyst_p90,
            color=BLUE, linewidth=4.0, alpha=0.7,
        )
        ax.scatter([analyst_mean], [0.5], color=BLUE, s=80, zorder=5)
        ax.scatter([analyst_p10, analyst_p90], [0.5, 0.5], color=GRAY,
                   s=40, zorder=5)
        for x, lbl in [
            (analyst_p10, f"P10\n{fmt(analyst_p10)}"),
            (analyst_mean, f"mean\n{fmt(analyst_mean)}"),
            (analyst_p90, f"P90\n{fmt(analyst_p90)}"),
        ]:
            ax.annotate(lbl, xy=(x, 0.5), xytext=(0, 18),
                        textcoords="offset points",
                        ha="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(_ratio_or_multiple_formatter(unit))
    ax.grid(axis="x", alpha=0.3)

    # 3. Warning banner
    if report.warnings:
        worst = max(
            report.warnings,
            key=lambda w: {"info": 0, "caution": 1, "warning": 2}.get(
                w.get("severity", "info"), 0
            ),
        )
        color = _severity_color(worst.get("severity", "info"))
        symbol = {"warning": "⚠", "caution": "•"}.get(worst.get("severity"), "ℹ")
        msg = worst.get("message", "")
        fig.text(
            0.02, 0.005,
            f"{symbol}  {msg}",
            fontsize=9, color=color, style="italic",
        )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  9. Calibration summary panel — small multiples
# --------------------------------------------------------------------------- #
def plot_calibration_summary(
    reports: list[CalibrationReport], out_path: str | Path,
) -> Path | None:
    """
    Single page showing all calibrated inputs as small multiples. Each subplot
    overlays the analyst's mean/P10/P90 onto the company's historical
    distribution. Sorted by warning severity (warnings first).
    """
    usable = [
        r for r in reports
        if r.company_context is not None or r.sector_context is not None
    ]
    if not usable:
        return None

    severity_rank = {"warning": 0, "caution": 1, "info": 2}
    usable.sort(key=lambda r: severity_rank.get(r.max_severity, 3))

    n = len(usable)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)

    _apply_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes).reshape(nrows, ncols)

    for i, report in enumerate(usable):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        ctx = report.company_context or report.sector_context
        spec = METRIC_REGISTRY.get(report.metric, {})
        unit = spec.get("unit", "ratio")
        fmt = (lambda x: f"{x:.0%}") if unit == "ratio" else (lambda x: f"{x:.1f}x")

        ax.hist(
            ctx.raw_values, bins=24, color=LIGHT_GRAY, alpha=0.85,
            edgecolor="white",
        )
        ax.axvline(ctx.median, color="black", linewidth=1.0)

        sm = report.input_distribution_summary
        if np.isfinite(sm.get("mean", float("nan"))):
            ax.axvline(sm["mean"], color=BLUE, linewidth=2.0)
        if np.isfinite(sm.get("p10", float("nan"))):
            ax.axvline(sm["p10"], color=RED, linestyle=":", linewidth=1.4)
        if np.isfinite(sm.get("p90", float("nan"))):
            ax.axvline(sm["p90"], color=GREEN, linestyle=":", linewidth=1.4)

        # Severity-tinted title
        title_color = _severity_color(report.max_severity)
        ax.set_title(
            f"{report.input_name} — {report.label}",
            fontsize=10, color=title_color,
        )
        ax.tick_params(axis="both", labelsize=8)
        ax.xaxis.set_major_formatter(_ratio_or_multiple_formatter(unit))
        ax.grid(axis="y", alpha=0.3)

        co_pct = report.analyst_mean_company_percentile
        sec_pct = report.analyst_mean_sector_percentile
        pieces = []
        if co_pct is not None:
            pieces.append(f"co {co_pct:.0f}%ile")
        if sec_pct is not None:
            pieces.append(f"sec {sec_pct:.0f}%ile")
        if pieces:
            ax.text(
                0.97, 0.95, "\n".join(pieces),
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC"),
            )

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis("off")

    fig.suptitle("Historical Calibration — All Inputs",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(out_path)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
#  Orchestrator
# --------------------------------------------------------------------------- #
def render_all(
    results: SimulationResults,
    model: ValuationModel,
    overlay: CatalystOverlay | None = None,
    output_dir: str | Path = "output",
    ticker: str | None = None,
    calibration_reports: list[CalibrationReport] | None = None,
) -> dict[str, Path]:
    """
    Generate all charts for a single simulation. Returns a dict mapping chart
    name → output path.
    """
    ticker = (ticker or results.ticker).lower().replace(" ", "_")
    out = _ensure_dir(output_dir)
    rendered: dict[str, Path] = {}

    rendered["fair_value_distribution"] = plot_fair_value_distribution(
        results, out / f"{ticker}_fair_value_distribution.png"
    )
    if results.tornado_data:
        rendered["tornado_chart"] = plot_tornado(
            results, out / f"{ticker}_tornado_chart.png"
        )
    if results.input_contributions:
        rendered["variance_contribution"] = plot_variance_contribution(
            results, out / f"{ticker}_variance_contribution.png"
        )
    rendered["input_distributions"] = plot_input_distributions(
        results, model, out / f"{ticker}_input_distributions.png"
    )
    rendered["cumulative_probability"] = plot_cumulative_probability(
        results, out / f"{ticker}_cumulative_probability.png"
    )
    if overlay and len(overlay):
        rr = plot_scenario_comparison(
            results, overlay, out / f"{ticker}_scenario_comparison.png"
        )
        if rr is not None:
            rendered["scenario_comparison"] = rr

    if calibration_reports:
        cal_out = out / "calibration"
        cal_out.mkdir(parents=True, exist_ok=True)
        for report in calibration_reports:
            slug = report.input_name.replace("/", "_").replace(" ", "_")
            p = plot_calibration_chart(
                report, cal_out / f"{ticker}_calibration_{slug}.png"
            )
            if p is not None:
                rendered[f"calibration_{slug}"] = p
        summary_path = plot_calibration_summary(
            calibration_reports,
            out / f"{ticker}_calibration_summary.png",
        )
        if summary_path is not None:
            rendered["calibration_summary"] = summary_path

    return {k: v for k, v in rendered.items() if v is not None}


__all__ = [
    "plot_fair_value_distribution",
    "plot_tornado",
    "plot_variance_contribution",
    "plot_input_distributions",
    "plot_cumulative_probability",
    "plot_scenario_comparison",
    "plot_calibration_chart",
    "plot_calibration_summary",
    "plot_risk_reward",
    "render_all",
]
