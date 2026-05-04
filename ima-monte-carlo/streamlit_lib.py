"""
Shared data loading + chart helpers for the Streamlit dashboard.

The Streamlit app reads the same JSON files exported by webapp_export.py
that power the React dashboard, so the two stay in sync automatically.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


PITCHES_DIR = Path(__file__).resolve().parent / "webapp" / "public" / "pitches"


# --------------------------------------------------------------------------- #
#  Global styling (called at the top of every page)
# --------------------------------------------------------------------------- #
_GLOBAL_CSS = """
<style>
    html, body, [class*="css"], [data-testid="stAppViewContainer"] {
        font-size: 22px;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-size: 1.02rem;
        line-height: 1.55;
    }
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        font-size: 0.92rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.85rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.92rem;
    }
    [data-testid="stMetricDelta"] {
        font-size: 0.95rem;
    }
    h1 { font-size: 2.1rem !important; }
    h2 { font-size: 1.55rem !important; }
    h3 { font-size: 1.25rem !important; }
    blockquote {
        font-size: 1.05rem;
        line-height: 1.6;
    }
    [data-testid="stDataFrame"] { font-size: 0.98rem; }
</style>
"""


def apply_global_styling() -> None:
    """Inject the dashboard's CSS. Call once at the top of each page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
#  Colors — keep aligned with webapp/src/lib/chartTheme.ts
# --------------------------------------------------------------------------- #
GREEN = "#2E7D32"
GREEN_SOFT = "rgba(50, 180, 80, 0.55)"
RED = "#C62828"
RED_SOFT = "rgba(220, 50, 50, 0.55)"
AMBER = "#F9A825"
BLUE = "#1565C0"
GRAY = "#4A4A4A"
GRAY_LIGHT = "#CCCCCC"
INK = "#1A1A1A"


# --------------------------------------------------------------------------- #
#  Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60)
def load_manifest() -> Optional[dict]:
    p = PITCHES_DIR / "_manifest.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


@st.cache_data(ttl=60)
def load_pitch(pitch_id: str) -> Optional[dict]:
    p = PITCHES_DIR / f"{pitch_id}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def list_pitch_summaries() -> list[dict]:
    m = load_manifest()
    if m is None:
        return []
    return list(m.get("pitches", []))


# --------------------------------------------------------------------------- #
#  Formatters
# --------------------------------------------------------------------------- #
def fmt_currency(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    abs_v = abs(value)
    if abs_v >= 1e9:
        return f"${value/1e9:.1f}B"
    if abs_v >= 1e6:
        return f"${value/1e6:.1f}M"
    if abs_v >= 1e3:
        return f"${value/1e3:.1f}K"
    return f"${value:.{digits}f}"


def fmt_price(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"${value:.{digits}f}"


def fmt_percent(
    value: Optional[float],
    digits: int = 1,
    with_sign: bool = False,
) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    sign = "+" if (with_sign and value > 0) else ""
    return f"{sign}{value*100:.{digits}f}%"


def fmt_multiplier(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}x"


def fmt_metric(value: Optional[float], unit: str, digits: int = 1) -> str:
    if unit == "ratio":
        return fmt_percent(value, digits)
    return fmt_multiplier(value, digits)


def metric_unit(metric: str) -> str:
    return "multiple" if "multiple" in (metric or "") else "ratio"


def model_type_label(t: str) -> str:
    return {
        "dcf": "DCF",
        "multiples": "Multiples",
        "sotp": "SOTP",
        "custom": "Custom",
    }.get((t or "").lower(), t or "—")


# --------------------------------------------------------------------------- #
#  Charts (plotly)
# --------------------------------------------------------------------------- #
def distribution_chart(pitch: dict) -> go.Figure:
    dist = pitch["distribution"]
    hist = dist["histogram"]
    edges = hist["edges"]
    counts = hist["counts"]
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    cp = pitch.get("current_price") or 0.0
    colors = [GREEN_SOFT if m >= cp else RED_SOFT for m in midpoints]
    width = (edges[-1] - edges[0]) / max(1, len(counts))

    fig = go.Figure()
    fig.add_bar(
        x=midpoints,
        y=counts,
        marker=dict(color=colors, line=dict(color="white", width=0.4)),
        width=[width] * len(counts),
        hovertemplate="Fair value: $%{x:.2f}<br>Draws: %{y}<extra></extra>",
        name="Fair Value",
    )

    pct = dist.get("percentiles", {})
    for q, label, color, dash in (
        ("10", "P10", GRAY, "dash"),
        ("50", "Median", BLUE, "dash"),
        ("90", "P90", GRAY, "dash"),
    ):
        v = pct.get(q)
        if v is not None and np.isfinite(v):
            fig.add_vline(
                x=v,
                line=dict(color=color, dash=dash, width=1.4),
                annotation_text=f"{label} {fmt_currency(v)}",
                annotation_position="top",
                annotation_font_size=10,
            )
    if cp:
        fig.add_vline(
            x=cp,
            line=dict(color=INK, width=2.0),
            annotation_text=f"Current {fmt_currency(cp)}",
            annotation_position="top",
        )

    fig.update_layout(
        title="Fair Value Distribution",
        xaxis_title="Fair Value per Share",
        yaxis_title="Frequency",
        bargap=0.0,
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


def cdf_chart(pitch: dict) -> go.Figure:
    pts = pitch["distribution"]["cdf_points"]
    xs = [p["value"] for p in pts]
    ys = [p["cumulative_prob"] for p in pts]
    cp = pitch.get("current_price") or 0.0

    fig = go.Figure()
    fig.add_scatter(
        x=xs,
        y=ys,
        mode="lines",
        line=dict(color=BLUE, width=2.4),
        fill="tozeroy",
        fillcolor="rgba(21, 101, 192, 0.10)",
        hovertemplate="$%{x:.2f}<br>%{y:.0%} cumulative<extra></extra>",
        name="CDF",
    )
    for h in (0.10, 0.25, 0.50, 0.75, 0.90):
        fig.add_hline(y=h, line=dict(color=GRAY_LIGHT, width=0.6))

    if cp:
        # Interpolate cumulative prob at current price
        cum = None
        for i in range(len(xs) - 1):
            if xs[i] <= cp <= xs[i + 1]:
                t = (cp - xs[i]) / max(1e-9, xs[i + 1] - xs[i])
                cum = ys[i] + t * (ys[i + 1] - ys[i])
                break
        fig.add_vline(
            x=cp,
            line=dict(color=INK, width=2),
            annotation_text=f"Current {fmt_currency(cp)}",
            annotation_position="top",
        )
        if cum is not None:
            fig.add_annotation(
                x=cp,
                y=cum,
                text=f"P(FV > current) = {fmt_percent(1 - cum, 0)}",
                showarrow=True,
                arrowhead=1,
                ax=60,
                ay=-30,
                bgcolor="white",
                bordercolor="#BBB",
            )

    fig.update_layout(
        title="Cumulative Probability",
        xaxis_title="Fair Value per Share",
        yaxis_title="Cumulative Probability",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        showlegend=False,
        margin=dict(l=10, r=10, t=50, b=10),
        height=380,
    )
    return fig


def tornado_chart(pitch: dict) -> Optional[go.Figure]:
    entries = pitch.get("tornado") or []
    usable = [
        e for e in entries
        if e.get("p10_fair_value") is not None
        and e.get("p90_fair_value") is not None
        and np.isfinite(e["p10_fair_value"])
        and np.isfinite(e["p90_fair_value"])
    ]
    if not usable:
        return None
    median = pitch["distribution"].get("median") or 0.0
    rows = []
    for e in usable:
        lo = min(e["p10_fair_value"], e["p90_fair_value"])
        hi = max(e["p10_fair_value"], e["p90_fair_value"])
        rows.append({
            "label": e["label"],
            "lo": lo,
            "hi": hi,
            "neg": -(max(0, median - lo)),
            "pos": max(0, hi - median),
            "span": hi - lo,
            "input_p10": e.get("input_p10"),
            "input_p90": e.get("input_p90"),
        })
    rows.sort(key=lambda r: r["span"])
    labels = [r["label"] for r in rows]

    fig = go.Figure()
    fig.add_bar(
        x=[r["neg"] for r in rows],
        y=labels,
        orientation="h",
        marker_color=RED,
        opacity=0.78,
        name="Low",
        hovertemplate="<b>%{y}</b><br>Low fair value: $%{customdata:.2f}<extra></extra>",
        customdata=[r["lo"] for r in rows],
    )
    fig.add_bar(
        x=[r["pos"] for r in rows],
        y=labels,
        orientation="h",
        marker_color=GREEN,
        opacity=0.78,
        name="High",
        hovertemplate="<b>%{y}</b><br>High fair value: $%{customdata:.2f}<extra></extra>",
        customdata=[r["hi"] for r in rows],
    )
    fig.update_layout(
        title="Tornado Sensitivity (deviation from median fair value)",
        barmode="relative",
        showlegend=True,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=10, r=10, t=50, b=40),
        height=max(220, 36 * len(rows) + 80),
        xaxis=dict(title=f"Δ vs median ({fmt_currency(median)})"),
    )
    return fig


def variance_contribution_chart(pitch: dict, min_share: float = 0.03) -> Optional[go.Figure]:
    entries = pitch.get("tornado") or []
    if not entries:
        return None
    total = sum(max(0, e.get("variance_share", 0)) for e in entries)
    if total <= 0:
        return None
    palette = [BLUE, GREEN, AMBER, "#6A1B9A", "#00838F", "#5D4037", "#455A64", "#AD1457"]
    items = []
    other = 0.0
    sorted_e = sorted(entries, key=lambda e: e.get("variance_share", 0), reverse=True)
    for e in sorted_e:
        share = e.get("variance_share", 0) / total
        if share < min_share:
            other += share
        else:
            items.append({"label": e["label"], "share": share})
    if other > 0:
        items.append({"label": "Other", "share": other})

    fig = go.Figure()
    cum = 0.0
    for i, it in enumerate(items):
        color = palette[i % len(palette)] if it["label"] != "Other" else GRAY
        fig.add_bar(
            x=[it["share"]],
            y=[""],
            orientation="h",
            marker_color=color,
            text=fmt_percent(it["share"], 0) if it["share"] >= 0.07 else "",
            textposition="inside",
            insidetextanchor="middle",
            hovertemplate=f"<b>{it['label']}</b><br>Share: %{{x:.1%}}<extra></extra>",
            name=it["label"],
            showlegend=True,
        )
        cum += it["share"]

    fig.update_layout(
        title="Variance Contribution (each input's share of output variance)",
        barmode="stack",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        yaxis=dict(showticklabels=False),
        margin=dict(l=10, r=10, t=50, b=10),
        height=200,
        legend=dict(orientation="h", y=-0.4),
    )
    return fig


def input_histogram_small(input_payload: dict) -> go.Figure:
    hist = input_payload["histogram"]
    edges = hist["edges"]
    counts = hist["counts"]
    if len(counts) == 0:
        return go.Figure()
    midpoints = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
    width = (edges[-1] - edges[0]) / max(1, len(counts))

    fig = go.Figure()
    fig.add_bar(
        x=midpoints,
        y=counts,
        marker_color=BLUE,
        opacity=0.55,
        width=[width] * len(counts),
        hovertemplate="%{x:.3g}<br>%{y} draws<extra></extra>",
    )
    sm = input_payload.get("sampled_summary", {})
    for key, color, dash, width_ in (
        ("p10", GRAY, "dash", 0.9),
        ("median", INK, "solid", 1.4),
        ("p90", GRAY, "dash", 0.9),
    ):
        v = sm.get(key)
        if v is not None and np.isfinite(v):
            fig.add_vline(x=v, line=dict(color=color, dash=dash, width=width_))

    fig.update_layout(
        showlegend=False,
        margin=dict(l=4, r=4, t=4, b=4),
        height=140,
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(visible=False),
    )
    return fig


def calibration_panel_charts(entry: dict) -> list[tuple[str, go.Figure]]:
    """One Plotly figure per available context (company, sector)."""
    out: list[tuple[str, go.Figure]] = []
    unit = metric_unit(entry.get("metric", ""))
    sm = entry.get("analyst_distribution_summary", {})
    a_mean = sm.get("mean")
    a_p10 = sm.get("p10")
    a_p90 = sm.get("p90")

    for label_key, ctx_key in (("Company history", "company_context"),
                                ("Sector peers", "sector_context")):
        ctx = entry.get(ctx_key)
        if not ctx:
            continue
        raw = [v for v in (ctx.get("raw_values") or []) if v is not None]
        if not raw:
            continue
        fig = go.Figure()
        fig.add_histogram(
            x=raw,
            nbinsx=24,
            marker_color=GRAY_LIGHT,
            opacity=0.85,
            hovertemplate="value: %{x:.3g}<br>n: %{y}<extra></extra>",
        )
        if ctx.get("median") is not None:
            fig.add_vline(x=ctx["median"], line=dict(color=INK, width=1.2),
                          annotation_text="median",
                          annotation_position="top",
                          annotation_font_size=10)
        if a_mean is not None:
            fig.add_vline(x=a_mean, line=dict(color=BLUE, width=2),
                          annotation_text="analyst base",
                          annotation_position="top",
                          annotation_font_size=10)
        if a_p90 is not None:
            fig.add_vline(x=a_p90, line=dict(color=GREEN, dash="dot", width=1.6))
        if a_p10 is not None:
            fig.add_vline(x=a_p10, line=dict(color=RED, dash="dot", width=1.6))

        n = ctx.get("n_observations", 0)
        ly = ctx.get("lookback_years")
        title = f"{label_key} — {ctx.get('entity', '')} (n={n}"
        if ly:
            title += f", {ly:.1f}y"
        title += ")"
        tickformat = ".0%" if unit == "ratio" else ".1f"
        fig.update_layout(
            title=title,
            margin=dict(l=10, r=10, t=50, b=10),
            height=240,
            xaxis=dict(tickformat=tickformat),
            yaxis=dict(visible=False),
            bargap=0.05,
            showlegend=False,
        )
        out.append((label_key, fig))
    return out


def severity_color(s: str) -> str:
    return {"warning": RED, "caution": AMBER, "info": BLUE}.get(s, GRAY)


# --------------------------------------------------------------------------- #
#  KPI helpers
# --------------------------------------------------------------------------- #
def expected_return_class(er: Optional[float]) -> str:
    if er is None or not np.isfinite(er):
        return "—"
    if er > 0.30:
        return "high"
    if er < 0:
        return "negative"
    return "mid"


__all__ = [
    "PITCHES_DIR",
    "apply_global_styling",
    "load_manifest",
    "load_pitch",
    "list_pitch_summaries",
    "fmt_currency",
    "fmt_price",
    "fmt_percent",
    "fmt_multiplier",
    "fmt_metric",
    "metric_unit",
    "model_type_label",
    "distribution_chart",
    "cdf_chart",
    "tornado_chart",
    "variance_contribution_chart",
    "input_histogram_small",
    "calibration_panel_charts",
    "severity_color",
    "expected_return_class",
    "GREEN", "RED", "AMBER", "BLUE", "GRAY",
]
