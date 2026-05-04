"""
Pitch detail page — single-pitch deep dive.

Streamlit auto-discovers any .py file in `pages/` and adds it to the sidebar.
The leading "1_" controls ordering; underscores become spaces in the title.
"""
from __future__ import annotations

import streamlit as st

from streamlit_lib import (
    apply_global_styling,
    calibration_panel_charts,
    cdf_chart,
    distribution_chart,
    fmt_currency,
    fmt_metric,
    fmt_multiplier,
    fmt_percent,
    fmt_price,
    input_histogram_small,
    list_pitch_summaries,
    load_pitch,
    metric_unit,
    model_type_label,
    tornado_chart,
    variance_contribution_chart,
)


st.set_page_config(page_title="Pitch Detail — IMA Monte Carlo", layout="wide")
apply_global_styling()


# --------------------------------------------------------------------------- #
#  Sidebar — pitch picker
# --------------------------------------------------------------------------- #
summaries = list_pitch_summaries()
if not summaries:
    st.warning(
        "No published pitches found. Generate one with "
        "`python main.py --pitch <path> --pitch-id <id>` first."
    )
    st.stop()

ids = [p["pitch_id"] for p in summaries]
preselected = st.session_state.get("selected_pitch_id")
default_idx = ids.index(preselected) if preselected in ids else 0

with st.sidebar:
    st.title("Pitch Detail")
    selected_id = st.selectbox(
        "Select pitch",
        options=ids,
        index=default_idx,
        format_func=lambda x: next(
            (f"{p['ticker']}  ({p['pitch_id']})" for p in summaries if p["pitch_id"] == x),
            x,
        ),
    )
    st.session_state["selected_pitch_id"] = selected_id

pitch = load_pitch(selected_id)
if pitch is None:
    st.error(f"Could not load pitch {selected_id}")
    st.stop()


# --------------------------------------------------------------------------- #
#  Header band
# --------------------------------------------------------------------------- #
header_left, header_right = st.columns([3, 1])
with header_left:
    draft_chip = "  *(draft)*" if pitch.get("is_draft") else ""
    st.markdown(f"# `{pitch['ticker']}`{draft_chip}")
    meta_line = (
        f"{model_type_label(pitch.get('model_type', ''))} • "
        f"Current price {fmt_price(pitch.get('current_price'))} • "
        f"{pitch.get('n_simulations', 0):,} simulations • "
        f"generated {pitch.get('generated_at', '')[:19].replace('T', ' ')}"
    )
    st.caption(meta_line)

with header_right:
    st.caption(f"`{pitch['pitch_id']}`")


# --------------------------------------------------------------------------- #
#  Key metrics
# --------------------------------------------------------------------------- #
pa = pitch.get("probability_analysis", {})
dist = pitch.get("distribution", {})

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        "Mean Fair Value",
        fmt_currency(dist.get("mean")),
        fmt_percent(pa.get("expected_return"), 1, with_sign=True),
    )
with m2:
    st.metric("Probability of Upside", fmt_percent(pa.get("probability_upside"), 0))
with m3:
    st.metric("Risk / Reward", fmt_multiplier(pa.get("risk_reward_ratio")))
with m4:
    st.metric("5% VaR", fmt_currency(pa.get("var_5")))

st.divider()


# --------------------------------------------------------------------------- #
#  Thesis
# --------------------------------------------------------------------------- #
st.subheader("Thesis")
thesis = (pitch.get("thesis_statement") or "").strip()
tv = pitch.get("thesis_validation", {}) or {}
if tv.get("is_placeholder") or not thesis:
    st.warning("No thesis statement provided.")
else:
    st.markdown(
        f"> {thesis}".replace("\n", "  \n"),
    )

chip_cols = st.columns([1, 1, 4])
with chip_cols[0]:
    st.caption(f"{tv.get('word_count', 0)} words")
with chip_cols[1]:
    drv = tv.get("driver_count_estimate", 0)
    st.caption(f"~{drv} driver{'s' if drv != 1 else ''}")
with chip_cols[2]:
    warns = tv.get("warnings") or []
    if not warns:
        st.success("Within target")
    else:
        for w in warns:
            st.warning(w)


# --------------------------------------------------------------------------- #
#  Distribution + CDF
# --------------------------------------------------------------------------- #
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(distribution_chart(pitch), use_container_width=True)
with c2:
    st.plotly_chart(cdf_chart(pitch), use_container_width=True)


# --------------------------------------------------------------------------- #
#  Driver concentration
# --------------------------------------------------------------------------- #
conc = pitch.get("concentration")
if conc:
    if conc.get("is_concentrated"):
        st.success(f"**Focused thesis** — {conc.get('message', '')}")
    else:
        st.warning(f"**Diffuse drivers** — {conc.get('message', '')}")
    drivers = conc.get("top_drivers") or []
    if drivers:
        rows = []
        for name, share in drivers:
            display = (
                f"Catalyst · {name[9:]}" if isinstance(name, str) and name.startswith("catalyst:")
                else name
            )
            rows.append({"Input": display, "Variance share": fmt_percent(share, 0)})
        st.dataframe(rows, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
#  Calibration
# --------------------------------------------------------------------------- #
calibration = pitch.get("calibration") or []
if calibration:
    st.subheader("Historical Calibration")
    sev_rank = {"warning": 0, "caution": 1, "info": 2}
    calibration_sorted = sorted(
        calibration, key=lambda e: sev_rank.get(e.get("max_severity", "info"), 3)
    )
    for entry in calibration_sorted:
        with st.expander(
            f"{entry['input']} — {entry.get('label', '')}  "
            f"({len(entry.get('warnings') or [])} warnings)",
            expanded=False,
        ):
            charts = calibration_panel_charts(entry)
            if charts:
                cols = st.columns(len(charts))
                for col, (_, fig) in zip(cols, charts):
                    with col:
                        st.plotly_chart(fig, use_container_width=True)

            unit = metric_unit(entry.get("metric", ""))
            sm = entry.get("analyst_distribution_summary") or {}
            st.markdown(
                f"**Analyst distribution:** bear {fmt_metric(sm.get('p10'), unit)} • "
                f"base {fmt_metric(sm.get('mean'), unit)} • "
                f"bull {fmt_metric(sm.get('p90'), unit)}"
            )
            pct_rows = []
            for label, key_low, key_mid, key_high in (
                ("Company",
                 "analyst_p10_company_percentile",
                 "analyst_mean_company_percentile",
                 "analyst_p90_company_percentile"),
                ("Sector",
                 "analyst_p10_sector_percentile",
                 "analyst_mean_sector_percentile",
                 "analyst_p90_sector_percentile"),
            ):
                lo = entry.get(key_low)
                mid = entry.get(key_mid)
                hi = entry.get(key_high)
                if all(v is not None for v in (lo, mid, hi)):
                    pct_rows.append({
                        "Reference": label,
                        "Bear (P10)": f"{lo:.0f}th",
                        "Base (mean)": f"{mid:.0f}th",
                        "Bull (P90)": f"{hi:.0f}th",
                    })
            if pct_rows:
                st.dataframe(pct_rows, use_container_width=True, hide_index=True)

            if entry.get("reversion_note"):
                st.info(entry["reversion_note"])

            warnings = entry.get("warnings") or []
            if not warnings:
                st.success("Within historical norms")
            else:
                for w in warnings:
                    sev = w.get("severity", "info")
                    body = w.get("message", "")
                    sug = w.get("suggestion") or ""
                    block = f"**{body}**"
                    if sug:
                        block += f"\n\n→ {sug}"
                    if sev == "warning":
                        st.error(block)
                    elif sev == "caution":
                        st.warning(block)
                    else:
                        st.info(block)


# --------------------------------------------------------------------------- #
#  Sensitivity
# --------------------------------------------------------------------------- #
torn = tornado_chart(pitch)
varc = variance_contribution_chart(pitch)
if torn or varc:
    st.subheader("Sensitivity")
    s1, s2 = st.columns(2)
    if torn:
        with s1:
            st.plotly_chart(torn, use_container_width=True)
    if varc:
        with s2:
            st.plotly_chart(varc, use_container_width=True)


# --------------------------------------------------------------------------- #
#  Sampled input distributions
# --------------------------------------------------------------------------- #
inputs = pitch.get("inputs") or []
if inputs:
    st.subheader("Sampled Input Distributions")
    cols_per_row = 3
    for i in range(0, len(inputs), cols_per_row):
        chunk = inputs[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, inp in zip(cols, chunk):
            with col:
                st.caption(f"**{inp.get('label', inp['name'])}**")
                sm = inp.get("sampled_summary") or {}
                st.caption(
                    f"μ={sm.get('mean'):.3g} • P10={sm.get('p10'):.3g} • "
                    f"P90={sm.get('p90'):.3g}"
                    if sm.get("mean") is not None else ""
                )
                st.plotly_chart(
                    input_histogram_small(inp),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )


# --------------------------------------------------------------------------- #
#  Catalysts / scenarios
# --------------------------------------------------------------------------- #
catalysts = pitch.get("catalysts") or []
scenarios = pitch.get("scenarios") or {}
if catalysts:
    st.subheader("Catalysts")
    for cat in catalysts:
        with st.expander(cat["name"], expanded=True):
            rows = []
            for label, oc in (cat.get("outcomes") or {}).items():
                key = f"{cat['name']} | {label}"
                sc = scenarios.get(key) or {}
                impact_type = oc.get("impact_type", "additive")
                impact = oc.get("value_impact")
                impact_str = (
                    f"{impact:.2f}x" if impact_type == "multiplicative" and impact is not None
                    else fmt_currency(impact)
                )
                rows.append({
                    "Outcome": label,
                    "Probability": fmt_percent(oc.get("probability"), 0),
                    "Impact": impact_str,
                    "Mean FV | outcome": fmt_currency(sc.get("mean_fair_value")),
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)
