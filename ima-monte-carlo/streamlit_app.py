"""
IMA Monte Carlo — Streamlit dashboard (landing page).

Run with:

    streamlit run streamlit_app.py

This page lists all published pitches (from webapp/public/pitches/_manifest.json).
The other pages — Pitch Detail, Compare, About — are auto-discovered by
Streamlit from the `pages/` directory.

Both this app and the React webapp read the same JSON files exported by
webapp_export.py, so they stay in sync automatically.
"""
from __future__ import annotations

import streamlit as st

from streamlit_lib import (
    apply_global_styling,
    expected_return_class,
    fmt_multiplier,
    fmt_percent,
    fmt_price,
    list_pitch_summaries,
    model_type_label,
)


st.set_page_config(
    page_title="IMA Monte Carlo",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_global_styling()


# --------------------------------------------------------------------------- #
#  Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("IMA Monte Carlo")
st.sidebar.caption("Pitch analysis dashboard")
st.sidebar.markdown(
    "Browse published pitches, drill into a single pitch, or compare side-by-side. "
    "Use the page selector above to navigate."
)
st.sidebar.divider()


# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.title("Pitch list")
st.caption(
    "Probability-weighted valuation analysis for Illinois Investment "
    "Management Academy pitches."
)

pitches = list_pitch_summaries()
if not pitches:
    st.info(
        "No published pitches yet.\n\n"
        "Run `python main.py --pitch your_pitch.py --pitch-id TICKER_DATE` "
        "to generate one. The pipeline writes the JSON to "
        "`webapp/public/pitches/` automatically."
    )
    st.stop()


# --------------------------------------------------------------------------- #
#  Filters + sort
# --------------------------------------------------------------------------- #
left, mid, right = st.columns([1, 1, 1.3])
with left:
    sort_key = st.selectbox(
        "Sort",
        options=[
            ("newest", "Newest first"),
            ("ticker", "Ticker A→Z"),
            ("expected_return", "Expected return (high→low)"),
            ("risk_reward", "Risk/Reward (high→low)"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )

with mid:
    model_types_present = sorted({p.get("model_type", "").lower() for p in pitches})
    model_filter = st.multiselect(
        "Model type",
        options=model_types_present,
        format_func=model_type_label,
    )

with right:
    bucket = st.multiselect(
        "Expected return",
        options=[(">30% upside", "high"), ("0–30%", "mid"), ("Negative", "neg")],
        format_func=lambda x: x[0],
    )


def in_bucket(er: float | None, key: str) -> bool:
    if er is None:
        return False
    if key == "high":
        return er > 0.30
    if key == "mid":
        return 0 <= er <= 0.30
    if key == "neg":
        return er < 0
    return True


filtered = []
for p in pitches:
    if model_filter and p.get("model_type", "").lower() not in model_filter:
        continue
    if bucket and not any(in_bucket(p.get("expected_return"), b[1]) for b in bucket):
        continue
    filtered.append(p)

sk = sort_key[0]
if sk == "ticker":
    filtered.sort(key=lambda x: (x.get("ticker") or ""))
elif sk == "expected_return":
    filtered.sort(key=lambda x: (x.get("expected_return") or float("-inf")), reverse=True)
elif sk == "risk_reward":
    filtered.sort(key=lambda x: (x.get("risk_reward_ratio") or float("-inf")), reverse=True)
else:
    filtered.sort(key=lambda x: (x.get("generated_at") or ""), reverse=True)


# --------------------------------------------------------------------------- #
#  Cards (3-up grid)
# --------------------------------------------------------------------------- #
if not filtered:
    st.warning("No pitches match the current filters.")
    st.stop()

st.write(f"**{len(filtered)} pitch{'es' if len(filtered) != 1 else ''}**")

cols_per_row = 3
for i in range(0, len(filtered), cols_per_row):
    chunk = filtered[i:i + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, p in zip(cols, chunk):
        with col:
            with st.container(border=True):
                head = st.columns([3, 2])
                with head[0]:
                    st.markdown(
                        f"### `{p.get('ticker', '—')}`"
                    )
                with head[1]:
                    st.caption(model_type_label(p.get("model_type", "")))

                er = p.get("expected_return")
                er_class = expected_return_class(er)
                er_color = (
                    "green" if er_class in ("high", "mid")
                    else "red" if er_class == "negative" else "gray"
                )
                er_str = fmt_percent(er, 1, with_sign=True)
                st.markdown(
                    f"**Expected return:** "
                    f"<span style='color:{er_color}; font-size:18px; font-weight:600'>"
                    f"{er_str}</span>",
                    unsafe_allow_html=True,
                )

                k1, k2 = st.columns(2)
                with k1:
                    st.metric(
                        "P(upside)",
                        fmt_percent(p.get("probability_upside"), 0),
                    )
                with k2:
                    st.metric(
                        "Risk / Reward",
                        fmt_multiplier(p.get("risk_reward_ratio"), 2),
                    )

                st.caption(
                    f"Current price: {fmt_price(p.get('current_price'))}"
                )

                thesis = (p.get("thesis_summary") or "").strip()
                if thesis:
                    st.markdown(
                        f"<small style='color:#666'>{thesis}…</small>",
                        unsafe_allow_html=True,
                    )

                st.page_link(
                    "pages/1_Pitch_Detail.py",
                    label=f"View {p['ticker']} →",
                    use_container_width=True,
                )
                # Stash selection for the detail page
                if st.button("Open", key=f"open_{p['pitch_id']}", use_container_width=True):
                    st.session_state["selected_pitch_id"] = p["pitch_id"]
                    st.switch_page("pages/1_Pitch_Detail.py")


st.divider()
st.caption(
    "Both this Streamlit app and the React webapp at `webapp/` read the same "
    "JSON files at `webapp/public/pitches/`. Push to git → Vercel deploys → "
    "this app picks up the changes on next reload."
)
