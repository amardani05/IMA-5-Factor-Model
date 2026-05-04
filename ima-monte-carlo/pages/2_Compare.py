"""
Compare page — side-by-side comparison of 2-4 pitches.
"""
from __future__ import annotations

import streamlit as st

from streamlit_lib import (
    apply_global_styling,
    fmt_currency,
    fmt_multiplier,
    fmt_percent,
    fmt_price,
    list_pitch_summaries,
    load_pitch,
    model_type_label,
)


st.set_page_config(page_title="Compare — IMA Monte Carlo", layout="wide")
apply_global_styling()
st.title("Compare pitches")
st.caption("Pick 2-4 pitches to compare side-by-side.")

summaries = list_pitch_summaries()
if not summaries:
    st.warning("No pitches found.")
    st.stop()

ids = [p["pitch_id"] for p in summaries]
default = st.session_state.get("compare_selection") or ids[: min(3, len(ids))]
selection = st.multiselect(
    "Pitches to compare",
    options=ids,
    default=default,
    max_selections=4,
    format_func=lambda x: next(
        (f"{p['ticker']}  ({p['pitch_id']})" for p in summaries if p["pitch_id"] == x),
        x,
    ),
)
st.session_state["compare_selection"] = selection

if len(selection) < 2:
    st.info("Select at least 2 pitches.")
    st.stop()

datas = [load_pitch(pid) for pid in selection]
datas = [d for d in datas if d is not None]
if not datas:
    st.error("Could not load selected pitches.")
    st.stop()


# --------------------------------------------------------------------------- #
#  Auto-generated banner — pick the leaders on each axis
# --------------------------------------------------------------------------- #
def safe_max(items, key):
    finite = [x for x in items if (x.get(*([key]) if isinstance(key, str) else key) or float("-inf")) > float("-inf")]
    return max(finite, key=lambda x: (x.get(key) or float("-inf"))) if finite else None


def warn_count(d: dict) -> int:
    return sum(len(c.get("warnings") or []) for c in (d.get("calibration") or []))


hi_er = max(datas, key=lambda d: (d.get("probability_analysis", {}).get("expected_return") or float("-inf")))
hi_rr = max(datas, key=lambda d: (d.get("probability_analysis", {}).get("risk_reward_ratio") or float("-inf")))
tightest = min(datas, key=warn_count)
banner_bits = [
    f"**{hi_er['ticker']}** has the highest expected return "
    f"({fmt_percent(hi_er['probability_analysis'].get('expected_return'), 0, with_sign=True)}).",
    f"**{hi_rr['ticker']}** has the best risk/reward "
    f"({fmt_multiplier(hi_rr['probability_analysis'].get('risk_reward_ratio'))}).",
    f"**{tightest['ticker']}** has the tightest calibration "
    f"({warn_count(tightest)} warning"
    f"{'s' if warn_count(tightest) != 1 else ''}).",
]
st.info(" ".join(banner_bits))


# --------------------------------------------------------------------------- #
#  Side-by-side comparison columns
# --------------------------------------------------------------------------- #
cols = st.columns(len(datas))
for col, d in zip(cols, datas):
    pa = d.get("probability_analysis", {})
    dist = d.get("distribution", {})
    tornado = d.get("tornado") or []
    total_var = sum(max(0, t.get("variance_share", 0)) for t in tornado) or 1.0
    top = sorted(tornado, key=lambda t: t.get("variance_share", 0), reverse=True)[:3]
    warns = warn_count(d)

    with col:
        with st.container(border=True):
            st.markdown(f"### `{d['ticker']}`")
            st.caption(
                f"{model_type_label(d.get('model_type', ''))} • "
                f"{fmt_price(d.get('current_price'))}"
            )
            rows = [
                ("Mean fair value", fmt_currency(dist.get("mean"))),
                ("Expected return", fmt_percent(pa.get("expected_return"), 1, with_sign=True)),
                ("P(upside)", fmt_percent(pa.get("probability_upside"), 0)),
                ("Risk / Reward", fmt_multiplier(pa.get("risk_reward_ratio"))),
                ("5% VaR", fmt_currency(pa.get("var_5"))),
                ("CVaR (5%)", fmt_currency(pa.get("cvar_5"))),
            ]
            st.dataframe(
                {"Metric": [r[0] for r in rows], "Value": [r[1] for r in rows]},
                use_container_width=True,
                hide_index=True,
            )

            st.caption("**Top drivers**")
            if not top:
                st.caption("—")
            else:
                for t in top:
                    share = (t.get("variance_share") or 0) / total_var
                    st.caption(f"• {t['label']} ({fmt_percent(share, 0)})")

            if warns == 0:
                st.success(f"{warns} calibration warnings")
            else:
                st.warning(f"{warns} calibration warning{'s' if warns != 1 else ''}")

            st.page_link(
                "pages/1_Pitch_Detail.py",
                label="Open detail →",
                use_container_width=True,
            )
