"""
About page — methodology, principles, limitations.
"""
import streamlit as st

from streamlit_lib import apply_global_styling

st.set_page_config(page_title="About — IMA Monte Carlo", layout="wide")
apply_global_styling()

st.title("About IMA Monte Carlo")
st.caption("Methodology, principles, and limitations.")

st.markdown(
    """
### What this is

The Illinois Investment Management Academy (IMA) is a student-run fundamental
investment program. This dashboard publishes Monte Carlo analyses of IMA
pitches: probability-weighted fair values, sensitivity decompositions, and
calibration of analyst inputs against historical distributions of the same
metrics.

Each pitch comes from a Python pipeline. Analysts declare distributional
assumptions for each model input — revenue growth, margins, multiples,
terminal growth, etc. — and the engine runs 50,000+ simulations to produce
a fair-value distribution.

### How simulations work

Each iteration draws one value per input from its declared distribution
(triangular, normal, uniform, lognormal, truncated, point, discrete,
empirical). Correlated inputs are sampled via a Gaussian copula — marginals
stay as declared, but the joint distribution respects the analyst-supplied
correlation matrix. Each draw is fed through the valuation model
(multiples, DCF, SOTP, or custom) to produce a fair-value-per-share. We
aggregate across all draws to get the output distribution and derived
statistics.

Two sensitivity views are computed. The **tornado** view freezes every other
input at its sampled median and varies one input from its P10 to P90,
recording the resulting fair-value swing. The **variance contribution** view
uses the squared correlation between each sampled input and the resulting
fair value as a Sobol-like proxy for the share of output variance the input
explains.

### Why historical calibration matters

For each input that maps to a historical metric (revenue growth, EBITDA
margin, ROIC, multiples), we pull the company's quarterly history and a
sector-peer pool from yfinance. The dashboard then shows where the analyst's
distribution sits inside both — a 90th-percentile bull case is flagged
differently from a 30th-percentile one.

This is descriptive, not prescriptive. The analyst is welcome to project
anything. Calibration just makes it explicit when a projection requires
structural-break justification.

### Why driver concentration is checked

A pitch where two inputs explain >80% of output variance is effectively a
two-driver bet. That's not a defect — it usually means the thesis is
focused — but it tells the audience which inputs deserve scrutiny.

The opposite case — diffuse drivers — usually means either the thesis is
multi-faceted (and should say so) or some inputs are over-specified as
distributions when they could be point estimates without changing the
conclusion.

### Limitations

- **Inputs are subjective.** Monte Carlo doesn't make opinions defensible,
  it makes them legible. Garbage in, garbage out still applies.
- **Historical data has gaps for small caps.** yfinance quarterly
  fundamentals can be sparse or missing for smaller IJR constituents.
  The dashboard surfaces "n=X" alongside every distribution so you can judge
  weight.
- **Mean reversion isn't universal.** Margins, ROIC, and multiples
  mean-revert empirically. But persistent moats and structural breaks can
  defeat reversion. Calibration warnings are a prompt to defend, not a veto.
- **Outputs are decision support, not predictions.** A 65% probability of
  upside is not a 65% chance of being right — it's a 65% chance, conditional
  on the analyst's input distributions being correct.

### Two dashboards, one source of truth

This Streamlit app and the React webapp at `webapp/` read the same JSON
files at `webapp/public/pitches/`. The Python pipeline writes them; both
dashboards display them. Pick the one that fits the task —
Streamlit is faster for analyst iteration, React is the public-facing
deployment.
"""
)
