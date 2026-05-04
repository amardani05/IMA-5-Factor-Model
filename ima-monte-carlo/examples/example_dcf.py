"""
Example: 5-year DCF pitch with terminal exit multiple.

Run with:

    python main.py --pitch examples/example_dcf.py

Uses HL (Hecla Mining, an S&P 600 small-cap) so historical calibration has
real yfinance data to compare against. The numbers are illustrative — this
is not investment advice.
"""

TICKER = "HL"
CURRENT_PRICE = 17.10
SHARES_OUTSTANDING = 600_000_000

MODEL_TYPE = "dcf"
PROJECTION_YEARS = 5

THESIS_STATEMENT = """
HL wins if (1) silver and gold prices stay near current levels long enough
for HL's per-share free cash flow to compound at mid-single digits as
incremental ounces from Keno Hill ramp. The stock is mispriced relative
to its precious-metals torque because the market still anchors to the
operational hiccups of 2022-2023.
"""

INPUTS = {
    "base_fcf": {
        "type": "normal",
        "mean": 150_000_000,
        "std": 30_000_000,
        "label": "Base Year FCF ($)",
        # base_fcf has no clean historical analog (it's a level, not a ratio)
    },
    "growth_rate": {
        "type": "triangular",
        "low": 0.02,
        "mode": 0.06,
        "high": 0.12,
        "label": "Annual FCF Growth Rate",
        "historical_metric": "fcf_growth_yoy",
    },
    "wacc": {
        "type": "normal",
        "mean": 0.10,
        "std": 0.01,
        "label": "WACC",
        # WACC is forward-looking — no historical analog
    },
    "terminal_multiple": {
        "type": "uniform",
        "low": 10.0,
        "high": 15.0,
        "label": "Terminal FCF Multiple",
        "historical_metric": "ev_ebitda_multiple",
    },
    "net_debt": {
        "type": "point",
        "value": 500_000_000,
        "label": "Net Debt ($)",
    },
    "shares": {
        "type": "point",
        "value": 600_000_000,
        "label": "Shares Outstanding",
    },
}

CORRELATIONS = {
    ("growth_rate", "wacc"): -0.15,
    ("growth_rate", "terminal_multiple"): 0.30,
}

CATALYSTS = []
