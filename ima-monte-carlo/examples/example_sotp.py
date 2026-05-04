"""
Example: Sum-of-the-Parts pitch — royalty segment (multiples) plus a pharma
operations segment (DCF), netted against holdco debt and corporate overhead.

Modeled after a healthcare holdco shape (e.g. INVA-style).

Run with:

    python main.py --pitch examples/example_sotp.py
"""

TICKER = "EXAMPLE_SOTP"
CURRENT_PRICE = 15.00
SHARES_OUTSTANDING = 70_000_000

MODEL_TYPE = "sotp"

SEGMENTS = [
    {
        "name": "Royalty Portfolio",
        "method": "multiples",
        "variant": "enterprise",
        "inputs": {
            "forward_metric": {
                "type": "triangular",
                "low": 300_000_000,
                "mode": 340_000_000,
                "high": 400_000_000,
                "label": "Royalty Revenue ($)",
            },
            "multiple": {
                "type": "uniform",
                "low": 8.0,
                "high": 12.0,
                "label": "EV/Revenue Multiple",
            },
        },
        "correlations": {},
    },
    {
        "name": "Pharma Operations",
        "method": "dcf",
        "projection_years": 5,
        "inputs": {
            "base_fcf": {
                "type": "normal",
                "mean": 50_000_000,
                "std": 10_000_000,
                "label": "Pharma Base FCF ($)",
            },
            "growth_rate": {
                "type": "triangular",
                "low": -0.05,
                "mode": 0.03,
                "high": 0.10,
                "label": "Pharma FCF Growth",
            },
            "wacc": {
                "type": "normal",
                "mean": 0.12,
                "std": 0.015,
                "label": "Pharma WACC",
            },
            "terminal_multiple": {
                "type": "uniform",
                "low": 6.0,
                "high": 10.0,
                "label": "Pharma Terminal Multiple",
            },
        },
        "correlations": {
            ("growth_rate", "terminal_multiple"): 0.25,
        },
    },
]

HOLDCO = {
    "net_debt": {
        "type": "point",
        "value": 100_000_000,
        "label": "HoldCo Net Debt ($)",
    },
    "corporate_costs": {
        "type": "point",
        "value": 20_000_000,
        "label": "Annual Corp Overhead ($)",
    },
    "corporate_costs_multiple": {
        "type": "uniform",
        "low": 5.0,
        "high": 8.0,
        "label": "Corp Cost Capitalization Multiple",
    },
    "shares": {
        "type": "point",
        "value": 70_000_000,
        "label": "Shares Outstanding",
    },
}

CATALYSTS = [
    {
        "name": "Pipeline Drug Approval",
        "outcomes": {
            "approval": {
                "probability": 0.40, "value_impact": 3.50, "impact_type": "additive"
            },
            "delay": {
                "probability": 0.35, "value_impact": -0.50, "impact_type": "additive"
            },
            "rejection": {
                "probability": 0.25, "value_impact": -2.00, "impact_type": "additive"
            },
        },
    },
]

# Top-level (cross-segment) correlations. Segment-local correlations are
# declared inside each segment's "correlations" field.
CORRELATIONS = {}
