"""
Example: Multiples-based pitch (forward EPS × P/E).

Replace the assumptions below with your own before pitching. Run with:

    python main.py --pitch examples/example_multiples.py
"""

TICKER = "EXAMPLE"
CURRENT_PRICE = 25.00
SHARES_OUTSTANDING = 0                # Not needed for equity-variant P/E

MODEL_TYPE = "multiples"
MULTIPLES_VARIANT = "equity"          # "equity" (P/E, P/FCF) or "enterprise" (EV/EBITDA)

INPUTS = {
    # Forward EPS estimate — asymmetric upside
    "forward_metric": {
        "type": "triangular",
        "low": 1.80,
        "mode": 2.10,
        "high": 2.50,
        "label": "Forward EPS (FY+1)",
    },
    # P/E multiple — triangular with trough/fair/expansion case
    "multiple": {
        "type": "triangular",
        "low": 10.0,
        "mode": 13.0,
        "high": 16.0,
        "label": "P/E Multiple",
    },
}

CORRELATIONS = {
    # Higher earnings prints tend to pull multiples up modestly
    ("forward_metric", "multiple"): 0.20,
}

CATALYSTS = [
    {
        "name": "Activist Involvement",
        "outcomes": {
            "activist_takes_stake": {
                "probability": 0.25, "value_impact": 4.0, "impact_type": "additive"
            },
            "no_activist": {
                "probability": 0.75, "value_impact": 0.0, "impact_type": "additive"
            },
        },
    },
]
