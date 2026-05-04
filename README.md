# IMA Monte Carlo Valuation Framework

Probabilistic valuation tool for the Illinois Investment Management Academy pitch process. The analyst expresses each valuation assumption as a probability distribution; the engine draws 50,000 correlated samples, feeds each draw through a valuation model, and returns a full probability distribution of fair value per share plus sensitivity diagnostics.

This is **not** a price-path simulator — there is no geometric Brownian motion or stochastic differential equation in sight. It simulates the *inputs* to a valuation model (revenue growth, margins, multiples, terminal assumptions, binary catalysts) and computes the implied fair value for each draw. The value is in properly propagating correlated uncertainty across multiple assumptions simultaneously, so the committee can see the full shape of the outcome distribution rather than a single point estimate.

## How It Works

```
  ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
  │ Pitch config     │───▶│ Gaussian copula │───▶│ Valuation model  │
  │ (input dists +   │    │ correlated      │    │ (multiples / DCF │
  │  correlations +  │    │ sampler         │    │  / SOTP / custom)│
  │  catalysts)      │    └─────────────────┘    └──────────────────┘
  └──────────────────┘                                    │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │ Catalyst overlay │
                                                 │ (additive + mul) │
                                                 └──────────────────┘
                                                          │
                                                          ▼
                                                 ┌──────────────────┐
                                                 │ 50,000 fair-     │
                                                 │ value draws      │
                                                 └──────────────────┘
                                                          │
                     ┌────────────────────────────────────┼────────────────────────────┐
                     ▼                                    ▼                            ▼
            ┌──────────────────┐                ┌──────────────────┐         ┌──────────────────┐
            │ Terminal report  │                │ Sensitivity      │         │ PNG charts       │
            │ + 3 CSVs         │                │ (tornado + Sobol │         │ (distribution,   │
            │                  │                │  + conditional)  │         │  tornado, CDF…)  │
            └──────────────────┘                └──────────────────┘         └──────────────────┘
```

Pipeline detail:

1. **Sample inputs.** Each input distribution from the pitch config is drawn 50,000 times. Continuous inputs are sampled through a Gaussian copula so requested correlations between inputs (e.g. revenue growth ↔ margin expansion, WACC ↔ growth) are preserved exactly at the rank level while the marginals stay identical to the analyst's specification. If the analyst's partial-correlation matrix is not positive semi-definite, it is projected to the nearest PSD matrix via eigenvalue clipping (a warning fires).
2. **Sample catalysts.** Binary/discrete catalysts (FDA outcomes, M&A scenarios, contract wins) are sampled independently from the continuous inputs. Catalysts can be grouped as mutually exclusive.
3. **Compute fair value per draw.** Each drawn row of inputs is passed to the chosen valuation model.
4. **Apply catalyst overlay.** Multiplicative impacts scale the base fair value; additive impacts adjust it per-share.
5. **Summarise and diagnose.** The engine computes the full distribution (mean, median, percentiles, VaR/CVaR), the probability of upside, the upside/downside capture, and the risk-reward ratio. Sensitivity is reported two ways — one-at-a-time tornado charts and variance-contribution scores.

## Project Structure

```
ima-monte-carlo/
├── config.py              # Simulation defaults (N_SIMULATIONS, seed, percentiles)
├── distributions.py       # 7 distribution types + make_distribution() factory
├── correlation.py         # Gaussian copula with PSD clipping
├── valuation_models.py    # MultiplesModel, DCFModel, SOTPModel, CustomModel
├── scenario_overlay.py    # Catalyst/outcome overlays
├── simulation.py          # MonteCarloSimulation engine + SimulationResults
├── sensitivity.py         # Tornado, variance contribution, scenario means
├── visualization.py       # 7 publication-quality charts
├── report.py              # Terminal report + 3 CSV exports
├── main.py                # CLI + interactive pitch builder
├── examples/              # Reference pitches
│   ├── example_multiples.py     # P/E pitch with activist catalyst
│   ├── example_dcf.py           # 5-year DCF with terminal multiple
│   └── example_sotp.py          # Royalty + pharma SOTP with pipeline catalyst
├── pitches/               # Your pitch configs (gitignored)
├── output/                # Generated reports and PNGs (gitignored)
└── requirements.txt
```

## Setup

```bash
cd ima-monte-carlo
pip install -r requirements.txt
```

Requires Python 3.9+. Depends on `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`.

## Quick Start

```bash
cd ima-monte-carlo

# Run the three reference pitches
python main.py --pitch examples/example_multiples.py
python main.py --pitch examples/example_dcf.py
python main.py --pitch examples/example_sotp.py

# Interactive pitch builder (multiples only — DCF/SOTP start from the examples)
python main.py --interactive

# Run every pitch in pitches/ and produce a risk-reward comparison scatter
python main.py --compare

# Overrides
python main.py --pitch pitches/INVA.py --n-sims 100000 --seed 7
python main.py --pitch pitches/INVA.py --no-charts
```

Each run prints a terminal report, writes three CSVs, and saves up to six PNG charts to `output/`.

## Writing a Pitch Config

A pitch is a Python module that declares module-level variables. Python beats YAML here because analysts can write expressions like `150e6` instead of `150000000`, and can reference shared constants. Files go in `pitches/` (gitignored so they stay local unless you explicitly share them).

Minimal multiples pitch:

```python
# pitches/ACME.py
TICKER = "ACME"
CURRENT_PRICE = 47.50

MODEL_TYPE = "multiples"
MULTIPLES_VARIANT = "equity"    # "equity" for P/E, P/FCF; "enterprise" for EV/EBITDA

INPUTS = {
    "forward_metric": {"type": "triangular", "low": 3.10, "mode": 3.60, "high": 4.20,
                       "label": "Forward EPS (FY+1)"},
    "multiple":       {"type": "triangular", "low": 12.0, "mode": 15.0, "high": 18.0,
                       "label": "P/E Multiple"},
}

CORRELATIONS = {
    ("forward_metric", "multiple"): 0.25,   # earnings beats tend to pull multiples up
}

CATALYSTS = []
```

Then:

```bash
python main.py --pitch pitches/ACME.py
```

## Valuation Models

| `MODEL_TYPE` | When to use | Required inputs |
|---|---|---|
| `multiples`  | The most common IMA pitch — forward metric × multiple. | `forward_metric`, `multiple` (+ `net_debt`, `cash`, `shares` for enterprise variant) |
| `dcf`        | Mature cash-generative names where the growth profile is the core thesis. | `base_fcf`, `wacc`, and either `growth_rate` or per-year `growth_year_1..N`, and either `terminal_growth` or `terminal_multiple` |
| `sotp`       | Holdcos, diversified businesses, pre-spinoff situations. Each segment is its own mini-valuation. | `SEGMENTS = [...]`, `HOLDCO = {...}` |
| `custom`     | Escape hatch — supply any Python callable that maps sampled inputs → fair value per share. | `FAIR_VALUE_FUNC` callable + an `INPUTS` dict |

### DCF edge cases

- If a draw produces `wacc ≤ terminal_growth` (Gordon Growth is undefined), the terminal value is capped at 30× terminal FCF and the draw is **kept** — one warning fires per model. This is expected behaviour when distributions overlap.
- Negative equity is floored at zero (long-only mandate — shareholders can't lose more than their investment).

## Input Distributions

`INPUTS` is a dict of `{name: distribution_spec}`. Every spec has a `"type"` field and an optional `"label"` (used on charts).

| Type | Spec | When to use |
|---|---|---|
| `triangular` | `low`, `mode`, `high` | **Default recommendation.** Asymmetric upside/downside with a clear base case. Matches how most analysts already think in worst/base/best. |
| `normal` | `mean`, `std` | Symmetric uncertainty (e.g. growth rates, margins) with a central estimate. |
| `uniform` | `low`, `high` | Genuinely don't know where in the range the truth lies (exit multiples, terminal assumptions). |
| `lognormal` | `mean`, `std` | Must be positive, right-skew potential. Parameterised by the *desired output mean and std* — the engine converts internally. |
| `truncnorm` | `mean`, `std`, `low`, `high` | Naturally bounded quantities (gross margin ∈ [0,1], ROE not extending past a physical cap). |
| `point` | `value` | Analyst treats this input as fixed (shares outstanding, tax rate). |
| `discrete` | `outcomes: {label: [prob, value]}` | Small finite set of outcomes — probabilities must sum to 1. |
| `empirical` | `historical_values: [...]` | Bootstrap from actual historical data (e.g., past quarterly revenue surprises). |

## Correlations

Specify only the pairs you believe are correlated. Unspecified pairs default to zero.

```python
CORRELATIONS = {
    ("revenue_growth", "ebitda_margin"): 0.4,   # operating leverage
    ("wacc", "growth_rate"):            -0.2,   # higher-growth names face higher discount rates
}
```

Mechanics:

- Unspecified pairs = 0 (independent), diagonal = 1.
- If the assembled matrix is not PSD (can happen with conflicting partial correlations), it is projected to the nearest PSD matrix via eigenvalue clipping — a warning notes that effective correlations may differ slightly from those specified.
- Gaussian copula preserves the **marginals exactly**. The achieved Pearson correlation in the output may differ slightly from the specified value depending on marginal shape (the copula specifies rank correlation). This is expected.
- Discrete catalysts are **always** sampled independently of continuous inputs. Dependencies between a catalyst and continuous inputs should be modelled via the catalyst's outcome impacts, not via `CORRELATIONS`.

## Catalysts

Binary and multi-outcome events that shift fair value additively (dollar-per-share) or multiplicatively (scale factor).

```python
CATALYSTS = [
    {
        "name": "FDA Decision",
        "outcomes": {
            "approval":  {"probability": 0.60, "value_impact":  8.00, "impact_type": "additive"},
            "crl":       {"probability": 0.25, "value_impact": -3.00, "impact_type": "additive"},
            "rejection": {"probability": 0.15, "value_impact": -6.00, "impact_type": "additive"},
        },
    },
    {
        "name": "Market Regime Compression",
        "outcomes": {
            "compression": {"probability": 0.30, "value_impact": 0.85, "impact_type": "multiplicative"},
            "no_change":   {"probability": 0.70, "value_impact": 1.00, "impact_type": "multiplicative"},
        },
    },
]
```

- Probabilities across the outcomes of a single catalyst must sum to 1.
- Multiple catalysts are independent by default. Adding `"group": "fda"` to two catalysts makes them mutually exclusive — then the probabilities of all outcomes across the whole group must sum to 1.
- Multiplicative impacts are applied first, then additive impacts are added on top.

## Outputs

Every run generates four things:

**1. Terminal report.** Mean/median/std, the full percentile grid (P5–P95), probability of upside, probability of >20% upside/downside, VaR(5), CVaR(5), upside/downside capture, risk-reward ratio, top-N variance contribution ranking, and per-catalyst conditional-mean FV.

**2. Three CSVs** in `output/`:

| File | Contents |
|---|---|
| `{ticker}_simulation_results.csv` | All N draws — fair values + sampled inputs + catalyst outcome labels |
| `{ticker}_summary.csv`            | All summary statistics in long (metric, value) format |
| `{ticker}_sensitivity.csv`        | Tornado data and variance-contribution shares |

**3. PNG charts** in `output/`:

| Chart | What it shows |
|---|---|
| `fair_value_distribution.png`   | Histogram of FV with current-price line, P10/P50/P90 markers, red below / green above, annotated P(undervalued) and expected return |
| `tornado_chart.png`             | Horizontal bars — fair value at input P10 vs. P90 with all other inputs at median. Sorted by span. |
| `variance_contribution.png`     | Normalised share of output variance per input (and per catalyst). Inputs <3% bucket into "Other". |
| `input_distributions.png`       | Small-multiples grid — lets the analyst visually verify every input distribution is sensible. |
| `cumulative_probability.png`    | CDF of FV with current-price crossing annotated as P(fair value > current). |
| `scenario_comparison.png`       | Overlaid histograms per catalyst outcome (only generated if catalysts exist). |
| `risk_reward.png`               | Only from `--compare`. Scatter of all pitches — downside (P10 vs. current) on X, expected return on Y, with a 1:1 diagonal. |

**4. In-process results.** If you're scripting things, `run_pitch(...)` in `main.py` returns a `SimulationResults` dataclass with all the raw distributions, so you can slot it into other analyses.

## Sensitivity — Two Methods, On Purpose

Both methods are reported because they answer different questions and sometimes disagree:

- **Tornado** (one-at-a-time): hold everything else at median, vary one input from its P10 to its P90. Answers: *"If this input surprises us, how much does fair value move?"*
- **Variance contribution** (Sobol-like approximation via R² of input vs. output, plus η² for catalysts): Answers: *"Of the uncertainty in our fair-value estimate, how much is driven by each input?"*

The tornado chart doesn't capture interactions — for a DCF, the nonlinearity between WACC and terminal growth is the obvious one. The variance-contribution score handles it better. Use both.

## Design Notes Worth Flagging

- **Reproducibility.** The default seed (`config.RANDOM_SEED = 42`) makes results identical across runs for committee review. Set it to `None` for true-random sampling. Override at the command line with `--seed N`.
- **Performance.** On a 2024-era laptop: multiples ≈ 0.03s, DCF ≈ 0.15s, SOTP ≈ 0.35s for 50,000 draws. The engine uses `numpy.random.Generator` throughout — the modern, thread-safe API — not the legacy `numpy.random.seed`.
- **PSD clipping of correlation matrices** is done via eigenvalue projection (set negative eigenvalues to `1e-8`, rescale diagonal back to 1). Alternatives like Higham's algorithm would be marginally more accurate but add latency for no practical gain at K ≤ ~20 inputs.
- **SOTP input namespacing.** Inputs inside a segment are internally stored as `"{segment}/{input}"` so two DCF segments can both have an input named `wacc` without colliding. Segment-local correlations declared inside a segment's `"correlations"` dict are re-namespaced and merged with the top-level `CORRELATIONS`.
- **Excluded draws.** If `compute_fair_value` returns NaN/inf for a draw (extreme input combos), the draw is dropped and logged. A warning fires if >5% of draws are excluded — that's usually a sign the input distributions are misconfigured.

## Historical Calibration (Project 6)

Each pitch can declare a `THESIS_STATEMENT` and a `historical_metric` on any
input. The pipeline then fetches the company's quarterly history (and a
sector-peer pool from the IJR universe) via yfinance and reports where the
analyst's distribution sits inside both. Severity-tagged warnings flag
projections that exceed historical extremes; a mean-reversion note appears
when the analyst projects a regime that diverges sharply from the long-run
average.

Available metrics: `revenue_growth_yoy`, `ebitda_margin`, `operating_margin`,
`gross_margin`, `fcf_margin`, `fcf_growth_yoy`, `roic`, `pe_multiple`,
`ev_ebitda_multiple`. WACC and terminal growth are forward-looking and have
no historical analog.

```python
INPUTS = {
    "growth_rate": {"type": "triangular", "low": 0.02, "mode": 0.06,
                     "high": 0.12, "label": "FCF Growth",
                     "historical_metric": "fcf_growth_yoy"},
    ...
}
```

Run `--skip-calibration` to bypass yfinance fetching for fast re-runs;
`--refresh-peers` to invalidate the (weekly) peer cache.

After the simulation, a **driver concentration** check fires: if 1-2 inputs
explain >80% of output variance, the pitch is flagged as "effectively a
2-driver bet" — not a defect, but worth surfacing.

## Web Dashboard (Project 8)

The dashboard at `webapp/` displays simulation outputs in a polished,
interactive format suitable for committee discussions and external sharing.
Architecture: analysts run the pipeline locally, the JSON gets committed
alongside code, Vercel auto-deploys on push.

### Publish a pitch

```bash
python main.py --pitch pitches/YOUR_PITCH.py --pitch-id TICKER_DATE
git add webapp/public/pitches/
git commit -m "Add TICKER pitch"
git push
```

Site auto-deploys on push, updates within ~60 seconds.

### Drafts (local-only iteration)

```bash
python main.py --pitch pitches/YOUR_PITCH.py --pitch-id TICKER_test --draft
# JSON saved to _drafts/ (gitignored)

# When ready:
python main.py --publish-draft TICKER_test
git add webapp/public/pitches/
git commit -m "Publish TICKER pitch"
git push
```

### Local development

```bash
cd webapp
npm install        # one-time
npm run dev        # opens localhost:5173
```

### Routes

- `/` — pitch list with sortable / filterable cards
- `/pitch/:id` — detailed view (distribution, CDF, calibration, sensitivity, scenarios)
- `/compare?pitches=A,B,C` — side-by-side, shareable URL
- `/about` — methodology page (the portfolio piece)

## Streamlit Dashboard

A second dashboard built in pure Python sits alongside the React webapp.
Both read the same JSON files at `webapp/public/pitches/`, so they stay
in sync. Use Streamlit when you want to iterate fast in Python without
the npm/Vite/React toolchain.

### Setup

```bash
cd ima-monte-carlo
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Pages auto-discovered from `pages/`:

- **Pitch list** (landing) — sortable, filterable cards
- **Pitch Detail** — full per-pitch dashboard (distribution, CDF, calibration, sensitivity, scenarios)
- **Compare** — side-by-side comparison of 2-4 pitches
- **About** — methodology page

### Optional: deploy to Streamlit Community Cloud

1. Push the repo to GitHub.
2. Sign in at <https://share.streamlit.io> with your GitHub account.
3. Click **New app**, point at `ima-monte-carlo/streamlit_app.py`, set
   the requirements path to `ima-monte-carlo/requirements-streamlit.txt`
   (Streamlit reads `requirements.txt` by default — combine the two files
   if your pipeline deps are needed at deploy time, but the dashboard
   itself only needs the streamlit/plotly file).
4. The app updates on every git push.
