"""
Default simulation settings for the IMA Monte Carlo valuation framework.

Analysts can override any of these at the command line (--n-sims, --seed) or
programmatically by passing arguments to MonteCarloSimulation.
"""

# --- Simulation settings ---
N_SIMULATIONS = 50_000          # Number of Monte Carlo draws
RANDOM_SEED = 42                # For reproducibility (set to None for true random)
CONFIDENCE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

# --- Display settings ---
HIST_BINS = 100
CURRENCY = "USD"

# --- Output paths ---
OUTPUT_DIR = "output"

# --- Quality thresholds ---
MAX_EXCLUDED_DRAW_FRACTION = 0.05   # Warn if >5% of draws produce NaN/inf
TERMINAL_VALUE_CAP_MULTIPLE = 30.0  # Cap terminal value at 30x terminal FCF when
                                    # Gordon Growth formula breaks (wacc <= g).
