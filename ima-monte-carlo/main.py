"""
Orchestrator and CLI for the IMA Monte Carlo valuation tool.

Usage
-----
    python main.py --pitch examples/example_multiples.py
    python main.py --pitch examples/example_dcf.py
    python main.py --pitch examples/example_sotp.py
    python main.py --pitch pitches/INVA.py

    python main.py --interactive
    python main.py --compare                          # runs everything in pitches/

    python main.py --pitch ... --n-sims 10000 --seed 123 --no-charts
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import numpy as np

from config import N_SIMULATIONS, OUTPUT_DIR, RANDOM_SEED
from distributions import make_distribution
from historical_calibration import (
    CalibrationReport,
    METRIC_REGISTRY,
    build_historical_context,
    calibrate_input,
    collect_calibration_inputs_from_pitch,
    fetch_bundle,
    fetch_price_history,
)
from peer_data import (
    build_peer_context,
    fetch_peer_data,
    identify_peers,
    load_universe,
)
from report import (
    print_calibration_report,
    print_concentration_verdict,
    print_report,
    write_calibration_csv,
    write_csvs,
)
from scenario_overlay import CatalystOverlay
from sensitivity import enrich_results
from simulation import (
    MonteCarloSimulation,
    SimulationResults,
    check_driver_concentration,
)
from thesis_validator import get_thesis, validate_thesis
from webapp_export import (
    export_pitch_to_webapp,
    publish_draft,
    update_pitch_manifest,
)
from valuation_models import (
    CustomModel,
    DCFModel,
    MultiplesModel,
    SOTPModel,
    ValuationModel,
)
from visualization import plot_risk_reward, render_all

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Pitch file loading
# --------------------------------------------------------------------------- #
def _load_pitch_module(path: str) -> ModuleType:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Pitch config not found: {p}")
    spec = importlib.util.spec_from_file_location(p.stem, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pitch module from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[p.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_model(mod: ModuleType) -> ValuationModel:
    model_type = getattr(mod, "MODEL_TYPE", "").lower()
    ticker = getattr(mod, "TICKER", "Unknown")
    current_price = float(getattr(mod, "CURRENT_PRICE"))
    shares = float(getattr(mod, "SHARES_OUTSTANDING", 0))

    if model_type == "multiples":
        variant = getattr(mod, "MULTIPLES_VARIANT", "equity")
        return MultiplesModel(
            name=ticker,
            inputs=mod.INPUTS,
            current_price=current_price,
            variant=variant,
            shares_outstanding=shares,
        )
    elif model_type == "dcf":
        years = int(getattr(mod, "PROJECTION_YEARS", 5))
        return DCFModel(
            name=ticker,
            inputs=mod.INPUTS,
            current_price=current_price,
            projection_years=years,
            shares_outstanding=shares,
        )
    elif model_type == "sotp":
        return SOTPModel(
            name=ticker,
            segments=mod.SEGMENTS,
            holdco_adjustments=mod.HOLDCO,
            current_price=current_price,
            shares_outstanding=shares,
        )
    elif model_type == "custom":
        return CustomModel(
            name=ticker,
            fair_value_func=mod.FAIR_VALUE_FUNC,
            inputs=mod.INPUTS,
            current_price=current_price,
            shares_outstanding=shares,
        )
    else:
        raise ValueError(
            f"Unknown MODEL_TYPE {model_type!r}. "
            f"Use 'multiples', 'dcf', 'sotp', or 'custom'."
        )


def _build_simulation(
    mod: ModuleType,
    model: ValuationModel,
    n_sims: int,
    seed: int | None,
) -> tuple[MonteCarloSimulation, CatalystOverlay]:
    correlations = getattr(mod, "CORRELATIONS", {}) or {}
    catalysts = getattr(mod, "CATALYSTS", []) or []
    cost_basis = getattr(mod, "COST_BASIS", None)
    ticker = getattr(mod, "TICKER", model.name)

    sim = MonteCarloSimulation(
        model=model,
        correlations=correlations,
        catalysts=catalysts,
        n_simulations=n_sims,
        seed=seed,
        ticker=ticker,
        cost_basis=cost_basis,
    )
    return sim, sim.overlay


# --------------------------------------------------------------------------- #
#  Calibration pipeline
# --------------------------------------------------------------------------- #
def build_calibration_for_pitch(
    mod: ModuleType,
    model: ValuationModel,
    *,
    refresh_peers: bool = False,
    n_peers: int = 30,
    quiet: bool = False,
) -> list[CalibrationReport]:
    """
    For each input on the pitch that declares a `historical_metric`, fetch
    the company's history + sector-peer history and compute a CalibrationReport.

    Returns an empty list if no inputs are eligible or if yfinance / universe
    data is unavailable.
    """
    pairs = collect_calibration_inputs_from_pitch(mod)
    if not pairs:
        if not quiet:
            print("  [calibration] No inputs declared `historical_metric` — skipped.")
        return []

    ticker = getattr(mod, "TICKER", model.name)
    if not quiet:
        print(f"  [calibration] {len(pairs)} input(s) flagged for calibration on {ticker}")

    # 1. Fetch primary company bundle + price history (price only if needed)
    needs_prices = any(
        METRIC_REGISTRY.get(metric, {}).get("needs_prices", False)
        for _, metric in pairs
    )

    if not quiet:
        print(f"  [calibration] Fetching {ticker} fundamentals from yfinance...")
    bundle = fetch_bundle(ticker)
    prices = fetch_price_history(ticker) if needs_prices else None

    # 2. Build company-level historical contexts (cache by metric to avoid
    #    re-extracting the same metric for multiple inputs).
    company_contexts: dict[str, object] = {}
    if bundle is not None:
        for _, metric in pairs:
            if metric in company_contexts:
                continue
            ctx = build_historical_context(metric, ticker, bundle, prices)
            company_contexts[metric] = ctx
    else:
        if not quiet:
            print(
                "  [calibration] Could not fetch yfinance data for company — "
                "calibration will rely on sector peers only."
            )

    # 3. Identify peers + fetch peer data
    universe = load_universe()
    peer_tickers = identify_peers(ticker, universe, n_peers=n_peers)
    sector_contexts: dict[str, object] = {}
    if peer_tickers:
        if not quiet:
            print(
                f"  [calibration] {len(peer_tickers)} sector peers identified "
                f"in {universe.loc[ticker, 'Sector'] if ticker in universe.index else 'N/A'}"
            )
        peer_bundles, peer_prices = fetch_peer_data(
            ticker, peer_tickers,
            needs_prices=needs_prices,
            refresh=refresh_peers,
        )
        for _, metric in pairs:
            if metric in sector_contexts:
                continue
            sector_contexts[metric] = build_peer_context(
                metric, peer_tickers, peer_bundles, peer_prices,
            )
    else:
        if not quiet:
            print(
                f"  [calibration] {ticker} not in IJR universe (or no sector "
                "match) — sector context unavailable."
            )

    # 4. Per-input calibration reports
    reports: list[CalibrationReport] = []
    for input_name, metric in pairs:
        dist = model.inputs.get(input_name)
        if dist is None:
            if not quiet:
                print(f"    [calibration] Input '{input_name}' not found on model — skipped.")
            continue
        report = calibrate_input(
            input_name=input_name,
            metric=metric,
            distribution=dist,
            company_context=company_contexts.get(metric),
            sector_context=sector_contexts.get(metric),
            label=getattr(dist, "label", "") or input_name,
        )
        reports.append(report)
    return reports


# --------------------------------------------------------------------------- #
#  Run one pitch
# --------------------------------------------------------------------------- #
def run_pitch(
    pitch_path: str,
    *,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = RANDOM_SEED,
    render_charts: bool = True,
    output_dir: str = OUTPUT_DIR,
    quiet: bool = False,
    skip_calibration: bool = False,
    refresh_peers: bool = False,
    pitch_id: str | None = None,
    as_draft: bool = False,
    no_export: bool = False,
) -> SimulationResults:
    mod = _load_pitch_module(pitch_path)
    model = _build_model(mod)
    sim, overlay = _build_simulation(mod, model, n_sims, seed)

    # 1. Thesis
    thesis_text = get_thesis(mod)
    thesis_validation = validate_thesis(thesis_text)

    if not quiet:
        print(f"\nLoaded pitch: {pitch_path}")
        print(f"  Ticker: {model.name}   Current price: ${model.current_price:.2f}")
        print(f"  Model:  {type(model).__name__}")
        print(f"  {model.describe_inputs()}")
        if sim.correlations:
            print(f"\n  Correlation spec: {len(sim.correlations)} pair(s)")
            print(sim.sampler.correlation_summary())
        if overlay:
            print("\n  Catalysts:")
            for line in overlay.describe().splitlines():
                print("    " + line)

    # 2. Calibration step
    calibration_reports: list[CalibrationReport] = []
    if not skip_calibration:
        if not quiet:
            print("\n=== Building historical calibration ===")
        try:
            calibration_reports = build_calibration_for_pitch(
                mod, model, refresh_peers=refresh_peers, quiet=quiet,
            )
        except Exception as e:
            logger.exception("Calibration step failed: %s", e)
            if not quiet:
                print(f"  [calibration] Failed: {e} — continuing without calibration.")

    if not quiet:
        print()
        print_calibration_report(thesis_text, thesis_validation, calibration_reports)

    # 3. Simulation
    if not quiet:
        print(f"\n  Running {n_sims:,} draws...")
    t0 = time.perf_counter()
    results = sim.run()
    elapsed = time.perf_counter() - t0
    if not quiet:
        print(f"  Simulation complete in {elapsed:.2f}s")

    enrich_results(results, model, overlay)

    # 4. Driver concentration
    concentration = check_driver_concentration(results)

    if not quiet:
        print_report(results, overlay)
        print_concentration_verdict(concentration)

    csvs = write_csvs(results, output_dir=output_dir, overlay=overlay)
    cal_csv = write_calibration_csv(
        calibration_reports, output_dir=output_dir, ticker=results.ticker,
    )
    if cal_csv is not None:
        csvs["calibration"] = cal_csv
    if not quiet:
        print(f"\n  CSVs written:")
        for name, path in csvs.items():
            print(f"    {name}: {path}")

    if render_charts:
        charts = render_all(
            results, model, overlay,
            output_dir=output_dir,
            calibration_reports=calibration_reports,
        )
        if not quiet:
            print(f"\n  Charts written:")
            for name, path in charts.items():
                print(f"    {name}: {path}")

    # Webapp JSON export
    if not no_export:
        ticker = getattr(mod, "TICKER", model.name)
        pid = pitch_id or f"{ticker}_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        export_path = export_pitch_to_webapp(
            pitch_id=pid,
            pitch_config=mod,
            simulation_results=results,
            calibration_reports=calibration_reports,
            thesis_text=thesis_text,
            thesis_validation=thesis_validation,
            concentration=concentration,
            as_draft=as_draft,
        )
        if not as_draft:
            update_pitch_manifest()
        if not quiet:
            location = "draft (gitignored)" if as_draft else "published"
            print(f"\n  Webapp export ({location}): {export_path}")
            if as_draft:
                print(f"    Promote with: python main.py --publish-draft {pid}")
            else:
                print("    Push to git to deploy.")

    return results


# --------------------------------------------------------------------------- #
#  Compare multiple pitches
# --------------------------------------------------------------------------- #
def run_compare(
    pitches_dir: str = "pitches",
    *,
    n_sims: int = N_SIMULATIONS,
    seed: int | None = RANDOM_SEED,
    output_dir: str = OUTPUT_DIR,
    skip_calibration: bool = False,
    refresh_peers: bool = False,
) -> list[SimulationResults]:
    d = Path(pitches_dir)
    if not d.exists():
        print(f"No pitches directory at {d} — nothing to compare.")
        return []
    files = sorted(p for p in d.glob("*.py") if not p.name.startswith("_"))
    if not files:
        print(f"No pitch files found in {d}/")
        return []

    all_results: list[SimulationResults] = []
    for f in files:
        print(f"\n--- {f.name} ---")
        try:
            r = run_pitch(
                str(f), n_sims=n_sims, seed=seed, render_charts=False,
                output_dir=output_dir, quiet=False,
                skip_calibration=skip_calibration,
                refresh_peers=refresh_peers,
            )
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR running {f.name}: {e}")

    if len(all_results) >= 2:
        out = plot_risk_reward(all_results, Path(output_dir) / "risk_reward.png")
        print(f"\nRisk-reward scatter written: {out}")
    return all_results


# --------------------------------------------------------------------------- #
#  Interactive CLI
# --------------------------------------------------------------------------- #
def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    return raw if raw else (default or "")


def _ask_float(prompt: str, default: float | None = None) -> float:
    while True:
        raw = _ask(prompt, str(default) if default is not None else None)
        try:
            return float(raw)
        except ValueError:
            print("  Please enter a number.")


def _ask_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    options = "/".join(choices)
    while True:
        raw = _ask(f"{prompt} ({options})", default or choices[0])
        raw = raw.lower()
        if raw in choices:
            return raw
        print(f"  Please pick one of: {options}")


def _ask_distribution(name: str) -> dict:
    print(f"\n  Input: {name}")
    dtype = _ask_choice(
        "    Distribution type",
        ["triangular", "normal", "uniform", "lognormal", "truncnorm", "point", "discrete"],
        default="triangular",
    )
    label = _ask("    Display label", default=name)
    spec: dict = {"type": dtype, "label": label}
    if dtype == "triangular":
        spec["low"] = _ask_float("    Low (worst case)")
        spec["mode"] = _ask_float("    Mode (base case)")
        spec["high"] = _ask_float("    High (best case)")
    elif dtype == "normal":
        spec["mean"] = _ask_float("    Mean")
        spec["std"] = _ask_float("    Std dev")
    elif dtype == "uniform":
        spec["low"] = _ask_float("    Low")
        spec["high"] = _ask_float("    High")
    elif dtype == "lognormal":
        spec["mean"] = _ask_float("    Desired mean")
        spec["std"] = _ask_float("    Desired std dev")
    elif dtype == "truncnorm":
        spec["mean"] = _ask_float("    Mean")
        spec["std"] = _ask_float("    Std dev")
        spec["low"] = _ask_float("    Lower bound")
        spec["high"] = _ask_float("    Upper bound")
    elif dtype == "point":
        spec["value"] = _ask_float("    Fixed value")
    elif dtype == "discrete":
        outcomes: dict[str, list[float]] = {}
        while True:
            lbl = _ask("    Outcome label (blank to stop)")
            if not lbl:
                break
            p = _ask_float(f"    P({lbl})")
            v = _ask_float(f"    Value of {lbl}")
            outcomes[lbl] = [p, v]
        spec["outcomes"] = outcomes
    # Validate
    try:
        make_distribution(spec)
    except Exception as e:
        print(f"    [warning] distribution invalid: {e}")
        retry = _ask_choice("    Retry?", ["y", "n"], "y")
        if retry == "y":
            return _ask_distribution(name)
    return spec


def _interactive_multiples() -> dict:
    print("\n=== Multiples-based pitch ===")
    ticker = _ask("Ticker", default="TICKER")
    price = _ask_float("Current price")
    variant = _ask_choice("Multiples variant", ["equity", "enterprise"], default="equity")

    inputs: dict[str, dict] = {}
    inputs["forward_metric"] = _ask_distribution("forward_metric (e.g. forward EPS, FCF, EBITDA)")
    inputs["multiple"] = _ask_distribution("multiple (e.g. P/E, EV/EBITDA)")
    shares = 0.0
    if variant == "enterprise":
        shares = _ask_float("Shares outstanding (number, not millions)")
        inputs["net_debt"] = _ask_distribution("net_debt")
        cash_str = _ask_choice("Model cash separately?", ["y", "n"], default="n")
        if cash_str == "y":
            inputs["cash"] = _ask_distribution("cash")

    catalysts: list[dict] = []
    while _ask_choice("Add a catalyst?", ["y", "n"], "n") == "y":
        name = _ask("  Catalyst name")
        outs = {}
        total = 0.0
        while True:
            lbl = _ask("    Outcome label (blank to stop)")
            if not lbl:
                break
            p = _ask_float(f"    P({lbl})")
            imp = _ask_float(f"    Value impact of {lbl} ($/share or ×)")
            tp = _ask_choice("    Impact type", ["additive", "multiplicative"], "additive")
            outs[lbl] = {"probability": p, "value_impact": imp, "impact_type": tp}
            total += p
        if abs(total - 1) > 1e-6:
            print(f"    [warning] probabilities sum to {total:.3f}; adjust before running.")
        catalysts.append({"name": name, "outcomes": outs})

    return {
        "TICKER": ticker,
        "CURRENT_PRICE": price,
        "SHARES_OUTSTANDING": shares,
        "MODEL_TYPE": "multiples",
        "MULTIPLES_VARIANT": variant,
        "INPUTS": inputs,
        "CORRELATIONS": {},
        "CATALYSTS": catalysts,
    }


def _write_pitch_file(cfg: dict, path: Path) -> None:
    """Write a minimal pitch config .py for later re-runs."""
    lines = [
        '"""Pitch config generated by interactive mode."""',
        "",
        f"TICKER = {cfg['TICKER']!r}",
        f"CURRENT_PRICE = {cfg['CURRENT_PRICE']!r}",
        f"SHARES_OUTSTANDING = {cfg['SHARES_OUTSTANDING']!r}",
        "",
        f"MODEL_TYPE = {cfg['MODEL_TYPE']!r}",
    ]
    if "MULTIPLES_VARIANT" in cfg:
        lines.append(f"MULTIPLES_VARIANT = {cfg['MULTIPLES_VARIANT']!r}")
    if "PROJECTION_YEARS" in cfg:
        lines.append(f"PROJECTION_YEARS = {cfg['PROJECTION_YEARS']!r}")
    lines.append("")
    lines.append(f"INPUTS = {cfg['INPUTS']!r}")
    lines.append("")
    lines.append(f"CORRELATIONS = {cfg.get('CORRELATIONS', {})!r}")
    lines.append("")
    lines.append(f"CATALYSTS = {cfg.get('CATALYSTS', [])!r}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_interactive() -> None:
    print("\nIMA Monte Carlo — Interactive Pitch Builder")
    method = _ask_choice(
        "Valuation method", ["multiples", "dcf", "sotp"], default="multiples"
    )
    if method != "multiples":
        print(
            f"\n  Interactive mode currently supports 'multiples' end-to-end. "
            f"For {method}, start from examples/example_{method}.py and edit "
            f"it into pitches/{_ask('ticker')}.py — that is the fastest path."
        )
        return

    cfg = _interactive_multiples()

    save = _ask_choice("Save this pitch to pitches/?", ["y", "n"], "y")
    out_path: Path | None = None
    if save == "y":
        fname = _ask("Filename", default=f"{cfg['TICKER']}.py")
        out_path = Path("pitches") / fname
        _write_pitch_file(cfg, out_path)
        print(f"  Saved: {out_path}")

    # Run the pitch either way
    if out_path is None:
        # Execute from a temp scratch file so run_pitch can use the same path
        tmp = Path("pitches") / f"_tmp_{cfg['TICKER']}.py"
        _write_pitch_file(cfg, tmp)
        out_path = tmp

    run_pitch(str(out_path))


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="IMA Monte Carlo valuation tool."
    )
    ap.add_argument("--pitch", type=str, help="Path to a pitch config .py file")
    ap.add_argument("--interactive", action="store_true", help="Interactive pitch builder")
    ap.add_argument("--compare", action="store_true",
                    help="Run all pitches in pitches/ and produce risk-reward scatter")
    ap.add_argument("--n-sims", type=int, default=N_SIMULATIONS,
                    help=f"Override simulation count (default {N_SIMULATIONS})")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help=f"Override random seed (default {RANDOM_SEED})")
    ap.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR, help="Output directory")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="Skip historical calibration (faster; useful for re-runs)")
    ap.add_argument("--refresh-peers", action="store_true",
                    help="Re-fetch sector peer fundamentals (slow; weekly cadence is enough)")
    ap.add_argument("--pitch-id", type=str, default=None,
                    help="Pitch ID for webapp export (default: TICKER_YYYYMMDD)")
    ap.add_argument("--draft", action="store_true",
                    help="Save webapp JSON as draft (gitignored). Default: publish.")
    ap.add_argument("--no-export", action="store_true",
                    help="Skip webapp JSON export entirely")
    ap.add_argument("--publish-draft", type=str, default=None,
                    help="Promote a draft to published (provide pitch_id) and exit")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.publish_draft:
        publish_draft(args.publish_draft)
        return 0

    if args.interactive:
        run_interactive()
        return 0

    if args.compare:
        run_compare(
            pitches_dir="pitches",
            n_sims=args.n_sims,
            seed=args.seed,
            output_dir=args.output_dir,
            skip_calibration=args.skip_calibration,
            refresh_peers=args.refresh_peers,
        )
        return 0

    if args.pitch:
        run_pitch(
            args.pitch,
            n_sims=args.n_sims,
            seed=args.seed,
            render_charts=not args.no_charts,
            output_dir=args.output_dir,
            skip_calibration=args.skip_calibration,
            refresh_peers=args.refresh_peers,
            pitch_id=args.pitch_id,
            as_draft=args.draft,
            no_export=args.no_export,
        )
        return 0

    print("Usage: python main.py --pitch <path>  |  --interactive  |  --compare")
    return 1


if __name__ == "__main__":
    sys.exit(main())
