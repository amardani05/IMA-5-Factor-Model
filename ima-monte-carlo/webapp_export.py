"""
Monte Carlo Webapp Export
=========================

Convert SimulationResults + CalibrationReport + ThesisValidation into JSON
files consumable by the React dashboard under webapp/public/pitches/.

The JSON shape is documented (and type-checked) in webapp/src/data/types.ts.
This module is the single source of truth for what the dashboard receives —
update both files together if the schema changes.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

import numpy as np
import pandas as pd

from historical_calibration import CalibrationReport
from simulation import SimulationResults

logger = logging.getLogger(__name__)


PITCHES_DIR = Path("webapp/public/pitches")
DRAFTS_DIR = PITCHES_DIR / "_drafts"


# --------------------------------------------------------------------------- #
#  Helpers — JSON shaping
# --------------------------------------------------------------------------- #

def _json_default(o: Any):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        if not np.isfinite(v):
            return None
        return v
    if isinstance(o, np.ndarray):
        return [_json_default(x) if isinstance(x, np.generic) else x for x in o.tolist()]
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    try:
        if pd.isna(o):
            return None
    except (TypeError, ValueError):
        pass
    raise TypeError(f"Cannot serialize {type(o).__name__}: {o!r}")


def _safe_float(x: Any) -> Optional[float]:
    """Convert to float; return None for non-finite values (JSON-friendly)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _compute_histogram(values, bins: int = 80) -> dict:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(arr, bins=bins)
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def _compute_cdf(values, n_points: int = 200) -> list[dict]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    sorted_values = np.sort(arr)
    n = len(sorted_values)
    n_points = min(n_points, n)
    indices = np.linspace(0, n - 1, n_points).astype(int)
    return [
        {
            "value": float(sorted_values[i]),
            "cumulative_prob": float((i + 1) / n),
        }
        for i in indices
    ]


# --------------------------------------------------------------------------- #
#  Pitch input flattening (handles all model types incl. SOTP)
# --------------------------------------------------------------------------- #

def _flatten_inputs(pitch_config: ModuleType) -> dict[str, dict]:
    """
    Return a dict {flat_input_name: spec_dict} matching the simulation
    sampled_inputs columns. For SOTP, segment inputs are prefixed with the
    segment name (e.g. "Royalty Portfolio/forward_metric").
    """
    model_type = (getattr(pitch_config, "MODEL_TYPE", "") or "").lower()

    if model_type == "sotp":
        flat: dict[str, dict] = {}
        for seg in getattr(pitch_config, "SEGMENTS", []) or []:
            seg_name = seg.get("name", "")
            for k, spec in (seg.get("inputs", {}) or {}).items():
                flat[f"{seg_name}/{k}"] = spec
        for k, spec in (getattr(pitch_config, "HOLDCO", {}) or {}).items():
            flat[f"holdco/{k}"] = spec
        return flat

    return dict(getattr(pitch_config, "INPUTS", {}) or {})


# --------------------------------------------------------------------------- #
#  Tornado entries (P10/P90 fair values per input)
# --------------------------------------------------------------------------- #

def _tornado_entries(
    simulation_results: SimulationResults,
    flat_inputs: dict[str, dict],
) -> list[dict]:
    """
    Emit one tornado entry per input. Pulls actual P10/P90 fair values from
    the existing tornado_data populated by sensitivity.py. Falls back to NaN
    if a particular input wasn't in the tornado set (e.g. degenerate).
    """
    span_lookup = {
        d["input"]: d for d in (simulation_results.tornado_data or [])
    }
    entries: list[dict] = []
    contribs = simulation_results.input_contributions or {}
    # Use the variance contributions as the canonical input list
    if not contribs:
        return entries
    total = sum(v for v in contribs.values() if v > 0)
    sorted_inputs = sorted(
        ((k, v) for k, v in contribs.items() if not k.startswith("catalyst:")),
        key=lambda kv: abs(kv[1]),
        reverse=True,
    )
    for input_name, raw in sorted_inputs:
        share = (raw / total) if total > 0 else 0.0
        td = span_lookup.get(input_name) or {}
        spec = flat_inputs.get(input_name, {})
        label = td.get("label") or spec.get("label") or input_name
        entries.append({
            "input": input_name,
            "label": label,
            "p10_fair_value": _safe_float(td.get("fv_low")) if td else None,
            "p90_fair_value": _safe_float(td.get("fv_high")) if td else None,
            "input_p10": _safe_float(td.get("low_value")) if td else None,
            "input_p90": _safe_float(td.get("high_value")) if td else None,
            "variance_share": _safe_float(share) or 0.0,
        })
    return entries


# --------------------------------------------------------------------------- #
#  Inputs payload (sampled summaries + histograms)
# --------------------------------------------------------------------------- #

def _input_payloads(
    simulation_results: SimulationResults,
    flat_inputs: dict[str, dict],
) -> list[dict]:
    sampled_df = simulation_results.sampled_inputs
    out: list[dict] = []
    for input_name, spec in flat_inputs.items():
        if input_name not in sampled_df.columns:
            continue
        sampled = sampled_df[input_name].to_numpy()
        finite = sampled[np.isfinite(sampled)]
        if finite.size == 0:
            continue
        out.append({
            "name": input_name,
            "label": spec.get("label", input_name),
            "type": spec.get("type"),
            "spec": {k: v for k, v in spec.items() if k != "label"},
            "sampled_summary": {
                "mean": _safe_float(np.mean(finite)),
                "median": _safe_float(np.median(finite)),
                "std": _safe_float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
                "p5": _safe_float(np.percentile(finite, 5)),
                "p10": _safe_float(np.percentile(finite, 10)),
                "p25": _safe_float(np.percentile(finite, 25)),
                "p75": _safe_float(np.percentile(finite, 75)),
                "p90": _safe_float(np.percentile(finite, 90)),
                "p95": _safe_float(np.percentile(finite, 95)),
            },
            "histogram": _compute_histogram(finite, bins=40),
        })
    return out


# --------------------------------------------------------------------------- #
#  Calibration payload
# --------------------------------------------------------------------------- #

def _calibration_payload(reports: list[CalibrationReport]) -> list[dict]:
    out: list[dict] = []
    for r in reports:
        entry: dict[str, Any] = {
            "input": r.input_name,
            "metric": r.metric,
            "label": r.label,
            "analyst_distribution_summary": {
                "mean": _safe_float(r.input_distribution_summary.get("mean")),
                "p10": _safe_float(r.input_distribution_summary.get("p10")),
                "p90": _safe_float(r.input_distribution_summary.get("p90")),
                "median": _safe_float(r.input_distribution_summary.get("median")),
                "std": _safe_float(r.input_distribution_summary.get("std")),
            },
            "warnings": [
                {
                    "severity": w.get("severity", "info"),
                    "message": w.get("message", ""),
                    "suggestion": w.get("suggestion"),
                }
                for w in (r.warnings or [])
            ],
            "reversion_note": r.reversion_note,
            "max_severity": r.max_severity,
            "analyst_mean_company_percentile": _safe_float(
                r.analyst_mean_company_percentile),
            "analyst_p90_company_percentile": _safe_float(
                r.analyst_p90_company_percentile),
            "analyst_p10_company_percentile": _safe_float(
                r.analyst_p10_company_percentile),
            "analyst_mean_sector_percentile": _safe_float(
                r.analyst_mean_sector_percentile),
            "analyst_p90_sector_percentile": _safe_float(
                r.analyst_p90_sector_percentile),
            "analyst_p10_sector_percentile": _safe_float(
                r.analyst_p10_sector_percentile),
        }

        if r.company_context is not None:
            cc = r.company_context
            entry["company_context"] = {
                "entity": cc.entity,
                "n_observations": cc.n_observations,
                "lookback_years": _safe_float(cc.lookback_years),
                "mean": _safe_float(cc.mean),
                "median": _safe_float(cc.median),
                "std": _safe_float(cc.std),
                "min": _safe_float(cc.min),
                "max": _safe_float(cc.max),
                "p5": _safe_float(cc.p5),
                "p10": _safe_float(cc.p10),
                "p25": _safe_float(cc.p25),
                "p75": _safe_float(cc.p75),
                "p90": _safe_float(cc.p90),
                "p95": _safe_float(cc.p95),
                "long_run_mean": _safe_float(cc.long_run_mean),
                "recent_mean": _safe_float(cc.recent_mean),
                "mean_reversion_implied": _safe_float(cc.mean_reversion_implied),
                "raw_values": [_safe_float(v) for v in cc.raw_values[:60]],
            }

        if r.sector_context is not None:
            sc = r.sector_context
            entry["sector_context"] = {
                "entity": sc.entity,
                "n_observations": sc.n_observations,
                "mean": _safe_float(sc.mean),
                "median": _safe_float(sc.median),
                "std": _safe_float(sc.std),
                "min": _safe_float(sc.min),
                "max": _safe_float(sc.max),
                "p5": _safe_float(sc.p5),
                "p10": _safe_float(sc.p10),
                "p25": _safe_float(sc.p25),
                "p75": _safe_float(sc.p75),
                "p90": _safe_float(sc.p90),
                "p95": _safe_float(sc.p95),
            }

        out.append(entry)
    return out


# --------------------------------------------------------------------------- #
#  Catalysts + scenarios
# --------------------------------------------------------------------------- #

def _catalysts_payload(pitch_config: ModuleType) -> list[dict]:
    raw = getattr(pitch_config, "CATALYSTS", []) or []
    out: list[dict] = []
    for cat in raw:
        if not isinstance(cat, dict):
            continue
        outcomes = {}
        for label, oc in (cat.get("outcomes", {}) or {}).items():
            if isinstance(oc, dict):
                outcomes[label] = {
                    "probability": _safe_float(oc.get("probability")),
                    "value_impact": _safe_float(oc.get("value_impact")),
                    "impact_type": oc.get("impact_type", "additive"),
                }
        out.append({"name": cat.get("name", ""), "outcomes": outcomes})
    return out


def _scenarios_payload(simulation_results: SimulationResults) -> dict:
    """
    Convert the catalyst conditional means into a scenario dict shape.
    Key: "catalyst_name | outcome_label" → {outcome_label, mean_fair_value, ...}.
    """
    out: dict[str, dict] = {}
    for cat_name, by_outcome in (simulation_results.scenario_conditional_means or {}).items():
        for outcome, mean_fv in by_outcome.items():
            key = f"{cat_name} | {outcome}"
            out[key] = {
                "catalyst": cat_name,
                "outcome_label": outcome,
                "mean_fair_value": _safe_float(mean_fv),
            }
    return out


# --------------------------------------------------------------------------- #
#  Concentration payload
# --------------------------------------------------------------------------- #

def _concentration_payload(concentration: Optional[dict]) -> Optional[dict]:
    if not concentration:
        return None
    drivers = concentration.get("top_drivers") or []
    return {
        "is_concentrated": bool(concentration.get("is_concentrated")),
        "top_drivers": [[name, _safe_float(share) or 0.0] for name, share in drivers],
        "concentration_pct": _safe_float(concentration.get("concentration_pct")) or 0.0,
        "message": concentration.get("message", ""),
    }


# --------------------------------------------------------------------------- #
#  Main export entry point
# --------------------------------------------------------------------------- #

def export_pitch_to_webapp(
    pitch_id: str,
    pitch_config: ModuleType,
    simulation_results: SimulationResults,
    *,
    calibration_reports: Optional[list[CalibrationReport]] = None,
    thesis_text: str = "",
    thesis_validation: Optional[dict] = None,
    concentration: Optional[dict] = None,
    as_draft: bool = False,
    pitches_dir: Optional[Path] = None,
) -> Path:
    """
    Write a pitch JSON to webapp/public/pitches/ (or _drafts/ if as_draft).

    `pitch_id` becomes the filename: {pitch_id}.json. URL-safe ascii ids work
    best (TICKER_YYYYMMDD or TICKER_QN style).
    """
    base_dir = (pitches_dir or PITCHES_DIR)
    target_dir = (base_dir / "_drafts") if as_draft else base_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = target_dir / f"{pitch_id}.json"

    fv = np.asarray(simulation_results.fair_values, dtype=float)
    finite_fv = fv[np.isfinite(fv)]
    current_price = float(getattr(pitch_config, "CURRENT_PRICE", 0.0) or 0.0)

    flat_inputs = _flatten_inputs(pitch_config)

    payload: dict[str, Any] = {
        "pitch_id": pitch_id,
        "ticker": getattr(pitch_config, "TICKER", "Unknown"),
        "current_price": _safe_float(current_price),
        "model_type": getattr(pitch_config, "MODEL_TYPE", ""),
        "shares_outstanding": _safe_float(getattr(pitch_config, "SHARES_OUTSTANDING", None)),
        "is_draft": bool(as_draft),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "n_simulations": int(simulation_results.n_simulations),
        "random_seed": simulation_results.seed,
        "thesis_statement": (thesis_text or "").strip(),
        "thesis_validation": thesis_validation or {},
    }

    payload["distribution"] = {
        "mean": _safe_float(simulation_results.mean),
        "median": _safe_float(simulation_results.median),
        "std": _safe_float(simulation_results.std),
        "min": _safe_float(np.min(finite_fv)) if finite_fv.size else None,
        "max": _safe_float(np.max(finite_fv)) if finite_fv.size else None,
        "percentiles": {
            f"{int(round(p * 100)):02d}": _safe_float(v)
            for p, v in (simulation_results.percentiles or {}).items()
        },
        "histogram": _compute_histogram(finite_fv, bins=80),
        "cdf_points": _compute_cdf(finite_fv, n_points=200),
    }

    payload["probability_analysis"] = {
        "expected_return": _safe_float(simulation_results.expected_return),
        "probability_upside": _safe_float(simulation_results.probability_upside),
        "probability_above_20pct": _safe_float(
            np.mean(finite_fv > current_price * 1.20)
        ) if current_price > 0 and finite_fv.size else None,
        "probability_below_20pct": _safe_float(
            np.mean(finite_fv < current_price * 0.80)
        ) if current_price > 0 and finite_fv.size else None,
        "upside_capture": _safe_float(simulation_results.upside_capture),
        "downside_capture": _safe_float(simulation_results.downside_capture),
        "risk_reward_ratio": _safe_float(simulation_results.risk_reward_ratio),
        "var_5": _safe_float(simulation_results.var_5),
        "var_10": _safe_float(simulation_results.var_10),
        "cvar_5": _safe_float(simulation_results.cvar_5),
        "max_loss_pct": _safe_float(simulation_results.max_loss_pct),
        "probability_above_cost": _safe_float(simulation_results.probability_above_cost),
    }

    payload["inputs"] = _input_payloads(simulation_results, flat_inputs)
    payload["tornado"] = _tornado_entries(simulation_results, flat_inputs)

    if calibration_reports:
        payload["calibration"] = _calibration_payload(calibration_reports)

    conc = _concentration_payload(concentration)
    if conc is not None:
        payload["concentration"] = conc

    catalysts = _catalysts_payload(pitch_config)
    if catalysts:
        payload["catalysts"] = catalysts
        scenarios = _scenarios_payload(simulation_results)
        if scenarios:
            payload["scenarios"] = scenarios

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)

    location = "draft" if as_draft else "published"
    size_kb = output_path.stat().st_size // 1024
    logger.info(
        "Exported pitch %s (%s) → %s (%d KB)",
        pitch_id, location, output_path, size_kb,
    )
    return output_path


# --------------------------------------------------------------------------- #
#  Manifest + draft promotion
# --------------------------------------------------------------------------- #

def update_pitch_manifest(pitches_dir: Optional[Path] = None) -> Path:
    """Scan published pitches (NOT drafts) and write _manifest.json."""
    base_dir = pitches_dir or PITCHES_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "pitches": [],
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    for json_path in sorted(base_dir.glob("*.json")):
        if json_path.name.startswith("_"):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
            manifest["pitches"].append({
                "pitch_id": data["pitch_id"],
                "ticker": data["ticker"],
                "model_type": data.get("model_type", ""),
                "current_price": data.get("current_price"),
                "expected_return": data.get(
                    "probability_analysis", {}).get("expected_return"),
                "probability_upside": data.get(
                    "probability_analysis", {}).get("probability_upside"),
                "risk_reward_ratio": data.get(
                    "probability_analysis", {}).get("risk_reward_ratio"),
                "generated_at": data.get("generated_at", ""),
                "thesis_summary": (data.get("thesis_statement") or "")[:200],
            })
        except Exception as exc:
            logger.warning("Skipping %s in manifest: %s", json_path.name, exc)

    manifest["pitches"].sort(
        key=lambda p: p.get("generated_at") or "", reverse=True,
    )

    out = base_dir / "_manifest.json"
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
    logger.info("Updated manifest: %d published pitches → %s",
                len(manifest["pitches"]), out)
    return out


def publish_draft(pitch_id: str, pitches_dir: Optional[Path] = None) -> Path:
    """Promote a draft to published. Updates manifest."""
    base_dir = pitches_dir or PITCHES_DIR
    drafts = base_dir / "_drafts"
    draft_path = drafts / f"{pitch_id}.json"
    pub_path = base_dir / f"{pitch_id}.json"

    if not draft_path.exists():
        raise FileNotFoundError(f"No draft at {draft_path}")

    with open(draft_path) as f:
        data = json.load(f)
    data["is_draft"] = False
    with open(pub_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    draft_path.unlink()
    update_pitch_manifest(base_dir)
    logger.info("Published %s → %s", pitch_id, pub_path)
    return pub_path


__all__ = [
    "export_pitch_to_webapp",
    "update_pitch_manifest",
    "publish_draft",
    "PITCHES_DIR",
    "DRAFTS_DIR",
]
