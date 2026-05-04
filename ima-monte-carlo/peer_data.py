"""
Sector-peer identification and pooled historical aggregation.

Two responsibilities:
    1. Given a target ticker, return a list of sector peers within a similar
       market cap band (loaded from the IJR holdings file or any caller-
       supplied universe DataFrame).
    2. Pool historical observations across peers into a single sector-level
       HistoricalContext, used for calibration alongside the company's own
       history.

Peer fundamentals fetching is expensive (many yfinance calls). Results are
cached as parquet files under data/peer_cache/ and refreshed on a weekly
cadence (or whenever --refresh-peers is passed).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from historical_calibration import (
    METRIC_REGISTRY,
    HistoricalContext,
    RawBundle,
    _build_context_from_values,
    fetch_bundle,
    fetch_price_history,
)

logger = logging.getLogger(__name__)


CACHE_DIR = Path("data/peer_cache")
CACHE_TTL_DAYS = 7

_HERE = Path(__file__).resolve().parent
_UNIVERSE_CANDIDATES = [
    _HERE / "IJR_holdings.csv",
    _HERE.parent / "IJR_holdings.csv",
    _HERE.parent.parent / "IJR_holdings.csv",
    Path("IJR_holdings.csv"),
]


def _default_universe_path() -> Optional[Path]:
    for p in _UNIVERSE_CANDIDATES:
        if p.exists():
            return p
    return None


DEFAULT_UNIVERSE_PATH = _default_universe_path() or _UNIVERSE_CANDIDATES[1]


# --------------------------------------------------------------------------- #
#  Universe loading
# --------------------------------------------------------------------------- #

def load_universe(path: Optional[Path | str] = None) -> pd.DataFrame:
    """
    Load the equity universe used for peer identification. Returns a DataFrame
    indexed by ticker with columns: Sector, MarketCap (in dollars).

    The IJR holdings file from iShares is the default. The schema:
        Ticker, Name, Type, Sector, ..., Market Value, ..., Quantity, Price
    Market cap is approximated as Quantity * Price for the ETF's holdings —
    a rough proxy for the company's total market cap, but adequate for the
    "0.3x to 3x of company market cap" peer band.
    """
    p = Path(path) if path else DEFAULT_UNIVERSE_PATH
    if not p.exists():
        logger.warning("Universe file not found at %s — peer search disabled.", p)
        return pd.DataFrame(columns=["Sector", "MarketCap"])

    # The IJR file has 9 metadata rows then a blank line then the header.
    # Use a simple heuristic: skiprows until we find a row starting with "Ticker,".
    with open(p, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Ticker,"):
            header_idx = i
            break
    if header_idx is None:
        logger.warning("Could not locate header row in %s", p)
        return pd.DataFrame(columns=["Sector", "MarketCap"])

    df = pd.read_csv(p, skiprows=header_idx, thousands=",")
    df = df[df.get("Type") == "EQUITY"]
    if "Quantity" in df.columns and "Price" in df.columns:
        df["MarketCap"] = pd.to_numeric(df["Quantity"], errors="coerce") * pd.to_numeric(
            df["Price"], errors="coerce"
        )
    elif "Market Value" in df.columns:
        df["MarketCap"] = pd.to_numeric(df["Market Value"], errors="coerce")
    else:
        df["MarketCap"] = np.nan

    df = df.dropna(subset=["Ticker", "Sector"])
    df = df.set_index("Ticker")
    return df[["Sector", "MarketCap"]]


# --------------------------------------------------------------------------- #
#  Peer identification
# --------------------------------------------------------------------------- #

def identify_peers(
    ticker: str,
    universe: pd.DataFrame,
    *,
    n_peers: int = 30,
    market_cap_band: tuple[float, float] = (0.3, 3.0),
    target_market_cap: Optional[float] = None,
    target_sector: Optional[str] = None,
) -> list[str]:
    """
    Identify sector peers within a similar market cap band.

    If `ticker` is in the universe, its sector and market cap are used.
    Otherwise the caller may pass `target_sector` and `target_market_cap`
    explicitly (useful for large-cap stocks not in the small-cap IJR file).
    """
    if universe is None or universe.empty:
        return []

    sector = target_sector
    mcap = target_market_cap
    if ticker in universe.index:
        sector = sector or universe.loc[ticker, "Sector"]
        mcap_val = universe.loc[ticker, "MarketCap"]
        if mcap is None and pd.notna(mcap_val):
            mcap = float(mcap_val)
    if sector is None:
        return []

    candidates = universe[universe["Sector"] == sector]
    candidates = candidates[candidates.index != ticker]

    if mcap is not None and "MarketCap" in candidates.columns:
        lo = mcap * market_cap_band[0]
        hi = mcap * market_cap_band[1]
        within_band = candidates[
            (candidates["MarketCap"] >= lo) & (candidates["MarketCap"] <= hi)
        ]
        # If band is too restrictive, fall back to the full sector list
        if len(within_band) >= max(5, n_peers // 2):
            candidates = within_band

    # Sort by proximity to target market cap (closest first)
    if mcap is not None and "MarketCap" in candidates.columns:
        candidates = candidates.assign(
            _dist=(candidates["MarketCap"] - mcap).abs()
        ).sort_values("_dist")

    return list(candidates.index[:n_peers])


# --------------------------------------------------------------------------- #
#  Caching
# --------------------------------------------------------------------------- #

def _cache_paths(ticker: str) -> dict[str, Path]:
    base = CACHE_DIR
    return {
        "peers": base / f"peers_{ticker}.json",
        "fundamentals": base / f"peer_fundamentals_{ticker}.parquet",
        "prices": base / f"peer_prices_{ticker}.parquet",
        "meta": base / f"peer_meta_{ticker}.json",
    }


def _cache_fresh(meta_path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    ts = meta.get("written_at")
    if not ts:
        return False
    try:
        written = datetime.fromisoformat(ts)
    except ValueError:
        return False
    return datetime.utcnow() - written < timedelta(days=ttl_days)


def _bundle_to_records(bundle: RawBundle) -> list[dict]:
    """Flatten a RawBundle's dataframes into long-form rows for parquet storage."""
    records: list[dict] = []
    for source, df in (
        ("financials", bundle.quarterly_financials),
        ("balance_sheet", bundle.quarterly_balance_sheet),
        ("cashflow", bundle.quarterly_cashflow),
    ):
        if df is None or df.empty:
            continue
        try:
            stacked = df.stack(dropna=False).reset_index()
        except Exception:
            continue
        stacked.columns = ["item", "period", "value"]
        for _, row in stacked.iterrows():
            records.append({
                "ticker": bundle.ticker,
                "source": source,
                "item": row["item"],
                "period": pd.Timestamp(row["period"]).isoformat()
                          if pd.notna(row["period"]) else "",
                "value": float(row["value"]) if pd.notna(row["value"]) else None,
            })
    return records


def _records_to_bundle(ticker: str, df: pd.DataFrame) -> RawBundle:
    """Re-hydrate a RawBundle from cached records."""
    bundle = RawBundle(ticker=ticker)
    sub = df[df["ticker"] == ticker]
    for source, attr in (
        ("financials", "quarterly_financials"),
        ("balance_sheet", "quarterly_balance_sheet"),
        ("cashflow", "quarterly_cashflow"),
    ):
        seg = sub[sub["source"] == source]
        if seg.empty:
            continue
        try:
            pivoted = seg.pivot_table(
                index="item", columns="period", values="value", aggfunc="first"
            )
        except Exception:
            continue
        if pivoted.empty:
            continue
        # Convert period strings back to Timestamps; sort newest first
        pivoted.columns = [pd.Timestamp(c) if c else pd.NaT for c in pivoted.columns]
        pivoted = pivoted.reindex(sorted(pivoted.columns, reverse=True), axis=1)
        setattr(bundle, attr, pivoted)
    return bundle


def fetch_peer_data(
    ticker: str,
    peer_tickers: list[str],
    *,
    needs_prices: bool = False,
    refresh: bool = False,
    pause_seconds: float = 0.0,
) -> tuple[dict[str, RawBundle], dict[str, pd.Series]]:
    """
    Fetch + cache fundamentals (and optionally price history) for peer tickers.

    Returns (bundles_by_ticker, prices_by_ticker). prices is empty if
    `needs_prices` is False.
    """
    paths = _cache_paths(ticker)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bundles: dict[str, RawBundle] = {}
    prices: dict[str, pd.Series] = {}

    fundamentals_fresh = _cache_fresh(paths["meta"]) and not refresh

    if fundamentals_fresh and paths["fundamentals"].exists():
        try:
            fund_df = pd.read_parquet(paths["fundamentals"])
            for tk in peer_tickers:
                bundles[tk] = _records_to_bundle(tk, fund_df)
            if needs_prices and paths["prices"].exists():
                price_df = pd.read_parquet(paths["prices"])
                for tk in peer_tickers:
                    sub = price_df[price_df["ticker"] == tk]
                    if sub.empty:
                        continue
                    s = pd.Series(
                        sub["close"].values,
                        index=pd.to_datetime(sub["date"]),
                    ).sort_index()
                    prices[tk] = s
            logger.info(
                "Loaded peer cache for %s: %d peers (fundamentals%s)",
                ticker, len(bundles),
                ", prices" if needs_prices and prices else "",
            )
            if needs_prices and not prices:
                # Cache lacked prices; fall through and refresh prices only
                pass
            else:
                return bundles, prices
        except Exception as e:
            logger.warning("Failed to load peer cache for %s: %s — refetching.", ticker, e)
            bundles = {}
            prices = {}

    # Fetch fresh data
    logger.info("Fetching peer data for %s: %d peers...", ticker, len(peer_tickers))
    all_records: list[dict] = []
    price_records: list[dict] = []
    for i, tk in enumerate(peer_tickers, 1):
        bundle = fetch_bundle(tk)
        if bundle is None:
            continue
        bundles[tk] = bundle
        all_records.extend(_bundle_to_records(bundle))
        if needs_prices:
            ph = fetch_price_history(tk)
            if ph is not None and not ph.empty:
                prices[tk] = ph
                for date, val in ph.items():
                    price_records.append({
                        "ticker": tk,
                        "date": pd.Timestamp(date).isoformat(),
                        "close": float(val),
                    })
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        if i % 5 == 0:
            logger.info("  fetched %d/%d peers", i, len(peer_tickers))

    # Persist
    try:
        if all_records:
            pd.DataFrame(all_records).to_parquet(paths["fundamentals"], index=False)
        if price_records:
            pd.DataFrame(price_records).to_parquet(paths["prices"], index=False)
        paths["meta"].write_text(json.dumps({
            "ticker": ticker,
            "peers": peer_tickers,
            "n_peers": len(bundles),
            "needs_prices": needs_prices,
            "written_at": datetime.utcnow().isoformat(),
        }))
        paths["peers"].write_text(json.dumps(peer_tickers))
    except Exception as e:
        logger.warning("Could not persist peer cache for %s: %s", ticker, e)

    return bundles, prices


# --------------------------------------------------------------------------- #
#  Pooling
# --------------------------------------------------------------------------- #

def build_peer_context(
    metric: str,
    peer_tickers: list[str],
    peer_bundles: dict[str, RawBundle],
    peer_prices: Optional[dict[str, pd.Series]] = None,
    *,
    min_observations: int = 30,
) -> Optional[HistoricalContext]:
    """
    Pool historical observations across peer tickers into a single
    HistoricalContext.
    """
    spec = METRIC_REGISTRY.get(metric)
    if spec is None or spec.get("extractor") is None:
        return None

    extractor = spec["extractor"]
    valid_range = spec.get("valid_range")
    needs_prices = spec.get("needs_prices", False)

    all_values: list[float] = []
    all_dates: list = []

    for tk in peer_tickers:
        bundle = peer_bundles.get(tk)
        if bundle is None:
            continue
        try:
            if needs_prices:
                if not peer_prices:
                    continue
                obs = extractor(bundle, peer_prices.get(tk))
            else:
                obs = extractor(bundle)
        except Exception:
            continue
        for date, val in obs or []:
            if val is None or not np.isfinite(val):
                continue
            if valid_range and not (valid_range[0] <= val <= valid_range[1]):
                continue
            all_values.append(val)
            all_dates.append(date)

    if len(all_values) < min_observations:
        return None

    arr = np.asarray(all_values)
    long_run = float(np.mean(arr))
    return _build_context_from_values(
        metric=metric,
        entity=f"Sector peers (n={len(peer_tickers)})",
        values=all_values,
        dates=all_dates,
        long_run_mean=long_run,
        recent_mean=long_run,  # peer pool has no meaningful "recent"
        autocorr=None,
    )


__all__ = [
    "load_universe",
    "identify_peers",
    "fetch_peer_data",
    "build_peer_context",
    "CACHE_DIR",
    "DEFAULT_UNIVERSE_PATH",
]
