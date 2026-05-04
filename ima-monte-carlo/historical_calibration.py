"""
Historical calibration of analyst-supplied distributions.

For every distributional input the analyst declares (revenue growth, EBITDA
margin, multiples, ...), this module computes:

    1. Where the analyst's distribution sits inside the company's own
       historical distribution of that metric.
    2. The same comparison against a sector peer group (peer_data.py).
    3. Mean-reversion gravity for empirically mean-reverting metrics
       (margins, ROIC, growth rates).
    4. Severity-tagged warnings when projections exceed historical extremes.

The tool is descriptive, not prescriptive — the analyst can still propose
anything they want. Calibration just makes it explicit when a projection
needs structural-break justification.

External data is fetched via yfinance. Companies with thin yfinance coverage
will produce sparse company contexts; sector peer aggregation fills the gap.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from distributions import Distribution

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  yfinance access — RawBundle wrapper
# --------------------------------------------------------------------------- #

@dataclass
class RawBundle:
    """Container for the raw yfinance dataframes we need per ticker."""
    ticker: str
    quarterly_financials: Optional[pd.DataFrame] = None
    quarterly_balance_sheet: Optional[pd.DataFrame] = None
    quarterly_cashflow: Optional[pd.DataFrame] = None
    info: dict = field(default_factory=dict)


def fetch_bundle(ticker: str, *, timeout: float = 15.0) -> Optional[RawBundle]:
    """Fetch a RawBundle for `ticker`. Returns None on any error."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning(
            "yfinance not installed — historical calibration disabled. "
            "Install with: pip install yfinance"
        )
        return None

    try:
        t = yf.Ticker(ticker)
        # yfinance attribute access can be slow + flaky; wrap each fetch
        bundle = RawBundle(ticker=ticker)
        try:
            bundle.quarterly_financials = t.quarterly_financials
        except Exception:
            bundle.quarterly_financials = None
        try:
            bundle.quarterly_balance_sheet = t.quarterly_balance_sheet
        except Exception:
            bundle.quarterly_balance_sheet = None
        try:
            bundle.quarterly_cashflow = t.quarterly_cashflow
        except Exception:
            bundle.quarterly_cashflow = None
        try:
            bundle.info = dict(t.info or {})
        except Exception:
            bundle.info = {}
        return bundle
    except Exception as e:
        logger.warning("Failed to fetch yfinance bundle for %s: %s", ticker, e)
        return None


def fetch_price_history(
    ticker: str,
    start: str = "2010-01-01",
    interval: str = "1mo",
) -> Optional[pd.Series]:
    """Fetch a closing-price series. Returns None on any error."""
    try:
        import yfinance as yf
    except ImportError:
        return None
    try:
        df = yf.download(
            ticker,
            start=start,
            interval=interval,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is None or df.empty:
            return None
        if "Close" in df.columns:
            ser = df["Close"]
        else:
            ser = df.iloc[:, 0]
        # Squeeze possible MultiIndex columns
        if isinstance(ser, pd.DataFrame):
            ser = ser.iloc[:, 0]
        ser = ser.dropna()
        ser.index = pd.to_datetime(ser.index)
        return ser
    except Exception as e:
        logger.warning("Failed to fetch price history for %s: %s", ticker, e)
        return None


# --------------------------------------------------------------------------- #
#  Field-name fallback (yfinance uses inconsistent line-item names)
# --------------------------------------------------------------------------- #

FIELDS = {
    "revenue": [
        "Total Revenue", "TotalRevenue", "Revenue",
        "Operating Revenue", "OperatingRevenue",
    ],
    "net_income": [
        "Net Income", "NetIncome",
        "Net Income Common Stockholders", "NetIncomeCommonStockholders",
        "Net Income From Continuing Operations",
    ],
    "operating_income": [
        "Operating Income", "OperatingIncome",
        "EBIT", "Operating Income Or Loss",
    ],
    "gross_profit": [
        "Gross Profit", "GrossProfit",
    ],
    "cost_of_revenue": [
        "Cost Of Revenue", "CostOfRevenue", "Cost Of Goods Sold",
    ],
    "depreciation": [
        "Reconciled Depreciation", "Depreciation",
        "Depreciation And Amortization",
        "DepreciationAmortizationDepletion",
    ],
    "ebitda": [
        "EBITDA", "Normalized EBITDA", "NormalizedEBITDA",
    ],
    "operating_cash_flow": [
        "Operating Cash Flow", "OperatingCashFlow",
        "Cash Flow From Continuing Operating Activities",
        "Total Cash From Operating Activities",
    ],
    "capex": [
        "Capital Expenditure", "CapitalExpenditures", "Capital Expenditures",
    ],
    "free_cash_flow": [
        "Free Cash Flow", "FreeCashFlow",
    ],
    "total_assets": [
        "Total Assets", "TotalAssets",
    ],
    "total_liab": [
        "Total Liabilities Net Minority Interest", "TotalLiab",
        "Total Liab", "Total Liabilities",
    ],
    "stockholder_equity": [
        "Stockholders Equity", "Total Stockholder Equity",
        "TotalStockholderEquity", "Common Stock Equity",
    ],
    "total_debt": [
        "Total Debt", "TotalDebt", "Long Term Debt",
    ],
    "cash": [
        "Cash And Cash Equivalents", "CashAndCashEquivalents",
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Short Term Investments",
    ],
    "shares": [
        "Share Issued", "ShareIssued",
        "Ordinary Shares Number", "OrdinarySharesNumber",
        "Common Stock Shares Outstanding", "CommonStockSharesOutstanding",
    ],
    "tax_rate": [
        "Tax Rate For Calcs", "TaxRateForCalcs",
    ],
}


def get_field(
    df: Optional[pd.DataFrame],
    field_names: list[str],
    column_index: int,
) -> Optional[float]:
    """
    Look up a value at column index `column_index` in `df`, trying each name
    in `field_names` until one matches. yfinance uses orientation:
    rows = line items, columns = period dates.
    """
    if df is None or df.empty:
        return None
    if column_index < 0 or column_index >= df.shape[1]:
        return None
    for name in field_names:
        if name in df.index:
            try:
                v = df.loc[name].iloc[column_index]
            except Exception:
                continue
            if pd.isna(v):
                continue
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def latest_field(
    df: Optional[pd.DataFrame],
    field_names: list[str],
) -> Optional[float]:
    """Most recent non-null value across columns for any matching field name."""
    if df is None or df.empty:
        return None
    for i in range(df.shape[1]):
        v = get_field(df, field_names, i)
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
#  Data structures
# --------------------------------------------------------------------------- #

@dataclass
class HistoricalContext:
    """Historical distribution of a single metric for one entity."""
    metric: str
    entity: str
    n_observations: int
    lookback_years: float

    mean: float
    median: float
    std: float
    min: float
    max: float
    p5: float
    p10: float
    p25: float
    p75: float
    p90: float
    p95: float

    long_run_mean: float
    recent_mean: float
    mean_reversion_implied: float
    autocorrelation_1y: Optional[float]

    raw_values: list[float] = field(default_factory=list)
    raw_dates: list[str] = field(default_factory=list)

    def percentile_of(self, value: float) -> float:
        """What percentile does `value` fall in within this distribution?"""
        if not self.raw_values or not np.isfinite(value):
            return float("nan")
        below = sum(1 for v in self.raw_values if v <= value)
        return 100.0 * below / len(self.raw_values)


@dataclass
class CalibrationReport:
    """Calibration assessment for a single input distribution."""
    input_name: str
    metric: str
    label: str
    input_distribution_summary: dict

    company_context: Optional[HistoricalContext]
    sector_context: Optional[HistoricalContext]

    analyst_mean_company_percentile: Optional[float]
    analyst_p90_company_percentile: Optional[float]
    analyst_p10_company_percentile: Optional[float]
    analyst_mean_sector_percentile: Optional[float]
    analyst_p90_sector_percentile: Optional[float]
    analyst_p10_sector_percentile: Optional[float]

    warnings: list[dict] = field(default_factory=list)
    reversion_note: Optional[str] = None

    @property
    def max_severity(self) -> str:
        sev = {"info": 0, "caution": 1, "warning": 2}
        if not self.warnings:
            return "info"
        return max(self.warnings, key=lambda w: sev.get(w.get("severity", "info"), 0))[
            "severity"
        ]


# --------------------------------------------------------------------------- #
#  Extractors — one per historical metric
# --------------------------------------------------------------------------- #

def _iter_quarters(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty:
        return 0
    return df.shape[1]


def extract_revenue_growth_yoy(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    fi = bundle.quarterly_financials
    nq = _iter_quarters(fi)
    if nq < 5:
        return []
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq - 4):
        cur_date = fi.columns[i]
        cur = get_field(fi, FIELDS["revenue"], i)
        prior = get_field(fi, FIELDS["revenue"], i + 4)
        if cur is None or prior is None or prior <= 0:
            continue
        out.append((cur_date, (cur - prior) / prior))
    return out


def extract_ebitda_margin(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    fi = bundle.quarterly_financials
    cf = bundle.quarterly_cashflow
    nq = _iter_quarters(fi)
    if nq == 0:
        return []
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq):
        date = fi.columns[i]
        rev = get_field(fi, FIELDS["revenue"], i)
        if rev is None or rev <= 0:
            continue
        ebitda = get_field(fi, FIELDS["ebitda"], i)
        if ebitda is None:
            # Build EBITDA = Operating Income + Depreciation
            op_inc = get_field(fi, FIELDS["operating_income"], i)
            dep = get_field(fi, FIELDS["depreciation"], i)
            if dep is None and cf is not None and i < cf.shape[1]:
                # cashflow columns are not guaranteed to align by index, but
                # they typically do for the same ticker since both come from
                # the same yfinance fetch; fall back gracefully.
                dep = get_field(cf, FIELDS["depreciation"], i)
            if op_inc is None:
                continue
            ebitda = op_inc + (dep or 0.0)
        out.append((date, ebitda / rev))
    return out


def extract_operating_margin(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    fi = bundle.quarterly_financials
    nq = _iter_quarters(fi)
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq):
        rev = get_field(fi, FIELDS["revenue"], i)
        op_inc = get_field(fi, FIELDS["operating_income"], i)
        if rev is None or rev <= 0 or op_inc is None:
            continue
        out.append((fi.columns[i], op_inc / rev))
    return out


def extract_gross_margin(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    fi = bundle.quarterly_financials
    nq = _iter_quarters(fi)
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq):
        rev = get_field(fi, FIELDS["revenue"], i)
        gp = get_field(fi, FIELDS["gross_profit"], i)
        if gp is None:
            cogs = get_field(fi, FIELDS["cost_of_revenue"], i)
            if rev is not None and rev > 0 and cogs is not None:
                gp = rev - cogs
        if rev is None or rev <= 0 or gp is None:
            continue
        out.append((fi.columns[i], gp / rev))
    return out


def extract_fcf_margin(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    cf = bundle.quarterly_cashflow
    fi = bundle.quarterly_financials
    nq_fi = _iter_quarters(fi)
    nq_cf = _iter_quarters(cf)
    if nq_fi == 0 or nq_cf == 0:
        return []
    # Build a date-indexed lookup for revenue
    rev_by_date: dict[pd.Timestamp, float] = {}
    for i in range(nq_fi):
        rev = get_field(fi, FIELDS["revenue"], i)
        if rev is not None:
            rev_by_date[pd.Timestamp(fi.columns[i])] = rev
    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq_cf):
        date = pd.Timestamp(cf.columns[i])
        fcf = get_field(cf, FIELDS["free_cash_flow"], i)
        if fcf is None:
            ocf = get_field(cf, FIELDS["operating_cash_flow"], i)
            capex = get_field(cf, FIELDS["capex"], i)
            if ocf is not None:
                fcf = ocf - abs(capex or 0.0)
        if fcf is None:
            continue
        rev = rev_by_date.get(date)
        if rev is None or rev <= 0:
            continue
        out.append((date, fcf / rev))
    return out


def extract_fcf_growth_yoy(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    cf = bundle.quarterly_cashflow
    nq = _iter_quarters(cf)
    if nq < 5:
        return []

    def _fcf(i: int) -> Optional[float]:
        v = get_field(cf, FIELDS["free_cash_flow"], i)
        if v is not None:
            return v
        ocf = get_field(cf, FIELDS["operating_cash_flow"], i)
        capex = get_field(cf, FIELDS["capex"], i)
        if ocf is None:
            return None
        return ocf - abs(capex or 0.0)

    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq - 4):
        cur = _fcf(i)
        prior = _fcf(i + 4)
        if cur is None or prior is None or prior == 0:
            continue
        # Only meaningful when prior FCF is positive
        if prior <= 0:
            continue
        out.append((cf.columns[i], (cur - prior) / prior))
    return out


def extract_roic(
    bundle: RawBundle, *_ignored,
) -> list[tuple[pd.Timestamp, float]]:
    fi = bundle.quarterly_financials
    bs = bundle.quarterly_balance_sheet
    nq_fi = _iter_quarters(fi)
    nq_bs = _iter_quarters(bs)
    if nq_fi < 4 or nq_bs == 0:
        return []
    # Index balance-sheet by date for nearest-on-or-before matching
    bs_dates = [pd.Timestamp(d) for d in bs.columns]

    def _ic_at(date: pd.Timestamp) -> Optional[float]:
        # Invested Capital = total debt + stockholder equity
        candidates = [(j, d) for j, d in enumerate(bs_dates) if d <= date]
        if not candidates:
            return None
        j, _ = max(candidates, key=lambda t: t[1])
        debt = get_field(bs, FIELDS["total_debt"], j) or 0.0
        equity = get_field(bs, FIELDS["stockholder_equity"], j)
        if equity is None:
            return None
        ic = debt + equity
        return ic if ic > 0 else None

    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq_fi - 3):
        # TTM operating income × (1 - tax_rate ≈ 0.25 fallback)
        ttm_op = 0.0
        ok = True
        for k in range(4):
            v = get_field(fi, FIELDS["operating_income"], i + k)
            if v is None:
                ok = False
                break
            ttm_op += v
        if not ok:
            continue
        tax = get_field(fi, FIELDS["tax_rate"], i)
        if tax is None or not (0.0 <= tax <= 0.6):
            tax = 0.25
        nopat = ttm_op * (1.0 - tax)
        date = pd.Timestamp(fi.columns[i])
        ic = _ic_at(date)
        if ic is None:
            continue
        out.append((date, nopat / ic))
    return out


def extract_pe_multiple(
    bundle: RawBundle, history_prices: Optional[pd.Series] = None,
) -> list[tuple[pd.Timestamp, float]]:
    if history_prices is None or history_prices.empty:
        return []
    fi = bundle.quarterly_financials
    bs = bundle.quarterly_balance_sheet
    nq = _iter_quarters(fi)
    if nq < 4:
        return []

    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq - 3):
        date = pd.Timestamp(fi.columns[i])
        ttm_ni = 0.0
        ok = True
        for k in range(4):
            v = get_field(fi, FIELDS["net_income"], i + k)
            if v is None:
                ok = False
                break
            ttm_ni += v
        if not ok or ttm_ni <= 0:
            continue
        # Shares: use the BS column nearest to this quarter
        shares = None
        if bs is not None and not bs.empty:
            bs_dates = [pd.Timestamp(d) for d in bs.columns]
            candidates = [(j, d) for j, d in enumerate(bs_dates) if d <= date]
            if candidates:
                j, _ = max(candidates, key=lambda t: t[1])
                shares = get_field(bs, FIELDS["shares"], j)
        if shares is None or shares <= 0:
            shares = bundle.info.get("sharesOutstanding")
        if not shares or shares <= 0:
            continue
        ttm_eps = ttm_ni / shares
        if ttm_eps <= 0:
            continue
        try:
            window = history_prices.loc[:date].tail(5)
        except Exception:
            continue
        if window.empty:
            continue
        try:
            close = float(window.iloc[-1])
        except (TypeError, ValueError):
            continue
        pe = close / ttm_eps
        if not np.isfinite(pe) or pe <= 0 or pe > 200:
            continue
        out.append((date, pe))
    return out


def extract_ev_ebitda(
    bundle: RawBundle, history_prices: Optional[pd.Series] = None,
) -> list[tuple[pd.Timestamp, float]]:
    if history_prices is None or history_prices.empty:
        return []
    fi = bundle.quarterly_financials
    bs = bundle.quarterly_balance_sheet
    cf = bundle.quarterly_cashflow
    nq = _iter_quarters(fi)
    if nq < 4 or bs is None or bs.empty:
        return []

    bs_dates = [pd.Timestamp(d) for d in bs.columns]

    def _bs_index(date: pd.Timestamp) -> Optional[int]:
        candidates = [(j, d) for j, d in enumerate(bs_dates) if d <= date]
        if not candidates:
            return None
        return max(candidates, key=lambda t: t[1])[0]

    out: list[tuple[pd.Timestamp, float]] = []
    for i in range(nq - 3):
        date = pd.Timestamp(fi.columns[i])
        # TTM EBITDA
        ttm_ebitda = 0.0
        ok = True
        for k in range(4):
            v = get_field(fi, FIELDS["ebitda"], i + k)
            if v is None:
                op = get_field(fi, FIELDS["operating_income"], i + k)
                dep = get_field(fi, FIELDS["depreciation"], i + k)
                if dep is None and cf is not None:
                    dep = get_field(cf, FIELDS["depreciation"], i + k)
                if op is None:
                    ok = False
                    break
                v = op + (dep or 0.0)
            ttm_ebitda += v
        if not ok or ttm_ebitda <= 0:
            continue
        j = _bs_index(date)
        if j is None:
            continue
        debt = get_field(bs, FIELDS["total_debt"], j) or 0.0
        cash = get_field(bs, FIELDS["cash"], j) or 0.0
        shares = get_field(bs, FIELDS["shares"], j) or bundle.info.get("sharesOutstanding")
        if not shares or shares <= 0:
            continue
        try:
            window = history_prices.loc[:date].tail(5)
        except Exception:
            continue
        if window.empty:
            continue
        try:
            close = float(window.iloc[-1])
        except (TypeError, ValueError):
            continue
        market_cap = close * shares
        ev = market_cap + debt - cash
        if ev <= 0:
            continue
        ratio = ev / ttm_ebitda
        if not np.isfinite(ratio) or ratio <= 0 or ratio > 100:
            continue
        out.append((date, ratio))
    return out


# --------------------------------------------------------------------------- #
#  Metric registry
# --------------------------------------------------------------------------- #

METRIC_REGISTRY: dict[str, dict] = {
    "revenue_growth_yoy": {
        "label": "Revenue Growth (YoY)",
        "unit": "ratio",
        "extractor": extract_revenue_growth_yoy,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (-0.50, 1.00),
        "needs_prices": False,
    },
    "ebitda_margin": {
        "label": "EBITDA Margin",
        "unit": "ratio",
        "extractor": extract_ebitda_margin,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (-0.50, 0.80),
        "needs_prices": False,
    },
    "operating_margin": {
        "label": "Operating Margin",
        "unit": "ratio",
        "extractor": extract_operating_margin,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (-0.50, 0.70),
        "needs_prices": False,
    },
    "gross_margin": {
        "label": "Gross Margin",
        "unit": "ratio",
        "extractor": extract_gross_margin,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (0.0, 1.0),
        "needs_prices": False,
    },
    "fcf_margin": {
        "label": "FCF Margin",
        "unit": "ratio",
        "extractor": extract_fcf_margin,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (-0.50, 0.50),
        "needs_prices": False,
    },
    "fcf_growth_yoy": {
        "label": "FCF Growth (YoY)",
        "unit": "ratio",
        "extractor": extract_fcf_growth_yoy,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (-1.00, 2.00),
        "needs_prices": False,
    },
    "roic": {
        "label": "Return on Invested Capital",
        "unit": "ratio",
        "extractor": extract_roic,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (-0.50, 0.80),
        "needs_prices": False,
    },
    "wacc": {
        "label": "Weighted Average Cost of Capital",
        "unit": "ratio",
        "extractor": None,
        "frequency": "n/a",
        "mean_reverting": False,
        "needs_prices": False,
        "comment": "WACC is forward-looking; no historical analog.",
    },
    "pe_multiple": {
        "label": "P/E Multiple",
        "unit": "multiple",
        "extractor": extract_pe_multiple,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (0.0, 100.0),
        "needs_prices": True,
    },
    "ev_ebitda_multiple": {
        "label": "EV/EBITDA Multiple",
        "unit": "multiple",
        "extractor": extract_ev_ebitda,
        "frequency": "quarterly",
        "mean_reverting": True,
        "lookback_quarters": 40,
        "valid_range": (0.0, 50.0),
        "needs_prices": True,
    },
}


# --------------------------------------------------------------------------- #
#  Building HistoricalContext
# --------------------------------------------------------------------------- #

def _build_context_from_values(
    metric: str,
    entity: str,
    values: list[float],
    dates: list,
    *,
    long_run_mean: Optional[float] = None,
    recent_mean: Optional[float] = None,
    autocorr: Optional[float] = None,
) -> HistoricalContext:
    arr = np.asarray(values, dtype=float)
    spec = METRIC_REGISTRY[metric]

    long_run = long_run_mean if long_run_mean is not None else float(np.mean(arr))
    recent = recent_mean if recent_mean is not None else float(np.mean(arr[: min(4, len(arr))]))
    if spec.get("mean_reverting", False):
        mr = 0.7 * long_run + 0.3 * recent
    else:
        mr = recent

    return HistoricalContext(
        metric=metric,
        entity=entity,
        n_observations=len(arr),
        lookback_years=len(arr) / 4.0,
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        p5=float(np.percentile(arr, 5)),
        p10=float(np.percentile(arr, 10)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p90=float(np.percentile(arr, 90)),
        p95=float(np.percentile(arr, 95)),
        long_run_mean=long_run,
        recent_mean=recent,
        mean_reversion_implied=mr,
        autocorrelation_1y=autocorr,
        raw_values=values,
        raw_dates=[str(d) for d in dates],
    )


def build_historical_context(
    metric: str,
    ticker: str,
    bundle: RawBundle,
    history_prices: Optional[pd.Series] = None,
    *,
    min_observations: int = 8,
) -> Optional[HistoricalContext]:
    """
    Compute the full historical distribution + reversion stats for one
    metric on one company. Returns None on insufficient data.
    """
    spec = METRIC_REGISTRY.get(metric)
    if spec is None or spec.get("extractor") is None:
        return None
    extractor: Callable = spec["extractor"]

    try:
        if spec.get("needs_prices"):
            obs = extractor(bundle, history_prices)
        else:
            obs = extractor(bundle)
    except Exception as e:
        logger.warning("Extractor failed for %s on %s: %s", metric, ticker, e)
        return None

    if not obs or len(obs) < min_observations:
        return None

    # Filter valid range
    valid_range = spec.get("valid_range")
    filtered: list[tuple[Any, float]] = []
    for date, val in obs:
        if val is None or not np.isfinite(val):
            continue
        if valid_range and not (valid_range[0] <= val <= valid_range[1]):
            continue
        filtered.append((date, val))
    if len(filtered) < min_observations:
        return None

    # Sort newest-first → oldest-last (matches yfinance convention; the first
    # 4 entries are the trailing year used for `recent_mean`).
    filtered.sort(key=lambda t: t[0], reverse=True)
    dates, values = zip(*filtered)
    values = list(values)

    arr = np.asarray(values)
    long_run = float(np.mean(arr))
    recent = float(np.mean(arr[: min(4, len(arr))]))

    autocorr: Optional[float] = None
    if len(arr) >= 12:
        try:
            ac = pd.Series(arr).autocorr(lag=4)
            autocorr = float(ac) if pd.notna(ac) else None
        except Exception:
            autocorr = None

    return _build_context_from_values(
        metric=metric,
        entity=ticker,
        values=values,
        dates=list(dates),
        long_run_mean=long_run,
        recent_mean=recent,
        autocorr=autocorr,
    )


# --------------------------------------------------------------------------- #
#  Calibration report
# --------------------------------------------------------------------------- #

def _summarize_distribution(
    distribution: Distribution, n_sample: int = 10_000, seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    try:
        samples = distribution.sample(n_sample, rng)
    except Exception:
        # Fallback to summary stats with a normal approximation
        try:
            m = distribution.mean()
            s = distribution.std()
            return {
                "mean": float(m),
                "p10": float(m - 1.28 * s),
                "p90": float(m + 1.28 * s),
                "std": float(s),
                "median": float(distribution.median()),
            }
        except Exception:
            return {"mean": float("nan"), "p10": float("nan"), "p90": float("nan"),
                    "std": float("nan"), "median": float("nan")}

    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "p10": float("nan"), "p90": float("nan"),
                "std": float("nan"), "median": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _generate_calibration_warnings(
    input_name: str,
    metric: str,
    summary: dict,
    company_context: Optional[HistoricalContext],
    sector_context: Optional[HistoricalContext],
    co_mean_pct: Optional[float],
    co_p90_pct: Optional[float],
    co_p10_pct: Optional[float],
    sec_mean_pct: Optional[float],
    sec_p90_pct: Optional[float],
    sec_p10_pct: Optional[float],
) -> list[dict]:
    warnings_list: list[dict] = []
    spec = METRIC_REGISTRY.get(metric, {})
    label = spec.get("label", metric)
    is_ratio = spec.get("unit") == "ratio"
    fmt = (lambda x: f"{x:.1%}") if is_ratio else (lambda x: f"{x:.1f}x")

    # Bull case exceeds historical max
    if company_context is not None and np.isfinite(summary.get("p90", float("nan"))):
        if summary["p90"] > company_context.max:
            warnings_list.append({
                "severity": "warning",
                "message": (
                    f"Bull case ({fmt(summary['p90'])}) exceeds the company's "
                    f"all-time high ({fmt(company_context.max)} over "
                    f"{company_context.lookback_years:.1f}y, "
                    f"n={company_context.n_observations})."
                ),
                "suggestion": (
                    "Defend explicitly: what structural change supports outperforming "
                    "every prior period?"
                ),
            })
        elif co_p90_pct is not None and co_p90_pct > 95:
            warnings_list.append({
                "severity": "caution",
                "message": (
                    f"Bull case sits at {co_p90_pct:.0f}th percentile of company "
                    f"history — in the top 5% of all observed outcomes."
                ),
                "suggestion": "Confirm the catalysts that would replicate top-5% periods.",
            })

    # Mean projection vs company history
    if company_context is not None and co_mean_pct is not None:
        if co_mean_pct > 80:
            warnings_list.append({
                "severity": "caution",
                "message": (
                    f"Base case ({fmt(summary['mean'])}) sits at {co_mean_pct:.0f}th "
                    f"percentile of company history. Historical median: "
                    f"{fmt(company_context.median)}."
                ),
                "suggestion": "Is the base case really 'base,' or a soft bull case?",
            })
        elif co_mean_pct < 20:
            warnings_list.append({
                "severity": "info",
                "message": (
                    f"Base case is conservative — {co_mean_pct:.0f}th percentile "
                    f"of company history."
                ),
                "suggestion": None,
            })

    # Sector comparison
    if sector_context is not None and sec_mean_pct is not None and sec_mean_pct > 90:
        warnings_list.append({
            "severity": "caution",
            "message": (
                f"Base case exceeds {sec_mean_pct:.0f}% of sector peers' history. "
                f"Sector median: {fmt(sector_context.median)}."
            ),
            "suggestion": (
                "If thesis is company-specific outperformance, confirm the moat. "
                "If sector tailwind, consider whether peers should project similarly."
            ),
        })

    # Bear case still optimistic
    if company_context is not None and co_p10_pct is not None and co_p10_pct > 50:
        warnings_list.append({
            "severity": "warning",
            "message": (
                f"Bear case ({fmt(summary['p10'])}) is still above the company's "
                f"historical median ({fmt(company_context.median)})."
            ),
            "suggestion": (
                "The downside scenario implies above-average performance even in the "
                "bear case. Is your distribution capturing real downside?"
            ),
        })

    # Mean reversion gravity (>2σ from long-run average)
    if (
        company_context is not None
        and company_context.long_run_mean is not None
        and company_context.std is not None
        and company_context.std > 0
        and np.isfinite(summary.get("mean", float("nan")))
        and abs(summary["mean"] - company_context.long_run_mean) > 2 * company_context.std
    ):
        direction = "above" if summary["mean"] > company_context.long_run_mean else "below"
        warnings_list.append({
            "severity": "caution",
            "message": (
                f"Base case is >2σ {direction} the company's "
                f"{company_context.lookback_years:.1f}-year average "
                f"({fmt(company_context.long_run_mean)}). Mean reversion is a "
                f"strong empirical force for {label}."
            ),
            "suggestion": (
                "Consider whether the projection should drift toward the long-run "
                "average over the projection horizon."
            ),
        })

    return warnings_list


def _generate_reversion_note(
    metric: str,
    summary: dict,
    context: Optional[HistoricalContext],
) -> Optional[str]:
    if context is None:
        return None
    spec = METRIC_REGISTRY.get(metric, {})
    if not spec.get("mean_reverting"):
        return None
    if not np.isfinite(summary.get("mean", float("nan"))):
        return None

    long_run = context.long_run_mean
    recent = context.recent_mean
    proposed = summary["mean"]
    label = spec.get("label", metric)
    fmt = (lambda x: f"{x:.1%}") if spec.get("unit") == "ratio" else (lambda x: f"{x:.1f}x")

    if context.std == 0 or abs(recent - long_run) < 0.5 * context.std:
        return None

    if recent > long_run and proposed > recent:
        return (
            f"Recent {label} ({fmt(recent)}) is elevated vs "
            f"{context.lookback_years:.1f}-year average ({fmt(long_run)}). Projecting "
            f"forward at {fmt(proposed)} assumes the elevated regime persists. "
            f"Mean-reversion-implied projection: {fmt(context.mean_reversion_implied)}."
        )
    if recent < long_run and proposed < recent:
        return (
            f"Recent {label} ({fmt(recent)}) is depressed vs "
            f"{context.lookback_years:.1f}-year average ({fmt(long_run)}). Projecting "
            f"forward at {fmt(proposed)} assumes the depressed regime persists. "
            f"Mean-reversion-implied projection: {fmt(context.mean_reversion_implied)}."
        )
    return None


def calibrate_input(
    input_name: str,
    metric: str,
    distribution: Distribution,
    company_context: Optional[HistoricalContext],
    sector_context: Optional[HistoricalContext],
    *,
    label: Optional[str] = None,
) -> CalibrationReport:
    """Generate a CalibrationReport for one input."""
    summary = _summarize_distribution(distribution)

    co_mean_pct = company_context.percentile_of(summary["mean"]) if company_context else None
    co_p90_pct = company_context.percentile_of(summary["p90"]) if company_context else None
    co_p10_pct = company_context.percentile_of(summary["p10"]) if company_context else None
    if co_mean_pct is not None and not np.isfinite(co_mean_pct):
        co_mean_pct = None
    if co_p90_pct is not None and not np.isfinite(co_p90_pct):
        co_p90_pct = None
    if co_p10_pct is not None and not np.isfinite(co_p10_pct):
        co_p10_pct = None

    sec_mean_pct = sector_context.percentile_of(summary["mean"]) if sector_context else None
    sec_p90_pct = sector_context.percentile_of(summary["p90"]) if sector_context else None
    sec_p10_pct = sector_context.percentile_of(summary["p10"]) if sector_context else None
    if sec_mean_pct is not None and not np.isfinite(sec_mean_pct):
        sec_mean_pct = None
    if sec_p90_pct is not None and not np.isfinite(sec_p90_pct):
        sec_p90_pct = None
    if sec_p10_pct is not None and not np.isfinite(sec_p10_pct):
        sec_p10_pct = None

    warnings_list = _generate_calibration_warnings(
        input_name, metric, summary,
        company_context, sector_context,
        co_mean_pct, co_p90_pct, co_p10_pct,
        sec_mean_pct, sec_p90_pct, sec_p10_pct,
    )

    reversion_note = _generate_reversion_note(metric, summary, company_context)

    spec = METRIC_REGISTRY.get(metric, {})
    return CalibrationReport(
        input_name=input_name,
        metric=metric,
        label=label or spec.get("label", metric),
        input_distribution_summary=summary,
        company_context=company_context,
        sector_context=sector_context,
        analyst_mean_company_percentile=co_mean_pct,
        analyst_p90_company_percentile=co_p90_pct,
        analyst_p10_company_percentile=co_p10_pct,
        analyst_mean_sector_percentile=sec_mean_pct,
        analyst_p90_sector_percentile=sec_p90_pct,
        analyst_p10_sector_percentile=sec_p10_pct,
        warnings=warnings_list,
        reversion_note=reversion_note,
    )


# --------------------------------------------------------------------------- #
#  Pitch-config integration helpers
# --------------------------------------------------------------------------- #

def collect_calibration_inputs(
    raw_inputs: dict, prefix: str = "",
) -> list[tuple[str, str]]:
    """
    Walk a pitch INPUTS dict (or a SOTP segments list) and return
    [(input_name, historical_metric)] pairs for every input that declares a
    `historical_metric` field.

    For SOTP, callers should call this on each segment with the segment name
    as `prefix` so the returned input names match the prefixed names exposed
    by the simulation engine.
    """
    out: list[tuple[str, str]] = []
    for name, spec in raw_inputs.items():
        if not isinstance(spec, dict):
            continue
        metric = spec.get("historical_metric")
        if not metric:
            continue
        if metric not in METRIC_REGISTRY:
            logger.warning(
                "Input %r references unknown historical_metric %r — skipped.",
                name, metric,
            )
            continue
        full_name = f"{prefix}/{name}" if prefix else name
        out.append((full_name, metric))
    return out


def collect_calibration_inputs_from_pitch(mod) -> list[tuple[str, str]]:
    """Extract (input_name, metric) pairs from a loaded pitch module."""
    pairs: list[tuple[str, str]] = []
    model_type = (getattr(mod, "MODEL_TYPE", "") or "").lower()
    if model_type == "sotp":
        segments = getattr(mod, "SEGMENTS", []) or []
        for seg in segments:
            seg_name = seg.get("name", "")
            seg_inputs = seg.get("inputs", {}) or {}
            pairs.extend(collect_calibration_inputs(seg_inputs, prefix=seg_name))
        # Also look at HOLDCO under "holdco/" prefix
        holdco = getattr(mod, "HOLDCO", {}) or {}
        pairs.extend(collect_calibration_inputs(holdco, prefix="holdco"))
    else:
        inputs = getattr(mod, "INPUTS", {}) or {}
        pairs.extend(collect_calibration_inputs(inputs))
    return pairs


__all__ = [
    "RawBundle",
    "fetch_bundle",
    "fetch_price_history",
    "FIELDS",
    "get_field",
    "latest_field",
    "HistoricalContext",
    "CalibrationReport",
    "METRIC_REGISTRY",
    "build_historical_context",
    "calibrate_input",
    "collect_calibration_inputs",
    "collect_calibration_inputs_from_pitch",
]
