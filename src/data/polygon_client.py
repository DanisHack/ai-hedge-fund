"""Polygon.io API wrapper using polygon-api-client."""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import date, datetime
from typing import Optional

from polygon import RESTClient

from src.config.settings import POLYGON_API_KEY, validate_polygon_key
from src.data.cache import get_cache
from src.data.models import CompanyDetails, CompanyNews, FinancialMetrics, Price

logger = logging.getLogger(__name__)

_client: Optional[RESTClient] = None


class _RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, max_calls: int = 5, period: float = 60.0):
        self._max_calls = max_calls
        self._period = period
        self._lock = threading.Lock()
        self._timestamps: deque[float] = deque()

    def acquire(self) -> None:
        """Block until a request slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Purge timestamps outside the window
                while self._timestamps and now - self._timestamps[0] >= self._period:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                # Calculate wait time until oldest timestamp expires
                wait = self._period - (now - self._timestamps[0]) + 0.1
            time.sleep(wait)


_rate_limiter = _RateLimiter(max_calls=5, period=65.0)  # Polygon free tier: 5 req/min


def _get_client() -> RESTClient:
    """Return or create the singleton Polygon RESTClient."""
    global _client
    if _client is None:
        validate_polygon_key()
        _client = RESTClient(api_key=POLYGON_API_KEY)
    return _client


def _throttled_call(fn, *args, **kwargs):
    """Call a Polygon API function with rate limiting and retry on 429."""
    max_retries = 3
    for attempt in range(max_retries + 1):
        _rate_limiter.acquire()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries:
                wait = 12 * (attempt + 1)
                logger.debug(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
                time.sleep(wait)
                continue
            raise


# ── Prices ──────────────────────────────────────────────────────────


def get_prices(
    ticker: str,
    start_date: str,
    end_date: str,
    multiplier: int = 1,
    timespan: str = "day",
) -> list[Price]:
    """Fetch OHLCV bars from Polygon Aggregates endpoint."""
    cache = get_cache()
    cached = cache.get("prices", ticker, start_date, end_date, str(multiplier), timespan)
    if cached is not None:
        return cached

    client = _get_client()
    logger.debug(f"Fetching prices for {ticker} from {start_date} to {end_date}")

    aggs = _throttled_call(
        client.get_aggs,
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=start_date,
        to=end_date,
        limit=50000,
    )

    prices = []
    for agg in aggs:
        prices.append(Price(
            open=agg.open,
            high=agg.high,
            low=agg.low,
            close=agg.close,
            volume=agg.volume,
            timestamp=datetime.fromtimestamp(agg.timestamp / 1000),
            vwap=getattr(agg, "vwap", None),
            transactions=getattr(agg, "transactions", None),
        ))

    cache.set("prices", ticker, start_date, end_date, str(multiplier), timespan, prices)
    logger.debug(f"Got {len(prices)} price bars for {ticker}")
    return prices


# ── Financials ──────────────────────────────────────────────────────


_MAX_FINANCIAL_LIMIT = 10  # Always fetch max, slice for smaller requests


def get_financial_metrics(
    ticker: str,
    end_date: Optional[str] = None,
    limit: int = 4,
) -> list[FinancialMetrics]:
    """Fetch stock financials from Polygon vX endpoint."""
    cache = get_cache()
    cache_key_date = end_date or "latest"

    # Check if we already have a full fetch cached
    cached = cache.get("financials", ticker, cache_key_date)
    if cached is not None:
        return cached[:limit]

    client = _get_client()
    fetch_limit = max(limit, _MAX_FINANCIAL_LIMIT)
    logger.debug(f"Fetching financials for {ticker} (limit={fetch_limit})")

    params: dict = {
        "ticker": ticker,
        "limit": fetch_limit,
        "order": "desc",
        "sort": "period_of_report_date",
    }
    if end_date:
        params["period_of_report_date_lte"] = end_date

    metrics = []
    try:
        results = _throttled_call(lambda: list(client.vx.list_stock_financials(**params)))
        for fin in results:
            m = _parse_financial(ticker, fin)
            if m is not None:
                metrics.append(m)
            if len(metrics) >= fetch_limit:
                break
    except Exception as e:
        logger.warning(f"Failed to fetch financials for {ticker}: {e}")

    # Cache with normalized key (no limit) so all agents share
    cache.set("financials", ticker, cache_key_date, metrics)
    logger.debug(f"Got {len(metrics)} financial periods for {ticker}")
    return metrics[:limit]


def _parse_financial(ticker: str, fin: object) -> Optional[FinancialMetrics]:
    """Parse a single Polygon financials response into our model."""
    try:
        financials = getattr(fin, "financials", None)
        if financials is None:
            return None

        income = getattr(financials, "income_statement", {}) or {}
        balance = getattr(financials, "balance_sheet", {}) or {}
        cash_flow = getattr(financials, "cash_flow_statement", {}) or {}

        # Handle both dict and object-like access
        if not isinstance(income, dict):
            income = income.__dict__ if hasattr(income, "__dict__") else {}
        if not isinstance(balance, dict):
            balance = balance.__dict__ if hasattr(balance, "__dict__") else {}
        if not isinstance(cash_flow, dict):
            cash_flow = cash_flow.__dict__ if hasattr(cash_flow, "__dict__") else {}

        revenue = _extract_value(income, "revenues")
        net_income = _extract_value(income, "net_income_loss")
        equity = _extract_value(balance, "equity")
        total_liabilities = _extract_value(balance, "liabilities")
        total_assets = _extract_value(balance, "assets")

        return FinancialMetrics(
            ticker=ticker,
            period=getattr(fin, "timeframe", "unknown"),
            fiscal_period=getattr(fin, "fiscal_period", "unknown"),
            filing_date=_parse_date(getattr(fin, "filing_date", None)),
            revenue=revenue,
            net_income=net_income,
            earnings_per_share=_extract_value(income, "basic_earnings_per_share"),
            total_assets=total_assets,
            total_liabilities=total_liabilities,
            shareholders_equity=equity,
            operating_cash_flow=_extract_value(cash_flow, "net_cash_flow_from_operating_activities"),
            free_cash_flow=None,
            gross_profit_margin=None,
            net_profit_margin=_safe_divide(net_income, revenue),
            return_on_equity=_safe_divide(net_income, equity),
            debt_to_equity=_safe_divide(total_liabilities, equity),
            current_ratio=None,
        )
    except Exception as e:
        logger.warning(f"Failed to parse financial for {ticker}: {e}")
        return None


# ── News ────────────────────────────────────────────────────────────


def get_company_news(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 10,
) -> list[CompanyNews]:
    """Fetch news articles for a ticker."""
    cache = get_cache()
    cached = cache.get("news", ticker, str(start_date), str(end_date), str(limit))
    if cached is not None:
        return cached

    client = _get_client()
    logger.debug(f"Fetching news for {ticker}")

    kwargs: dict = {"ticker": ticker, "limit": limit, "order": "desc"}
    if start_date:
        kwargs["published_utc_gte"] = start_date
    if end_date:
        kwargs["published_utc_lte"] = end_date

    articles = []
    try:
        news_results = _throttled_call(lambda: list(client.list_ticker_news(**kwargs)))
        for n in news_results:
            articles.append(CompanyNews(
                title=n.title,
                author=getattr(n, "author", None),
                published_utc=n.published_utc,
                article_url=n.article_url,
                description=getattr(n, "description", None),
                tickers=getattr(n, "tickers", []),
            ))
            if len(articles) >= limit:
                break
    except Exception as e:
        logger.warning(f"Failed to fetch news for {ticker}: {e}")

    cache.set("news", ticker, str(start_date), str(end_date), str(limit), articles)
    logger.debug(f"Got {len(articles)} news articles for {ticker}")
    return articles


# ── Company Details ─────────────────────────────────────────────────


def get_company_details(ticker: str) -> Optional[CompanyDetails]:
    """Fetch ticker details (company info, market cap, etc.)."""
    cache = get_cache()
    cached = cache.get("details", ticker)
    if cached is not None:
        return cached

    client = _get_client()
    logger.debug(f"Fetching company details for {ticker}")

    try:
        details = _throttled_call(client.get_ticker_details, ticker)
        if details is None:
            return None

        result = CompanyDetails(
            ticker=ticker,
            name=details.name,
            market_cap=getattr(details, "market_cap", None),
            description=getattr(details, "description", None),
            sic_code=getattr(details, "sic_code", None),
            sic_description=getattr(details, "sic_description", None),
            homepage_url=getattr(details, "homepage_url", None),
            total_employees=getattr(details, "total_employees", None),
            list_date=getattr(details, "list_date", None),
            share_class_shares_outstanding=getattr(details, "share_class_shares_outstanding", None),
            weighted_shares_outstanding=getattr(details, "weighted_shares_outstanding", None),
        )

        cache.set("details", ticker, result, ttl_minutes=120)
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch details for {ticker}: {e}")
        return None


# ── Prefetch ───────────────────────────────────────────────────


def prefetch_ticker_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> None:
    """Pre-fetch all data for tickers to warm the cache before agents run.

    This makes a small number of sequential API calls so that when 6+ agents
    run in parallel, they all hit cache instead of flooding the API.
    """
    for ticker in tickers:
        logger.debug(f"Prefetching data for {ticker}")
        try:
            get_prices(ticker, start_date, end_date)
        except Exception as e:
            logger.warning(f"Prefetch prices failed for {ticker}: {e}")
        try:
            get_financial_metrics(ticker, end_date=end_date, limit=_MAX_FINANCIAL_LIMIT)
        except Exception as e:
            logger.warning(f"Prefetch financials failed for {ticker}: {e}")
        try:
            get_company_details(ticker)
        except Exception as e:
            logger.warning(f"Prefetch details failed for {ticker}: {e}")
        try:
            get_company_news(ticker, end_date=end_date, limit=20)
        except Exception as e:
            logger.warning(f"Prefetch news failed for {ticker}: {e}")

    # Also prefetch SPY prices (used by macro_regime agent)
    try:
        get_prices("SPY", start_date, end_date)
    except Exception as e:
        logger.warning(f"Prefetch SPY prices failed: {e}")


# ── Helpers ─────────────────────────────────────────────────────────


def _extract_value(statement: dict, key: str) -> Optional[float]:
    """Extract a numeric value from a Polygon financials statement dict."""
    entry = statement.get(key)
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("value")
    if hasattr(entry, "value"):
        return getattr(entry, "value", None)
    try:
        return float(entry)
    except (ValueError, TypeError):
        return None


def _safe_divide(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    """Safe division returning None if inputs are missing or denominator is zero."""
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _parse_date(val: Optional[str]) -> Optional[date]:
    """Parse a date string, returning None on failure."""
    if val is None:
        return None
    try:
        return date.fromisoformat(val)
    except (ValueError, TypeError):
        return None
