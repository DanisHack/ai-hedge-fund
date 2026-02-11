"""Polygon.io API wrapper using polygon-api-client."""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from polygon import RESTClient

from src.config.settings import POLYGON_API_KEY, validate_polygon_key
from src.data.cache import get_cache
from src.data.models import CompanyDetails, CompanyNews, FinancialMetrics, Price

logger = logging.getLogger(__name__)

_client: Optional[RESTClient] = None


def _get_client() -> RESTClient:
    """Return or create the singleton Polygon RESTClient."""
    global _client
    if _client is None:
        validate_polygon_key()
        _client = RESTClient(api_key=POLYGON_API_KEY)
    return _client


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

    aggs = client.get_aggs(
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


def get_financial_metrics(
    ticker: str,
    end_date: Optional[str] = None,
    limit: int = 4,
) -> list[FinancialMetrics]:
    """Fetch stock financials from Polygon vX endpoint."""
    cache = get_cache()
    cache_key_date = end_date or "latest"
    cached = cache.get("financials", ticker, cache_key_date, str(limit))
    if cached is not None:
        return cached

    client = _get_client()
    logger.debug(f"Fetching financials for {ticker}")

    params: dict = {
        "ticker": ticker,
        "limit": limit,
        "order": "desc",
        "sort": "period_of_report_date",
    }
    if end_date:
        params["period_of_report_date_lte"] = end_date

    metrics = []
    try:
        for fin in client.vx.list_stock_financials(**params):
            m = _parse_financial(ticker, fin)
            if m is not None:
                metrics.append(m)
            if len(metrics) >= limit:
                break
    except Exception as e:
        logger.warning(f"Failed to fetch financials for {ticker}: {e}")

    cache.set("financials", ticker, cache_key_date, str(limit), metrics)
    logger.debug(f"Got {len(metrics)} financial periods for {ticker}")
    return metrics


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
        kwargs["published_utc.gte"] = start_date
    if end_date:
        kwargs["published_utc.lte"] = end_date

    articles = []
    try:
        for n in client.list_ticker_news(**kwargs):
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
        details = client.get_ticker_details(ticker)
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
