"""Financial entity recognition — tickers, companies, and sectors.

Provides regex-based ticker recognition, company name normalization,
and sector classification for financial documents.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any


# ---------------------------------------------------------------------------
# Ticker recognition patterns
# ---------------------------------------------------------------------------

# $TICKER style mentions (e.g. $AAPL, $MSFT, $TSLA)
DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{1,5})\b")

# Parenthetical ticker mentions (e.g. "Apple Inc. (AAPL)")
PAREN_TICKER_RE = re.compile(r"\(([A-Z]{1,5})\)")

# Exchange:tick patterns (e.g. "NASDAQ:AAPL", "NYSE:MSFT")
EXCHANGE_TICKER_RE = re.compile(
    r"(?:NASDAQ|NYSE|AMEX|ARCA|BATS|OTC):([A-Z]{1,5})\b",
    re.IGNORECASE,
)

# Ticker-only patterns: 1-5 uppercase letters preceded by whitespace/start
# and followed by whitespace/punctuation. Must NOT be a common English word.
_TICKER_ONLY_RE = re.compile(
    r"(?:^|(?<=\s))([A-Z]{2,5})(?=[.,;:!\?\s]|$)"
)

# Common English words that look like tickers but aren't
_TICKER_BLACKLIST: set[str] = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "MAN", "NEW", "NOW",
    "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "LET", "SAY", "SHE",
    "TOO", "USE", "DAD", "MOM", "CEO", "CFO", "COO", "CTO", "IPO", "ETF",
    "SEC", "EPS", "GDP", "APR", "APY", "IRA", "ROA", "ROE", "DCF", "NAV",
    "YES", "NOR", "BIT", "RUN", "END", "ADD", "TOP", "SET", "TRY", "JOB",
    "PAY", "BUY", "TAX", "GAP", "RED", "BIG", "BAD", "AGO", "AGE", "AGO",
    "FIT", "HIT", "SIT", "CUT", "PUT", "LOT", "HIT", "ITS", "USD", "EUR",
    "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "INR", "BRL", "KRW",
}


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Technology": [
        "software", "hardware", "semiconductor", "cloud", "saas", "ai",
        "artificial intelligence", "machine learning", "cybersecurity",
        "chip", "processor", "data center", "tech", "computing", "internet",
        "platform", "enterprise software", "devops", "blockchain", "crypto",
    ],
    "Healthcare": [
        "pharmaceutical", "biotech", "medical", "health", "drug",
        "clinical", "therapeutic", "diagnostic", "hospital", "insurance",
        "healthcare", "biotechnology", "genomics", "telemedicine",
    ],
    "Financial Services": [
        "bank", "banking", "investment", "insurance", "fintech",
        "lending", "credit", "mortgage", "wealth management", "broker",
        "exchange", "payment", "financial", "capital markets",
    ],
    "Consumer Discretionary": [
        "retail", "e-commerce", "restaurant", "hotel", "entertainment",
        "media", "streaming", "gaming", "luxury", "automotive",
        "apparel", "travel", "leisure", "consumer",
    ],
    "Consumer Staples": [
        "food", "beverage", "household", "personal care", "tobacco",
        "grocery", "packaged goods", "consumer products", "staples",
    ],
    "Energy": [
        "oil", "gas", "petroleum", "refinery", "pipeline", "solar",
        "wind", "renewable", "energy", "nuclear", "coal", "lng",
        "drilling", "exploration", "upstream", "downstream",
    ],
    "Industrials": [
        "manufacturing", "aerospace", "defense", "construction",
        "transportation", "logistics", "railroad", "aviation",
        "industrial", "engineering", "machinery", "infrastructure",
    ],
    "Real Estate": [
        "reit", "real estate", "property", "commercial real estate",
        "residential", "mortgage", "office", "retail space", "warehouse",
    ],
    "Utilities": [
        "electric", "utility", "power", "water", "gas utility",
        "regulated", "grid", "transmission", "distribution",
    ],
    "Materials": [
        "chemical", "steel", "mining", "metals", "lumber",
        "packaging", "paper", "construction materials", "minerals",
    ],
    "Communication Services": [
        "telecom", "media", "entertainment", "social media",
        "broadcasting", "advertising", "publishing", "gaming",
    ],
}


class TickerRecognizer:
    """Recognize and classify financial entities in text.

    Identifies stock tickers (in various formats), company names,
    and classifies documents/mentions by sector.
    """

    def __init__(self, extra_blacklist: set[str] | None = None) -> None:
        self._blacklist = _TICKER_BLACKLIST.copy()
        if extra_blacklist:
            self._blacklist.update(extra_blacklist)

    def recognize(self, text: str) -> list[str]:
        """Find all ticker symbols mentioned in text.

        Deduplicates and filters out common English words.

        Args:
            text: The text to search.

        Returns:
            Sorted list of unique ticker symbols (e.g. ['AAPL', 'MSFT']).
        """
        tickers: set[str] = set()

        for pattern in [DOLLAR_TICKER_RE, PAREN_TICKER_RE, EXCHANGE_TICKER_RE]:
            for match in pattern.finditer(text):
                ticker = match.group(1).upper()
                if ticker not in self._blacklist:
                    tickers.add(ticker)

        # Also check bare tickers (more aggressive — apply blacklist)
        for match in _TICKER_ONLY_RE.finditer(text):
            ticker = match.group(1).upper()
            if ticker not in self._blacklist:
                tickers.add(ticker)

        return sorted(tickers)

    def extract_companies(self, text: str) -> list[str]:
        """Extract likely company names from text.

        Looks for common corporate suffixes.

        Args:
            text: The text to search.

        Returns:
            List of unique company names found.
        """
        company_pattern = re.compile(
            r"([A-Z][\w\s&.,'-]{2,60}?)\s*"
            r"(?:Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|LLC|"
            r"PLC|Group|Holdings|Partners|Associates|SA|AG|NV|SE)\b"
        )
        companies: list[str] = []
        seen: set[str] = set()
        for match in company_pattern.finditer(text):
            name = match.group(0).strip()
            # Normalize whitespace
            name = re.sub(r"\s+", " ", name)
            if name.lower() not in seen:
                seen.add(name.lower())
                companies.append(name)
        return companies

    def classify_sector(self, text: str) -> str:
        """Classify the most likely sector for a text excerpt.

        Uses keyword matching against SECTOR_KEYWORDS.

        Args:
            text: The text to classify.

        Returns:
            Best-matching sector name, or "Unknown".
        """
        text_lower = text.lower()
        scores: dict[str, int] = {}

        for sector, keywords in SECTOR_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[sector] = score

        if not scores:
            return "Unknown"

        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def classify_sector_for_ticker(self, ticker: str) -> str:
        """Look up the sector for a ticker using yfinance.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Sector name, or "Unknown" if lookup fails.
        """
        try:
            import yfinance as yf

            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get("sector", "Unknown")
        except Exception:
            return "Unknown"


def normalize_company_name(name: str) -> str:
    """Normalize a company name for consistent matching.

    - Strips trailing corporate suffixes (Inc., Corp., etc.)
    - Normalizes unicode and whitespace
    - Lowercases

    Args:
        name: Raw company name.

    Returns:
        Normalized lowercase name without suffixes.
    """
    # Normalize unicode
    name = unicodedata.normalize("NFKD", name)

    # Strip common suffixes
    suffixes = [
        r",?\s*(?:Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|LLC|"
        r"PLC|Group|Holdings|Partners|Associates|SA|AG|NV|SE)\.?\s*$"
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)

    # Normalize whitespace
    name = re.sub(r"\s+", " ", name).strip()

    return name.lower()


def find_ticker_for_company(company_name: str, tickers_in_text: list[str]) -> str | None:
    """Try to match a company name to a ticker from the document.

    Uses a simple heuristic: if the first letters of major words in the
    company name match a ticker, return it.

    Args:
        company_name: The company name to match.
        tickers_in_text: List of tickers found in the same document.

    Returns:
        Best-matching ticker, or None.
    """
    # Build acronym from capitalized words
    words = company_name.split()
    if len(words) < 2:
        return None

    # Try first-letter acronym
    acronym = "".join(w[0].upper() for w in words if w and w[0].isalpha())
    if acronym in tickers_in_text:
        return acronym

    # Try first two letters + first letter of next word
    if len(words) >= 3:
        candidate = words[0][:2].upper() + words[1][0].upper()
        if candidate in tickers_in_text:
            return candidate

    return None
