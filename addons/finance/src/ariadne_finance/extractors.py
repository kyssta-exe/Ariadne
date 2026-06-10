"""Document extractors for finance research files."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from arriadne.addons import ExtractionError, ExtractorBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ticker recognition
# ---------------------------------------------------------------------------

# Matches $TSLA, $AAPL or bare tickers like AAPL, TSLA (2-5 uppercase letters)
# in typical financial text contexts
_TICKER_RE = re.compile(
    r"""
    (?:
        \$([A-Z]{1,5})           |  # $TICKER format
        \b([A-Z]{2,5})\b           # bare TICKER (must be 2-5 chars)
    )
    """,
    re.VERBOSE,
)

# Words that look like tickers but aren't
_FALSE_POSITIVES = frozenset({
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN",
    "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS",
    "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET",
    "LET", "SAY", "SHE", "TOO", "USE", "DFS", "API", "PDF", "CSV",
    "GDP", "CEO", "CFO", "CTO", "IPO", "SEC", "EPS", "FYI", "FAQ",
    "USA", "UK", "EUR", "GBP", "USD", "JPY", "CHF", "AUD", "CAD",
    "NYSE", "FAQ", "HDL", "LLC", "INC", "LTD", "Corp", "Ltd",
})


def recognize_tickers(text: str) -> list[dict[str, Any]]:
    """Extract stock ticker symbols from text.

    Args:
        text: Input text to scan.

    Returns:
        List of dicts with 'value', 'start', 'end', 'confidence' keys.
    """
    entities = []
    seen = set()

    for match in _TICKER_RE.finditer(text):
        ticker = match.group(1) or match.group(2)
        if ticker and ticker not in _FALSE_POSITIVES and ticker not in seen:
            seen.add(ticker)
            entities.append({
                "type": "ticker",
                "value": ticker,
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9 if match.group(1) else 0.7,
            })

    return entities


# ---------------------------------------------------------------------------
# Sector classification
# ---------------------------------------------------------------------------

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Technology": [
        "software", "semiconductor", "cloud", "saas", "ai", "machine learning",
        "data center", "cybersecurity", "fintech", "e-commerce", "internet",
        "chip", "processor", "gpu", "cpu", "digital",
    ],
    "Healthcare": [
        "pharmaceutical", "biotech", "medical device", "health insurance",
        "clinical trial", "drug", "therapy", "diagnostic", "hospital",
        "biopharma", "genomics", "vaccine",
    ],
    "Financial Services": [
        "bank", "insurance", "asset management", "hedge fund", "broker",
        "credit", "lending", "mortgage", "payment", "wealth management",
        "capital market", "investment bank",
    ],
    "Consumer Discretionary": [
        "retail", "restaurant", "hotel", "entertainment", "automotive",
        "luxury", "apparel", "gaming", "media", "streaming",
    ],
    "Consumer Staples": [
        "food", "beverage", "household", "personal care", "tobacco",
        "grocery", "consumer goods",
    ],
    "Energy": [
        "oil", "gas", "petroleum", "renewable", "solar", "wind",
        "pipeline", "refinery", "lng", "drilling", "exploration",
    ],
    "Industrials": [
        "aerospace", "defense", "construction", "manufacturing",
        "transportation", "logistics", "railroad", "airline", "shipping",
    ],
    "Communication Services": [
        "telecom", "broadcast", "publishing", "gaming", "social media",
        "advertising", "content",
    ],
    "Utilities": [
        "electric", "water", "gas utility", "power grid", "nuclear",
        "renewable energy",
    ],
    "Real Estate": [
        "reit", "commercial real estate", "residential", "property",
        "mortgage", "real estate investment",
    ],
    "Materials": [
        "mining", "chemical", "steel", "aluminum", "lumber",
        "packaging", "paper",
    ],
}


def classify_sector(text: str) -> dict[str, float]:
    """Classify text into GICS sectors based on keyword matching.

    Args:
        text: Input text to classify.

    Returns:
        Dict of sector → confidence score (0.0-1.0), sorted by score desc.
    """
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for sector, keywords in SECTOR_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text_lower)
        if hits > 0:
            scores[sector] = min(1.0, hits / 3.0)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))


# ---------------------------------------------------------------------------
# PDF Extractor
# ---------------------------------------------------------------------------

class PDFExtractor(ExtractorBase):
    """Extract text and tables from PDF financial documents.

    Uses marker-pdf for text extraction and pdfplumber for table extraction.
    Both are optional dependencies — install with ``pip install ariadne-finance[pdf]``.
    """

    @property
    def name(self) -> str:
        return "finance-pdf"

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content_parts: list[str] = []
        tables: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {"source": str(file_path), "type": "pdf"}

        # Extract text with marker-pdf
        try:
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(str(file_path))
            content_parts.append(md_text)
            metadata["extractor"] = "marker-pdf"
        except ImportError:
            try:
                import pymupdf
                doc = pymupdf.open(str(file_path))
                for page in doc:
                    content_parts.append(page.get_text())
                doc.close()
                metadata["extractor"] = "pymupdf"
            except ImportError:
                raise ExtractionError(
                    "No PDF library available. Install marker-pdf or pymupdf: "
                    "pip install 'ariadne-finance[pdf]'"
                )

        # Extract tables with pdfplumber
        try:
            import pdfplumber
            with pdfplumber.open(str(file_path)) as pdf:
                for i, page in enumerate(pdf.pages):
                    for table in page.extract_tables():
                        if table:
                            tables.append({
                                "page": i + 1,
                                "rows": len(table),
                                "data": table,
                            })
            metadata["tables_found"] = len(tables)
        except ImportError:
            logger.debug("pdfplumber not available, skipping table extraction")

        content = "\n\n".join(content_parts)

        # Extract entities
        entities = recognize_tickers(content)
        sectors = classify_sector(content)
        if sectors:
            top_sector = list(sectors.keys())[0]
            entities.append({
                "type": "sector",
                "value": top_sector,
                "confidence": sectors[top_sector],
            })

        return {
            "content": content,
            "metadata": metadata,
            "entities": entities,
            "tables": tables,
        }


# ---------------------------------------------------------------------------
# Excel Extractor
# ---------------------------------------------------------------------------

class ExcelExtractor(ExtractorBase):
    """Extract data from Excel spreadsheets (.xlsx, .xls)."""

    @property
    def name(self) -> str:
        return "finance-excel"

    @property
    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls"]

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pandas as pd
        except ImportError:
            raise ExtractionError(
                "pandas is required for Excel extraction: pip install pandas"
            )

        try:
            # Read all sheets
            xls = pd.ExcelFile(str(file_path))
            sheets_data: dict[str, Any] = {}
            content_parts: list[str] = []

            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                sheets_data[sheet_name] = df.to_dict(orient="records")
                content_parts.append(
                    f"## Sheet: {sheet_name}\n{df.to_markdown(index=False)}"
                )

            content = "\n\n".join(content_parts)
            entities = recognize_tickers(content)
            sectors = classify_sector(content)
            if sectors:
                top_sector = list(sectors.keys())[0]
                entities.append({
                    "type": "sector",
                    "value": top_sector,
                    "confidence": sectors[top_sector],
                })

            return {
                "content": content,
                "metadata": {
                    "source": str(file_path),
                    "type": "excel",
                    "sheets": list(xls.sheet_names),
                    "extractor": "pandas",
                },
                "entities": entities,
                "tables": sheets_data,
            }
        except Exception as e:
            raise ExtractionError(f"Failed to extract Excel file: {e}") from e


# ---------------------------------------------------------------------------
# CSV Extractor
# ---------------------------------------------------------------------------

class CSVExtractor(ExtractorBase):
    """Extract data from CSV files."""

    @property
    def name(self) -> str:
        return "finance-csv"

    @property
    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]

    def extract(self, file_path: str | Path) -> dict[str, Any]:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            import pandas as pd
        except ImportError:
            raise ExtractionError(
                "pandas is required for CSV extraction: pip install pandas"
            )

        try:
            sep = "\t" if file_path.suffix == ".tsv" else ","
            df = pd.read_csv(str(file_path), sep=sep)
            content = df.to_markdown(index=False)
            entities = recognize_tickers(content)
            sectors = classify_sector(content)
            if sectors:
                top_sector = list(sectors.keys())[0]
                entities.append({
                    "type": "sector",
                    "value": top_sector,
                    "confidence": sectors[top_sector],
                })

            return {
                "content": content,
                "metadata": {
                    "source": str(file_path),
                    "type": "csv",
                    "columns": list(df.columns),
                    "rows": len(df),
                    "extractor": "pandas",
                },
                "entities": entities,
                "tables": {file_path.stem: df.to_dict(orient="records")},
            }
        except Exception as e:
            raise ExtractionError(f"Failed to extract CSV file: {e}") from e
