# ariadne-finance

Finance research add-on for [Ariadne](https://github.com/kyssta-exe/Ariadne) memory system.

## Features

- **PDF extraction** — marker-pdf for text, pdfplumber for tables
- **Excel/CSV extraction** — pandas-based with sheet detection
- **Ticker recognition** — regex-based with false positive filtering
- **Sector classification** — GICS sector keyword matching
- **Knowledge graph** — company → sector → ticker relationships
- **CLI tools** — `ariadne finance ingest/search/tickers`
- **API endpoints** — `/api/finance/ingest`, `/api/finance/tickers`, `/api/finance/classify`

## Installation

```bash
# Core (Excel + CSV only)
pip install ariadne-finance

# With PDF support
pip install "ariadne-finance[pdf]"

# Full (PDF + yfinance)
pip install "ariadne-finance[full]"
```

## Usage

### CLI

```bash
# Ingest a financial report
ariadne finance ingest report.pdf --importance 0.8

# Search finance memories
ariadne finance search "NVDA revenue Q3"

# Extract tickers from a file
ariadne finance tickers report.txt
```

### Python API

```python
from arriadne_finance.extractors import PDFExtractor, recognize_tickers

# Extract from PDF
extractor = PDFExtractor()
result = extractor.extract("report.pdf")
print(result["content"][:200])
print(result["entities"])  # tickers, sectors

# Recognize tickers in text
tickers = recognize_tickers("AAPL reported strong EPS, beating $MSFT")
```

### Dashboard API

```bash
# Ingest a document
curl -X POST http://localhost:8765/api/finance/ingest \
  -F "file=@report.pdf" -F "importance=0.8"

# Extract tickers
curl "http://localhost:8765/api/finance/tickers?text=AAPL+beat+MSFT+estimates"

# Classify sector
curl "http://localhost:8765/api/finance/classify?text=semiconductor+chip+manufacturing"
```

## License

MIT
