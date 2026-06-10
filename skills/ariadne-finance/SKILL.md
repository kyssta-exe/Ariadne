---
name: ariadne-finance
description: "Finance research add-on for Ariadne — PDF/Excel extraction, ticker recognition, financial knowledge graph."
version: 0.10.0
author: kyssta
license: MIT
category: productivity
metadata:
  hermes:
    tags: [finance, research, pdf, tickers, knowledge-graph, ariadne-addon]
    requires: [ariadne-memory, ariadne-finance]
---

# Ariadne Finance — Finance Research Add-on

Finance research tools for the Ariadne memory system. Extracts financial data from PDFs, Excel, and CSV files, recognizes stock tickers, classifies market sectors, and builds a financial knowledge graph.

## Installation

```bash
pip install ariadne-finance           # Excel + CSV
pip install "ariadne-finance[pdf]"    # + PDF support
pip install "ariadne-finance[full]"   # + yfinance market data
```

## Tools

### Document Ingestion

| Tool | Purpose | Key Params |
|------|---------|------------|
| `ariadne_finance_ingest` | Ingest a financial document (PDF/Excel/CSV) into Ariadne | `file_path`, `importance` (0-1) |
| `ariadne_finance_search` | Search finance memories | `query`, `limit`, `ticker`, `sector` |
| `ariadne_finance_tickers` | Extract tickers from text or file | `text` or `file_path` |

### Entity Recognition

| Tool | Purpose | Key Params |
|------|---------|------------|
| `ariadne_finance_classify` | Classify text into GICS market sectors | `text` |
| `ariadne_finance_graph` | Query financial knowledge graph | `entity`, `hops` |

## CLI Usage

```bash
# Ingest a financial report
ariadne finance ingest report.pdf --importance 0.8

# Search finance memories
ariadne finance search "NVDA revenue Q3"

# Extract tickers from a file
ariadne finance tickers report.txt
```

## Dashboard API

```bash
# Ingest a document
curl -X POST http://localhost:8765/api/finance/ingest \
  -F "file=@report.pdf" -F "importance=0.8"

# Extract tickers from text
curl "http://localhost:8765/api/finance/tickers?text=AAPL+beat+MSFT+estimates"

# Classify sector
curl "http://localhost:8765/api/finance/classify?text=semiconductor+chip+manufacturing"
```

## Python API

```python
from arriadne_finance.extractors import PDFExtractor, recognize_tickers, classify_sector

# Extract from PDF
extractor = PDFExtractor()
result = extractor.extract("report.pdf")
print(result["content"][:200])      # extracted text
print(result["entities"])           # tickers, sectors found
print(result["tables"])             # extracted tables

# Recognize tickers in text
tickers = recognize_tickers("AAPL reported strong EPS, beating $MSFT")
# [{'type': 'ticker', 'value': 'AAPL', 'confidence': 0.7}, ...]

# Classify into sectors
sectors = classify_sector("semiconductor chip manufacturing GPU processor")
# {'Technology': 0.67, ...}
```

## Workflows

### Ingest and Search Financial Reports

```bash
# 1. Ingest reports
ariadne finance ingest q3-report.pdf --importance 0.8
ariadne finance ingest competitor-analysis.xlsx --importance 0.7

# 2. Search across all reports
ariadne finance search "revenue growth YoY"
ariadne finance search "NVDA" --k 5
```

### Build Financial Knowledge Graph

```python
from arriadne import AriadneMemory, AriadneConfig

config = AriadneConfig(db_path="finance.db")
mem = AriadneMemory(config=config)

# After ingesting reports, link entities
mem.add_edge("NVDA", "Technology", "belongs_to_sector", weight=1.0)
mem.add_edge("NVDA", "AMD", "competes_with", weight=0.8)
mem.add_edge("AAPL", "Technology", "belongs_to_sector", weight=1.0)

# Query the graph
result = mem.graph("NVDA", hops=2)
```

## Entity Types

| Entity | Description | Attributes |
|--------|-------------|------------|
| `ticker` | Stock ticker symbol | exchange, company_name |
| `company` | Company or corporation | sector, industry, country |
| `sector` | GICS market sector | — |
| `financial_metric` | Financial KPI | value, period |
| `earnings_report` | Quarterly/annual report | quarter, year |

## Tips

- Use `importance=0.8+` for key reports, `0.5` for routine data
- The addon auto-discovers installed extractors — PDF requires optional deps
- Ticker recognition uses regex with false-positive filtering for common words
- Sector classification is keyword-based — works best with financial terminology
