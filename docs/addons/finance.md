---
title: "Finance Addon — Ariadne"
description: "Track financial records, portfolios, and transactions in Ariadne. CLI, dashboard, API, and Hermes integration."
---

# Finance Addon

The **finance addon** extends Ariadne with financial memory: portfolio tracking, transaction history, balance snapshots, and investment context. It stores structured financial data alongside your existing memories and provides tools for CLI, dashboard, API, and Hermes agent usage.

## Installation

```bash
pip install ariadne-finance
```

Or with Ariadne extras:

```bash
pip install "ariadne[finance]"
```

::: tip
The finance addon requires Ariadne ≥ 0.10.0. Install Ariadne first if you haven't already.
:::

### Verify

```bash
ariadne finance --help
```

You should see:

```
usage: ariadne finance [-h] {import-csv,portfolio,balance,transactions,search} ...

Finance addon for Ariadne memory system.

commands:
  import-csv   Import transactions from a CSV file
  portfolio    Show current portfolio snapshot
  balance      Show balance history
  transactions Search transaction history
  search       Search financial memories
```

## Data Model

The finance addon adds three tables to the Ariadne database:

### `finance_transactions`

```sql
CREATE TABLE finance_transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,              -- ISO 8601 date (YYYY-MM-DD)
    description TEXT NOT NULL,
    amount REAL NOT NULL,            -- Positive = credit, negative = debit
    currency TEXT NOT NULL DEFAULT 'USD',
    category TEXT,                   -- User-defined category
    account TEXT,                    -- Account name (e.g. "checking", "investment")
    tags TEXT,                       -- JSON array of tags
    created_at REAL NOT NULL
);
```

### `finance_portfolios`

```sql
CREATE TABLE finance_portfolios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,     -- ISO 8601 date
    asset TEXT NOT NULL,             -- e.g. "AAPL", "BTC", "cash"
    quantity REAL NOT NULL,
    price_per_unit REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    total_value REAL NOT NULL,       -- quantity * price_per_unit
    account TEXT,
    created_at REAL NOT NULL
);
```

### `finance_balances`

```sql
CREATE TABLE finance_balances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    account TEXT NOT NULL,
    balance REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'USD',
    notes TEXT,
    created_at REAL NOT NULL
);
```

## CLI Usage

### Import Transactions

```bash
# Import from CSV (auto-detects columns)
ariadne finance import-csv transactions.csv

# With explicit column mapping
ariadne finance import-csv transactions.csv \
    --date-col "Transaction Date" \
    --amount-col "Amount" \
    --description-col "Description" \
    --account "checking"
```

**CSV format** (auto-detected columns):

```csv
Date,Description,Amount,Category
2024-01-15,Amazon Purchase,-89.99,Shopping
2024-01-15,Direct Deposit,3200.00,Income
2024-01-16,Electric Bill,-120.50,Utilities
```

### Portfolio Snapshot

```bash
# Show current portfolio
ariadne finance portfolio

# Add an asset
ariadne finance portfolio add \
    --asset AAPL \
    --quantity 10 \
    --price 185.50 \
    --account investment

# Show portfolio as of a specific date
ariadne finance portfolio --date 2024-06-01
```

### Balance Tracking

```bash
# Record a balance
ariadne finance balance set \
    --account checking \
    --balance 5420.30 \
    --date 2024-01-15

# Show balance history
ariadne finance balance --account checking --last 30d
```

### Search Transactions

```bash
# Search by description
ariadne finance search "electric"

# Search by category
ariadne finance search --category shopping

# Search by date range
ariadne finance search --from 2024-01-01 --to 2024-01-31

# Search by amount range
ariadne finance search --min-amount 100 --max-amount 500
```

## Python API

```python
from arriadne import AriadneMemory
from ariadne_finance import FinanceAddon

# Initialize core memory
mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Initialize the finance addon
finance = FinanceAddon(memory=mem)

# ── Import transactions ─────────────────────────────────
finance.import_transactions([
    {"date": "2024-01-15", "description": "Amazon Purchase", "amount": -89.99, "category": "shopping"},
    {"date": "2024-01-15", "description": "Direct Deposit", "amount": 3200.00, "category": "income"},
])

# ── Record portfolio snapshot ────────────────────────────
finance.set_portfolio([
    {"asset": "AAPL", "quantity": 10, "price_per_unit": 185.50, "account": "investment"},
    {"asset": "BTC", "quantity": 0.5, "price_per_unit": 42000.00, "account": "crypto"},
    {"asset": "cash", "quantity": 1, "price_per_unit": 5420.30, "account": "checking"},
])

# ── Search transactions ──────────────────────────────────
results = finance.search_transactions("shopping", k=5)
for t in results:
    print(f"{t['date']} | {t['description']}: ${t['amount']:.2f}")

# ── Get portfolio summary ────────────────────────────────
portfolio = finance.get_portfolio()
print(f"Total value: ${portfolio['total_value']:.2f}")
for asset in portfolio['assets']:
    print(f"  {asset['asset']}: ${asset['total_value']:.2f}")
```

## Dashboard

The finance addon adds a **Finance** tab to the Ariadne dashboard:

```
ariadne dashboard
# Open http://localhost:8000 → Finance tab
```

Dashboard features:

| Panel | Description |
|-------|-------------|
| **Portfolio Overview** | Pie chart of asset allocation, total value |
| **Transaction Timeline** | Bar chart of income vs expenses by month |
| **Balance History** | Line chart of account balances over time |
| **Transaction Search** | Full-text search across all transactions |
| **Category Breakdown** | Spending by category (donut chart) |

## Hermes Integration

The finance addon registers tools that Hermes agents can use automatically:

### Available Tools

| Tool | Description |
|------|-------------|
| `ariadne_finance_add_transaction` | Record a single transaction |
| `ariadne_finance_import_csv` | Import transactions from a CSV file path |
| `ariadne_finance_search_transactions` | Search transactions by description, category, date, or amount |
| `ariadne_finance_portfolio_add` | Add or update an asset in the portfolio |
| `ariadne_finance_portfolio_get` | Get current portfolio snapshot |
| `ariadne_finance_portfolio_history` | Get portfolio value over time |
| `ariadne_finance_balance_set` | Record an account balance |
| `ariadne_finance_balance_get` | Get balance history for an account |
| `ariadne_finance_summary` | Get a financial summary (income, expenses, net) |

### Tool Definitions

```json
{
  "name": "ariadne_finance_add_transaction",
  "description": "Record a financial transaction",
  "parameters": {
    "type": "object",
    "properties": {
      "date": {"type": "string", "description": "Transaction date (YYYY-MM-DD)"},
      "description": {"type": "string", "description": "Transaction description"},
      "amount": {"type": "number", "description": "Amount (positive=credit, negative=debit)"},
      "currency": {"type": "string", "default": "USD"},
      "category": {"type": "string", "description": "Transaction category"},
      "account": {"type": "string", "description": "Account name"},
      "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags"}
    },
    "required": ["date", "description", "amount"]
  }
}
```

### Agent Usage

Once the addon is installed, Hermes agents can use it in conversation:

> "Add a $45.50 transaction for groceries on January 15th"
>
> → Hermes calls `ariadne_finance_add_transaction`

> "What was my portfolio worth last month?"
>
> → Hermes calls `ariadne_finance_portfolio_history`

> "Show me all my shopping expenses this year"
>
> → Hermes calls `ariadne_finance_search_transactions` with `category: "shopping"` and date filter

## Examples

### Monthly Budget Tracking

```python
from arriadne import AriadneMemory
from ariadne_finance import FinanceAddon

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)
finance = FinanceAddon(memory=mem)

# Import January transactions
finance.import_transactions_from_csv("jan-2024.csv", account="checking")

# Get spending by category
summary = finance.get_summary(start_date="2024-01-01", end_date="2024-01-31")
print(f"Income:  ${summary['income']:.2f}")
print(f"Expenses: ${summary['expenses']:.2f}")
print(f"Net:      ${summary['net']:.2f}")
for cat, total in summary['by_category'].items():
    print(f"  {cat}: ${total:.2f}")
```

### Portfolio Drift Analysis

```python
# Record portfolio at two points in time
finance.set_portfolio([
    {"asset": "AAPL", "quantity": 10, "price_per_unit": 185.50},
    {"asset": "GOOGL", "quantity": 5, "price_per_unit": 140.00},
], snapshot_date="2024-01-01")

finance.set_portfolio([
    {"asset": "AAPL", "quantity": 10, "price_per_unit": 195.00},
    {"asset": "GOOGL", "quantity": 5, "price_per_unit": 135.00},
], snapshot_date="2024-06-01")

# Compare
history = finance.get_portfolio_history(start_date="2024-01-01", end_date="2024-06-01")
for snap in history:
    print(f"{snap['date']}: ${snap['total_value']:.2f}")
```

### Searching Financial Context

```bash
# Search for memories about financial decisions
ariadne finance search "quarterly rebalance"

# This searches both structured transactions AND
# natural-language memories stored by core Ariadne
# (e.g. "Decided to rebalance portfolio to 60/40 split on 2024-01-15")
```

## Configuration

Configure the finance addon via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARIADNE_FINANCE_DEFAULT_CURRENCY` | `USD` | Default currency for new transactions |
| `ARIADNE_FINANCE_DEFAULT_ACCOUNT` | `main` | Default account name |
| `ARIADNE_FINANCE_CSV_ENCODING` | `utf-8` | CSV file encoding |
| `ARIADNE_FINANCE_RETENTION_DAYS` | `365` | Days to keep transaction detail (older → summary only) |

Or in `~/.ariadne/config.yaml`:

```yaml
addons:
  finance:
    default_currency: USD
    default_account: main
    retention_days: 365
```

## Data Backup

The finance data lives in the same SQLite database as core Ariadne. Back up with:

```bash
ariadne backup --output finance-backup-$(date +%Y%m%d).db
```

Or programmatically:

```python
from arriadne import AriadneMemory
mem = AriadneMemory(db_path="memory.db")
mem.export_json()  # Includes finance tables
```

## Related

- [Addon System](./index) — How addons work
- [Installation](/guide/installation) — Install Ariadne
- [Configuration](/guide/configuration) — AriadneConfig reference
- [CLI Reference](/api/cli) — All CLI commands
