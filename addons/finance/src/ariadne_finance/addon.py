"""Ariadne Finance addon — main entry point."""

from __future__ import annotations

import logging
from typing import Any

from arriadne.addons import (
    APIRoute,
    BaseAddon,
    CLICommand,
    EntityType,
    ExtractorBase,
    GraphRelationship,
    SearchFilter,
)

logger = logging.getLogger(__name__)


class Addon(BaseAddon):
    """Finance research add-on for Ariadne.

    Provides PDF/Excel/CSV extraction, ticker recognition, sector
    classification, and financial knowledge graph relationships.
    """

    def __init__(self) -> None:
        self._extractors: list[ExtractorBase] = []
        self._initialized = False

    @property
    def name(self) -> str:
        return "ariadne-finance"

    @property
    def version(self) -> str:
        return "0.10.0"

    @property
    def description(self) -> str:
        return "Finance research — PDF/Excel extraction, ticker recognition, financial knowledge graph"

    def initialize(self, config: Any = None) -> None:
        """Initialize extractors based on available dependencies."""
        from arriadne_finance.extractors import (
            CSVExtractor,
            ExcelExtractor,
        )

        self._extractors = [ExcelExtractor(), CSVExtractor()]

        # Try to load PDF extractor (requires optional deps)
        try:
            from arriadne_finance.extractors import PDFExtractor
            self._extractors.insert(0, PDFExtractor())
            logger.info("Finance addon: PDF extraction enabled")
        except ImportError:
            logger.info(
                "Finance addon: PDF extraction unavailable "
                "(install with: pip install 'ariadne-finance[pdf]')"
            )

        self._initialized = True

    def shutdown(self) -> None:
        self._extractors.clear()
        self._initialized = False

    def get_extractors(self) -> list[ExtractorBase]:
        return self._extractors

    def get_entity_types(self) -> list[EntityType]:
        return [
            EntityType(
                name="ticker",
                display_name="Stock Ticker",
                description="Stock market ticker symbol (e.g. AAPL, TSLA)",
                attributes={"exchange": "str", "company_name": "str"},
            ),
            EntityType(
                name="company",
                display_name="Company",
                description="Company or corporation",
                attributes={"sector": "str", "industry": "str", "country": "str"},
            ),
            EntityType(
                name="sector",
                display_name="Market Sector",
                description="GICS market sector (e.g. Technology, Healthcare)",
            ),
            EntityType(
                name="financial_metric",
                display_name="Financial Metric",
                description="Financial metric or KPI (e.g. revenue, EPS, P/E ratio)",
                attributes={"value": "float", "period": "str"},
            ),
            EntityType(
                name="earnings_report",
                display_name="Earnings Report",
                description="Quarterly or annual earnings report",
                attributes={"quarter": "str", "year": "int"},
            ),
        ]

    def get_cli_commands(self) -> list[CLICommand]:
        from arriadne_finance.cli import cmd_finance
        return [
            CLICommand(
                name="finance",
                help_text="Finance research tools (ingest, search, tickers)",
                handler=cmd_finance,
            ),
        ]

    def get_api_routes(self) -> list[APIRoute]:
        try:
            from arriadne_finance.api import router
            return [
                APIRoute(
                    path="/api/finance",
                    router=router,
                    prefix="/api/finance",
                    tags=["finance"],
                ),
            ]
        except ImportError:
            return []

    def get_search_filters(self) -> list[SearchFilter]:
        return [
            SearchFilter(
                name="ticker",
                display_name="Ticker Symbol",
                description="Filter by stock ticker",
                filter_type="string",
            ),
            SearchFilter(
                name="sector",
                display_name="Sector",
                description="Filter by market sector",
                filter_type="string",
            ),
            SearchFilter(
                name="date_from",
                display_name="Date From",
                description="Start date (YYYY-MM-DD)",
                filter_type="date",
            ),
            SearchFilter(
                name="date_to",
                display_name="Date To",
                description="End date (YYYY-MM-DD)",
                filter_type="date",
            ),
        ]

    def get_graph_relationships(self) -> list[GraphRelationship]:
        return [
            GraphRelationship(
                name="belongs_to_sector",
                description="Company belongs to a market sector",
            ),
            GraphRelationship(
                name="reports_earnings",
                description="Company reports earnings for a period",
            ),
            GraphRelationship(
                name="mentions_ticker",
                description="Document mentions a stock ticker",
            ),
            GraphRelationship(
                name="competes_with",
                description="Company competes with another company",
                bidirectional=True,
            ),
            GraphRelationship(
                name="supplies",
                description="Company supplies another company",
            ),
        ]
