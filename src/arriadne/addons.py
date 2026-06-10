"""Ariadne addon system — plugin registry with entry_points discovery.

Addons extend Ariadne with domain-specific extractors, entity types,
CLI commands, API routes, search filters, and graph relationships.

Each addon is a separate Python package that registers itself via the
``ariadne.addons`` entry_points group.  The :class:`AddonRegistry` scans
all installed entry points, instantiates the addon classes, and exposes
them through a unified interface.

Example entry_points (in pyproject.toml)::

    [project.entry-points."ariadne.addons"]
    legal = "ariadne_legal:LegalAddon"

Or in setup.cfg / setup.py::

    [options.entry_points]
    ariadne.addons =
        legal = ariadne_legal:LegalAddon
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class EntityType:
    """Domain-specific entity type definition.

    Attributes:
        name: Short identifier (e.g. ``"patent"``, ``"protein"``).
        display_name: Human-readable label (e.g. ``"Patent"``).
        description: What this entity type represents.
        attributes: Schema for entity-specific attributes (keys → types).
        searchable: Whether this entity type participates in search.
    """

    name: str
    display_name: str
    description: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    searchable: bool = True


@dataclass(frozen=True, slots=True)
class CLICommand:
    """CLI command provided by an addon.

    Attributes:
        name: Subcommand name (e.g. ``"legal-search"``).
        help_text: Help string shown in ``--help``.
        handler: Callable that executes the command.  Receives the parsed
            ``argparse.Namespace`` as its only argument.
    """

    name: str
    help_text: str
    handler: Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class APIRoute:
    """API route provided by an addon.

    Attributes:
        path: URL path prefix (e.g. ``"/api/legal"``).
        router: An ``APIRouter`` (Starlette/FastAPI compatible) instance
            that the host application should mount.
        prefix: Route prefix for the router.
        tags: OpenAPI tags for documentation grouping.
    """

    path: str
    router: Any
    prefix: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SearchFilter:
    """Custom search filter contributed by an addon.

    Attributes:
        name: Filter identifier (e.g. ``"jurisdiction"``).
        display_name: Human-readable label.
        description: What the filter does.
        filter_type: Value type (``"string"``, ``"number"``, ``"boolean"``,
            ``"date"``, or a custom domain type).
        default: Default value, or ``None`` for no default.
    """

    name: str
    display_name: str
    description: str = ""
    filter_type: str = "string"
    default: Any = None


@dataclass(frozen=True, slots=True)
class GraphRelationship:
    """Knowledge-graph relationship type contributed by an addon.

    Attributes:
        name: Edge type (e.g. ``"cites"``).
        description: Human-readable description.
        bidirectional: If ``True`` the relationship is symmetric.
    """

    name: str
    description: str = ""
    bidirectional: bool = False


# ---------------------------------------------------------------------------
# ABCs
# ---------------------------------------------------------------------------

class ExtractorBase(ABC):
    """Abstract base class for document extractors.

    An extractor reads a file from disk and returns structured content,
    metadata, and recognised entities.  Each addon can contribute one or
    more extractors via :meth:`BaseAddon.get_extractors`.

    Subclasses must implement :meth:`extract`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the extractor (e.g. ``"pdf"``)."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this extractor handles (e.g. ``[".pdf"]``)."""
        ...

    @abstractmethod
    def extract(self, file_path: str | Path) -> dict[str, Any]:
        """Extract structured data from a document.

        Args:
            file_path: Path to the document to extract.

        Returns:
            Dictionary with at least these keys:

            - ``"content"`` — extracted text content (``str``).
            - ``"metadata"`` — document metadata (``dict[str, Any]``).
            - ``"entities"`` — list of extracted entity dicts
              (``list[dict[str, Any]]``), each optionally containing:
              ``"type"``, ``"value"``, ``"start"``, ``"end"``,
              ``"confidence"``.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ExtractionError: On parse or format errors.
        """
        ...

    def can_handle(self, file_path: str | Path) -> bool:
        """Return ``True`` if this extractor handles the given file type."""
        ext = Path(file_path).suffix.lower()
        return ext in self.supported_extensions


class ExtractionError(Exception):
    """Raised when an extractor fails to process a document."""


class BaseAddon(ABC):
    """Abstract base class for Ariadne addons.

    Every addon must subclass ``BaseAddon`` and implement at least the
    required properties (:meth:`name`, :meth:`version`, :meth:`description`).
    All ``get_*`` methods have sensible no-op defaults so that minimal
    addons only need to override the hooks they actually use.

    Subclasses are instantiated by :class:`AddonRegistry` when the entry
    point is loaded.  The registry calls :meth:`initialize` after
    construction and :meth:`shutdown` on teardown.
    """

    # -- Required properties --------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique addon identifier (e.g. ``"ariadne-legal"``)."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version string (e.g. ``"1.2.0"``)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line human-readable description."""
        ...

    # -- Hook methods (override as needed) -----------------------------------

    def get_extractors(self) -> list[ExtractorBase]:
        """Return extractors contributed by this addon."""
        return []

    def get_entity_types(self) -> list[EntityType]:
        """Return domain-specific entity types."""
        return []

    def get_cli_commands(self) -> list[CLICommand]:
        """Return CLI subcommands contributed by this addon."""
        return []

    def get_api_routes(self) -> list[APIRoute]:
        """Return API routes contributed by this addon."""
        return []

    def get_search_filters(self) -> list[SearchFilter]:
        """Return custom search filters."""
        return []

    def get_graph_relationships(self) -> list[GraphRelationship]:
        """Return knowledge-graph relationship types."""
        return []

    # -- Lifecycle ------------------------------------------------------------

    def initialize(self, config: Any = None) -> None:
        """Called once after instantiation.  *config* is the active
        :class:`~arriadne.config.AriadneConfig` (or ``None``).

        Override this to perform setup that depends on the host config
        (opening connections, loading models, etc.).
        """

    def shutdown(self) -> None:
        """Called when the host is tearing down.

        Override this to release resources (close connections, flush
        caches, etc.).
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class AddonRegistry:
    """Discovers, loads, and manages Ariadne addons.

    Addons are discovered via the ``ariadne.addons`` entry_points group.
    Each entry point must reference a callable (typically a class) that
    returns a :class:`BaseAddon` instance when called with no arguments.

    The registry provides aggregated views of all registered addons'
    extractors, entity types, CLI commands, etc.  Call :meth:`initialize`
    to instantiate and activate all addons, and :meth:`shutdown` to tear
    them down.

    Example::

        registry = AddonRegistry()
        registry.discover()
        registry.initialize(config)
        extractors = registry.get_all_extractors()
        registry.shutdown()

    You can also register addons manually for testing::

        registry.register(MyTestAddon())
    """

    def __init__(self) -> None:
        self._addons: dict[str, BaseAddon] = {}
        self._entry_points: dict[str, Any] = {}
        self._initialized: bool = False

    # -- Discovery ------------------------------------------------------------

    def discover(self) -> list[str]:
        """Scan installed packages for ``ariadne.addons`` entry points.

        Each entry point is resolved to a callable (class or factory)
        and called with no arguments.  If it returns a
        :class:`BaseAddon` instance, the addon is registered.

        Returns:
            List of addon names that were successfully discovered.

        Raises:
            Does **not** raise on individual import failures — those are
            logged and skipped so that one broken addon cannot prevent
            others from loading.
        """
        found: list[str] = []

        try:
            # Python 3.12+ has importlib.metadata.entry_points(group=...)
            # Python 3.10/3.11 also support this via the backport behaviour.
            from importlib.metadata import entry_points
            eps = entry_points(group="ariadne.addons")
        except Exception:
            # Fallback for unusual environments where entry_points() may
            # behave differently (e.g. older setuptools).
            try:
                from importlib.metadata import entry_points
                all_eps = entry_points()
                eps = all_eps.get("ariadne.addons", [])  # type: ignore[arg-type]
            except Exception as exc:
                logger.error("Failed to load entry_points: %s", exc)
                return found

        for ep in eps:
            name = ep.name
            if name in self._addons:
                logger.warning(
                    "Addon %r already registered, skipping duplicate entry point",
                    name,
                )
                continue

            try:
                addon_cls = ep.load()
                addon = addon_cls()
                if not isinstance(addon, BaseAddon):
                    logger.warning(
                        "Entry point %r loaded %r which is not a BaseAddon subclass; "
                        "skipping.",
                        ep,
                        addon_cls,
                    )
                    continue
                self._addons[addon.name] = addon
                self._entry_points[addon.name] = ep
                found.append(addon.name)
                logger.info(
                    "Discovered addon %r (v%s) from entry point %r",
                    addon.name,
                    addon.version,
                    ep,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load addon from entry point %r: %s", ep, exc,
                )

        return found

    def register(self, addon: BaseAddon) -> None:
        """Manually register an addon instance (useful for testing).

        Args:
            addon: A :class:`BaseAddon` instance to register.

        Raises:
            ValueError: If an addon with the same name is already registered.
        """
        if addon.name in self._addons:
            raise ValueError(
                f"Addon {addon.name!r} is already registered"
            )
        self._addons[addon.name] = addon
        logger.info("Manually registered addon %r (v%s)", addon.name, addon.version)

    def unregister(self, name: str) -> bool:
        """Remove a registered addon by name.

        If the addon is initialized, :meth:`BaseAddon.shutdown` is called
        first.

        Returns:
            ``True`` if the addon was found and removed.
        """
        addon = self._addons.pop(name, None)
        if addon is None:
            return False
        if self._initialized:
            try:
                addon.shutdown()
            except Exception as exc:
                logger.warning(
                    "Error shutting down addon %r during unregister: %s",
                    name,
                    exc,
                )
        self._entry_points.pop(name, None)
        logger.info("Unregistered addon %r", name)
        return True

    # -- Lifecycle ------------------------------------------------------------

    def initialize(self, config: Any = None) -> None:
        """Instantiate and initialize all discovered addons.

        Args:
            config: Optional :class:`~arriadne.config.AriadneConfig` to
                pass through to each addon's :meth:`BaseAddon.initialize`.

        Raises:
            If an addon's :meth:`initialize` raises, the exception is
            logged and that addon is removed from the registry; other
            addons continue initializing.
        """
        failed: list[str] = []
        for name, addon in list(self._addons.items()):
            try:
                addon.initialize(config)
                logger.info("Initialized addon %r", name)
            except Exception as exc:
                logger.error("Failed to initialize addon %r: %s", name, exc)
                failed.append(name)

        for name in failed:
            del self._addons[name]
            self._entry_points.pop(name, None)

        self._initialized = True
        logger.info(
            "AddonRegistry initialized: %d addon(s) active",
            len(self._addons),
        )

    def shutdown(self) -> None:
        """Shut down all registered addons and clear the registry."""
        for name, addon in self._addons.items():
            try:
                addon.shutdown()
                logger.info("Shut down addon %r", name)
            except Exception as exc:
                logger.warning("Error shutting down addon %r: %s", name, exc)
        self._addons.clear()
        self._entry_points.clear()
        self._initialized = False
        logger.info("AddonRegistry shut down")

    # -- Accessors ------------------------------------------------------------

    @property
    def addon_names(self) -> list[str]:
        """Sorted list of registered addon names."""
        return sorted(self._addons.keys())

    @property
    def count(self) -> int:
        """Number of registered addons."""
        return len(self._addons)

    def get_addon(self, name: str) -> BaseAddon | None:
        """Retrieve a specific addon by name.

        Returns:
            The addon, or ``None`` if not found.
        """
        return self._addons.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._addons

    def __len__(self) -> int:
        return len(self._addons)

    def __iter__(self) -> Iterator[BaseAddon]:
        return iter(self._addons.values())

    # -- Aggregated views -----------------------------------------------------

    def get_all_extractors(self) -> list[ExtractorBase]:
        """Collect extractors from every registered addon."""
        extractors: list[ExtractorBase] = []
        for addon in self._addons.values():
            extractors.extend(addon.get_extractors())
        return extractors

    def get_extractor_for_file(self, file_path: str | Path) -> ExtractorBase | None:
        """Find the first extractor that can handle *file_path*.

        Searches addons in registration order and returns the first
        matching extractor, or ``None``.
        """
        for extractor in self.get_all_extractors():
            if extractor.can_handle(file_path):
                return extractor
        return None

    def get_all_entity_types(self) -> list[EntityType]:
        """Collect entity types from every registered addon."""
        types: list[EntityType] = []
        seen: set[str] = set()
        for addon in self._addons.values():
            for et in addon.get_entity_types():
                if et.name not in seen:
                    seen.add(et.name)
                    types.append(et)
        return types

    def get_all_cli_commands(self) -> list[CLICommand]:
        """Collect CLI commands from every registered addon."""
        commands: list[CLICommand] = []
        seen: set[str] = set()
        for addon in self._addons.values():
            for cmd in addon.get_cli_commands():
                if cmd.name not in seen:
                    seen.add(cmd.name)
                    commands.append(cmd)
        return commands

    def get_all_api_routes(self) -> list[APIRoute]:
        """Collect API routes from every registered addon."""
        routes: list[APIRoute] = []
        for addon in self._addons.values():
            routes.extend(addon.get_api_routes())
        return routes

    def get_all_search_filters(self) -> list[SearchFilter]:
        """Collect search filters from every registered addon."""
        filters: list[SearchFilter] = []
        seen: set[str] = set()
        for addon in self._addons.values():
            for f in addon.get_search_filters():
                if f.name not in seen:
                    seen.add(f.name)
                    filters.append(f)
        return filters

    def get_all_graph_relationships(self) -> list[GraphRelationship]:
        """Collect graph relationships from every registered addon."""
        rels: list[GraphRelationship] = []
        seen: set[str] = set()
        for addon in self._addons.values():
            for r in addon.get_graph_relationships():
                if r.name not in seen:
                    seen.add(r.name)
                    rels.append(r)
        return rels

    # -- Discovery helpers for file extraction -------------------------------

    def extract_document(self, file_path: str | Path) -> dict[str, Any]:
        """Extract structured data from a document using a registered extractor.

        Finds an appropriate extractor for *file_path* via
        :meth:`get_extractor_for_file` and calls its
        :meth:`ExtractorBase.extract` method.

        Args:
            file_path: Path to the document.

        Returns:
            The extraction result dict.

        Raises:
            FileNotFoundError: If *file_path* does not exist.
            ExtractionError: If no extractor can handle the file type,
                or the extractor fails.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No such file: {path}")

        extractor = self.get_extractor_for_file(path)
        if extractor is None:
            supported = set()
            for ext in self.get_all_extractors():
                supported.update(ext.supported_extensions)
            raise ExtractionError(
                f"No extractor found for {path.suffix!r} file. "
                f"Supported extensions: {sorted(supported) or '(none)'}"
            )

        logger.info(
            "Extracting %s with extractor %r", path, extractor.name
        )
        return extractor.extract(path)

    # -- Context manager support ---------------------------------------------

    def __enter__(self) -> AddonRegistry:
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.shutdown()

    # -- Repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        state = "initialized" if self._initialized else "discovered"
        return (
            f"AddonRegistry({state}, addons={self.addon_names!r})"
        )
