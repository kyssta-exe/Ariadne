"""Dashboard routes — serves the SPA shell and static assets."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# Path to static files
STATIC_DIR = Path(__file__).parent / "static"


def _read_html() -> str:
    """Read the SPA shell HTML."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Ariadne Console</h1><p>Dashboard files not found.</p>"


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_index():
    """Serve the dashboard SPA shell."""
    return HTMLResponse(_read_html())


# Mount static files at /dashboard/static/
# This must be done AFTER the route registration by the app factory.
