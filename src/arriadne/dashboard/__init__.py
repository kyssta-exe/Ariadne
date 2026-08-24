"""Ariadne Console — Browser-based management dashboard."""

__all__ = ["create_app"]


def __getattr__(name: str):
    # Lazy: importing the package does not require fastapi.
    if name == "create_app":
        from arriadne.dashboard.server import create_app
        return create_app
    raise AttributeError(name)
