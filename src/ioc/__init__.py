"""
Dependency injection container configuration using Dishka.
"""

from src.ioc.service_provider import ServiceProvider


class AppProvider(ServiceProvider):
    """Main dependency injection provider for the application."""
    pass


__all__ = ["AppProvider"]
