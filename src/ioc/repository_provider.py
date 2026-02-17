"""
Repository provider for dependency injection.

This module provides all repository dependencies.
"""

from dishka import Provider, Scope, provide
from sqlalchemy.ext.asyncio import AsyncSession



class RepositoryProvider(Provider):
    """
    Provider for repository dependencies.

    All repositories are provided at REQUEST scope.
    """
