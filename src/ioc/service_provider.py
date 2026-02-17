"""
Service provider for dependency injection.

This module provides all service dependencies.
"""

from dishka import Provider, Scope, provide

from src.services.detection_service import DetectionService, GigaCheckService, RuBertService


class ServiceProvider(Provider):
    """
    Provider for service dependencies.

    All services are provided at APP scope (singleton).
    """

    @provide(scope=Scope.APP)
    def provide_gigacheck_service(self) -> GigaCheckService:
        return GigaCheckService()

    @provide(scope=Scope.APP)
    def provide_rubert_service(self) -> RuBertService:
        return RuBertService()

    @provide(scope=Scope.APP)
    def provide_detection_service(
        self,
        gigacheck: GigaCheckService,
        rubert: RuBertService,
    ) -> DetectionService:
        return DetectionService(gigacheck, rubert)
