"""
Service provider for dependency injection.

This module provides all service dependencies.
"""

from dishka import Provider, Scope, provide

from src.services.detection_service import DetectionService, GigaCheckService, KazBertService, RuBertService


class ServiceProvider(Provider):
    """
    Provider for service dependencies.

    All services are provided at APP scope (singleton).
    """

    @provide(scope=Scope.APP)
    def provide_rubert_service(self) -> RuBertService:
        return RuBertService()

    @provide(scope=Scope.APP)
    def provide_gigacheck_service(self) -> GigaCheckService:
        return GigaCheckService()

    @provide(scope=Scope.APP)
    def provide_kazbert_service(self) -> KazBertService:
        return KazBertService()

    @provide(scope=Scope.APP)
    def provide_detection_service(
        self,
        rubert: RuBertService,
        gigacheck: GigaCheckService,
        kazbert: KazBertService,
    ) -> DetectionService:
        return DetectionService(rubert, gigacheck, kazbert)
