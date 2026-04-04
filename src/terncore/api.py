"""
terncore.api — Cubey REST API definition.

Maps cube addresses to HTTP endpoints.
finance.banking.transfer → POST /cube/finance/banking/transfer

This module defines the API contract. The transport layer (Flask, FastAPI, etc.)
is configured by the host application.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from terncore.confidence import RoutingConfidence
from terncore.cube import (
    CubeAction,
    CubeyClient,
    Guardian,
    GuardianVerdict,
    CUBE_ADDRESS_SPACE,
    validate_address,
)
from terncore.analytics import analyze, AnalyticsWindow


# MARK: - API Request / Response


@dataclass
class CubeAPIRequest:
    """Inbound API request — maps to CubeAction."""
    address: str            # "finance.banking"
    action: str             # "transfer"
    params: dict[str, Any]  # action parameters
    confidence: str         # "sure" / "unsure" / "unknown"
    weight: float
    tenant_id: str
    user_id: str

    def to_confidence(self) -> RoutingConfidence:
        return {
            "sure": RoutingConfidence.SURE,
            "unsure": RoutingConfidence.UNSURE,
            "unknown": RoutingConfidence.UNKNOWN,
        }.get(self.confidence, RoutingConfidence.UNKNOWN)


@dataclass
class CubeAPIResponse:
    """Outbound API response — serialisable."""
    action_id: str
    verdict: str            # "execute" / "gate" / "rollback" / "anomaly"
    reason: str
    confidence: str         # "sure" / "unsure" / "unknown"
    weight: float
    can_execute: bool

    @staticmethod
    def from_verdict(v: GuardianVerdict) -> CubeAPIResponse:
        return CubeAPIResponse(
            action_id=v.action_id,
            verdict=v.verdict,
            reason=v.reason,
            confidence={1: "sure", 0: "unsure", -1: "unknown"}[v.confidence.value],
            weight=v.weight,
            can_execute=v.can_execute,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "confidence": self.confidence,
            "weight": self.weight,
            "can_execute": self.can_execute,
        }


# MARK: - API Router


@dataclass
class APIRoute:
    method: str     # "POST" / "GET"
    path: str       # "/cube/finance/banking/transfer"
    address: str    # "finance.banking"
    action: str     # "transfer"


class CubeAPIRouter:
    """
    Maps cube addresses to HTTP routes.

    Usage:
        router = CubeAPIRouter(client)
        response = router.handle_request(request)

    Route generation:
        routes = router.generate_routes()
        # [APIRoute(POST, /cube/finance/banking, finance.banking, execute), ...]
    """

    def __init__(self, client: CubeyClient):
        self.client = client

    def handle_request(self, request: CubeAPIRequest) -> CubeAPIResponse:
        """Handle an inbound API request through Guardian."""
        verdict = self.client.execute(
            address=request.address,
            action=request.action,
            params=request.params,
            confidence=request.to_confidence(),
            weight=request.weight,
        )
        return CubeAPIResponse.from_verdict(verdict)

    def generate_routes(self) -> list[APIRoute]:
        """Generate all possible API routes from the address space."""
        routes = []
        for domain, functions in CUBE_ADDRESS_SPACE.items():
            for fn in functions:
                routes.append(APIRoute(
                    method="POST",
                    path=f"/cube/{domain}/{fn}",
                    address=f"{domain}.{fn}",
                    action="execute",
                ))
        return routes

    def openapi_paths(self) -> dict[str, Any]:
        """Generate OpenAPI paths object."""
        paths = {}
        for route in self.generate_routes():
            paths[route.path] = {
                "post": {
                    "summary": f"Execute action on {route.address}",
                    "operationId": f"cube_{route.address.replace('.', '_')}",
                    "tags": [route.address.split(".")[0]],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CubeAPIRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Guardian verdict",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/CubeAPIResponse"}
                                }
                            }
                        }
                    }
                }
            }
        return paths

    def openapi_spec(self) -> dict[str, Any]:
        """Generate complete OpenAPI 3.0 specification."""
        return {
            "openapi": "3.0.3",
            "info": {
                "title": "Cubey Agent³ eOS API",
                "description": "CubeAction address protocol — 63 cells, 7 domains, ternary confidence",
                "version": "0.4.0",
                "contact": {"name": "Gamma Seeds Pte Ltd"},
            },
            "paths": self.openapi_paths(),
            "components": {
                "schemas": {
                    "CubeAPIRequest": {
                        "type": "object",
                        "required": ["address", "action", "params", "confidence", "weight"],
                        "properties": {
                            "address": {"type": "string", "example": "finance.banking"},
                            "action": {"type": "string", "example": "transfer"},
                            "params": {"type": "object"},
                            "confidence": {"type": "string", "enum": ["sure", "unsure", "unknown"]},
                            "weight": {"type": "number", "minimum": -1, "maximum": 1},
                            "tenant_id": {"type": "string"},
                            "user_id": {"type": "string"},
                        }
                    },
                    "CubeAPIResponse": {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "string"},
                            "verdict": {"type": "string", "enum": ["execute", "gate", "rollback", "anomaly"]},
                            "reason": {"type": "string"},
                            "confidence": {"type": "string", "enum": ["sure", "unsure", "unknown"]},
                            "weight": {"type": "number"},
                            "can_execute": {"type": "boolean"},
                        }
                    }
                }
            }
        }
