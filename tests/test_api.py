"""Tests for Cubey REST API."""

import json
from terncore.confidence import RoutingConfidence
from terncore.cube import CubeyClient, Guardian, CUBE_ADDRESS_SPACE
from terncore.api import CubeAPIRequest, CubeAPIResponse, CubeAPIRouter


class TestCubeAPIRequest:
    def test_to_confidence(self):
        r = CubeAPIRequest(address="sales.invoices", action="create", params={},
                            confidence="sure", weight=0.9, tenant_id="t", user_id="u")
        assert r.to_confidence() == RoutingConfidence.SURE

    def test_unknown_default(self):
        r = CubeAPIRequest(address="sales.crm", action="r", params={},
                            confidence="invalid", weight=0.0, tenant_id="t", user_id="u")
        assert r.to_confidence() == RoutingConfidence.UNKNOWN


class TestCubeAPIResponse:
    def test_from_verdict(self):
        client = CubeyClient("t", "u")
        verdict = client.execute("sales.invoices", "create", {},
                                  RoutingConfidence.SURE, 0.92)
        response = CubeAPIResponse.from_verdict(verdict)
        assert response.verdict == "execute"
        assert response.can_execute is True
        assert response.confidence == "sure"

    def test_to_dict(self):
        client = CubeyClient("t", "u")
        verdict = client.execute("finance.bullion", "buy", {},
                                  RoutingConfidence.UNSURE, 0.5)
        response = CubeAPIResponse.from_verdict(verdict)
        d = response.to_dict()
        assert d["verdict"] == "gate"
        assert d["can_execute"] is False
        assert isinstance(d["weight"], float)


class TestCubeAPIRouter:
    def test_handle_request(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        req = CubeAPIRequest(address="sales.invoices", action="create", params={},
                              confidence="sure", weight=0.92, tenant_id="t", user_id="u")
        resp = router.handle_request(req)
        assert resp.can_execute is True

    def test_handle_unsure_gates(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        req = CubeAPIRequest(address="operations.inventory", action="delete", params={},
                              confidence="unsure", weight=0.5, tenant_id="t", user_id="u")
        resp = router.handle_request(req)
        assert resp.verdict == "gate"

    def test_generate_routes_count(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        routes = router.generate_routes()
        assert len(routes) == 63  # 7 domains × 9 cells

    def test_route_paths(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        routes = router.generate_routes()
        paths = [r.path for r in routes]
        assert "/cube/finance/bullion" in paths
        assert "/cube/sales/invoices" in paths
        assert "/cube/hr/probation" in paths

    def test_openapi_spec_structure(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        spec = router.openapi_spec()
        assert spec["openapi"] == "3.0.3"
        assert "paths" in spec
        assert len(spec["paths"]) == 63
        assert "/cube/finance/banking" in spec["paths"]

    def test_openapi_spec_serializable(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        spec = router.openapi_spec()
        # Must be JSON-serializable
        j = json.dumps(spec)
        assert len(j) > 100

    def test_openapi_paths_have_tags(self):
        client = CubeyClient("t", "u")
        router = CubeAPIRouter(client)
        spec = router.openapi_spec()
        finance_path = spec["paths"]["/cube/finance/bullion"]["post"]
        assert "finance" in finance_path["tags"]
