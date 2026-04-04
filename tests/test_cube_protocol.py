"""
Tests for tern-core v0.4.0 — CubeAction Address Protocol.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

import pytest

from terncore.confidence import RoutingConfidence
from terncore.routing import TernaryRouter
from terncore.cube import (
    CubeAction,
    CubeyClient,
    Guardian,
    GuardianVerdict,
    CUBE_ADDRESS_SPACE,
    validate_address,
)


# ── TestValidateAddress ─────────────────────────────────────────────


class TestValidateAddress:
    def test_valid_finance_swift(self):
        domain, function = validate_address("finance.bullion")
        assert domain == "finance"
        assert function == "bullion"

    def test_valid_all_54_addresses(self):
        count = 0
        for domain, functions in CUBE_ADDRESS_SPACE.items():
            for fn in functions:
                d, f = validate_address(f"{domain}.{fn}")
                assert d == domain
                assert f == fn
                count += 1
        assert count == 54

    def test_invalid_domain_raises(self):
        with pytest.raises(ValueError, match="Unknown domain"):
            validate_address("invalid.something")

    def test_invalid_function_raises(self):
        with pytest.raises(ValueError, match="Unknown function"):
            validate_address("finance.nonexistent")

    def test_missing_dot_raises(self):
        with pytest.raises(ValueError, match="expected domain.function"):
            validate_address("financeswift")


# ── TestCubeAction ──────────────────────────────────────────────────


class TestCubeAction:
    def test_construction_parses_address(self):
        action = CubeAction(
            address="finance.bullion", action="buy",
            params={"metal": "XAU"}, confidence=RoutingConfidence.SURE, weight=0.92,
        )
        assert action.domain == "finance"
        assert action.function == "bullion"

    def test_weight_clamped(self):
        action = CubeAction(
            address="sales.invoices", action="create",
            params={}, confidence=RoutingConfidence.SURE, weight=5.0,
        )
        assert action.weight == 1.0

        action2 = CubeAction(
            address="sales.invoices", action="create",
            params={}, confidence=RoutingConfidence.UNKNOWN, weight=-5.0,
        )
        assert action2.weight == -1.0

    def test_name_property(self):
        action = CubeAction(
            address="finance.banking", action="transfer",
            params={}, confidence=RoutingConfidence.SURE, weight=0.9,
        )
        assert action.name == "finance.banking.transfer"

    def test_is_sure_unsure_unknown(self):
        sure = CubeAction(
            address="sales.crm", action="read", params={},
            confidence=RoutingConfidence.SURE, weight=0.9,
        )
        assert sure.is_sure and not sure.is_unsure and not sure.is_unknown

        unsure = CubeAction(
            address="sales.crm", action="read", params={},
            confidence=RoutingConfidence.UNSURE, weight=0.5,
        )
        assert unsure.is_unsure

        unknown = CubeAction(
            address="sales.crm", action="read", params={},
            confidence=RoutingConfidence.UNKNOWN, weight=-0.3,
        )
        assert unknown.is_unknown


# ── TestGuardian ────────────────────────────────────────────────────


class TestGuardian:
    def test_sure_executes(self):
        g = Guardian()
        action = CubeAction(
            address="sales.invoices", action="create", params={},
            confidence=RoutingConfidence.SURE, weight=0.92,
        )
        verdict = g.evaluate(action)
        assert verdict.can_execute
        assert verdict.verdict == "execute"

    def test_unsure_gates(self):
        g = Guardian()
        action = CubeAction(
            address="operations.inventory", action="bulk_delete", params={},
            confidence=RoutingConfidence.UNSURE, weight=0.51,
        )
        verdict = g.evaluate(action)
        assert verdict.is_gated
        assert "human review" in verdict.reason

    def test_unknown_rolls_back(self):
        g = Guardian()
        action = CubeAction(
            address="compliance.access_rights", action="elevate", params={},
            confidence=RoutingConfidence.UNKNOWN, weight=-0.3,
        )
        verdict = g.evaluate(action)
        assert verdict.is_rolled_back

    def test_three_unknowns_locks_domain(self):
        g = Guardian(unknown_threshold=3)
        for i in range(3):
            g.evaluate(CubeAction(
                address="compliance.data_privacy", action=f"probe_{i}", params={},
                confidence=RoutingConfidence.UNKNOWN, weight=-0.3,
            ))
        assert g.is_protected("compliance")
        assert len(g.recent_anomalies) >= 1

    def test_protected_domain_blocks_sure(self):
        g = Guardian()
        g._protected_domains.add("finance")
        action = CubeAction(
            address="finance.accounting", action="read", params={},
            confidence=RoutingConfidence.SURE, weight=0.99,
        )
        verdict = g.evaluate(action)
        assert verdict.is_rolled_back
        assert "protected" in verdict.reason

    def test_clear_domain_restores(self):
        g = Guardian()
        g._protected_domains.add("finance")
        g.clear_domain("finance")
        assert not g.is_protected("finance")

    def test_pending_gates_list(self):
        g = Guardian()
        g.evaluate(CubeAction(
            address="sales.quotes", action="send", params={},
            confidence=RoutingConfidence.UNSURE, weight=0.5,
        ))
        assert len(g.pending_gates) == 1

    def test_recent_anomalies_list(self):
        g = Guardian(unknown_threshold=2)
        for i in range(2):
            g.evaluate(CubeAction(
                address="cx.escalations", action=f"probe_{i}", params={},
                confidence=RoutingConfidence.UNKNOWN, weight=-0.5,
            ))
        assert len(g.recent_anomalies) >= 1


# ── TestCubeyClient ─────────────────────────────────────────────────


class TestCubeyClient:
    def test_execute_sure_path(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        verdict = client.execute(
            address="finance.bullion", action="buy",
            params={"metal": "XAU"}, confidence=RoutingConfidence.SURE, weight=0.92,
        )
        assert verdict.can_execute

    def test_execute_unsure_gate(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        verdict = client.execute(
            address="operations.inventory", action="delete",
            params={}, confidence=RoutingConfidence.UNSURE, weight=0.51,
        )
        assert verdict.is_gated

    def test_execute_unknown_rollback(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        verdict = client.execute(
            address="compliance.fraud_detection", action="override",
            params={}, confidence=RoutingConfidence.UNKNOWN, weight=-0.3,
        )
        assert verdict.is_rolled_back

    def test_route_and_execute(self):
        router = TernaryRouter(deferral_band=(0.3, 0.6))
        router.register("tool", lambda p: 0.9)
        client = CubeyClient(tenant_id="t1", user_id="u1", router=router)
        verdict = client.route_and_execute(
            address="sales.invoices", action="create",
            params={"amount": 42000}, prompt="Create invoice",
        )
        assert verdict.can_execute

    def test_route_and_execute_no_router_raises(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        with pytest.raises(ValueError, match="Router required"):
            client.route_and_execute("sales.crm", "read", {}, "test")

    def test_valid_addresses_all_54(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        assert len(client.valid_addresses()) == 54

    def test_valid_addresses_filtered(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        finance = client.valid_addresses("finance")
        assert len(finance) == 9
        assert all(a.startswith("finance.") for a in finance)

    def test_invalid_address_raises(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        with pytest.raises(ValueError):
            client.execute("invalid.bad", "x", {}, RoutingConfidence.SURE, 0.9)


# ── TestIntegration ─────────────────────────────────────────────────


class TestIntegration:
    def test_finance_swift_sure_executes(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        verdict = client.execute(
            address="finance.banking", action="transfer",
            params={"amount": 50000, "currency": "AUD"},
            confidence=RoutingConfidence.SURE, weight=0.92,
        )
        assert verdict.can_execute

    def test_finance_unknown_blocked(self):
        client = CubeyClient(tenant_id="t1", user_id="u1")
        verdict = client.execute(
            address="finance.banking", action="transfer",
            params={"amount": 800000, "currency": "EUR"},
            confidence=RoutingConfidence.UNKNOWN, weight=-0.3,
        )
        assert not verdict.can_execute

    def test_three_unknown_finance_locks_domain(self):
        guardian = Guardian(unknown_threshold=3)
        client = CubeyClient(tenant_id="t1", user_id="u1", guardian=guardian)
        for i in range(3):
            client.execute(
                address="finance.bullion", action=f"probe_{i}", params={},
                confidence=RoutingConfidence.UNKNOWN, weight=-0.3,
            )
        assert guardian.is_protected("finance")

        # Now SURE action is blocked
        verdict = client.execute(
            address="finance.accounting", action="read", params={},
            confidence=RoutingConfidence.SURE, weight=0.99,
        )
        assert verdict.is_rolled_back

    def test_address_space_complete(self):
        total = sum(len(fns) for fns in CUBE_ADDRESS_SPACE.values())
        assert total == 54
        assert len(CUBE_ADDRESS_SPACE) == 6
