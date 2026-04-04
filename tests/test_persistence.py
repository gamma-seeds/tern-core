"""
Tests for tern-core persistence — Guardian state survives restart.
"""

import json
import tempfile
from pathlib import Path

import pytest

from terncore.confidence import RoutingConfidence
from terncore.cube import CubeAction, Guardian, CubeyClient
from terncore.persistence import GuardianPersistence, CubeySessionPersistence


class TestGuardianPersistence:

    def test_save_and_load_empty(self, tmp_path):
        path = tmp_path / "guardian.json"
        g = Guardian()
        GuardianPersistence(path).save(g)
        restored = GuardianPersistence(path).load()
        assert restored is not None
        assert len(restored.event_log) == 0

    def test_save_and_load_with_events(self, tmp_path):
        path = tmp_path / "guardian.json"
        g = Guardian()
        g.evaluate(CubeAction(address="sales.invoices", action="create", params={},
                               confidence=RoutingConfidence.SURE, weight=0.92))
        g.evaluate(CubeAction(address="finance.bullion", action="buy", params={},
                               confidence=RoutingConfidence.UNSURE, weight=0.50))
        GuardianPersistence(path).save(g)

        restored = GuardianPersistence(path).load()
        assert len(restored.event_log) == 2

    def test_protected_domains_survive(self, tmp_path):
        path = tmp_path / "guardian.json"
        g = Guardian(unknown_threshold=2)
        for i in range(2):
            g.evaluate(CubeAction(address="compliance.data_privacy", action=f"p{i}",
                                   params={}, confidence=RoutingConfidence.UNKNOWN, weight=-0.3))
        assert g.is_protected("compliance")

        GuardianPersistence(path).save(g)
        restored = GuardianPersistence(path).load()
        assert restored.is_protected("compliance")

    def test_config_survives(self, tmp_path):
        path = tmp_path / "guardian.json"
        g = Guardian(unknown_threshold=5, correlation_window=120.0)
        GuardianPersistence(path).save(g)

        restored = GuardianPersistence(path).load()
        assert restored._unknown_threshold == 5
        assert restored._correlation_window == 120.0

    def test_load_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert GuardianPersistence(path).load() is None

    def test_exists(self, tmp_path):
        path = tmp_path / "guardian.json"
        p = GuardianPersistence(path)
        assert not p.exists()
        Guardian()  # don't save
        p.save(Guardian())
        assert p.exists()

    def test_delete(self, tmp_path):
        path = tmp_path / "guardian.json"
        p = GuardianPersistence(path)
        p.save(Guardian())
        assert p.exists()
        p.delete()
        assert not p.exists()

    def test_verdicts_round_trip(self, tmp_path):
        path = tmp_path / "guardian.json"
        g = Guardian()
        # All three verdict types
        g.evaluate(CubeAction(address="sales.crm", action="r", params={},
                               confidence=RoutingConfidence.SURE, weight=0.9))
        g.evaluate(CubeAction(address="operations.inventory", action="d", params={},
                               confidence=RoutingConfidence.UNSURE, weight=0.5))
        g.evaluate(CubeAction(address="compliance.access_rights", action="e", params={},
                               confidence=RoutingConfidence.UNKNOWN, weight=-0.3))

        GuardianPersistence(path).save(g)
        restored = GuardianPersistence(path).load()

        verdicts = [v.verdict for v in restored.event_log]
        assert "execute" in verdicts
        assert "gate" in verdicts
        assert "rollback" in verdicts

    def test_recent_actions_round_trip(self, tmp_path):
        path = tmp_path / "guardian.json"
        g = Guardian()
        g.evaluate(CubeAction(address="finance.banking", action="transfer", params={"amount": "50000"},
                               confidence=RoutingConfidence.SURE, weight=0.92))

        GuardianPersistence(path).save(g)
        restored = GuardianPersistence(path).load()
        assert len(restored._recent_actions) == 1
        assert restored._recent_actions[0].address == "finance.banking"

    def test_json_is_readable(self, tmp_path):
        """Saved file is human-readable JSON."""
        path = tmp_path / "guardian.json"
        g = Guardian()
        g.evaluate(CubeAction(address="sales.invoices", action="create", params={},
                               confidence=RoutingConfidence.SURE, weight=0.92))
        GuardianPersistence(path).save(g)

        data = json.loads(path.read_text())
        assert data["version"] == 1
        assert "saved_at" in data
        assert len(data["event_log"]) == 1


class TestCubeySessionPersistence:

    def test_save_and_load_tenant(self, tmp_path):
        sp = CubeySessionPersistence(tmp_path / "sessions")
        g = Guardian()
        g.evaluate(CubeAction(address="sales.crm", action="read", params={},
                               confidence=RoutingConfidence.SURE, weight=0.9))
        sp.save_guardian("tenant-001", g)

        restored = sp.load_guardian("tenant-001")
        assert restored is not None
        assert len(restored.event_log) == 1

    def test_list_tenants(self, tmp_path):
        sp = CubeySessionPersistence(tmp_path / "sessions")
        sp.save_guardian("t1", Guardian())
        sp.save_guardian("t2", Guardian())
        tenants = sp.list_tenants()
        assert "t1" in tenants
        assert "t2" in tenants
        assert len(tenants) == 2

    def test_load_missing_tenant(self, tmp_path):
        sp = CubeySessionPersistence(tmp_path / "sessions")
        assert sp.load_guardian("nonexistent") is None

    def test_full_session_round_trip(self, tmp_path):
        """Full demo cycle: create client → execute actions → save → restart → load → verify."""
        sp = CubeySessionPersistence(tmp_path / "sessions")

        # Session 1 — execute actions
        client = CubeyClient(tenant_id="demo", user_id="rob")
        client.execute("finance.bullion", "buy", {"metal": "XAU"},
                        RoutingConfidence.SURE, 0.92)
        client.execute("compliance.data_privacy", "audit", {},
                        RoutingConfidence.UNSURE, 0.45)
        sp.save_guardian("demo", client.guardian)

        # Simulate restart — new process
        restored_guardian = sp.load_guardian("demo")
        assert restored_guardian is not None
        assert len(restored_guardian.event_log) == 2
        assert len(restored_guardian.pending_gates) == 1

        # Session 2 — continues with restored state
        client2 = CubeyClient(tenant_id="demo", user_id="rob", guardian=restored_guardian)
        verdict = client2.execute("sales.invoices", "create", {},
                                   RoutingConfidence.SURE, 0.90)
        assert verdict.can_execute
        assert len(restored_guardian.event_log) == 3
