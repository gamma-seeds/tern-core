"""
terncore.persistence — Guardian state persistence.

Saves and restores Guardian state to JSON so demos survive restarts.
Protected domains, event log, and action history are preserved.

CNS Synaptic™ by Synapticode Co., Ltd.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from terncore.confidence import RoutingConfidence
from terncore.cube import (
    CubeAction,
    Guardian,
    GuardianVerdict,
    CUBE_ADDRESS_SPACE,
)


def _serialize_verdict(v: GuardianVerdict) -> dict:
    return {
        "action_id": v.action_id,
        "verdict": v.verdict,
        "reason": v.reason,
        "confidence": v.confidence.value,
        "weight": v.weight,
        "timestamp": v.timestamp.isoformat(),
    }


def _deserialize_verdict(d: dict) -> GuardianVerdict:
    return GuardianVerdict(
        action_id=d["action_id"],
        verdict=d["verdict"],
        reason=d["reason"],
        confidence=RoutingConfidence(d["confidence"]),
        weight=d["weight"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
    )


def _serialize_action(a: CubeAction) -> dict:
    return {
        "id": a.id,
        "address": a.address,
        "action": a.action,
        "params": a.params,
        "confidence": a.confidence.value,
        "weight": a.weight,
        "tenant_id": a.tenant_id,
        "user_id": a.user_id,
        "timestamp": a.timestamp.isoformat(),
    }


def _deserialize_action(d: dict) -> CubeAction:
    return CubeAction(
        address=d["address"],
        action=d["action"],
        params=d.get("params", {}),
        confidence=RoutingConfidence(d["confidence"]),
        weight=d["weight"],
        tenant_id=d.get("tenant_id", ""),
        user_id=d.get("user_id", ""),
        id=d["id"],
        timestamp=datetime.fromisoformat(d["timestamp"]),
    )


class GuardianPersistence:
    """Save and restore Guardian state to/from JSON."""

    def __init__(self, path: Path):
        self.path = path

    def save(self, guardian: Guardian) -> None:
        """Serialize Guardian state to JSON file."""
        state = {
            "version": 1,
            "saved_at": datetime.now().isoformat(),
            "protected_domains": list(guardian._protected_domains),
            "event_log": [_serialize_verdict(v) for v in guardian._event_log],
            "recent_actions": [_serialize_action(a) for a in guardian._recent_actions],
            "config": {
                "unknown_threshold": guardian._unknown_threshold,
                "correlation_window": guardian._correlation_window,
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, default=str))

    def load(self) -> Optional[Guardian]:
        """Deserialize Guardian state from JSON file. Returns None if file missing."""
        if not self.path.exists():
            return None

        data = json.loads(self.path.read_text())

        guardian = Guardian(
            unknown_threshold=data["config"]["unknown_threshold"],
            correlation_window=data["config"]["correlation_window"],
        )
        guardian._protected_domains = set(data.get("protected_domains", []))
        guardian._event_log = [_deserialize_verdict(v) for v in data.get("event_log", [])]
        guardian._recent_actions = [_deserialize_action(a) for a in data.get("recent_actions", [])]

        return guardian

    def exists(self) -> bool:
        return self.path.exists()

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()


class CubeySessionPersistence:
    """Save and restore full CubeyClient session state."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_guardian(self, tenant_id: str, guardian: Guardian) -> Path:
        path = self.base_dir / f"guardian_{tenant_id}.json"
        GuardianPersistence(path).save(guardian)
        return path

    def load_guardian(self, tenant_id: str) -> Optional[Guardian]:
        path = self.base_dir / f"guardian_{tenant_id}.json"
        return GuardianPersistence(path).load()

    def list_tenants(self) -> list[str]:
        return [
            p.stem.replace("guardian_", "")
            for p in self.base_dir.glob("guardian_*.json")
        ]
