"""
pytest configuration for tern-core-kokoro tests.

Routes ``terncore`` imports through this worktree's ``src/`` rather
than the venv editable install (which points at the canonical
``~/synapticode/tern-core/`` R12 sprint working tree).

This is load-bearing for the Kokoro 82M attachment work — the new
``adapters/kokoro.py`` + ``integration3/`` module tree lives in this
worktree only; the canonical tern-core checkout stays untouched per
the Phase 0 brief §6 no-breakage constraint.

The ``integration3`` canonical package (Wave 5 closure
v1.0-wave5-closure / HEAD 67c7f74) resolves as a proper installed
distribution — locally via ``pip install -e ~/synapticode/agents/integration3``,
in CI via a separate checkout + ``pip install -e`` of
``gamma-seeds/integration3`` at the same pin.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Worktree src/ wins over venv editable install pointing at sibling
# canonical tern-core checkout.
_WORKTREE_ROOT = Path(__file__).resolve().parent.parent
_WORKTREE_SRC = _WORKTREE_ROOT / "src"
if _WORKTREE_SRC.exists() and str(_WORKTREE_SRC) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_SRC))
# Drop any pre-cached terncore module loaded from the venv editable
# install so the worktree version reloads cleanly.
for _mod in list(sys.modules):
    if _mod == "terncore" or _mod.startswith("terncore."):
        del sys.modules[_mod]
