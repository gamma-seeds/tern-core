"""
Standalone CLI wrapper for the per-group symmetric (ternary_g128) pack.

Packs an unpacked-FP16 ternary model (e.g. PrismML
``Ternary-Bonsai-*-unpacked``, a dense Qwen3 checkpoint) into tern-core's
lossless ``ternary_g128`` format, preserving the per-group (128) scale
granularity.

Usage:
    python tools/pack_qwen3_g128.py prism-ml/Ternary-Bonsai-8B-unpacked \
        -o bonsai-8b-g128.tern-model
    python tools/pack_qwen3_g128.py /path/to/unpacked-dir \
        -o out.tern-model --adapter qwen3 --report report.json

See also: python -m terncore.pack_g128 (equivalent entry point).

The ``--adapter`` defaults to ``qwen3``; any dense adapter that
classifies 2-D block projections as ternary-eligible works.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure tern-core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from terncore.pack_g128 import main

if __name__ == "__main__":
    raise SystemExit(main())
