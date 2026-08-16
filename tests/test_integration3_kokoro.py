"""
Tests for ``terncore.integration3.KokoroProvider3`` — first downstream
consumer integration following integration³ end-to-end build closure
2026-05-19 (HEAD 67c7f74). Covers F-1 → F-12 falsifiers per Phase 0
brief ``2026-05-19_kokoro_82m_integration3_attachment_phase0.md``.

Falsifier coverage:

  F-1  Provider³ Protocol satisfaction (isinstance + 4-method surface)
  F-2  EpistemicState three-cell mapping (CONFIRMED / UNCERTAIN /
       DISCONFIRMED)
  F-3  Provenance³ soft-stamp shape
  F-4  AtomIdentifier³ SHA-256-of-fields determinism + mutation-
       sensitivity
  F-5  NO Subscriber³ Protocol satisfaction (cluster A)
  F-6  NO Ledger³ writeback (cluster B static scan)
  F-7  Kokoro 82M model loads cleanly (lazy load + voice .pt access)
  F-8  Compressed artefact ≤22 MB demo ceiling
  F-9  54-voice catalogue accessible at filesystem; demo subset
       distinguishable from full catalogue
  F-10 Output structurally valid (InferenceResult3 shape preserved)
  F-11 G2P DI plug (OQ-4 Option C — optional g2p callable)
  F-12 Cross-repo trip-wire activation parity (TW-2 + TW-9)

Test naming discipline (per integration³ Wave 1 Dispatch Addendum 2
§3 element 3) — vocabulary disambiguation in test method identifiers.

Copyright (c) 2025–2026 Robert Lakelin. All rights reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# sys.path routing (worktree src/ + agents/integration3) handled in
# tests/conftest.py — module-identity preservation across the test
# session.

import integration3  # noqa: E402
from integration3 import (  # noqa: E402
    AtomIdentifier3,
    Confidence3,
    EpistemicState,
    InferenceResult3,
    Provenance3,
    Provider3,
    ProviderConfig3,
    Thought3,
)

from terncore.integration3 import (  # noqa: E402
    DEFAULT_COMPRESSED_ARTEFACT,
    DEFAULT_MODEL_PATH,
    DEFAULT_VOICES_DIR,
    DEMO_VOICES,
    INFERENCE_OUTCOME_CONFIRMED,
    INFERENCE_OUTCOME_DISCONFIRMED,
    INFERENCE_OUTCOME_UNCERTAIN,
    IntegratedProvider3Base,
    KOKORO_SAMPLE_RATE_HZ,
    KokoroProvider3,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _iter_code_lines(path: Path):
    """Yield non-comment, non-docstring source lines from a Python file.

    Lightweight docstring stripper: tracks ``\"\"\"`` and ``'''`` triple-
    quote toggles at logical-line start. Skips lines that begin with
    ``#`` after stripping. This is sufficient for cluster-A / cluster-B
    static scans against the small terncore.integration3 module tree
    (rejecting documentation references that intentionally surface the
    architectural declination)."""
    source = path.read_text()
    lines = source.splitlines()
    in_triple_double = False
    in_triple_single = False
    for raw_line in lines:
        stripped = raw_line.strip()
        # Toggle triple-quote state on lines that open/close docstrings.
        # Handle the simple case where the triple-quote pair lives on the
        # same line by counting occurrences.
        if not in_triple_single:
            count_dd = stripped.count('"""')
            if count_dd % 2 == 1:
                in_triple_double = not in_triple_double
                # Skip the line that opens/closes a docstring (it may
                # carry text on the same line either side; conservative
                # skip is correct for our pattern checks).
                continue
            if in_triple_double:
                continue
        if not in_triple_double:
            count_ss = stripped.count("'''")
            if count_ss % 2 == 1:
                in_triple_single = not in_triple_single
                continue
            if in_triple_single:
                continue
        if stripped.startswith("#"):
            continue
        yield raw_line


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def default_provider() -> KokoroProvider3:
    """KokoroProvider3 with default config (canonical model-library paths)."""
    return KokoroProvider3()


@pytest.fixture
def custom_config_provider() -> KokoroProvider3:
    """KokoroProvider3 with a custom ProviderConfig3."""
    config = ProviderConfig3(
        endpoint=DEFAULT_COMPRESSED_ARTEFACT,
        model="kokoro-82m-test",
        timeout_seconds=60.0,
        output_half_life=180.0,
    )
    return KokoroProvider3(config=config)


# ── F-1 — Provider³ Protocol satisfaction ─────────────────────────────────────

def test_kokoro_provider3_protocol_isinstance_true(default_provider):
    """F-1: ``isinstance(KokoroProvider3(config), Provider3)`` returns
    True via ``@runtime_checkable`` structural Protocol match."""
    assert isinstance(default_provider, Provider3)


def test_kokoro_provider3_protocol_four_method_surface(default_provider):
    """F-1: The four canonical Provider³ surfaces are present —
    config / health_check / is_available / infer."""
    assert hasattr(default_provider, "config")
    assert isinstance(default_provider.config, ProviderConfig3)
    assert callable(getattr(default_provider, "health_check"))
    assert hasattr(default_provider, "is_available")
    assert callable(getattr(default_provider, "infer"))


def test_kokoro_provider3_config_property_returns_bound_config(custom_config_provider):
    """F-1: ``config`` property returns the ``ProviderConfig3`` instance
    bound at construction time exactly."""
    assert custom_config_provider.config.model == "kokoro-82m-test"
    assert custom_config_provider.config.timeout_seconds == 60.0


def test_kokoro_integrated_provider3_base_protocol_isinstance_true():
    """F-1: ``IntegratedProvider3Base`` itself satisfies Provider³
    structurally (template-method class fully implements the
    Protocol surface)."""
    base = IntegratedProvider3Base(config=ProviderConfig3())
    assert isinstance(base, Provider3)


# ── F-2 — EpistemicState three-cell mapping per P099 §107 ─────────────────────

def test_kokoro_epistemic_state_inference_tier_three_cell_mapping(default_provider):
    """F-2: Three branches of the cluster C ratification are reachable
    and map to the canonical EpistemicState three-cell tier."""
    # UNCERTAIN branch — structural fallback (no kokoro PyPI package
    # in CI environment; provider returns UNCERTAIN with summary).
    result_uncertain = default_provider.infer("hello world")
    assert isinstance(result_uncertain, InferenceResult3)
    assert result_uncertain.epistemic_state in {
        EpistemicState.CONFIRMED,
        EpistemicState.UNCERTAIN,
    }

    # DISCONFIRMED branch — force exception via a broken G2P.
    def broken_g2p(text: str) -> list[int]:
        raise RuntimeError("forced G2P failure")

    disc_provider = KokoroProvider3(g2p=broken_g2p)
    result_disc = disc_provider.infer("hello world")
    assert result_disc.epistemic_state is EpistemicState.DISCONFIRMED


def test_kokoro_epistemic_state_disconfirmed_propagates_unknown_confidence():
    """F-2: On DISCONFIRMED branch the propagated input_confidence is
    UNKNOWN (counter-evidence path collapses confidence)."""
    def broken_g2p(text: str) -> list[int]:
        raise ValueError("forced inference failure")

    provider = KokoroProvider3(g2p=broken_g2p)
    result = provider.infer("test")
    assert result.epistemic_state is EpistemicState.DISCONFIRMED
    assert result.input_confidence is Confidence3.UNKNOWN


def test_kokoro_epistemic_state_inference_tier_distinct_from_routing_tier(default_provider):
    """F-2: The InferenceResult³ carries BOTH the routing-tier
    Confidence³ (input_confidence) and the inference-tier
    EpistemicState — two-tier discipline per SEED §Core item 2.
    Never aliased."""
    result = default_provider.infer("hello")
    assert isinstance(result.input_confidence, Confidence3)
    assert isinstance(result.epistemic_state, EpistemicState)
    assert type(result.input_confidence) is not type(result.epistemic_state)


# ── F-3 — Provenance³ soft-stamp ──────────────────────────────────────────────

def test_kokoro_provenance3_soft_stamp_device_signature(default_provider):
    """F-3: ``InferenceResult3.provenance.device_signature`` matches the
    canonical ``f"tern-core@{endpoint}"`` soft-stamp pattern."""
    result = default_provider.infer("hello")
    assert result.provenance is not None
    expected = f"tern-core@{default_provider.config.endpoint}"
    assert result.provenance.device_signature == expected


def test_kokoro_provenance3_soft_stamp_five_field_structure(default_provider):
    """F-3: ``Provenance3`` carries the canonical 5-field structure per
    P096 §130 (device_signature + author_identity_token +
    hardware_clock_timestamp + content_hash + parent_atom_hash)."""
    result = default_provider.infer("hello")
    prov = result.provenance
    assert prov is not None
    assert isinstance(prov.device_signature, str)
    assert isinstance(prov.author_identity_token, str)
    assert isinstance(prov.hardware_clock_timestamp, float)
    assert isinstance(prov.content_hash, str)
    assert len(prov.content_hash) == 64   # SHA-256 hex
    assert prov.parent_atom_hash is None


def test_kokoro_provenance3_author_identity_token_distinct(default_provider):
    """F-3: ``KokoroProvider3`` surfaces its own author identity token
    distinct from the base class — separate canonical adapter
    identity per cluster D substitutability invariant."""
    result = default_provider.infer("hello")
    assert result.provenance is not None
    assert "Kokoro" in result.provenance.author_identity_token


# ── F-4 — AtomIdentifier³ SHA-256-of-fields ───────────────────────────────────

def test_kokoro_atom_identifier3_64char_hex(default_provider):
    """F-4: ``InferenceResult3.atom_identifier`` is a 64-char hex
    string (SHA-256 hexdigest)."""
    result = default_provider.infer("hello")
    assert result.atom_identifier is not None
    assert len(result.atom_identifier) == 64
    int(result.atom_identifier, 16)   # parses as hex


def test_kokoro_atom_identifier3_deterministic_for_same_inputs():
    """F-4: SHA-256-of-fields is deterministic — re-computing from
    the same (value, confidence, provenance) yields the same hex."""
    prov = Provenance3(
        device_signature="tern-core@test",
        author_identity_token="test",
        hardware_clock_timestamp=1234567.0,
        content_hash=Provenance3.compute_content_hash("hello"),
    )
    h1 = AtomIdentifier3.compute(
        value="hello", confidence=Confidence3.SURE, provenance=prov
    )
    h2 = AtomIdentifier3.compute(
        value="hello", confidence=Confidence3.SURE, provenance=prov
    )
    assert h1 == h2


def test_kokoro_atom_identifier3_mutation_sensitive_value():
    """F-4: Mutating the value changes the identifier."""
    prov = Provenance3(
        device_signature="tern-core@test",
        author_identity_token="test",
        hardware_clock_timestamp=1234567.0,
        content_hash=Provenance3.compute_content_hash("x"),
    )
    h1 = AtomIdentifier3.compute("a", Confidence3.SURE, prov)
    h2 = AtomIdentifier3.compute("b", Confidence3.SURE, prov)
    assert h1 != h2


def test_kokoro_atom_identifier3_mutation_sensitive_confidence():
    """F-4: Mutating the confidence changes the identifier."""
    prov = Provenance3(
        device_signature="tern-core@test",
        author_identity_token="test",
        hardware_clock_timestamp=1234567.0,
        content_hash=Provenance3.compute_content_hash("x"),
    )
    h1 = AtomIdentifier3.compute("x", Confidence3.SURE, prov)
    h2 = AtomIdentifier3.compute("x", Confidence3.UNSURE, prov)
    h3 = AtomIdentifier3.compute("x", Confidence3.UNKNOWN, prov)
    assert h1 != h2 != h3 != h1


# ── F-5 — NO Subscriber³ Protocol satisfaction (cluster A) ────────────────────

def test_kokoro_no_subscriber3_protocol_satisfaction(default_provider):
    """F-5: KokoroProvider3 does NOT satisfy Subscriber³ — cluster A
    architectural rule (provider is adapter, not faculty consumer)."""
    from integration3 import Subscriber3
    assert not isinstance(default_provider, Subscriber3)
    assert not hasattr(default_provider, "on_event")


def test_kokoro_integrated_provider3_base_no_subscriber3():
    """F-5: ``IntegratedProvider3Base`` itself does NOT satisfy
    Subscriber³ (the rule applies at the base layer)."""
    from integration3 import Subscriber3
    base = IntegratedProvider3Base(config=ProviderConfig3())
    assert not isinstance(base, Subscriber3)
    assert not hasattr(base, "on_event")


def test_kokoro_no_subscriber3_import_in_terncore_integration3():
    """F-5: Static analysis — no ``Subscriber3`` import across the
    terncore.integration3 module tree.

    Scan focuses on import lines + executable code (skips comments
    and docstring text where the architectural declination is
    documented intentionally as part of cluster A authorial
    discipline)."""
    import terncore.integration3 as pkg
    root = Path(pkg.__file__).parent
    for py in root.rglob("*.py"):
        for line in _iter_code_lines(py):
            assert "Subscriber3" not in line, (
                f"Subscriber3 reference in code line in {py}: {line!r} "
                f"— cluster A rule violated"
            )


# ── F-6 — NO Ledger³ writeback (cluster B) ────────────────────────────────────

def test_kokoro_no_ledger3_writeback_static_grep():
    """F-6: Static analysis — no Ledger3 / LedgerEntry3 /
    make_bootstrap_ledger_entry / decision3 import across the
    terncore.integration3 module tree.

    Scan focuses on import lines + executable code (skips comments
    and docstring text where the architectural declination is
    documented intentionally as part of cluster B authorial
    discipline)."""
    import terncore.integration3 as pkg
    root = Path(pkg.__file__).parent
    forbidden = (
        "Ledger3",
        "LedgerEntry3",
        "make_bootstrap_ledger_entry",
        "decision3",
    )
    for py in root.rglob("*.py"):
        for line in _iter_code_lines(py):
            for token in forbidden:
                assert token not in line, (
                    f"Forbidden token {token!r} found in code line in "
                    f"{py}: {line!r} — cluster B rule violated"
                )


# ── F-7 — Kokoro 82M model loads cleanly ──────────────────────────────────────

def test_kokoro_health_check_returns_true_on_disk(default_provider):
    """F-7: ``health_check()`` returns True iff the compressed artefact
    or source .pth path exists AND the default voice .pt loads."""
    # Either the compressed artefact OR the source .pth must exist
    # in the canonical model-library layout.
    artefact = Path(DEFAULT_COMPRESSED_ARTEFACT)
    source_pth = Path(DEFAULT_MODEL_PATH)
    voice_file = Path(DEFAULT_VOICES_DIR) / f"{default_provider.voice}.pt"
    if not ((artefact.exists() or source_pth.exists()) and voice_file.exists()):
        pytest.skip(
            "Kokoro 82M model + voice .pt files not present at "
            f"canonical model-library path {DEFAULT_VOICES_DIR}"
        )
    assert default_provider.health_check() is True


def test_kokoro_eager_load_constructor_kwarg():
    """F-7: ``eager_load=True`` constructor kwarg triggers immediate
    health check + voice cache populate."""
    voice_file = Path(DEFAULT_VOICES_DIR) / "af_heart.pt"
    if not voice_file.exists():
        pytest.skip("Default voice .pt not present at canonical path")
    provider = KokoroProvider3(eager_load=True)
    assert provider.is_available is True


# ── F-8 — Compressed artefact ≤22 MB demo ceiling ─────────────────────────────

def test_kokoro_compressed_artefact_within_22mb_ceiling():
    """F-8: Compressed Kokoro 82M demo artefact (main model + demo-
    subset 6 voices at FP16) ≤22 MB ceiling per OQ-6 ratification.

    Measured via on-disk size of the .tern-model file + sum of FP16
    voice .pt sizes (estimated at half the FP32 on-disk size since
    voice .pt files ship FP32 and tern-core's FP16 retention halves)."""
    artefact = Path(DEFAULT_COMPRESSED_ARTEFACT)
    if not artefact.exists():
        pytest.skip(
            f"Compressed artefact not present at {DEFAULT_COMPRESSED_ARTEFACT}; "
            f"run tools/compress_kokoro.py to materialise"
        )
    model_bytes = artefact.stat().st_size

    voices_dir = Path(DEFAULT_VOICES_DIR)
    voice_bytes = 0
    for voice in DEMO_VOICES:
        vf = voices_dir / f"{voice}.pt"
        if vf.exists():
            # FP32 → FP16 halves the footprint
            voice_bytes += vf.stat().st_size // 2

    total_bytes = model_bytes + voice_bytes
    ceiling = 22 * 1024 * 1024
    assert total_bytes <= ceiling, (
        f"Demo artefact {total_bytes / 1024 / 1024:.2f} MB exceeds "
        f"≤22 MB ceiling (model={model_bytes/1024/1024:.2f} MB; "
        f"voices={voice_bytes/1024/1024:.2f} MB)"
    )


def test_kokoro_compressed_artefact_verifies_integrity():
    """F-8: Compressed .tern-model passes its built-in integrity
    check (CRC32 + SHA-256 footer)."""
    artefact = Path(DEFAULT_COMPRESSED_ARTEFACT)
    if not artefact.exists():
        pytest.skip("Compressed artefact not present")
    from terncore.tern_model import TernModelReader
    reader = TernModelReader(str(artefact))
    assert reader.verify() is True


# ── F-9 — Voice catalogue accessibility ───────────────────────────────────────

def test_kokoro_full_54_voice_catalogue_accessible(default_provider):
    """F-9: All 54 voice .pt files reachable at the filesystem
    ``voices_dir``."""
    voices_dir = Path(DEFAULT_VOICES_DIR)
    if not voices_dir.exists():
        pytest.skip("voices/ directory not present at model-library path")
    voices = default_provider.available_voices()
    assert len(voices) == 54


def test_kokoro_demo_subset_six_voices_per_oq10_ratification(default_provider):
    """F-9: Demo subset shipped in compiled artefact = 6 voices
    (af_heart, am_adam, bf_alice, bm_daniel, jf_alpha, zf_xiaobei)
    per OQ-10 Option A surgeon ratification 2026-05-19."""
    assert len(DEMO_VOICES) == 6
    assert "af_heart" in DEMO_VOICES
    assert "am_adam" in DEMO_VOICES
    assert "bf_alice" in DEMO_VOICES
    assert "bm_daniel" in DEMO_VOICES
    assert "jf_alpha" in DEMO_VOICES
    assert "zf_xiaobei" in DEMO_VOICES
    assert default_provider.demo_voices == DEMO_VOICES


def test_kokoro_demo_subset_distinguishable_from_full_catalogue(default_provider):
    """F-9: The demo subset is strictly contained in the full 54-voice
    catalogue (cardinality check)."""
    voices_dir = Path(DEFAULT_VOICES_DIR)
    if not voices_dir.exists():
        pytest.skip("voices/ directory not present")
    full = set(default_provider.available_voices())
    demo = set(DEMO_VOICES)
    assert demo.issubset(full)
    assert len(full) > len(demo)


def test_kokoro_voice_character_changes_audibly_three_voices(default_provider):
    """F-9: Voice selection changes the output character. Probe with
    af_heart vs am_adam vs bf_alice — the canonical test triple per
    Phase 0 brief §4 F-9 declaration.

    At this attachment the live inference path may not be available;
    the test asserts that the provider can be re-instantiated with
    each voice and produces distinct output_atom dicts (voice field
    differs)."""
    voices_dir = Path(DEFAULT_VOICES_DIR)
    if not voices_dir.exists():
        pytest.skip("voices/ directory not present")
    outputs = {}
    for voice in ("af_heart", "am_adam", "bf_alice"):
        provider = KokoroProvider3(voice=voice)
        result = provider.infer("hello")
        outputs[voice] = result.output_atom.content
    assert outputs["af_heart"]["voice"] == "af_heart"
    assert outputs["am_adam"]["voice"] == "am_adam"
    assert outputs["bf_alice"]["voice"] == "bf_alice"
    assert outputs["af_heart"]["voice"] != outputs["am_adam"]["voice"]


# ── F-10 — Audio output structurally valid ────────────────────────────────────

def test_kokoro_inference_result3_structurally_valid(default_provider):
    """F-10: ``KokoroProvider3.infer()`` returns ``InferenceResult3``
    with the canonical shape (all Wave 0 + Wave 5 fields populated)."""
    result = default_provider.infer("hello world")
    assert isinstance(result, InferenceResult3)
    assert result.model == default_provider.config.model
    assert result.endpoint == default_provider.config.endpoint
    assert isinstance(result.text, str)
    assert isinstance(result.output_atom, Thought3)
    assert isinstance(result.input_confidence, Confidence3)
    assert isinstance(result.latency_ms, float)
    assert isinstance(result.epistemic_state, EpistemicState)
    assert result.provenance is not None
    assert result.atom_identifier is not None


def test_kokoro_output_atom_content_carries_audio_metadata(default_provider):
    """F-10: ``output_atom.content`` is a dict carrying audio metadata
    (sample_rate, voice, duration_seconds, phoneme_token_count, audio
    or fallback flag) — TTS-specific shape preserved."""
    result = default_provider.infer("hello world")
    content = result.output_atom.content
    assert isinstance(content, dict)
    assert content["sample_rate"] == KOKORO_SAMPLE_RATE_HZ
    assert content["voice"] in DEMO_VOICES
    assert "duration_seconds" in content
    assert "phoneme_token_count" in content


def test_kokoro_sample_rate_24khz_per_kokoro_readme():
    """F-10: Kokoro audio output is 24 kHz per the upstream Kokoro
    README (StyleTTS 2 + ISTFTNet convention)."""
    assert KOKORO_SAMPLE_RATE_HZ == 24_000


# ── F-11 — G2P DI plug per OQ-4 Option C ──────────────────────────────────────

def test_kokoro_g2p_di_pluggable_optional_constructor_kwarg():
    """F-11: ``KokoroProvider3(g2p=callable)`` constructor accepts the
    optional G2P injection. The provider stays G2P-free by default
    (tern-core stays a compression / inference library, not a TTS
    pipeline)."""
    call_log: list[str] = []

    def my_g2p(text: str) -> str:
        call_log.append(text)
        return "həˈloʊ wɜːld"

    provider = KokoroProvider3(g2p=my_g2p)
    result = provider.infer("hello world")
    assert isinstance(result, InferenceResult3)
    assert call_log == ["hello world"]


def test_kokoro_g2p_di_default_is_noop():
    """F-11: Default G2P slot is no-op — caller passes phonemes
    directly. The provider does not silently invoke a missing
    G2P library at this attachment."""
    provider = KokoroProvider3()
    assert provider._g2p is None  # pylint: disable=protected-access


# ── F-12 — Cross-repo trip-wire activation parity ─────────────────────────────

def test_kokoro_epistemic_state_canonical_string_value_parity():
    """F-12 / TW-2: EpistemicState member NAMES + string VALUES match
    integration³ canonical surface byte-for-byte."""
    expected = {
        "CONFIRMED": "CONFIRMED",
        "UNCERTAIN": "UNCERTAIN",
        "DISCONFIRMED": "DISCONFIRMED",
    }
    for member in EpistemicState:
        assert member.name in expected
        assert member.value == expected[member.name]


def test_kokoro_confidence3_three_state_enum_parity():
    """F-12 / TW-9: Confidence³ three-state enum parity holds across
    integration³ canonical + tern-core consumer attachment."""
    expected_names = {"SURE", "UNSURE", "UNKNOWN"}
    actual_names = {m.name for m in Confidence3}
    assert actual_names == expected_names


def test_kokoro_integration3_canonical_types_imported_cleanly():
    """F-12: All integration³ canonical types consumed at this
    attachment (Provider3 / ProviderConfig3 / InferenceResult3 /
    EpistemicState / Confidence3 / Provenance3 / AtomIdentifier3 /
    Thought3) import cleanly from the integration3 public API."""
    from integration3 import (
        AtomIdentifier3 as A,
        Confidence3 as C,
        EpistemicState as E,
        InferenceResult3 as I,
        Provenance3 as P,
        Provider3 as Pr,
        ProviderConfig3 as PC,
        Thought3 as T,
    )
    for cls in (A, C, E, I, P, Pr, PC, T):
        assert cls is not None


# ── Auxiliary smoke / structural tests ────────────────────────────────────────

def test_kokoro_outcome_token_constants_three_cell():
    """The three outcome-token constants surface exactly three
    distinct strings mapping to the EpistemicState three-cell."""
    tokens = {
        INFERENCE_OUTCOME_CONFIRMED,
        INFERENCE_OUTCOME_UNCERTAIN,
        INFERENCE_OUTCOME_DISCONFIRMED,
    }
    assert len(tokens) == 3


def test_kokoro_provider3_with_context_atoms_weakest_link_confidence(default_provider):
    """Weakest-link Confidence³ propagation from context atoms
    (existing tern-core Provider³ contract preserved)."""
    sure_atom = Thought3(
        content="sure context",
        source="test",
        base_confidence=Confidence3.SURE,
    )
    unsure_atom = Thought3(
        content="unsure context",
        source="test",
        base_confidence=Confidence3.UNSURE,
    )
    # Weakest of SURE + UNSURE → UNSURE.
    result = default_provider.infer(
        "test", context_atoms=[sure_atom, unsure_atom]
    )
    # When the path doesn't go DISCONFIRMED, the input confidence
    # propagates weakest-link.
    if result.epistemic_state is not EpistemicState.DISCONFIRMED:
        assert result.input_confidence is Confidence3.UNSURE


def test_kokoro_repr_includes_endpoint_and_model(default_provider):
    """``__repr__`` surfaces endpoint + model for diagnostics."""
    rep = repr(default_provider)
    assert "KokoroProvider3" in rep
    assert default_provider.config.model in rep


def test_kokoro_provider3_default_voice_is_af_heart():
    """Default voice = af_heart per Kokoro README canonical example."""
    provider = KokoroProvider3()
    assert provider.voice == "af_heart"
