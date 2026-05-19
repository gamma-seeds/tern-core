# PHASE 0 DESIGN BRIEF — Kokoro 82M tern-core Attachment via integration³ Provider³ Protocol

**Author:** CC direct (per surgeon's dispatch 2026-05-19 — single-consumer scope; sub-agent overhead not justified)
**Date:** 2026-05-19
**Scope:** First downstream consumer integration following integration³ end-to-end build closure (Wave 5 closure 2026-05-19; HEAD `67c7f74`)
**Status:** Awaiting surgeon ratification before implementation dispatch
**Working tree:** `~/synapticode/tern-core-kokoro/` (worktree on branch `feat/integration3-kokoro-82m` tracking `origin/main` at `0ac8148`; R12 sprint working tree preserved at `~/synapticode/tern-core/`)

---

## 0. Provenance and reading log

Brief drafted against:

- `INTEGRATION3_WAVE1_DISPATCH_ADDENDUM_2.md` §3 (brief structure mandate), §4 (order of operations), §5 (gate criteria pattern), §9 (halt-at-gate triggers) — all carry forward unchanged at downstream-consumer scale
- `~/synapticode/agents/integration3/integration3/provider3.py` post-Wave 5 closure (HEAD `67c7f74`) — canonical Provider³ Protocol + TernCoreProvider³ concrete adapter + InferenceResult³ frozen dataclass + ProviderConfig³ closed-set 5-field config; surfaces read at lines 130–467 inclusive
- `~/synapticode/agents/integration3/INTEGRATION3_TYPE_INDEX.md` §5.13 — provider3 4-type row enumeration (Provider³ Protocol + TernCoreProvider³ + InferenceResult³ + ProviderConfig³)
- `~/synapticode/agents/integration3/INTEGRATION3_PATENT_BRIDGE_v1.md` §4.13 — provider3 reading index (SEED Q7 rename discipline + P099 §107 EpistemicState mapping + P096 §130 Provenance³ + P096 §140 AtomIdentifier³)
- `~/synapticode/agents/integration3/phase0_briefs/2026-05-19_provider3_wave5_phase0.md` — Wave 5 ratified Phase 0 brief (822 LOC; 10 OQs; cluster A/B/C/D dispositions inline) — this brief inherits Wave 5 ratifications verbatim
- Wave 5 cluster dispositions ratified 2026-05-19 (carried forward inline):
  - **Cluster (A)** Subscriber³ DECLINE — Option B pull-by-consumer; provider3 + Kokoro attachment do NOT implement publisher3.Subscriber3
  - **Cluster (B)** Ledger³ writeback DECLINE — Option C caller-controlled; provider3 + Kokoro attachment do NOT touch decision3.Ledger³
  - **Cluster (C)** EpistemicState three-cell mapping — high-confidence → CONFIRMED; partial-evidence → UNCERTAIN; counter-evidence / exception → DISCONFIRMED (per P099 §107)
  - **Cluster (D)** Provenance³ soft-stamp — `device_signature = f"tern-core@{endpoint}"`; AtomIdentifier³ SHA-256-of-fields per Wave 2 OQ-1 canonical identity scheme
  - **Cluster (G)** sync-only `infer()` — async deferred to Phase 0.x per Wave 5 cluster G
- `~/synapticode/gs-intel/GammaSeeds_IntelNote_Kokoro82M_TernaryCompilation_Apr2026.docx` (12 April 2026; CONFIDENTIAL) — strategic framing across three vectors (KAIST NPU proof-point + Seal³ attested TTS + Korean TTS market signal); ternary compression target authoritative figure 15–20 MB from 164 MB FP16
- `~/synapticode/model-library/Kokoro-82M/` (retrieved from HuggingFace `hexgrad/Kokoro-82M` 2026-05-19; 339 MB total on disk; SHA256 `496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4`) — kokoro-v1_0.pth (327.2 MB main weights) + config.json + README.md + EVAL.md + VOICES.md + 54 voice embeddings
- `~/synapticode/model-library/Kokoro-82M/config.json` — full architecture spec (StyleTTS 2 derived; ISTFTNet vocoder; plbert text encoder 768/12/12; main acoustic stack hidden_dim 512 / n_layer 3; n_token 178 IPA vocabulary; 80 mel channels; multispeaker 54 voices × 128 style_dim)
- `~/synapticode/model-library/Kokoro-82M/README.md` — architecture references (StyleTTS 2 https://arxiv.org/abs/2306.07691; ISTFTNet https://arxiv.org/abs/2203.02395); inference pattern (`kokoro` PyPI package + misaki G2P)
- `~/synapticode/tern-core-kokoro/src/terncore/adapters/` — existing 5-adapter cohort (base.py + gemma3.py + gemma4.py + llama.py + phi3.py + qwen3_moe.py)
- `~/synapticode/tern-core-kokoro/CLAUDE.md` — repo-level architecture document (mixed ternary/INT4 with layer-sensitivity routing per v0.6.0+)
- Memory: `pattern_cross_wave_protocol_stub_v1_taxonomy_v2.md` + `pattern_dual_direction_dissolution_lifecycle_v1.md` + `pattern_cross_wave_protocol_stub_v1_boundary_bidirectional_sibling_rank_prerequisite_v1.md` (all banked at Wave 5 closure)

---

## 1. Surface-by-surface scope

This is a **consumer attachment**, not a Type Index surface extension. Zero new Type Index types land at this attachment. The attachment exercises existing integration³ surfaces:

| Integration³ surface | Status | Attachment role |
|---|---|---|
| `integration3.Provider3` Protocol (§5.13 row 1) | EXISTING (Wave 5) | KokoroProvider3 satisfies structurally |
| `integration3.TernCoreProvider3` (§5.13 row 2) | EXISTING (Wave 0 + Wave 5 ext.) | Reference precedent; Kokoro is sibling adapter via local-direct inference path, not HTTP |
| `integration3.InferenceResult3` (§5.13 row 3) | EXISTING (Wave 0 + Wave 5 ext.) | KokoroProvider3 produces; epistemic_state + provenance + atom_identifier populated per cluster C/D |
| `integration3.ProviderConfig3` (§5.13 row 4) | EXISTING (Wave 5) | KokoroProvider3 consumes; closed-set 5-field shape (endpoint repurposed as model-path; model = "kokoro-82m") |
| `integration3.EpistemicState` (thought3 §5.1 row 19) | EXISTING (Wave 2 canonical) | Three-cell mapping per cluster C |
| `integration3.Provenance3` (thought3 §5.1 row 4) | EXISTING (Wave 2) | Soft-stamp per cluster D |
| `integration3.AtomIdentifier3` (thought3 §5.1 row 5) | EXISTING (Wave 2 OQ-1) | SHA-256-of-fields identity scheme |

### 1.1 New tern-core surfaces (Option B scaffolding ratified)

Three new modules at `src/terncore/integration3/` per surgeon's Option B ratification 2026-05-19:

| Surface | File | Role |
|---|---|---|
| `IntegratedProvider3Base` | `src/terncore/integration3/base.py` | Base class implementing canonical disposition wiring (EpistemicState mapping + Provenance³ stamp + AtomIdentifier³ identity); wraps a tern-core inference path via dependency injection; concrete subclasses bind specific models |
| `KokoroProvider3` | `src/terncore/integration3/kokoro.py` | First canonical Provider³-satisfying instance; wraps Kokoro 82M inference (StyleTTS 2 + ISTFTNet); first downstream consumer integration |
| `__init__.py` exports | `src/terncore/integration3/__init__.py` | Public surface exposing `IntegratedProvider3Base` + `KokoroProvider3` |

### 1.2 New tern-core adapter (location decision; see OQ-1)

A Kokoro-aware adapter handling weight extraction from `kokoro-v1_0.pth` + layer-sensitivity routing for mixed ternary/INT4 compression. Location dispositioned at OQ-1 (sibling to existing 5-adapter cohort at `adapters/kokoro.py` vs local-to-integration3 at `integration3/kokoro_adapter.py`).

---

## 2. Design decisions

### 2.1 IntegratedProvider3Base — concrete class vs ABC vs Protocol-only

#### Option A — concrete base class with template-method pattern

`IntegratedProvider3Base` ships as a concrete class with:
- `__init__(config: ProviderConfig3, adapter: <KokoroAdapter | future siblings>)` constructor
- `config` property returning `self._config`
- `health_check()` + `is_available()` default implementations (lazy adapter probe)
- `infer(input_text, input_atoms, input_confidence)` template method calling `self._adapter.run_inference(...)` + applying cluster C/D wiring (EpistemicState mapping + Provenance³ stamp + AtomIdentifier³ compute) + producing `InferenceResult3`

Concrete subclasses (KokoroProvider3 + future MLXProvider3 + ...) override hook points: `_run_inference_path()` + `_map_outcome_to_epistemic_state()` + adapter wiring at construction.

**Pro:** template-method pattern centralises canonical disposition wiring (cluster C/D) so each concrete provider satisfies cluster ratifications by inheritance; reduces drift surface; analogous to `seal3.Seal3` umbrella facade pattern from Wave 1 closure.
**Con:** introduces inheritance hierarchy in tern-core (existing adapter cohort uses base class + concrete adapters pattern, so the convention is consistent).

#### Option B — abstract base (ABC) with required overrides

Same shape as Option A but `IntegratedProvider3Base` declared `abc.ABC` with `@abstractmethod` on `_run_inference_path()`. Forces concrete subclasses to implement the inference hook explicitly.

**Pro:** static enforcement of subclass contract.
**Con:** Python's @runtime_checkable Protocol satisfaction already handles the "must satisfy Provider3" contract structurally; @abstractmethod adds dynamic-only enforcement at instantiation time. Marginal value.

#### Option C — Protocol-only with helper functions

Drop `IntegratedProvider3Base` entirely; ship helper functions (`stamp_provenance` + `compute_atom_identifier` + `map_to_epistemic_state`) that concrete providers compose into their own `infer()` methods.

**Pro:** maximum flexibility; minimal scaffolding.
**Con:** disposition wiring becomes per-provider boilerplate; cluster C/D drift surface grows with each new provider.

#### Recommended: Option A (concrete base class with template-method pattern)

Patent / SEED anchor — SEED Q7 rename discipline + Build Plan §3.5 line 211 substitutability invariant. The substitutability invariant ("future MockProvider³ for tests; future MLXProvider³ for direct-on-device inference") requires canonical disposition wiring at the base layer so substitution preserves cluster C/D contracts. Template-method pattern centralises this exactly once.

Concrete `IntegratedProvider3Base` matches existing tern-core adapter convention (`adapters/base.py` is a similarly-shaped concrete base for the model-adapter cohort).

### 2.2 Kokoro adapter location (sibling cohort vs integration3-local)

The Kokoro-aware adapter handles weight extraction from `kokoro-v1_0.pth` + layer-sensitivity routing per the mixed ternary/INT4 v0.6.0+ pipeline. Location options:

#### Option A — sibling to existing 5-adapter cohort

Lands at `src/terncore/adapters/kokoro.py` alongside `base.py + gemma3.py + gemma4.py + llama.py + phi3.py + qwen3_moe.py`. Discoverable via existing `adapters/__init__.py` patterns; uses the established `AdapterInfo` + `WeightClassification` registry shape from Group A PR #8 (Sessions 4-8).

**Pro:** consistent with existing tern-core convention; adapter cohort stays the canonical "model knowledge" location; KokoroProvider3 imports from `terncore.adapters.kokoro` exactly like the cohort precedents.
**Con:** Kokoro is a TTS architecture (StyleTTS 2 + ISTFTNet) — structurally different from the LLM cohort (llama/gemma/phi3/qwen3). The base class + AdapterInfo registry may need extension for TTS-specific concerns (mel spectrogram input; ISTFTNet vocoder output; voice embedding handling).

#### Option B — local to integration3 directory

Lands at `src/terncore/integration3/kokoro_adapter.py` co-located with KokoroProvider3. Stays self-contained within the integration³ attachment scope.

**Pro:** clean separation between the LLM-focused adapter cohort and TTS attachment work; no risk of base class drift to accommodate TTS-specific concerns.
**Con:** divergent convention; future TTS adapters (additional voice models) would need to decide whether to sit at adapter cohort level or stay in integration3 sub-directory; risk of forking the adapter pattern.

#### Recommended: Option A (sibling cohort at `adapters/kokoro.py`)

Rationale — the existing adapter cohort base.py is already extensible (Group A PR #8 schema widening with `architectures: list[str]` and `expert_pattern` and `attention_type_pattern` per A3). Adding TTS-specific fields to `AdapterInfo` (e.g., `vocoder_pattern: Optional[str] = None` + `acoustic_stack_pattern: Optional[str] = None`) extends the base cleanly under the established conservative-allow-list discipline. The adapter cohort stays the canonical model-knowledge location; future TTS adapters (additional voice models per Korean TTS market signal) inherit from the same base.

Surfaced as **OQ-1 halt-at-gate**: surgeon disposition gates implementation start.

### 2.3 Voice embedding compression tier — ternary vs INT4 vs FP16 (per-voice that ships)

Kokoro 82M ships 54 voice embeddings (.pt files, 0.52 MB each = ~28 MB total at FP16). These are speaker-style embeddings (style_dim=128) consumed by the StyleTTS 2 acoustic stack as conditioning. The compression tier choice applies per-voice; the voice-subset cardinality decision (which voices ship at all) is a separate concern dispositioned at §2.10 below.

#### Option A — voice embeddings stay FP16 (no compression)

Matches v0.6.0+ pipeline's "FP16 for embeddings, norms, and lm_head" pattern. Voice embeddings are functionally analogous to LLM token embeddings — small, sensitive, used as starting representations. Each voice that ships occupies 0.52 MB FP16.

**Pro:** preserves speaker identity fidelity (load-bearing user-visible characteristic of Kokoro); matches existing v0.6.0+ embedding policy; couples cleanly with the demo-subset disposition (§2.10) — small voice count × 0.52 MB FP16 stays within the OQ-6 footprint band.
**Con:** at the full 54-voice catalogue, FP16 sums to ~28 MB additive overhead — handled at §2.10 subset disposition rather than at the compression tier.

#### Option B — voice embeddings INT4

Apply INT4 quantisation per voice (~0.13 MB each; ~7 MB total at full 54-voice catalogue).

**Pro:** smaller per-voice cost; full catalogue stays under 10 MB voice overhead.
**Con:** speaker identity may degrade audibly; voice embeddings are arguably MORE sensitive than mid-stack LLM weights (they encode the speaker's full spectral character); degrades the KAIST NPU strategic-vector demonstration.

#### Option C — voice embeddings ternary

Apply ternary quantisation per voice (~0.03 MB each; ~1.5 MB total at full 54-voice catalogue).

**Pro:** smallest footprint.
**Con:** very likely to break speaker identity; high regression risk.

#### Recommended: Option A (voice embeddings stay FP16) coupled with §2.10 subset shipping

Patent / pipeline anchor — v0.6.0+ "FP16 for embeddings, norms, and lm_head" canonical pipeline policy. Speaker embeddings are functionally embeddings. Audio output fidelity is the load-bearing demonstration characteristic for KAIST NPU proof-point vector — degradation here breaks the strategic value.

OQ-2 compression-tier disposition (per-voice) is **coherent only when paired with the §2.10 subset-shipping decision** (which voices ship at all). The footprint composition math is shown explicitly at §2.7.

Surfaced as **OQ-2 halt-at-gate**.

### 2.4 ISTFTNet vocoder compression scope — full ternary vs selective INT4

The ISTFTNet vocoder is convolutional (3 ResBlocks with dilation [1,3,5] + kernel sizes [3,7,11], upsample rates [10,6], FFT 20, hop size 5, upsample_initial_channel 512). Vocoder weights drive the final mel→audio conversion. Compression scope options:

#### Option A — ISTFTNet full ternary (matching main acoustic stack)

Apply ternary to all ISTFTNet convolutional weights. Maximum compression.

#### Option B — ISTFTNet ResBlocks ternary + upsample layers INT4

Layer-sensitivity routing: ResBlock dilated convs ternary-tolerant (analogous to mid-stack LLM weights); upsample layers (initial channel 512 → audio) INT4 (analogous to "sensitive" pipeline category).

#### Option C — ISTFTNet INT4 throughout

Conservative: vocoder weights are arguably more sensitive than text-encoder weights since they directly produce audio output.

#### Recommended: Option B (layer-sensitivity routing)

Anchor — tern-core v0.6.0+ mixed ternary/INT4 with layer-sensitivity routing is the canonical compression discipline. The vocoder's mixed sensitivity profile (dilated convs tolerant; upsample/output layers sensitive) suits the established routing approach.

Surfaced as **OQ-3 halt-at-gate**: brief author proposes Option B; surgeon ratifies. May refine at implementation time if empirical layer-sensitivity analysis (autoscan/sensitivity pipeline) surfaces specific layer signals.

### 2.5 Phoneme tokeniser handling (misaki G2P)

Kokoro 82M consumes phoneme sequences (IPA, n_token=178), not raw text. The `kokoro` PyPI package wraps a G2P (grapheme-to-phoneme) library called `misaki` (https://github.com/hexgrad/misaki). Options:

#### Option A — misaki G2P in scope at Wave 5 closure

KokoroProvider3.infer() accepts raw text input; internally invokes misaki G2P to produce phoneme sequence; runs Kokoro acoustic stack + ISTFTNet to produce audio. Self-contained TTS pipeline.

**Pro:** consumer-friendly surface (raw text in; audio out); demo-ready.
**Con:** adds a third-party dependency (misaki) to tern-core's integration³ attachment; tern-core stays G2P-free as a core principle.

#### Option B — misaki G2P out-of-scope; phoneme input expected

KokoroProvider3.infer() accepts phoneme sequence as input (encoded per the n_token=178 IPA vocabulary in config.json). Callers responsible for G2P.

**Pro:** tern-core stays G2P-free; integration3 attachment scope minimal; caller picks G2P implementation.
**Con:** demo path requires caller to set up misaki separately; consumer-attachment surface less obvious.

#### Option C — misaki G2P optional via dependency-injection

KokoroProvider3 accepts an optional G2P function in constructor; default is no-op (expects phoneme input); test/demo paths can inject misaki.

**Pro:** flexibility; tern-core stays G2P-free at the core level; consumers can inject what they need.
**Con:** slight constructor surface bloat.

#### Recommended: Option C (DI-injected G2P)

Anchor — tern-core stays a compression / inference library, not a TTS pipeline; misaki is a separate concern. DI matches the established adapter-pattern dependency-injection posture (per Wave 5 cluster A architectural sharpening: provider3 is an adapter, not a faculty).

Surfaced as **OQ-4 halt-at-gate**: surgeon disposition. Brief author leans Option C; Option A is a viable demo-path alternative.

### 2.6 Health check load timing — eager vs lazy

`Provider3.health_check()` returns `bool`. Two implementation timings:

#### Option A — eager: load Kokoro 82M weights at `__init__()`

`KokoroProvider3.__init__(config)` loads `kokoro-v1_0.pth` immediately; `health_check()` returns `self._model is not None`.

**Pro:** simple; deterministic failure at construction time; matches "fail fast" discipline.
**Con:** 327 MB load at construction; slow tests; bridge3 IPC consumers attaching multiple providers serially incur cumulative load latency.

#### Option B — lazy: defer load until first `infer()`

`__init__()` stores config only; first `infer()` call triggers model load; `health_check()` probes config validity (model path exists + readable).

**Pro:** fast construction; lighter test surface; aligns with deferred-evaluation patterns used elsewhere (cell3 DeferredBlockingList3 precedent at Wave 3).
**Con:** first-inference latency is high (~1–2s for model load + inference); `health_check()` truthfulness about availability is shallower.

#### Recommended: Option B (lazy load) with eager-load opt-in

Default to lazy; constructor accepts optional `eager_load: bool = False` kwarg for eager loading (consumers that want construction-time validation opt in explicitly).

Surfaced as **OQ-5 (non-halt; refinement within §6 envelope)** for implementation-time refinement.

### 2.7 Ternary footprint tolerance band — composition math

Intel note authoritative figure (15–20 MB compressed from 164 MB FP16) refers to the **compressed main-model footprint** (kokoro-v1_0.pth ternary-compiled), NOT total deployable footprint including voices. The intel-note projection is based on the Mistral-7B compression ratio (96.4% ternary; 14.5 GB → 2.27 GB at ~1.58 bits/parameter). Kokoro 82M has architectural differences (StyleTTS 2 + ISTFTNet vs decoder-only LLM); layer-sensitivity routing decisions (§2.3 + §2.4) materially affect achievable main-model footprint.

**Total deployable footprint = main-model (ternary) + shipped-voices (FP16 per §2.3) + ISTFTNet vocoder (per §2.4)**.

Composition under brief recommendations:

| Component | Recommendation | Footprint |
|---|---|---|
| Main acoustic stack (ternary) | §2.4 layer-sensitivity routing | ~15–20 MB |
| Voice embeddings (FP16) | §2.3 Option A; §2.10 subset N | N × 0.52 MB |
| ISTFTNet vocoder | §2.4 Option B routing (ResBlocks ternary + upsample INT4) | included in main-model figure |
| Sum (demo subset N=4) | — | ~17–22 MB |
| Sum (demo subset N=8) | — | ~19–24 MB |
| Sum (full catalogue N=54) | — | ~43–48 MB |

Brief author proposes **≤22 MB ceiling** as acceptance criterion for the **demo-shipped artefact** (KAIST NPU proof-point path; intel-note 15–20 MB main-model band + ~2–4 MB voice subset). Falsifier F-8 codifies against this ceiling.

Full 54-voice catalogue deployment is **out of OQ-6 ceiling scope** — it's a deployment-time configuration decision (§2.10 disposition: subset for demo; full catalogue available via voice-by-voice dynamic loading per consumer's deployment posture).

Surfaced as **OQ-6 halt-at-gate**: surgeon ratifies the ceiling band scope (demo-shipped artefact ≤22 MB vs total deployment with all voices vs other).

### 2.8 Sync vs async `infer()` (reaffirms Wave 5 cluster G)

Wave 5 cluster G ratification: sync-only `infer()`; async deferred to Phase 0.x. This brief inherits the ratification verbatim. No new dispatched OQ at Wave 5-derivative scope.

Falsifier F-1 verifies Provider³ Protocol satisfaction (synchronous `infer()` signature matches).

### 2.10 Voice subset shipping cardinality (build-time vs deployment-time)

The 54-voice catalogue × 0.52 MB FP16 = ~28 MB voice overhead — overshoots the OQ-6 ≤22 MB ceiling on its own. The compression-tier disposition (§2.3 Option A FP16) does not resolve the cardinality question: how many voices ship in the demo artefact vs deployment artefact?

Three resolution shapes per surgeon's clarification 2026-05-19:

#### Option A — demo-subset shipping (4–8 voices at FP16; full catalogue deferred)

Demo path ships a curated voice subset (e.g., 4–8 voices spanning English / non-English coverage); ~2–4 MB voice budget. Full 54-voice catalogue stays in `~/synapticode/model-library/Kokoro-82M/voices/` available for deployment-time configuration (consumers' KokoroProvider3 instances load specific voices on demand).

**Demo subset proposal** (illustrative; surgeon may refine at ratification):
- `af_heart` (English-American female; demo canonical per Kokoro README example)
- `am_adam` (English-American male)
- `bf_alice` (English-British female)
- `bm_daniel` (English-British male)
- `jf_alpha` (Japanese female; non-English coverage)
- `zf_xiaobei` (Chinese female; non-English coverage)

6 voices × 0.52 MB FP16 = ~3.1 MB voice budget. Combined with main-model 15–20 MB: ~18–23 MB total. Within OQ-6 ceiling.

**Pro:** matches intel-note KAIST proof-point strategic vector (small demonstrable artefact); preserves voice fidelity for the demo voices; full catalogue stays available at filesystem for deployment-time selection without re-compilation.
**Con:** demo artefact ≠ deployment artefact; surface for distinguishing "compiled set" vs "available set" at the KokoroProvider3 + adapter layer.

#### Option B — voice subset cardinality controlled at ProviderConfig3 layer

KokoroProvider3 loads voices specified in config (e.g., `ProviderConfig3.model = "kokoro-82m"` + new disposition for voice selection). Default config loads demo subset; consumers can override to load any voice subset including full 54.

**Pro:** flexibility; one artefact serves both demo and deployment paths.
**Con:** ProviderConfig3 is closed-set (Wave 5 OQ-3 ratification); voice selection field would need to land at the KokoroProvider3 layer rather than ProviderConfig3 — `voice: str` parameter on `infer()` or sibling field on a KokoroProvider3-specific config wrapper.

#### Option C — full-catalogue compiled (raises OQ-6 ceiling)

All 54 voices ship in the compiled artefact at FP16 (~28 MB). Total footprint ~43–48 MB. Diverges from intel-note 15–20 MB target framing; KAIST NPU proof-point strategic vector compromised.

**Pro:** simplest implementation; consumers get the full catalogue out of the box.
**Con:** breaks the demo footprint claim; OQ-6 ceiling needs revision to ~50 MB.

#### Recommended: Option A (demo-subset shipping; full catalogue deferred to deployment-time configuration)

Per surgeon's instinct 2026-05-19 + intel-note KAIST proof-point strategic vector. Demo path is the load-bearing near-term use; full-catalogue deployment is a downstream concern handled via Kokoro 82M's filesystem layout (voices live as separate .pt files; loading is per-voice; the compiled main-model + demo-subset voice bundle is a discrete deliverable).

Surfaced as **OQ-10 halt-at-gate**: surgeon disposes demo-subset cardinality (which voices? how many? which language coverage?) and the compile-time vs runtime boundary (do voices outside the demo subset get included in the compiled artefact at all, or just stay as filesystem-accessible .pt files for deployment-time loading?).

---

## 3. Cross-module interface declarations

### 3.1 integration³ canonical surfaces consumed

KokoroProvider3 + IntegratedProvider3Base import:
- `from integration3 import Provider3, ProviderConfig3, InferenceResult3` (Wave 5 canonical exports)
- `from integration3 import Confidence3, EpistemicState, Thought3, Provenance3, AtomIdentifier3` (Wave 2 canonical)
- (Optional) `from integration3 import FacultyId3` for `ProviderConfig3.faculty_id` field if cross-faculty attribution is in scope

### 3.2 integration³ canonical surfaces NOT consumed (cluster A + B declines)

KokoroProvider3 + IntegratedProvider3Base do NOT import:
- `from integration3 import Subscriber3` — cluster A DECLINE (provider3 is adapter, not faculty consumer)
- `from integration3 import Publisher3` — adapters do not emit events
- `from integration3 import Ledger3, LedgerEntry3, make_bootstrap_ledger_entry` — cluster B DECLINE (caller-controlled writeback)
- `from integration3 import Seal3` — Seal³ attestation is caller-side per intel note strategic vector 2 (callers seal Kokoro inference results via integration3.seal3, not provider3 itself)

### 3.3 tern-core internal surfaces consumed

- `terncore.adapters.kokoro` (per OQ-1 Option A) or local `kokoro_adapter` (per OQ-1 Option B)
- `terncore.compression` pipeline entry points (`convert.py` + `autoscan.py` per existing v0.6.0+ pipeline; verify exact import surface at implementation time)
- `terncore.adapters.base.AdapterInfo` + `WeightClassification` (per Group A PR #8 schema)

### 3.4 Existing `integration3.TernCoreProvider3` relationship

KokoroProvider3 is a **sibling** to integration3.TernCoreProvider3 — both satisfy Provider³ Protocol. Differences:
- TernCoreProvider3: HTTP-based inference (calls tern-core endpoint over network)
- KokoroProvider3: local-direct inference (loads Kokoro 82M weights in-process; no network)

Consumers choose based on use case (HTTP for service-mesh deployment; local-direct for embedded / on-device / demo). Both satisfy the same Provider³ contract per Wave 5 substitutability invariant.

### 3.5 Cross-repo trip-wire activation

Wave 4 + Wave 5 trip-wire arms transitioning SKIPPED → LIVE at this attachment:

| TW | Stub location | Arm transition | Rationale |
|---|---|---|---|
| TW-2 | `tests/test_cross_repo_invariants.py::test_epistemic_state_string_match` | LIVE (was integration3-only LIVE since Wave 2; cross-repo arm activates when tern-core consumes EpistemicState canonical) | tern-core (this attachment) imports `integration3.EpistemicState`; member NAMES + string VALUES checked against canonical |
| TW-9 | `test_phi_can_three_state_enum_parity` | LIVE (similar; φ_can three-state enum parity now extends to tern-core consumption) | Confidence³ + EpistemicState consumed live |

Per Wave 5 cluster (G) directive: brief author identifies which trip-wires activate. Brief proposes the 2 above; surgeon ratifies or adds.

**Surfaced as OQ-7 halt-at-gate**: surgeon disposes the trip-wire activation set.

### 3.6 No new Protocol stubs (cluster H pattern boundary)

Per cluster H ratification (negative disposition at Wave 5 closure): pattern_cross_wave_protocol_stub_v1 applies only at bidirectional sibling-rank parallel implementation. KokoroProvider3 → integration³ is unidirectional consumption — direct import works. No Protocol stub at this attachment.

---

## 4. Falsifier statements

**F-1 (Provider³ Protocol satisfaction).** `isinstance(KokoroProvider3(config), Provider3)` returns True via `@runtime_checkable` structural match. The four-method surface (config / health_check / is_available / infer) is satisfied. Falsifies if KokoroProvider3 misses any Protocol method.

**F-2 (EpistemicState three-cell mapping per P099 §107).** For a KokoroProvider3 inference call producing non-empty audio output, `InferenceResult3.epistemic_state is EpistemicState.CONFIRMED`. For an inference call returning empty/silence output (stub-fallback path), `EpistemicState.UNCERTAIN`. For an inference call raising an exception (model load failure / G2P failure), `EpistemicState.DISCONFIRMED`. Test: `test_kokoro_epistemic_state_three_cell_mapping` enumerates all three branches.

**F-3 (Provenance³ soft-stamp).** `InferenceResult3.provenance.device_signature == f"tern-core@{endpoint}"` where endpoint is `KokoroProvider3.config.endpoint` (or model-path string per OQ-1). Falsifies if soft-stamp shape drifts. Anchor: P096 §130; Wave 5 cluster D.

**F-4 (AtomIdentifier³ SHA-256-of-fields).** `InferenceResult3.atom_identifier == AtomIdentifier3.compute(value=text, confidence=output_atom.base_confidence, provenance=provenance)`. 64-char hex; deterministic; mutation-sensitive across all three input fields. Anchor: P096 §140; Wave 2 OQ-1.

**F-5 (NO Subscriber³ Protocol satisfaction).** `isinstance(KokoroProvider3(config), integration3.Subscriber3)` returns False. KokoroProvider3 + IntegratedProvider3Base do NOT define `on_event` method. Cluster A architectural rule enforced: adapters do not satisfy Subscriber³.

**F-6 (NO Ledger³ writeback).** Static-analysis check: `grep -rn "Ledger3\|LedgerEntry3\|make_bootstrap_ledger_entry" src/terncore/integration3/` returns zero matches. KokoroProvider3 + IntegratedProvider3Base do NOT import decision3 surfaces. Cluster B architectural rule enforced.

**F-7 (Kokoro 82M model loads cleanly).** `KokoroProvider3(config).health_check() is True` once model loaded (eager or after first lazy load). Underlying `kokoro-v1_0.pth` deserialises without exception. Audio inference produces non-empty mel spectrogram (assertion via `numpy.ndarray.shape[0] > 0`).

**F-8 (Ternary footprint within target band — demo-shipped artefact).** Post-compression Kokoro 82M demo artefact (main model + demo-subset voices per §2.10) footprint ≤ 22 MB (per OQ-6 proposed ceiling: 15–20 MB intel-note main-model target + ~2–4 MB demo-subset voice budget). Measured via tern-core compression pipeline output size + shipped voice .pt files. Falsifies if demo artefact misses the band. Full-catalogue deployment footprint is out of F-8 scope (§2.10 OQ-10 disposition).

**F-9 (Voice embedding handling — 54 voices accessible).** All 54 voice .pt files loadable; KokoroProvider3 exposes voice selection per `ProviderConfig3.model` or a dedicated `voice` field (OQ refinement). Audio output character changes audibly across at least 3 voice selections (af_heart, am_adam, bf_alice as canonical test triple).

**F-10 (Audio output structurally valid).** KokoroProvider3.infer(text="hello world") produces `InferenceResult3` with `output_atom.content` containing audio data (numpy float array; sample rate 24000 Hz per Kokoro README; duration >0).

**F-11 (Phoneme tokeniser interface — per OQ-4 disposition).** If OQ-4 Option C ratified (DI G2P): KokoroProvider3 accepts an optional `g2p: Callable[[str], list[int]]` constructor argument; defaults to a no-op assuming phoneme input. Test: `test_kokoro_g2p_di_pluggable` verifies the optional G2P injection path.

**F-12 (Cross-repo trip-wire activation parity).** Wave 4 + Wave 5 trip-wire arms identified in §3.5 activate when tern-core integration3 module loads. EpistemicState + Confidence³ member NAMES + string VALUES match integration³ canonical surface. Verified via direct import + introspection.

---

## 5. Test plan

### 5.1 New test file

`~/synapticode/tern-core-kokoro/tests/test_integration3_kokoro.py` — new file. Tests F-1 → F-12 above. Estimated 400–550 LOC.

### 5.2 Test fixtures

- Module-scope fixture loading Kokoro 82M weights once (eager) for tests that require live inference
- Test-scope fixture using lazy-load to verify the deferred path
- Mock provider fixture exercising IntegratedProvider3Base independent of Kokoro-specific concerns

### 5.3 Test naming discipline (per integration³ Wave 1 Dispatch Addendum 2 §3 element 3)

Vocabulary disambiguation in test names — three-tier φ_can discipline preserved:
- `test_kokoro_confidence_routing_tier_propagation` (Confidence³ routing-tier)
- `test_kokoro_epistemic_state_inference_tier_three_cell` (EpistemicState inference-tier)
- `test_kokoro_no_subscriber3_protocol_satisfaction` (cluster A architectural enforcement)
- `test_kokoro_no_ledger3_writeback` (cluster B architectural enforcement)

### 5.4 Existing test baseline preserved

`~/synapticode/tern-core-kokoro/tests/` existing suite passes unchanged. The new `tests/test_integration3_kokoro.py` is additive; no existing tern-core test file modified.

### 5.5 Cross-repo trip-wire activation tests

If OQ-7 ratifies the proposed activation set (TW-2 + TW-9 cross-repo arms), this brief's implementation may **not** edit integration³'s `tests/test_cross_repo_invariants.py` directly (integration³ at 67c7f74 is at rest per build-closure stand-down). Instead:
- A new tern-core-local test `tests/test_cross_repo_invariants_kokoro_arm.py` verifies the integration3 canonical surfaces import cleanly + EpistemicState/Confidence³ member parity holds at this attachment
- Future integration³ touch (housekeeping commit) may activate the upstream TW-N stubs' cross-repo arms post-attachment

Surfaced for surgeon ratification at OQ-7.

### 5.6 Forecast

Baseline (per tern-core current main): exact count to verify at implementation start (estimated 391+ tests post Group A PR #8 closure). Target post-attachment: baseline + 12–18 new unit tests. No regressions in existing tern-core test surface.

---

## 6. No-breakage constraints

- **integration³ at 67c7f74 unchanged.** No edits to `~/synapticode/agents/integration3/` files. This attachment is downstream-consumer only.
- **tern-core existing adapter cohort preserved exact-shape.** `adapters/base.py + gemma3.py + gemma4.py + llama.py + phi3.py + qwen3_moe.py` unchanged. If OQ-1 Option A ratified, `adapters/kokoro.py` lands as a sibling addition with `AdapterInfo` extensions (additive fields with defaults; no existing-adapter field removal).
- **tern-core CLI + harness paths preserved.** `src/terncore/api.py` + `src/terncore/convert.py` + `src/terncore/autoscan.py` public surfaces preserved exact-shape; KokoroProvider3 + IntegratedProvider3Base extend without modifying.
- **R12 sprint working tree untouched.** Worktree posture (this attachment lands on `feat/integration3-kokoro-82m` branch off main; R12 work stays on `feat/r12-eval-loop-instrumentation-and-mps-mitigation-2026-05-18`) preserved.
- **`integration3.TernCoreProvider3` not touched.** KokoroProvider3 is a sibling Provider3-satisfying class; the existing TernCoreProvider3 HTTP path stays canonical for service-mesh use cases.

---

## 7. Open questions

### Halt-at-gate items (surgeon disposition required before implementation)

**OQ-1 — Kokoro adapter location (cluster 2.2)**. Sibling cohort `adapters/kokoro.py` (Option A; brief recommended) vs integration3-local `integration3/kokoro_adapter.py` (Option B). Halt-at-gate: choice affects AdapterInfo schema extension scope + future TTS adapter convention.

**OQ-2 — Voice embedding compression tier per shipped voice (cluster 2.3)**. FP16 (Option A; brief recommended) vs INT4 (Option B) vs ternary (Option C). Halt-at-gate: audio fidelity (KAIST NPU strategic vector). Couples with OQ-10 voice subset cardinality + OQ-6 footprint band.

**OQ-3 — ISTFTNet vocoder compression scope (cluster 2.4)**. Full ternary (Option A) vs ResBlocks ternary + upsample INT4 (Option B; brief recommended) vs full INT4 (Option C). Halt-at-gate: layer-sensitivity routing decision affects F-8 footprint + audio fidelity.

**OQ-4 — Phoneme tokeniser scope (cluster 2.5)**. misaki G2P in-scope (Option A) vs out-of-scope (Option B) vs DI-injected (Option C; brief recommended). Halt-at-gate: tern-core core-principle (G2P-free) vs demo-path consumer surface.

**OQ-6 — Ternary footprint tolerance band — demo artefact scope (cluster 2.7)**. Strict 15–20 MB main-model only (intel-note authoritative) vs ≤22 MB ceiling for demo artefact (brief recommended: main 15–20 MB + demo-subset voices 2–4 MB) vs broader ceiling for full-catalogue deployment. Halt-at-gate: F-8 acceptance criterion + scope (demo artefact vs deployment).

**OQ-7 — Cross-repo trip-wire activation set (cluster 3.5)**. TW-2 + TW-9 cross-repo arms activate at this attachment (brief recommended) vs broader set (any other applicable arms) vs deferred (no activation at this attachment; future housekeeping). Halt-at-gate: defines drift-defence boundary growth at this milestone.

**OQ-10 — Voice subset shipping cardinality + compile-time/runtime boundary (cluster 2.10)**. Demo-subset 4–8 voices ship in compiled artefact (Option A; brief recommended; full catalogue deferred to filesystem-loaded deployment-time configuration) vs ProviderConfig3-controlled cardinality (Option B) vs full 54-voice catalogue compiled (Option C; raises OQ-6 ceiling to ~50 MB). Halt-at-gate: defines OQ-6 footprint composition + demo artefact shape + future deployment configurability. Couples tightly with OQ-2 + OQ-6.

### Non-halt OQs (refinable within §6 envelope at implementation time)

**OQ-5 — Health check load timing (cluster 2.6)**. Lazy with eager opt-in (brief recommended). Implementation-time refinement.

### Additional open items (surfaced for awareness; not blocking)

- **OQ-8 — Voice selection surface shape (runtime per-call)**. Voice selection happens at `infer()` time? Per-provider-instance via `ProviderConfig3.model = "kokoro-82m-af_heart"`? Per-call via additional `voice: str` parameter on infer()? Implementation-time disposition; defaults to per-call `voice: str` parameter on a KokoroProvider3 method (since ProviderConfig3 is closed-set per Wave 5 OQ-3 ratification). Couples with OQ-10 (only voices in the compiled subset selectable at runtime).
- **OQ-9 — Audio output shape on InferenceResult3.output_atom.content**. Numpy float array? Bytes blob? AudioSegment-style structured? Implementation-time disposition; brief proposes numpy float array at 24000 Hz per Kokoro README convention.

---

## 8. Estimated LOC

| Surface | Path | LOC estimate |
|---|---|---|
| `IntegratedProvider3Base` | `src/terncore/integration3/base.py` | 180–260 (template-method + cluster C/D wiring + helpers) |
| `KokoroProvider3` | `src/terncore/integration3/kokoro.py` | 220–340 (Provider3 satisfaction + Kokoro inference dispatch + voice handling) |
| Kokoro adapter (OQ-1 dependent) | `src/terncore/adapters/kokoro.py` (Option A) OR `src/terncore/integration3/kokoro_adapter.py` (Option B) | 180–280 (weight extraction + layer-sensitivity routing + AdapterInfo registration) |
| `__init__.py` extensions | `src/terncore/integration3/__init__.py` + (Option A) `src/terncore/adapters/__init__.py` | 10–20 |
| Test file | `tests/test_integration3_kokoro.py` | 400–550 (12 falsifier-anchored tests + fixtures + helpers) |
| Cross-repo arm test | `tests/test_cross_repo_invariants_kokoro_arm.py` | 60–120 |
| **Total** | — | **~1050–1570 new LOC** |

Forecast: smaller than integration³ Wave 5 provider3 brief implementation (~840–1140 LOC) due to single-consumer scope; larger than integration³ Wave 5 bridge3 due to TTS-specific surface plus Kokoro adapter authoring.

---

*End of brief.*

*Awaiting surgeon ratification per Addendum 2 §4 order-of-operations: brief → ratification → implementation → tests → bank as canonical LedgerEntry³ record (first downstream-consumer ledger entry; bootstraps post-build ledger continuity per surgeon's dispatch §5).*
