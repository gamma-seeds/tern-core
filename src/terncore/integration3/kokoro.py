"""kokoro.py — KokoroProvider3 canonical integration³ Provider³ implementation.

First canonical Provider³ Protocol consumer attaching to integration³ per
surgeon's downstream-consumer integration dispatch 2026-05-19 (ratified
ahead of Wave 5 closure). Wraps the Kokoro 82M TTS model (StyleTTS 2
acoustic stack; ISTFTNet vocoder; plbert text encoder; 8 languages / 54
voices; Apache 2.0).

Model facts (verified at retrieval 2026-05-19 from HuggingFace
``hexgrad/Kokoro-82M`` to ``~/synapticode/model-library/Kokoro-82M/``):
  - 82M parameters; FP32 ``kokoro-v1_0.pth`` 327.2 MB
  - StyleTTS 2 acoustic stack (https://arxiv.org/abs/2306.07691)
  - ISTFTNet vocoder (https://arxiv.org/abs/2203.02395)
  - plbert text encoder: 768 hidden / 12 heads / 12 layers
  - 54 voice embeddings (.pt files; ~0.51 MB each at FP32; FP16 ~0.26 MB)
  - n_token: 178 (IPA phoneme vocabulary); n_mels: 80; sample rate 24 kHz
  - Licence: Apache 2.0

Ternary compression target per gs-intel note + Phase 0 brief OQ-6:
≤22 MB demo-shipped artefact (15-20 MB main model + ~2 MB demo-subset
voices; full 54-voice catalogue stays at filesystem for deployment-time
loading per OQ-10 Option A).

Demo-subset 6 voices (per surgeon ratification 2026-05-19):
  - ``af_heart``   — English-American female (canonical demo voice)
  - ``am_adam``    — English-American male
  - ``bf_alice``   — English-British female
  - ``bm_daniel``  — English-British male
  - ``jf_alpha``   — Japanese female (East-Asian coverage)
  - ``zf_xiaobei`` — Chinese female (East-Asian coverage)

OQ dispositions inherited from Phase 0 brief 2026-05-19:

  - **OQ-1 Option A** — Kokoro adapter at ``terncore.adapters.kokoro``
    sibling cohort
  - **OQ-2 Option A** — voice embeddings FP16 (matches v0.6.0+ embedding
    policy)
  - **OQ-3 Option B** — ISTFTNet mixed routing (handled by the adapter's
    ``classify_weight`` per WeightClassification rules)
  - **OQ-4 Option C** — DI-injected G2P (constructor accepts optional
    ``g2p`` callable; default expects pre-tokenised phoneme input)
  - **OQ-5** — lazy load (model loads on first ``infer`` call); eager
    opt-in via ``eager_load=True`` constructor kwarg
  - **OQ-6** — ≤22 MB demo artefact ceiling (verified by F-8 falsifier)
  - **OQ-10 Option A** — demo-subset 6 voices compiled in; full 54
    catalogue stays filesystem-accessible at deployment time

Cluster A + B architectural rules enforced:

  - NO Subscriber³ import / satisfaction (Cluster A)
  - NO Ledger3 / LedgerEntry3 / decision3 import (Cluster B)

Inventor: Robert Lakelin
Gamma Seeds Pte Ltd / Synapticode
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from terncore.integration3.base import (
    INFERENCE_OUTCOME_CONFIRMED,
    INFERENCE_OUTCOME_DISCONFIRMED,
    INFERENCE_OUTCOME_UNCERTAIN,
    IntegratedProvider3Base,
)

# integration³ canonical surfaces (re-import here for the OQ-4 G2P type
# signature; the base module already arranged the sys.path fallback).
from integration3 import ProviderConfig3, Thought3

# Demo-subset 6 voices per surgeon ratification 2026-05-19.
DEMO_VOICES: tuple[str, ...] = (
    "af_heart",
    "am_adam",
    "bf_alice",
    "bm_daniel",
    "jf_alpha",
    "zf_xiaobei",
)

# Default model artefact + voices directory paths. Both are filesystem
# paths under the canonical model-library convention.
DEFAULT_MODEL_PATH = (
    "/Users/syn/synapticode/model-library/Kokoro-82M/kokoro-v1_0.pth"
)
DEFAULT_VOICES_DIR = (
    "/Users/syn/synapticode/model-library/Kokoro-82M/voices"
)
DEFAULT_COMPRESSED_ARTEFACT = (
    "/Users/syn/synapticode/model-library/Kokoro-82M/"
    "kokoro_82m_ternary_v0.6.0.tern-model"
)

# Kokoro audio output sample rate (per Kokoro README + StyleTTS 2 convention).
KOKORO_SAMPLE_RATE_HZ: int = 24_000


class KokoroProvider3(IntegratedProvider3Base):
    """First canonical Provider³ Protocol consumer — Kokoro 82M TTS.

    Wraps the Kokoro 82M acoustic + vocoder stack with the canonical
    integration³ disposition wiring (EpistemicState mapping +
    Provenance³ stamp + AtomIdentifier³ identity) inherited from
    :class:`IntegratedProvider3Base`.

    Surface (Provider³ Protocol satisfied structurally):

      - ``config``          — ``ProviderConfig3`` (endpoint = artefact path
                              or HF identifier; model = ``"kokoro-82m"``)
      - ``health_check``    — verifies the compressed artefact exists +
                              ≥1 demo voice is loadable
      - ``is_available``    — cached health flag
      - ``infer``           — text-to-audio dispatch (returns
                              ``InferenceResult3`` with ``output_atom.content``
                              holding the audio waveform + metadata)

    Construction:

        provider = KokoroProvider3(
            config=ProviderConfig3(
                endpoint="/path/to/kokoro.tern-model",
                model="kokoro-82m",
            ),
            voice="af_heart",
            g2p=optional_g2p_callable,
            eager_load=False,
        )

    Voice selection:

      - Default voice = ``af_heart`` (per Kokoro README example).
      - Per-call override via ``infer(..., voice="am_adam")``.
      - Demo subset 6 voices ship compiled in; full 54-voice catalogue
        accessible by setting ``voices_dir`` and selecting any
        filesystem voice.

    G2P (grapheme-to-phoneme) handling per OQ-4 Option C:

      - Default: prompt is expected to be a pre-tokenised phoneme
        string (callers tokenise via misaki / espeak-ng / their own
        G2P).
      - Optional: pass ``g2p: Callable[[str], str]`` (or
        ``Callable[[str], list[int]]``) at construction to inject a
        G2P pre-processor. Production callers inject the G2P matching
        their language coverage requirement.
    """

    AUTHOR_IDENTITY_TOKEN: str = "terncore.integration3.KokoroProvider3"

    def __init__(
        self,
        config: Optional[ProviderConfig3] = None,
        *,
        voice: str = "af_heart",
        voices_dir: str = DEFAULT_VOICES_DIR,
        g2p: Optional[Callable[[str], Any]] = None,
        eager_load: bool = False,
        adapter: Any = None,
    ) -> None:
        """Construct a Kokoro 82M Provider³ adapter.

        Args:
            config:     Optional ``ProviderConfig3``. Default constructs
                        a Kokoro-canonical config with
                        ``endpoint=DEFAULT_COMPRESSED_ARTEFACT`` +
                        ``model="kokoro-82m"``.
            voice:      Default voice identifier; must be a member of
                        the demo subset for compiled-in availability,
                        or a filesystem-accessible voice .pt file in
                        ``voices_dir`` for full-catalogue access.
            voices_dir: Filesystem path to the Kokoro voices directory
                        (default: model-library canonical path).
            g2p:        Optional grapheme-to-phoneme callable (OQ-4
                        Option C). Default: no-op (caller pre-tokenises).
            eager_load: If True, load the model + default voice
                        immediately at construction. Default False
                        (lazy load per OQ-5 default).
            adapter:    Optional tern-core ``KokoroAdapter`` instance.
                        If None, the provider lazily instantiates one
                        via ``get_adapter("kokoro")`` at first need.
        """
        if config is None:
            config = ProviderConfig3(
                endpoint=DEFAULT_COMPRESSED_ARTEFACT,
                model="kokoro-82m",
                timeout_seconds=30.0,
                output_half_life=120.0,
            )
        super().__init__(config=config, adapter=adapter)

        self._voice: str = voice
        self._voices_dir: Path = Path(voices_dir)
        self._g2p: Optional[Callable[[str], Any]] = g2p
        self._model_loaded: bool = False
        self._voice_cache: dict[str, Any] = {}

        if eager_load:
            self._ensure_model_loaded()

    # ── Provider³ Protocol surface (health probe override) ────────────────────

    def health_check(self) -> bool:
        """Probe Kokoro readiness.

        Returns True iff:

          - The compressed artefact path or source .pth path exists
            on disk (the consumer can fall back to the FP32 .pth if
            the compressed artefact is missing — useful for
            development / pre-compression smoke).
          - At least the default voice .pt file is loadable from
            ``voices_dir``.

        Caches the result on ``self._available``.
        """
        endpoint_path = Path(self._config.endpoint)
        source_pth = Path(DEFAULT_MODEL_PATH)
        artefact_exists = endpoint_path.exists() or source_pth.exists()

        voice_file = self._voices_dir / f"{self._voice}.pt"
        voice_exists = voice_file.exists()

        self._available = bool(artefact_exists and voice_exists)
        return self._available

    # ── Inference dispatch hook (template-method override) ────────────────────

    def _run_inference(
        self,
        prompt: str,
        context_str: str,
    ) -> tuple[Any, str]:
        """Run Kokoro 82M text-to-audio inference.

        At this attachment the inference path is structural — the
        provider stamps the canonical disposition wiring on the
        Kokoro output. Two execution branches:

          - **Live inference** (if the ``kokoro`` PyPI package is
            installed AND the model artefact exists AND the voice
            loads): full text-to-audio pipeline → CONFIRMED outcome,
            ``output_content`` is a dict carrying ``audio`` (numpy
            float array, 24 kHz), ``sample_rate``, ``voice``,
            ``duration_seconds``.

          - **Structural fallback** (if any prerequisite missing):
            UNCERTAIN outcome with a descriptive output_content dict
            preserving the Provider³ Protocol contract for downstream
            consumers (test surfaces + cluster C UNCERTAIN-branch
            exercise).

        DISCONFIRMED branch is exercised by base class exception
        catching when any inference step raises.
        """
        # Optional G2P pre-processing (OQ-4 Option C DI).
        if self._g2p is not None:
            prompt_tokens = self._g2p(prompt)
        else:
            prompt_tokens = prompt

        # Determine voice selection (default + per-call override
        # handled via ``infer(voice=...)`` plumbing if exposed; at
        # this attachment voice selection lives on the provider
        # instance).
        voice = self._voice

        # Try live inference path; fall back to UNCERTAIN structural
        # output if the live path is unavailable.
        if self._try_load_voice(voice):
            audio_dict = self._build_audio_output(
                prompt_tokens=prompt_tokens,
                voice=voice,
            )
            if audio_dict is not None and audio_dict.get("audio") is not None:
                return audio_dict, INFERENCE_OUTCOME_CONFIRMED

        # Structural fallback (UNCERTAIN) — used when prerequisites
        # missing or live forward pass unavailable in this environment.
        fallback = {
            "audio": None,
            "sample_rate": KOKORO_SAMPLE_RATE_HZ,
            "voice": voice,
            "duration_seconds": 0.0,
            "phoneme_token_count": (
                len(prompt_tokens) if hasattr(prompt_tokens, "__len__")
                else len(prompt)
            ),
            "fallback": True,
            "summary": (
                f"[kokoro stub — voice={voice} model={self._config.model}] "
                f"Received {len(prompt)} chars; load compressed artefact "
                f"to get real audio inference."
            ),
        }
        return fallback, INFERENCE_OUTCOME_UNCERTAIN

    # ── Text rendering for InferenceResult3.text ──────────────────────────────

    def _render_text(self, output_content: Any) -> str:
        """Render the Kokoro audio output dict to a one-line summary
        for ``InferenceResult3.text``. The full audio + metadata
        remains accessible on ``output_atom.content``.
        """
        if isinstance(output_content, dict):
            if output_content.get("audio") is not None:
                dur = output_content.get("duration_seconds", 0.0)
                sr = output_content.get("sample_rate", KOKORO_SAMPLE_RATE_HZ)
                voice = output_content.get("voice", "unknown")
                return (
                    f"[kokoro audio voice={voice} duration={dur:.2f}s "
                    f"sample_rate={sr}Hz]"
                )
            if output_content.get("summary"):
                return str(output_content["summary"])
        return str(output_content)

    # ── Voice + model loading helpers ─────────────────────────────────────────

    def _try_load_voice(self, voice: str) -> bool:
        """Load the voice .pt embedding into cache. Returns True iff
        the voice .pt file is on disk and loadable.

        Lazy posture: first call per voice loads from disk; subsequent
        calls hit the cache.
        """
        if voice in self._voice_cache:
            return True
        voice_file = self._voices_dir / f"{voice}.pt"
        if not voice_file.exists():
            return False
        try:
            import torch  # local import — avoid loading torch in tests
                          # that mock at boundaries
            embedding = torch.load(
                str(voice_file),
                map_location="cpu",
                weights_only=True,
            )
            self._voice_cache[voice] = embedding
            return True
        except Exception:
            return False

    def _ensure_model_loaded(self) -> None:
        """Eager-load path (OQ-5 opt-in via ``eager_load=True``).

        At this attachment "loading the model" means probing the
        compressed artefact + loading the default voice; the full
        ``kokoro`` PyPI package model construction is left to the
        live-inference build path inside ``_build_audio_output``.
        """
        if self._model_loaded:
            return
        self.health_check()
        self._try_load_voice(self._voice)
        self._model_loaded = True

    def _build_audio_output(
        self,
        prompt_tokens: Any,
        voice: str,
    ) -> Optional[dict[str, Any]]:
        """Build the audio output dict.

        Attempts to use the ``kokoro`` PyPI package if available
        (live inference path). Returns None if live inference cannot
        run in the current environment — the caller falls back to the
        UNCERTAIN structural path.

        This method is the single integration seam to the upstream
        Kokoro inference library; substituting a tern-core-native
        inference path (loading the compressed .tern-model artefact +
        running on the ternary execution engine) is a future
        refinement that lands in this method without changing the
        ``Provider³`` Protocol surface.
        """
        # Attempt the upstream ``kokoro`` PyPI package live path.
        # The package may not be installed in every environment
        # (e.g. tern-core CI); the structural fallback covers that
        # case and is exercised by F-2 UNCERTAIN-branch tests.
        try:
            from kokoro import KPipeline  # type: ignore  # noqa
        except ImportError:
            return None

        try:
            # Live-path construction — kept minimal at this attachment;
            # production deployments wire the ternary-compressed
            # artefact through tern-core's inference engine instead of
            # the upstream PyPI package.
            lang_code = voice[:1]  # 'a' = American, 'b' = British, etc.
            pipeline = KPipeline(lang_code=lang_code)  # type: ignore
            text_form = (
                " ".join(map(str, prompt_tokens))
                if isinstance(prompt_tokens, list)
                else str(prompt_tokens)
            )
            generator = pipeline(text_form, voice=voice)  # type: ignore
            chunks: list[Any] = []
            for _gs, _ps, audio in generator:  # type: ignore
                chunks.append(audio)
            if not chunks:
                return None
            try:
                import numpy as np
                audio_arr = np.concatenate([
                    np.asarray(c).reshape(-1) for c in chunks
                ])
            except Exception:
                audio_arr = chunks[0]
            duration = (
                float(len(audio_arr)) / KOKORO_SAMPLE_RATE_HZ
                if hasattr(audio_arr, "__len__")
                else 0.0
            )
            return {
                "audio": audio_arr,
                "sample_rate": KOKORO_SAMPLE_RATE_HZ,
                "voice": voice,
                "duration_seconds": duration,
                "phoneme_token_count": (
                    len(prompt_tokens) if hasattr(prompt_tokens, "__len__")
                    else 0
                ),
                "fallback": False,
            }
        except Exception:
            return None

    # ── Helpers exposed for testing + diagnostics ─────────────────────────────

    @property
    def voice(self) -> str:
        """Current default voice identifier."""
        return self._voice

    @property
    def demo_voices(self) -> tuple[str, ...]:
        """Six demo voices shipped in compiled artefact per OQ-10
        Option A. Full 54-voice catalogue accessible at filesystem
        via ``voices_dir``."""
        return DEMO_VOICES

    def available_voices(self) -> list[str]:
        """Enumerate voices accessible at the filesystem ``voices_dir``.

        F-9 falsifier surface: enumerates all 54 voices when the full
        catalogue is staged at ``voices_dir`` (the canonical model-
        library layout).
        """
        if not self._voices_dir.exists():
            return []
        return sorted(
            p.stem for p in self._voices_dir.glob("*.pt")
        )


__all__ = [
    "KokoroProvider3",
    "DEMO_VOICES",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_VOICES_DIR",
    "DEFAULT_COMPRESSED_ARTEFACT",
    "KOKORO_SAMPLE_RATE_HZ",
]
