# Per-expert tolerance analysis — results

Each dated subdirectory holds the human-readable `summary_table.md` for one
per-expert tolerance run (model attribution in the directory name).

The raw artefacts for each run — `per_expert_tolerance_analysis.json` and the
four `plot_*.png` figures — are evidence-of-record on the **private**
`gamma-seeds/ecc-ternary` release `evidence-tern-core-scans-2026-05-28`
(off-site, access-gated; durable, not single-drive). All five runs are bundled,
preserving this dated-directory structure, in:

- `per_expert_tolerance_analysis_2026-05-28.tar.gz` —
  https://github.com/gamma-seeds/ecc-ternary/releases/download/evidence-tern-core-scans-2026-05-28/per_expert_tolerance_analysis_2026-05-28.tar.gz

Regenerate any run with `benchmarks/analyse_per_expert_tolerance.py` (it writes
the JSON + plots into the run's output directory). The top-level
`tq_bench_results_*.json` benchmark dumps are assets on the same release:

- https://github.com/gamma-seeds/ecc-ternary/releases/download/evidence-tern-core-scans-2026-05-28/tq_bench_results_gemma_4_e4b_it_20260512T222533Z.json
- https://github.com/gamma-seeds/ecc-ternary/releases/download/evidence-tern-core-scans-2026-05-28/tq_bench_results_llama_3_2_1b_instruct_ternpacked_20260513T041543Z.json

SHA-256 for every asset is in the release MANIFEST.
