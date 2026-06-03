# Per-expert tolerance analysis — results

Each dated subdirectory holds the human-readable `summary_table.md` for one
per-expert tolerance run (model attribution in the directory name).

The raw artefacts for each run — `per_expert_tolerance_analysis.json` and the
four `plot_*.png` figures — live on the archive, excluded from version control
for size:

```
/Volumes/Syn Archive/models/compressed/analysis_results/<dated-subdir>/
```

Regenerate any run with `benchmarks/analyse_per_expert_tolerance.py` (it writes
the JSON + plots into the run's output directory). The top-level
`tq_bench_results_*.json` benchmark dumps are archived alongside, under
`/Volumes/Syn Archive/models/compressed/analysis_results/`.
