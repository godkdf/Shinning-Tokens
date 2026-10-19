# Contribute

Make one change with a measurable question. Preserve existing published inputs and protocol versions.

For a tokenizer, add a unique registry ID, exact upstream revision, terms link and native adapter. Document normalization, special IDs and piece-byte semantics. Run all existing diagnostics and report failures. Do not silently substitute a converted tokenizer for its native implementation.

For an analysis, state its denominator and exclusions. Preserve per-language results; report regressions alongside gains. Categories must have explicit rules and versions. Do not turn compression, heuristic code pieces or script counts into model capability claims.

For an experiment, follow [the experiment format](docs/experiments/README.md). Link the parent, candidate, deterministic recipe, selection data, evaluation data and artifact hashes. Distinguish proposed, historical, paper-reported and locally reproduced evidence. Keep model-weight training separate from the cheap intrinsic pipeline.

Run `uv run pytest -q`. For a release bundle, run the complete pipeline and `uv run tokenizer-lab verify`. Changes to corpus sampling or token semantics require a protocol revision and refreshed evidence. Contributions should include a concrete before/after number, an affected example, or a clearly defined new question.
