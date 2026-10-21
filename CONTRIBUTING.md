# Contribute

Start with one measurable question. Keep frozen inputs/results intact; put new corpus or tokenizer semantics in a new versioned cohort or explicit local run.

## Tokenizer or family

1. Pin the official repository revision, exact artifact hashes and upstream terms in the registry/catalog. Download data assets only; do not fetch weights or enable `trust_remote_code`.
2. Record the native implementation, regex, normalization, byte decoding, special-token map/policy, reserved ranges and automatic BOS/EOS behavior. State any unverified wrapper parity.
3. Group aliases only when the complete declared measured pipeline matches. Identical ranks, a chat-template change or matching dimensions alone do not establish equivalence.
4. Add meaningful tests for reference IDs, byte fidelity, special policies, invalid transport and cached-only behavior. Include exact input/asset/code identities in measured results.
5. Run the selected diagnostics and language partitions. Keep mismatches, errors and unknown emissions in their denominators. Report both gains and regressions.

## Analysis or experiment

State the denominator, weighting, sampling, exclusions and what the result does not establish. Vocabulary category rules need versions. Do not label script coverage, code-associated pieces or compression as model capability.

For changed vocabularies, follow [the experiment format](docs/experiments/README.md): parent identity, hypothesis, deterministic recipe, selection/evaluation split, additions/deletions, gains, regressions, compute and limitations. Separate proposed, historical, paper-reported and freshly reproduced evidence. Keep model training separate from intrinsic evaluation.

## Development checks

```sh
uv sync --frozen --extra test
uv run --frozen pytest -q
uv run --frozen python -m tokenizer_lab.publication audit --source-only
```

Fresh checkouts skip integration tests whose explicit tokenizer assets/reference caches are absent. Report skips; never describe them as verified tokenizers. Core tests use local fixtures and should not download a model or full benchmark silently.

Before release, follow [release preparation](docs/RELEASE.md), run asset integrity checks and regenerate the source/explorer distributions. Native portability checks are manual; their workflow definition is not evidence that a platform has been exercised. Local user files and unrelated experiments must not enter a release by accident.

Include a concrete before/after number, an affected example or a sharply defined new question in a pull request. File a [bug report](https://github.com/godkdf/Shinning-Tokens/issues) with release, platform, selected IDs and reproducible steps. Follow [SECURITY.md](SECURITY.md) for sensitive reports.
