# Reproduce the Lab

1. Clone the repository, run `uv sync --frozen --extra test`, and run `uv run pytest -q`.
2. Run `uv run tokenizer-lab doctor`. Missing assets are reported without downloading model weights.
3. Accept upstream tokenizer terms and authenticate to Hugging Face for gated assets. `run` downloads exact revisions from `configs/registry.json` into `.cache/assets/`. The `o200k_base` encoding is loaded by native tiktoken.
4. For V3.1, download **Frozen Gemma V3.1 inputs** from the explorer's Reproduce view and save the archive at `artifacts/gemma-v31-inputs.tar.gz`. The package includes the supplied candidate, reconstructed control, agreement and notices. Verify and install its two input files, then verify the deterministic expansion rebuild:

```sh
uv run tokenizer-lab install-experiment --bundle artifacts/gemma-v31-inputs.tar.gz
uv run tokenizer-lab build-expansion --parent .cache/assets/gemma3_parent/tokenizer.json
```

The installer checks the pinned archive and input file hashes. It only writes the two named inputs; it never extracts arbitrary archive paths. The builder validates a structural hash before adding any IDs and validates the complete child afterward. It refuses a different parent. The supplied exploratory parent was reconstructed from the candidate; an official gated replay was not possible during launch preparation. A downloaded parent that differs needs an explicit audited experiment revision, not bypassing this check.

5. Run `uv run tokenizer-lab prepare`. First preparation pins the source revisions. Subsequent runs reuse pinned samples whose file hashes match. To intentionally change corpus selection, create a new corpus/config version; do not overwrite a published version's recipe.
6. Run `uv run tokenizer-lab run --config configs/launch.json`. This produces 7 profiles, all document observations, isolated runtime measurements, 15 overlap comparisons, and 30 directional maps. Checkpoints are reusable only for matching artifact/input/runtime identities.
7. Run `uv run tokenizer-lab verify`. It independently checks document hashes, complete observation identities, aggregate denominators, fidelity partitions, vocabulary IDs and the comparison graph, then writes `results/bundle-manifest.json`.
8. Run `uv run tokenizer-lab serve` and open http://localhost:8765. The local converter accepts arbitrary text or ordinary source IDs. The static hosted explorer uses saved examples and downloadable vocabulary maps; it labels that limitation.

`run --skip-maps` and `run --skip-runtime` are development commands only. Such runs do not pass release verification.

## Outputs

- `results/v0.1/hub.json`: summaries, identities, runtime samples, paired experiment results.
- `results/raw/observations.jsonl.gz`: complete per-document evidence, including failures.
- `results/v0.1/checkpoints/`: per-tokenizer observation files.
- `site/data/vocab/`: full vocabulary inventories, gzip JSON.
- `site/data/mappings/`: 30 context-labelled standalone mapping files.
- `site/data/examples.json`: original text with saved encodings for every tokenizer.
- `reports/v0.1.md`: generated human-readable tables.

Large generated outputs and downloaded upstream assets are kept outside source Git. A published explorer carries the vocabulary and mapping downloads. Source-only cloning is not equivalent to having the complete result bundle. The result manifest lists hashes for independent verification.

## Transfer baselines

These commands consume explicitly provided NumPy arrays. They never download a model or train one implicitly.

```sh
uv run tokenizer-lab transfer-average --embeddings parent.npy --sequences parent_ids.json --output initialized.npy
uv run tokenizer-lab transfer-fit --source train_source.npy --target train_target.npy --output mapping.npz
uv run tokenizer-lab transfer-apply --source heldout_source.npy --mapping mapping.npz --output predicted.npy
```

`parent_ids.json` is an ordered JSON list of nonempty parent-ID lists. Matched mapping anchors have equal row count; source and target vector widths may differ. Use disjoint fit/validation/test items, select regularization on validation only, and compare held-out reconstruction against the target training-mean baseline. Input embeddings and output heads can require different treatment; tied heads must preserve their tying policy. Initial vectors are not evidence of improved generation. See the [research library](research/index.md).
