# Reproduce Tokenizer Lab

Use application **0.2.3**, protocol **intrinsic-v0.2** and the pinned inputs in `benchmarks/catalog.json`. Result cohorts v0.1–v0.4 retain separate sources and evidence; v0.4 is the current 54-profile aggregate.

## Open saved results or run a selection

From the source checkout:

```sh
uv sync --frozen --extra test
uv run --frozen tokenizer-lab launch
```

Open **Downloads & storage**. Select tokenizer IDs, datasets and languages explicitly, review the plan, and download only that selection. Selecting none downloads none. Normal language names and continent browsing map to exact language/script IDs; they do not change the frozen text.

A download is a separate action. Gated assets require upstream acceptance and credentials; local experiment artifacts require their exact input bundle. Neither a benchmark nor the converter downloads a missing tokenizer. A hash mismatch fails rather than substituting another revision. Model weights are not required.

Once the desired inputs are cached, select them in the launcher and run. The application records vocabulary, original-text fidelity, throughput and memory. To run an exact small CLI selection using already cached English FineWeb and GPT-2:

```sh
uv run --frozen tokenizer-lab run-selected --tokenizers gpt2 --datasets fineweb-mixture-v0.1 --languages eng_Latn --vocabulary --runtime --output results/local/gpt2-english
```

That command selects **368** original English FineWeb documents. It does not select all languages or download inputs. Results belong to that local run, not the frozen published cohort.

Import custom tokenizer JSON in the launcher using its absolute local path and a display name. It receives the same selected document IDs, hashes and fidelity checks. No custom repository code is executed. Native tiktoken and SentencePiece profiles use the recorded built-in adapters; custom import currently accepts Hugging Face-compatible JSON only.

## Rebuild current or family results

`configs/expanded-selection.json` selects all **54** registered profiles and all **867** prepared partitions. With every required asset and corpus partition already cached:

```sh
uv run --frozen tokenizer-lab run-selected --selection configs/expanded-selection.json --vocabulary --runtime --output results/local/full-matrix
```

The run is intentionally large: **23,304,780 observations**. The normal launcher default remains empty. Keep independent timing jobs sequential; do not compare concurrent timing runs as though they were controlled measurements.

`configs/reference-selection.json` selects GPT-2 and Llama 3. `configs/kimi-selection.json` selects the 8 Kimi/Moonlight profiles. After explicitly preparing the Kimi selection and full corpus:

```sh
uv run --frozen python -m tokenizer_lab.publish_kimi --run
```

This writes each profile separately under `results/v0.4/<id>`, saves measurement sources, checks all **6,936** new checkpoints, extends the frozen prior aggregate and updates the local explorer. It uses existing cached inputs and does not publish a website. To assemble and audit already completed per-profile outputs without re-running tokenization, omit `--run`.

The reference assembler `python -m tokenizer_lab.publish_references` expects the separate completed `results/v0.3/gpt2/result.json` and `results/v0.3/llama3/result.json` files. Both assemblers depend on their frozen base aggregate and vocabulary inventories; a minimal source archive does not contain every generated inventory or observation checkpoint.

## Inspect and audit evidence

- `site/data/hub.json.gz`: current compact UI aggregate; **schema 2.0**.
- `results/v0.4/hub.json`: current expanded aggregate, scientific provenance and per-profile result paths.
- `site/data/benchmarks.csv.gz`: **46,818** dataset/language/tokenizer aggregate rows.
- `results/v0.4/verification.json`: the combined audit chain, with hashes linking prior verification.
- `results/v0.4/verification-new.json`: the fresh **3,452,560-observation** Kimi audit.
- `results/v0.3/verification-new.json`: the fresh **863,140-observation** GPT-2/Llama 3 audit.
- `results/v0.2/verification.json`: the original **18,989,080-observation** expanded audit.
- Per-profile `result.json`: input selection, artifact/config/code/package identities, checkpoint hashes, runtime and transport probes.
- `results/v0.*/<id>/checkpoints/`: generated compressed per-document observations; intentionally not committed or included in the small source archive.

The audit chain retains earlier verified observations. It does not claim that all **46,818** checkpoints were re-audited simultaneously during every extension. [Formats and hash scopes](SCHEMAS.md).

For release metadata, assets, doc links and integrity checks:

```sh
uv run --frozen python -m tokenizer_lab.publication audit
```

A source checkout or source archive is not the complete generated result payload. The explorer archive supplies saved inventories, maps and compressed text; complete observations can be regenerated from the pinned inputs. A local run never silently becomes a published result.

## Open the latest saved explorer

```sh
uv run --frozen tokenizer-lab build-site
uv run --frozen tokenizer-lab serve
```

Open **http://localhost:8765**. `build-site` selects the latest saved versioned aggregate, refuses pending audits and refreshes its compressed JSON/CSV. It does not measure tokenizers or fetch data. Local conversion supports arbitrary ordinary text or source IDs with cached tokenizers; static hosting supports saved examples without the Python service.

## Frozen v0.1 and Gemma V3.1

`configs/launch.json`, `tokenizer-lab run` and `tokenizer-lab verify` describe the **original seven-profile v0.1 pipeline**, not the current 54-profile release. Its `run` command can fetch its selected tokenizer revisions. These are explicit historical replay commands, not the default start flow.

The Gemma V3.1 experiment appends **2,000 IDs**. Download its exact input bundle, retain its agreement/notices, and install it before replay:

```sh
uv run --frozen tokenizer-lab install-experiment --bundle artifacts/gemma-v31-inputs.tar.gz
uv run --frozen tokenizer-lab build-expansion --parent .cache/assets/gemma3_parent/tokenizer.json
```

The installer accepts only the named files and pinned hashes. The builder refuses a different structural parent. This parent is reconstructed from the supplied candidate; an independently verified official gated replay remains unavailable. Do not bypass the check or present it as a verified official artifact.

## Transfer baselines

These commands consume explicit NumPy inputs; they do not download or train a model implicitly:

```sh
uv run --frozen tokenizer-lab transfer-average --embeddings parent.npy --sequences parent_ids.json --output initialized.npy
uv run --frozen tokenizer-lab transfer-fit --source train_source.npy --target train_target.npy --output mapping.npz
uv run --frozen tokenizer-lab transfer-apply --source heldout_source.npy --mapping mapping.npz --output predicted.npy
```

Use disjoint fit/validation/test anchors, choose regularization on validation only, and compare held-out error with a training-mean baseline. Vector initialization is not evidence of better generation. See the [research library](research/index.md).
