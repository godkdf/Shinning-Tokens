# Data formats and provenance

The current **hub and run schema is 2.0**. Vocabulary inventories retain schema **1.0**. Application version **0.2.3**, measurement protocol **intrinsic-v0.2**, data cohort **v0.4** and schema versions identify different things.

JSON is UTF-8. `null` means unavailable, not zero. SHA-256 uses lowercase hex. A `.json.gz` contains one JSON value; `.jsonl.gz` contains one observation per line. Consumers must decompress gzip before parsing unless their HTTP client has already done so.

## Main records

| Artifact | Fields and meaning |
|---|---|
| Document | `id`, `dataset`, `language`, original `text`, `utf8_bytes`, original-text `sha256` and source identity. `domain`/other source fields depend on the corpus. |
| Observation | Document identity, `tokenizer`, `tokens`, original UTF-8 and UTF-16 sizes, `exact`, `utf16le_exact`, `utf16be_exact`, `canonical_only`, `error`, unknown emissions and decoded identity. |
| Aggregate | `dataset`, `language`, `tokenizer`, full `documents` denominator, `utf8_bytes`, `tokens`, UTF-16 counts, exact/canonical/mismatch/error counts, `bytes_per_token` and token-tail statistics. |
| Profile | ID/name, upstream revision/terms, artifact identity, pipeline `components`, `vocabulary`, `provenance`, optional families and exact checkpoint `aliases`. |
| Vocabulary | `tokenizer`, `artifact_sha256`, `fields` and ordered `rows`; schema 1.0. |
| Vocabulary row | `[id, native_surface, bytes_hex_or_null, script, types, byte_length_or_null]`. |
| Pair | `left`, `right`, shared/union byte-piece counts, Jaccard, directional coverage, category overlap and definition. |
| Runtime | Tokenizer, fresh-process RSS/load measurements, five encode/decode duration samples, MB/s, input SHA-256, implementation and hardware/cohort information. |
| Run result | Explicit `selection`, profiles/aggregates, runtime, 16 transport probes per tokenizer, checkpoints, measurement-code hashes, package versions, errors and network-download count. |
| Checkpoint | Gzip JSONL observations plus `.meta.json` with input/config/artifact/protocol/code/package identities and the checkpoint hash. |
| Current hub | Profiles, benchmarks, runtime, pairs, corpora, mappings, experiments, checks, protocol, reproduction, transport probes and verification. |

Use `result_paths` and `measurement_sources` to find the correct frozen result and code for each profile. `profile_metadata_updates` records the Qwen 2.5 alias/name addition for Kimi Dev; its scientific measurements were not replaced. `verification` explicitly describes which observations were audited freshly and which prior audit was retained.

## Hash scopes

Do not interchange these hashes:

- A registry `sha256` or `offline_sha256` verifies the exact downloaded/exported **file bytes**.
- A JSON adapter's profile `artifact_sha256` hashes its JSON file. A native tiktoken adapter hashes **canonical mergeable ranks plus the special-token map**; regex, normalization, chunking and marker policy are recorded separately.
- A checkpoint's `configuration_sha256` hashes its tokenizer specification. `measurement_code` identifies the saved measuring implementation.
- A document `sha256` hashes original UTF-8 text, not compressed storage or normalized text.
- Catalog and text-index hashes verify their corresponding compressed files, including exact records and ordering.

Two profiles sharing an ordinary rank file can still have different special IDs, reserved names or recognition rules. Kimi illustrates this distinction. An alias requires the declared complete measured pipeline to match, not just equal vocabulary size or common pieces.

## Counts and missing data

`documents` always retains failed inputs. Exactly one UTF-8 fidelity category applies to each observation: exact, canonical-only change, other mismatch or execution error. Unknown emissions are separate. Canonical-only changes remain byte-exact failures.

`bytes_per_token` is pooled original UTF-8 bytes divided by pooled emitted tokens. An error can make the ratio unavailable. Never average language ratios and label that number as the pooled ratio. Fidelity loss makes apparent compression diagnostic rather than a valid lossless gain.

Vocabulary categories overlap and count **IDs**. Pairwise overlap counts **unique ordinary byte strings**, excluding control IDs/unrecoverable bytes. An incomplete UTF-8 piece is valid vocabulary data and can decode correctly when neighboring pieces are present. Equal pieces do not establish equal contextual segmentation.

Vocabulary ID count, maximum ID + 1, ID holes and a model's embedding rows are separate quantities. Some Kimi wrappers expose two reserved IDs beyond their checkpoint's declared rows; the inventory records this rather than promising model compatibility.

UTF-16LE/BE fields encode valid original/reconstructed Unicode without a BOM. They do not change token counts. `A😀` has **3 code units / 6 UTF-16 bytes**. Six malformed transport cases are rejected before tokenization; invalid surrogate strings are not silently repaired.

## Conversion and maps

The converter decodes a complete ordinary source sequence, strictly validates UTF-8, retokenizes it and checks reconstructed bytes. It rejects special/control ID streams without an explicit model-specific policy. `exact` compares source-decoded bytes with target-decoded bytes; `source_roundtrip_exact` separately reports whether original input text survived source normalization.

Historical standalone maps have `exact_byte_piece`, `standalone_one_to_many`, `standalone_changes_bytes`, `byte_fragment_context_required` and `special_or_unavailable` statuses. `matching_piece_ids` can coexist with different standalone `target_ids`. `context_required: true` prohibits interpreting these diagnostics as universal token-ID substitution tables. Thirty historical directions are retained; arbitrary whole-sequence conversion is available locally.

## Compressed text, selections and distributions

The separate realistic code cohort exports `site/data/code-study/study.json.gz` (schema **1.0**). It contains `summary`, `profiles`, `languages`, `workloads`, `coverage`, `provenance` and `rows`. Each row is one **tokenizer × workload × programming language × split**, restricted to `size_band=all` so size-band subtotals cannot double-count files. Counts include `documents`, `utf8_bytes`, `tokens`, `exact`, `canonical_only`, `mismatches`, `errors`, `unknown_emissions`, `utf16le_exact` and `utf16be_exact`. The current export has **20,736 rows** representing **2,419,200 observations**. Missing groups are rejected rather than filled with zeros.

The six workload IDs are `code-realistic-v0.1`, `real-edit-before-v0.1`, `real-edit-after-v0.1`, `real-edit-diff-v0.1`, `repository-context-v0.1` and `code-stress-v0.1`. Splits are `development` and `evaluation`. Ratios in the app are recomputed from pooled counts. Fidelity is scoped to the selected rows; the global exact-profile count covers all workloads and both splits. The independent validation ledger and hashes of all 108 input aggregate files accompany the export. `measurements.csv.gz` contains the same rows. This bundle is separate from the frozen intrinsic hub and corpus catalog. [Code study guide](CODE_BENCHMARK.md).

`benchmarks/catalog.json` fixes **867** original partitions. `site/data/text-index.json` records compressed UI text and available saved-example sidecars, hashes and byte sizes. The **689** sidecars contain **771** small saved examples; this is separate from full corpus coverage. Unavailable saved token IDs do not remove source documents.

Selection objects contain explicit arrays: `tokenizers`, `datasets`, `languages`, and optionally local `custom_tokenizers` paths for a run. Missing/empty lists mean none. Catalog presence is not proof that an asset is installed. Download plans distinguish cached bytes, installed bytes and unknown/transient network cost.

Source and explorer archives have member manifests; starter kits have `offline-manifest.json`. Distribution SHA-256/size is distinct from scientific input identity. Release checksums cover the exact published bytes, not every file a user later generates locally.
