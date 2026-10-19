# Data schemas v1.0

All JSON text is UTF-8. `null` means unavailable, never zero. SHA-256 values use lowercase hex. Gzip files contain ordinary JSON or one JSON object per line.

| Artifact | Required content |
|---|---|
| Corpus document | `id`, `dataset`, `language`, `domain`, `source_id`, original `text`, `utf8_bytes`, `sha256`, source metadata |
| Observation | Document identity/hash, `tokenizer`, original byte count, emitted `tokens`, `exact`, `canonical_only`, `error`, unknown count, decoded hash or mismatch example |
| Aggregate | Dataset/language/tokenizer, full denominator, bytes, tokens, ratio, four fidelity partitions, unknown count, tail statistics |
| Vocabulary | `tokenizer`, artifact hash, `fields` and ordered `rows` arrays |
| Vocabulary row | `[id, native_surface, bytes_hex_or_null, script, types, byte_length_or_null]` |
| Pair | `left`, `right`, shared/union unique byte-piece counts, Jaccard, directional coverage, category overlap and definition |
| Directional map | Source/target identities/hashes, context warning, one record per source ID, status and available target diagnostics |
| Hub | Profiles, aggregates, runtime, pairs, mappings, corpora, experiments, checks, protocol and reproduction |
| Bundle manifest | Relative file paths, byte sizes, hashes and verification counts |

Map statuses include `exact_byte_piece`, `standalone_one_to_many`, `standalone_changes_bytes`, `byte_fragment_context_required`, and `special_or_unavailable`. `matching_piece_ids` can coexist with a different `target_ids` standalone segmentation. `context_required: true` means the standalone result cannot be substituted blindly into an arbitrary sequence.

Vocabulary categories count IDs; overlap counts unique byte strings. An incomplete byte piece can be a valid vocabulary entry while being invalid as standalone Unicode. Special/control IDs have no ordinary-text conversion policy in v0.1.
