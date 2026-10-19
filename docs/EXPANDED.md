# Expanded intrinsic protocol v0.2

The Lab compares tokenizer behavior on original text. It does not train a model, rank generation quality, or estimate downstream gains from compression alone.

## Fixed inputs

| Dataset | Inputs | Scope |
|---|---:|---|
| FineWeb mixture | 5,941 | Existing pinned 24-variety sample retained |
| FLORES-200 dev | 203,388 | 204 script varieties × 997 sentences |
| FLORES-200 devtest | 206,448 | 204 script varieties × 1,012 sentences |
| MADLAD-400 sample | 5,237 | 418 clean configurations; one bounded prefix per configuration |
| Unicode diagnostics | 500 | Constructed original/transformed pairs |
| Numbers | 10,000 | Constructed formats, not arithmetic ability |
| Code | 56 | Handwritten diagnostics, not representative code coverage |
| **Total** | **431,570** | **44 profiles → 18,989,080 observations** |

FLORES uses the [official public Meta FLORES-200 archive](https://github.com/facebookresearch/flores/blob/main/flores200/README.md), SHA-256 `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`, **25,585,843 bytes**. Both public splits are retained in full with original sentence order. There is no FLORES+ result in this release; that newer resource requires separate access acceptance. FLORES is evaluation data, not tokenizer training data. License: CC-BY-SA-4.0.

MADLAD uses [allenai/MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400), revision `9d886a76bd8fa69b294f2dd3843dacb8388ee5a5`. For each configuration, choose the lexicographically first clean JSONL gzip shard, read its original order, retain unique nonempty text documents of at most 100,000 UTF-8 bytes, and stop after **65,536 retained bytes or 32 whole documents**. The last document may overshoot the byte target; the retained size is strictly below 165,536 bytes. A parser stops at 10,000 source lines and rejects lines over 2,000,000 bytes. The declared dirty/noise configuration is excluded. This is a bounded prefix sample, not a uniform sample. It is not established as held out from historical MADLAD token selection. License: ODC-BY-1.0.

`benchmarks/catalog.json` fixes every partition's document count, original byte count, compressed JSONL hash, source revision and sampling description. `site/data/text-index.json` fixes the corresponding compressed UI payload hashes and sizes. Every document retains its ID, exact UTF-8 text hash and source identity. Partitioning changes storage, not text.

## Tokenizer identity and fidelity

`configs/registry.json` records pinned data assets. `configs/tokenizer-candidates.json` records the 105 official repositories inspected, 104 included checkpoints, 38 distinct profiles and one excluded video-only checkpoint. JSON profiles deduplicate the recorded native Rust pipeline including IDs, additions, flags, normalization and decoding. Chat-template changes alone do not create another plain-text tokenizer. The combined registry has 44 profiles because one of the 38 is the existing Qwen3 profile.

Legacy GLM wrappers use explicit data-only ports. Their components label reference-wrapper parity as **not established**. Missing configured special strings are disclosed rather than assigned invented official IDs. Ordinary text results are measurements of the declared adapter. Do not equate these ports with an independently verified upstream Transformers wrapper.

All tokenizers receive the same Unicode strings, with no truncation, padding or automatically added BOS/EOS. Tokenizer-owned normalization still applies and is measured. Exact UTF-8 byte equality is required; canonical equivalence is a separate diagnostic. Unknown emissions and encode/decode errors remain visible. Compression is diagnostic when fidelity fails.

UTF-16LE and UTF-16BE encode original and reconstructed Unicode as strict transport bytes without a BOM. A supplementary character such as 😀 occupies **2 code units / 4 bytes** in either UTF-16 form; `A😀` occupies **3 code units / 6 bytes**. Valid UTF-16 is decoded before tokenization. Ten valid transport cases and six malformed cases are recorded per tokenizer. Odd-length byte strings and unpaired high/low surrogates must be rejected before tokenization. These checks do not imply that tokenizers consume UTF-16 code units directly.

## Runs, timing and storage

Local runs require explicit nonempty tokenizer, dataset and language selections, read only cached inputs, and save per-partition observations. Empty or omitted download selections mean none. There are no background tokenizer or corpus downloads. Hashes are checked before a restored payload is installed. Failed downloads leave no partially installed payload.

After explicitly preparing the selected inputs, replay the full expanded matrix with:

```sh
uv run --frozen tokenizer-lab run-selected --selection configs/expanded-selection.json --vocabulary --runtime --output results/replay-v0.2
```

That command intentionally selects all 44 configured tokenizers and all prepared benchmark partitions. The local launcher's normal default remains an empty selection. For everyday work, select a smaller scope in the launcher. The published runtime table is measured sequentially; the expanded corpus execution can be partitioned into separate jobs without pooling timing from concurrent jobs.

The expanded run retains the frozen v0.1 release and writes separate v0.2 checkpoints. Native runtime samples use the same interleaved FineWeb text, one warm-up, five repeats and a fresh process. Final runtime measurements are collected sequentially after corpus jobs finish. RSS, incremental loaded RSS, sampled peak RSS, loading time and encode/decode MB/s have distinct definitions; numbers apply to the recorded host. No aggregate winner score is defined.

Vocabulary categories overlap. Script, numeric and code-associated token counts describe pieces; they are not ability scores. Pairwise overlap uses unique ordinary byte pieces without normalization. Whole-sequence conversion retokenizes reconstructed text and checks fidelity. Full standalone maps are available for the original 30 directions; new pairs can be converted on demand locally rather than silently downloading every possible map.

Language names and aliases use the recorded FLORES names, langcodes/language-data and ISO names. Continent browsing uses overlapping CLDR and Glottolog country metadata. Multiple groups are intentional; Unspecified means insufficient metadata. Metadata file SHA-256 values are stored in `benchmarks/languages.json`.
