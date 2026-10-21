# Intrinsic benchmark protocol

Current protocol: **intrinsic-v0.2**. Application **0.2.3** publishes result cohorts v0.1–v0.4 separately. The current matrix has **54 profiles × 431,570 inputs = 23,304,780 observations**, across **867 partitions per tokenizer**. Earlier cohorts and their saved measurement sources remain frozen.

The expanded inputs and Unicode transport rules are specified in [docs/EXPANDED.md](docs/EXPANDED.md); formats and hash scopes are in [docs/SCHEMAS.md](docs/SCHEMAS.md). Result extensions retain prior audit evidence rather than claiming every historical checkpoint was re-audited during each addition. The initial FineWeb/diagnostic definitions below remain part of the protocol.

## Inputs and scope

All implementations receive the same original UTF-8 document strings. No automatic BOS/EOS insertion, chat template, normalization, padding, truncation, or cleaning is added by this Lab. Native normalization already present inside an upstream tokenizer remains enabled and is part of its measured behavior. Arbitrary invalid UTF-8 byte sequences and unpaired Python surrogates are outside this text benchmark.

FineWeb2 source revisions are frozen in `benchmarks/fineweb-v0.1/source-lock.json`. Language inventory and selection are in `src/tokenizer_lab/corpora.py`. The seeded recipe chooses one test shard and a row-group order; it shuffles documents within read groups and retains whole nonempty documents, at most 100,000 bytes each, until the byte target. Duplicate text hashes within a language are excluded. This is bounded sampling, not a uniform sample of the entire dataset. English uses original FineWeb's 10BT sample; it is labelled separately in provenance. The selected shard can be too small to reach the target. Missing or short sources are never silently replaced.

The corpus manifest records each repository revision, shard, sampled document count, actual bytes and sample hash. Each original document records its source URL/ID, row group, text hash and UTF-8 size. Source-group resampling uses URL/ID when available. This does not remove near duplicates and does not prove clean separation from all earlier exploratory selection data. FineWeb is a different corpus from the earlier MADLAD selection sample; the report makes no stronger decontamination claim.

Constructed diagnostics have 25 Unicode bases × 20 transformations = **500 cases**, 1,000 seeded number families × 10 formats = **10,000 cases**, and 14 handwritten code snippets × 4 indentation prefixes = **56 cases**. Unicode normalization variants are explicitly labelled. Numeric formats include ASCII, Arabic-Indic, signed, decimal, scientific, hexadecimal, leading zeros and expressions. Six code formats are Python, JavaScript, C, SQL, JSON and HTML. These suites identify segmentation behavior, not task competence.

## Fidelity and efficiency

For every input, compare original `text.encode('utf-8')` with bytes reconstructed by the native decoder. Tiktoken supplies bytes directly; the Hugging Face decoder supplies Unicode, which is encoded to UTF-8. A decoder's replacement or normalization therefore counts as changed bytes.

Each observation belongs to exactly one fidelity category: **exact**, **canonical-only difference** (same NFC form), **other mismatch**, or **error**. Canonical-only differences are failures of byte-exact fidelity. Unknown-token emissions are counted independently. Mismatch examples preserve original and reconstructed text; counts and hashes preserve full identity.

**Bytes/token = sum(original UTF-8 bytes) ÷ sum(emitted tokens).** For example, 1,000,000 bytes / 400,000 tokens = **2.5 bytes/token**. A tokenizer producing 320,000 tokens has **3.125 bytes/token** and emits **20% fewer tokens** for those same bytes. This says nothing by itself about model accuracy. An error makes the group's ratio unavailable; lossy compression remains visible but is explicitly diagnostic. Per-language rows are the primary evidence. Any pooled UI number is a ratio of pooled counts, not a mean of document ratios or a universal quality score.

## Vocabulary semantics

Vocabulary size counts distinct usable IDs returned by the native inventory, including special IDs. `id_space` is maximum ID + 1; holes and model embedding padding rows are different quantities. BLOOM tokenizer entries must not be reported as its model's padded embedding rows.

Overlap compares **unique ordinary byte pieces**, excluding special IDs and pieces without recoverable bytes. Jaccard = shared / union. Directional coverage of A = shared / unique pieces in A. With 100 pieces in A, 200 in B, and 50 shared, coverage is **50% of A**, **25% of B**, and Jaccard is **20%**. Duplicate IDs with identical bytes count once for overlap, but individually in vocabulary ID counts.

ByteLevel markers are decoded to bytes; byte-fallback pieces can be incomplete UTF-8; SentencePiece's `▁` is represented as its space boundary. AddedTokens outside the model vocabulary use literal bytes. Equal byte pieces do not imply equal context behavior. Native surfaces, hex bytes, IDs, byte fragments, and special IDs remain distinct in exports.

Scripts use pinned Unicode **15.1.0 Scripts.txt**. Common/Inherited characters are not treated as evidence for a language. A piece with multiple strong scripts is Mixed. Numeric properties use the recorded Python Unicode version. Code-associated, identifier and emoji-candidate labels are documented heuristics. Categories overlap, so their percentages need not sum to 100%.

## Runtime and memory

Each tokenizer runs in a fresh subprocess using only its native runtime, without the vocabulary classification tables. The input interleaves the same frozen language documents to at least 1 MB. A warm-up precedes **5 encode/decode repeats**; MB/s uses median elapsed time, decimal MB, original input bytes for encoding and produced bytes for decoding. All five timings and fidelity counts are saved. Tokenizers parallelism is disabled; tiktoken's batch API uses one thread. These are machine-specific measurements, not portable speed rankings.

RSS before loading, RSS after loading, incremental RSS, and peak RSS sampled every **5 ms** are separate fields. Peak sampling can miss brief spikes. File size is separate from process memory. Load time includes runtime import and cached artifact loading; it is not a cold network download measurement.

## Expansion and conversion

V3.1 is compared to the **Gemma 3 control**, not Gemma4. The local control removes exactly 2,000 appended AddedTokens while preserving the inherited model, merges and components. Its upstream gated replay is unverified. The rebuild recipe reproduces the supplied candidate file exactly. Fresh reports include gains and regressions per language, source-group bootstrap **95% intervals with 1,000 replicates**, and their exact-pair scope. These intervals have no multiple-comparison correction and do not establish model-quality gains.

Whole-sequence conversion rejects special/control IDs and incomplete byte streams, decodes the complete ordinary sequence, encodes the target, then compares reconstructed bytes. It cannot undo normalization already performed by the source. Standalone vocabulary maps describe piece matching and one-to-many encodings; they are not universal token-ID substitution tables or model-weight conversion.

Embedding initialization, embedding-space mapping and cross-tokenizer distillation are separate studies. Published results, historical local reports, executable baselines and fresh local reproductions have explicit evidence labels.

## Expanded Unicode and family behavior

FLORES-200 dev/devtest contribute **409,836 sentences / 204 script varieties**. MADLAD contributes **5,237 bounded-prefix documents / 418 configurations**. These augment the **5,941 FineWeb documents** and **10,556 diagnostics** without modifying them. FLORES+ was not run; MADLAD is not a uniform or established historical holdout sample.

UTF-16LE and BE compare strict encodings of the original and reconstructed Unicode without BOMs. They are transport checks, not alternative tokenizer code-unit inputs. Each profile has **10 valid probes and 6 malformed inputs rejected before tokenization**. Token counts do not depend on this transport switch.

Full native tokenizer identity includes special-token policy and wrapper limits. Kimi's shared ordinary rank file does not collapse its 8 profiles: special names/ranges or direct marker-recognition policies differ. Four expose two reserved IDs beyond their model configs' declared rows. Native tiktoken data paths are inspected; complete Transformers wrapper parity is not independently established. [Kimi details](docs/KIMI.md).

Runtime cohorts use the same frozen **216-document / 1,112,864-byte** interleaved FineWeb input with one warm-up and five timed repeats in sequential fresh processes. The 54 historical rows were not measured simultaneously. Loaded RSS, incremental RSS, sampled peak, tokenizer file bytes and browser memory are distinct quantities.
