# Tokenizer Lab

Compare how tokenizers represent the same text: what they preserve, how many tokens they use, what their vocabularies contain, and how fast they run. **No model training or model weights are required.**

**Release 0.2.3 · 54 tokenizer profiles · 23,304,780 observations · 1,431 vocabulary comparisons.**

The lab includes Gemma, BLOOM, JAIS, GPT-2, native o200k, Llama 3, GLM, Qwen, DeepSeek, MiMo, MiniMax and Kimi/Moonlight generations. Exact tokenizer aliases are grouped; different special IDs or processing policies remain distinct. See [the inventory](docs/tokenizer-families.md) and [Kimi identities](docs/KIMI.md).

## Tokenizers during model training

The separate [Gemma training pilots](experiments/gemma-270m-colab/README.md) train the same Gemma transformer with different vocabulary tables for **20 minutes on one TPU v5e-1**. The [training comparison](experiments/gemma-270m-colab/training-results/README.md) records **7 completed runs across 7 tokenizer variants**, with BPB, throughput, memory, operation timings and estimated CU. Llama 3 reports **1.782555 original-byte FineWiki BPB**; DeepSeek-V3-Base reports **1.786081**. Qwen3 uses separately labelled NFC-normalized scoring. Native o200k reports **1.825707 original-byte FineWiki BPB**.

These are single-run pilot observations, with source evidence and limitations recorded in the comparison. The intrinsic lab below still requires no model training. Only **V4+ training notebooks** remain in the current Colab directory; pre-V4 notebooks are retained in Git history.

## Start the local app

1. Download the **0.2.3 starter ZIP for your computer** from the project's [release assets](https://github.com/godkdf/Shinning-Tokens/releases).
2. Extract the complete ZIP. On Windows open **Launch Tokenizer Lab.cmd**; on either Mac architecture open **Launch Tokenizer Lab.command**; on Linux run **launch-tokenizer-lab.sh**.
3. In **Downloads & storage**, select only the tokenizers, benchmarks and languages you need. Review the sizes, then download explicitly.
4. Run the selected benchmark, compare saved results, inspect vocabularies, or convert text between installed tokenizers.

Starters contain **0 tokenizer assets and 0 benchmark-text payloads**. Python, native dependencies, the interface and saved aggregate results are included. After preparing selected inputs online once, runs work offline. Language text stays compressed until selected. You can search languages by normal name and browse overlapping continent groups.

Custom tokenizers: import a local Hugging Face-compatible **tokenizer.json**, give it a name, and select the benchmarks and languages to run. Import does not execute repository Python. [Offline setup and system requirements](docs/OFFLINE.md).

**Verification:** the current Apple Silicon kit is exercised locally. Windows, Linux and Intel Mac packages have pinned runtime/wheel hashes and archive checks; native execution on those three platforms remains deferred. The manual portability workflow is available for later verification.

## Run from source

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The project runs from its source checkout rather than as a standalone PyPI installation.

```sh
git clone https://github.com/godkdf/Shinning-Tokens.git
cd Shinning-Tokens
uv sync --frozen --extra test
uv run --frozen tokenizer-lab launch
```

The launcher opens at **http://localhost:8765/launcher.html**. The explorer uses the same local service. The repository currently requires GitHub access while private; publishing or changing visibility is a separate owner action.

`uv run --frozen tokenizer-lab doctor` checks installed assets without fetching them. `uv run --frozen pytest -q` runs the tests. Asset integration tests skip when their explicitly cached inputs are absent; skips are not evidence that those tokenizer implementations passed.

## What has been measured

The separate [realistic code study](reports/code-realistic-v0.1.md) adds **32 languages, 32,000 real source files, 3,200 actual edit pairs, 1,600 constructed contexts and 1,600 controlled stress cases**. All **54 profiles** completed **2,419,200 measurements with 0 execution errors**; **28 profiles** preserved every input byte-for-byte. Its corpus is provisional, with remaining size-coverage and validation gaps documented in the report. The historical totals below describe the original benchmark.

Open **Realistic code** in Tokenizer Lab to compare profiles by workload, programming language and split, inspect coverage and export CSV tables. It defaults to held-out evaluation. [Preparation, rerun commands and limits](docs/CODE_BENCHMARK.md). Rebuild the explorer with `uv run --frozen tokenizer-lab build-site` after preparing the audited study. Saved code results require no tokenizer assets to browse; code reruns use the dedicated runners described in the guide.

Every profile receives the same **431,570 original inputs across 867 dataset/language partitions**.

| Dataset | Inputs | Coverage |
|---|---:|---|
| FineWeb mixture | 5,941 | 24 pinned language/script varieties |
| FLORES-200 dev | 203,388 | 204 varieties × 997 sentences |
| FLORES-200 devtest | 206,448 | 204 varieties × 1,012 sentences |
| MADLAD sample | 5,237 | 418 bounded clean-source configurations |
| Unicode, number and code diagnostics | 10,556 | 500 + 10,000 + 56 constructed cases |
| **Total per tokenizer** | **431,570** | **867 partitions** |

The original 44-profile expanded measurements remain frozen in v0.2. GPT-2 and Llama 3 add 2 profiles in v0.3; Kimi/Moonlight adds 8 in v0.4. These result-cohort names are separate from application version **0.2.3** and measurement protocol **intrinsic-v0.2**.

Measured properties:

- **Fidelity:** exact original UTF-8, UTF-16LE and UTF-16BE bytes; canonical-only changes, other mismatches, errors and unknown emissions remain visible.
- **Efficiency:** emitted tokens and original UTF-8 bytes/token, by dataset and language.
- **Vocabulary:** ordinary/special IDs, ID gaps, scripts, overlapping numeric/code categories and unique byte pieces.
- **Overlap and conversion:** 1,431 undirected vocabulary pairs; verified whole-sequence decode → encode; 30 historical standalone maps with context warnings.
- **Runtime:** native encode/decode throughput, load time, process RSS and sampled peak, with one warm-up and five repeats in fresh processes.

**Compression is not model quality.** FLORES-200 was run; FLORES+ was not. MADLAD uses bounded prefixes, not uniform samples, and is not established as held out from historical token selection. Runtime rows were collected in separate cohorts, not one simultaneous 54-profile rerun. Some legacy wrappers have inspected data-only ports without independently established full upstream wrapper parity.

Examples of findings:

- GPT-2, Llama 3 and all **8 new Kimi/Moonlight profiles** reconstruct their sampled inputs exactly in all three encodings.
- Kimi Dev is an exact Qwen 2.5 tokenizer alias. Its NFC normalization changes **11,521 / 431,570 inputs (2.6696%)**, all canonically equivalent but not byte-exact.
- Gemma V3.1 appends **2,000 IDs** to the recorded Gemma 3 parent. Its frozen FineWeb analysis saves **1.8464%** pooled tokens, with **17 languages improving and 7 regressing**. No model-quality gain is established. The reconstructed parent is not independently verified against Google's gated artifact.

## Results, inputs and reuse

| Read | Purpose |
|---|---|
| [Latest results: v0.4](reports/v0.4.md) | Kimi tables, inherited audit chain and current totals |
| [Expanded v0.2](reports/v0.2.md), [references v0.3](reports/v0.3.md) | Earlier frozen cohorts and their timing scopes |
| [Benchmark definitions](BENCHMARK.md) | Denominators, fidelity, bytes/token, vocabulary and runtime |
| [Reproduce measurements](docs/REPRODUCE.md) | Explicit downloads, cached runs, custom inputs and audits |
| [Data formats](docs/SCHEMAS.md) | Read aggregates, vocabulary rows, checkpoints and provenance |
| [Transfer research](docs/research/index.md) | FVT, WECHSEL, FOCUS, CW2V, mapping, ZeTT and distillation |
| [Contribute](CONTRIBUTING.md) | Add a tokenizer, diagnostic, analysis or experiment |
| [Release preparation](docs/RELEASE.md) | Build, verify and publish the concrete release assets |

The **source archive** contains code, manifests, documentation, notices and aggregates; it omits original-text payloads, tokenizer assets and per-document checkpoints. The **explorer archive** contains saved vocabularies, mappings and compressed texts. The **starter ZIPs** contain the local application with zero selected inputs. Choose the artifact that matches the job; full release sizes and SHA-256 values are recorded in its manifest.

Licensing: original source and diagnostics use **Apache-2.0**. Upstream tokenizer, dataset and metadata terms remain applicable; the lab does not relicense them. See [THIRD_PARTY.md](THIRD_PARTY.md), [licenses](licenses/), [CITATION.cff](CITATION.cff) and the exact result manifest when citing a result. [Security and local-service scope](SECURITY.md).
