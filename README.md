# Tokenizer Lab

A reproducible hub for understanding tokenizer strengths, weaknesses, and tradeoffs across languages. Inspect the same UTF-8 text, vocabulary pieces, and numeric/code diagnostics across **6 primary tokenizers**, with **1 parent control**, **15 vocabulary pairs**, and **30 directional vocabulary maps**.

This is intrinsic analysis: **no model training or model weights are required**. Compression is not model quality. Embedding transfer and distillation have a separate, cited [research library](docs/research/index.md).

| Tokenizer | Purpose |
|---|---|
| Original Gemma4 | General multilingual comparator |
| BLOOM | General multilingual comparator |
| JAIS Family 256M | Arabic/English comparator |
| Qwen3 | General multilingual comparator; normalization is measured explicitly |
| Native o200k | Native `tiktoken` implementation, not a converted tokenizer |
| Gemma V3.1 | Experimental Gemma 3 expansion: **2,000 appended IDs** |
| Gemma 3 parent control | Expansion control, reconstructed by removing those 2,000 additions |

Exact repository revisions are in [the registry](configs/registry.json). The reconstructed parent has not been verified against Google's gated download. Its identity and exact candidate rebuild are recorded in [the expansion experiment](experiments/gemma-v31/identity.json).

## Initial measured results

- **115,479 observations**: 7 tokenizers × 16,497 original inputs.
- **5,941 FineWeb documents**, **23,437,127 UTF-8 bytes**, **24 language/script varieties**. All 7 tokenizers reconstruct every sampled FineWeb document exactly.
- On **10,556 constructed diagnostics**, Gemma4, Gemma V3.1 and its parent each change 20 inputs; Qwen3 changes 24 through canonical Unicode normalization. BLOOM, JAIS and native o200k reconstruct all diagnostics exactly. These are observed samples, not a proof of losslessness for every possible string.
- V3.1 saves **104,899 / 5,681,387 tokens = 1.8464%** on the pooled FineWeb sample. **17 languages improve; 7 regress**. Igbo uses **11.68% fewer** tokens; Portuguese uses **3.97% more**. No model-quality gain is established.

See [the complete generated report](reports/v0.1.md), per-language tables and artifact identities. Runtime numbers apply to the recorded host; no combined winner score is assigned.

## Start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Run from the cloned repository; the editable research project keeps configuration, inputs, and outputs together.

```sh
git clone https://github.com/godkdf/Shinning-Tokens.git
cd Shinning-Tokens
uv sync --frozen --extra test
uv run tokenizer-lab doctor
uv run pytest -q
```

The repository may be private. GitHub access is required until its owner changes that audience.

The published result bundle can be explored without downloading model weights. For a fresh measurement:

```sh
uv run tokenizer-lab prepare
uv run tokenizer-lab run --config configs/launch.json
uv run tokenizer-lab verify
uv run tokenizer-lab serve
```

Open **http://localhost:8765**. Downloaded tokenizer files are cached outside tracked source. Official Gemma 3 access requires accepting upstream terms and authenticating with Hugging Face. The frozen V3.1 input package provides the exact experimental inputs with their agreement and notices; follow [artifact setup](docs/REPRODUCE.md) before the full run.

## What is measured

- **Original UTF-8 fidelity:** exact bytes, canonical-normalization differences, other mismatches, errors, and unknown-token emissions. Failures remain in the denominator.
- **Capacity:** ordinary/special IDs, ID space and holes, unique byte pieces, file size, scripts, overlapping numerical/code/whitespace categories.
- **Text efficiency:** bytes/token and emitted token counts by language and diagnostic suite. Pooled counts are explicitly labelled.
- **Runtime:** native encode/decode MB/s, 5 repeats after warm-up, load time, process RSS and sampled peak in fresh subprocesses.
- **Overlap:** unique ordinary byte-piece intersection, Jaccard, directional coverage, and category overlap. Matching pieces do not imply matching segmentation.
- **Converters:** verified whole-sequence decode → encode, plus standalone vocabulary maps labelled for context dependence.
- **Experiments:** V3.1 versus its parent, paired uncertainty, regressions, and a historical embedding mapping case.

The launch corpus targets **1,000,000 bytes × 24 language/script varieties**, using pinned FineWeb2 test shards and original FineWeb for English. Whole-document sampling can overshoot; a small source shard can fall short. Actual sizes, source revisions, exclusions, and shortfalls are in [the corpus manifest](benchmarks/fineweb-v0.1/manifest.json). Additional diagnostics contain **500 Unicode**, **10,000 number**, and **56 code** cases. These constructed cases are not representative arithmetic or coding ability tests.

## Read and extend

| Location | Contents |
|---|---|
| [BENCHMARK.md](BENCHMARK.md) | Definitions, sampling, fidelity and runtime protocol |
| [reports/v0.1.md](reports/v0.1.md) | Generated measured results |
| [results/v0.1/hub.json](results/v0.1/hub.json) | Machine-readable aggregates and provenance |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Inputs, commands, checks and limitations |
| [docs/SCHEMAS.md](docs/SCHEMAS.md) | Documents, observations, vocabulary and mapping formats |
| [docs/research/](docs/research/index.md) | FVT, WECHSEL, FOCUS, CW2V, mapping, ZeTT, ULD and ALM |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Add a tokenizer, analysis or experiment |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Upstream terms and data attribution |

Source code and original diagnostics: Apache-2.0. Upstream tokenizer and corpus terms remain applicable. Cite [CITATION.cff](CITATION.cff), the exact result manifest, and the relevant upstream work.
