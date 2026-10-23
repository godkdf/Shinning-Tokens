# Gemma tokenizer training pilots — V4 series

Train the same Gemma3_270M transformer from random initialization with different
tokenizers, then compare learning and throughput within a **1,200-second /
20-minute training window on one TPU v5e-1**. The vocabulary table changes
size; all variants keep **100,326,016 backbone parameters** and width **640**.

**[Training statistics: BPB, throughput, memory, timings, cost and evidence](training-results/README.md)**
contains **6 completed runs**, an earlier incomplete attempt and the missing
o200k result. [CSV](training-results/comparison.csv) and
[JSON](training-results/comparison.json) are available for reuse.

## Notebooks

Only upload the chosen notebook to Colab, select TPU v5e-1, use your existing
Drive account and run top to bottom. Each notebook embeds its runtime code.
No supporting source files need uploading. Preprocessing, initial compilation,
evaluation, saving and profiling add time outside the timed training window.

| Tokenizer | Notebook | Native IDs | Tied table rows | Total parameters | Batch policy |
|---|---|---:|---:|---:|---|
| Gemma 3 | [V4](gemma_270m_colab_v5e1_V4.ipynb) | 262,145 | 262,148 | 268,100,736 | Discover both contexts immediately, then lock |
| Gemma 4 | [V4.1.1](gemma_270m_colab_v5e1_V4_1_1.ipynb) | 262,144 | 262,147 | 268,100,096 | Discover both contexts immediately, then lock |
| Native o200k_base | [V4.2.1](gemma_270m_colab_v5e1_V4_2_1.ipynb) | 200,000 | 200,003 | 228,327,936 | Fixed 3×1,024 /1×2,048 |
| Native o200k_base | [V4.2.2](gemma_270m_colab_v5e1_V4_2_2.ipynb) | 200,000 | 200,003 | 228,327,936 | Discover after 1,000 updates, then lock |
| BLOOM | [V4.3.1](gemma_270m_colab_v5e1_V4_3_1.ipynb) | 250,680 | 250,683 | 260,763,136 | Discover after 1,000 updates, then lock |
| Original Llama 3 | [V4.4.1](gemma_270m_colab_v5e1_V4_4_1.ipynb) | 128,256 | 128,259 | 182,411,776 | Discover after 1,000 updates, then lock |
| Qwen3, official NFC | [V4.5.1](gemma_270m_colab_v5e1_V4_5_1.ipynb) | 151,669 | 151,672 | 197,396,096 | Discover after 1,000 updates, then lock |
| DeepSeek-V3-Base | [V4.6.1](gemma_270m_colab_v5e1_V4_6_1.ipynb) | 128,815 | 128,818 | 182,769,536 | Discover after 1,000 updates, then lock |

Three additional vocabulary rows are reserved for packing PAD/BOS/EOS. Native
IDs are mapped away from PAD=0, including tokenizers whose original ID 0 is
a real text or special token. Input embeddings and output head remain tied.
Tokenizer reference, revision and SHA256 are recorded by each notebook.

The delayed-discovery versions start with **3×1,024 /1×2,048**, measure
independent short/long batch candidates after **1,000 completed updates**, and
lock once. Recoverable memory failures can trigger discovery earlier. Search
uses at most **256 real updates**, **16-update windows**, **85% memory
admission**, at least **2% useful-throughput improvement**, and batch cap **32**.
These are measured reasonable batches, not a proof of the physical OOM limit.
No rechecking after locking. Outside discovery, use the cumulative repeating
**3M short /1M long processed-token schedule**. No checkpoint restoration.

## Data and comparison limits

Keep the frozen approximately **256 MiB** raw corpus, source identities and
held-out splits. Encode documents once, then pack both contexts. Package and
source caches can be reused; encoded/packed cache identities are tokenizer
specific. Training uses BF16 computation, FP32 master weights, AdamW learning
rate **0.0003**, warmup **5 updates**, clipping **1.0**, decay **0.01**, seed **42**.

Human languages: Arabic, Mandarin Chinese, German, English, French, Hindi,
Igbo, Japanese, Portuguese, Russian, Southern Sotho, Spanish, Swahili,
Setswana, Xhosa and Zulu. Programming/markup domains: C++, HTML, Java,
JavaScript, Python and Rust. Other categories include educational PDF text,
DCLM web text, educational web text, multilingual web, math and translations.

Primary evaluation is mean BPB across **16 FineWiki languages**, at fixed
**1,024-token context**. Token loss and tokens/s use different token units
across tokenizers. Original-byte BPB and attributed original bytes/s support
cross-tokenizer comparisons, subject to the recorded run differences.
**Qwen3 preserves its official NFC normalization and scores normalized text**;
its BPB/bytes/s need a separate label. Raw source bytes remain preserved.
BLOOM rejects inputs containing unsupported byte-losing text rather than
silently dropping it; the supplied completed corpus passed its checks.

The total parameter counts, batch discovery timing, long batch and compilation
overhead differ. These are single-run pilot observations, not independently
verified rankings or a fixed-parameter segmentation comparison. This pilot
uses 20-minute FineWiki evaluation, separate from the draft Shining the Gem
competition's proposed 30-minute hidden FineWeb 2 protocol.

## Source and history

Version folders contain source builders, embedded runners, requirements and
available build provenance. Previously delivered V4+ notebooks are preserved
unchanged. Builders refuse to overwrite delivered notebooks; inspect or derive
a new version instead. The original V4 builder can retrieve its exact parent
from Git history, so a full-history clone is needed to reconstruct that parent.
Running a delivered notebook or runner does not require historical notebooks.

At the user's request on **23 October 2026**, **18 pre-V4 notebooks** from this
Colab training series were removed from the current files. They remain at
commit [`d1143a10cf5ecef2ed4b8eeb8cdfe5f2edc74a81`](https://github.com/godkdf/Shinning-Tokens/tree/d1143a10cf5ecef2ed4b8eeb8cdfe5f2edc74a81/experiments/gemma-270m-colab).
Earlier source/helper folders remain historical engineering material; their
notebook references refer to that history. Other intrinsic tokenizer studies
and their notebook/version names are separate experiments.

This publication was checked with static/CPU validation only. No TPU tests,
accelerator initialization or new model training were performed.
