# Tokenizer training comparison — V4 series

Updated **23 October 2026**. **6 completed runs, 7 tokenizer variants, 1 missing result**.
Each completed run used approximately **1,200 seconds / 20 minutes on one TPU v5e-1**.
These are exploratory training results from one run per completed tokenizer, without independent replay.
Machine reports exist for Gemma 4 and BLOOM; the other four completed rows come from user-supplied cell outputs and retain their displayed precision.

All variants keep the Gemma3_270M transformer: **100,326,016 backbone parameters**, width **640**, fresh random initialization and AdamW.
The tied vocabulary table changes size, so total parameters differ. This measures tokenizer plus vocabulary cost under a time budget; it does not isolate segmentation at fixed parameter count.
The frozen source corpus is approximately **256 MiB**, with 16 human languages and 6 programming/markup domains.
Evaluation uses the existing **16-language FineWiki macro BPB at 1,024-token context**, plus the in-mixture holdout.
These pilots are separate from the proposed competition's 30-minute budget and hidden FineWeb 2 evaluation.

## Training and quality

BPB is lower-is-better. Bytes/s includes repeated corpus exposure and is not unique data consumed.
**Qwen3 uses NFC-normalized scoring bytes and is listed separately from original-byte results.** Its BPB and bytes/s cannot be treated as measurements on identical original bytes without a common-text rescore.

| Tokenizer / notebook | Parameters | Updates | Scored tokens | Tokens/s | Scoring bytes/s | FineWiki BPB | In-mixture BPB | Evidence |
|---|---|---|---|---|---|---|---|---|
| [Gemma 3 V4](../gemma_270m_colab_v5e1_V4.ipynb) | 268,100,736 | 16,319 | 44,560,465 | 37,133.6 | 156,460.0 | 1.944411 | 1.540723 | [cell output](evidence/V4/cell-output.txt) |
| [Gemma 4 V4.1.1](../gemma_270m_colab_v5e1_V4_1_1.ipynb) | 268,100,096 | 15,340 | 42,052,321 | 35,043.4 | 147,479.6 | 1.963398 | 1.541087 | [machine report](evidence/V4.1.1-complete/result.json) |
| [o200k_base V4.2.2](../gemma_270m_colab_v5e1_V4_2_2.ipynb) | 228,327,936 | — | — | — | — | — | — | Missing report |
| [BLOOM V4.3.1](../gemma_270m_colab_v5e1_V4_3_1.ipynb) | 260,763,136 | 17,112 | 47,053,541 | 39,208.3 | 172,778.3 | 1.955738 | 1.543808 | [machine report](evidence/V4.3.1/result.json) |
| [Llama 3 V4.4.1](../gemma_270m_colab_v5e1_V4_4_1.ipynb) | 182,411,776 | 24,544 | 80,307,432 | 66,921.9 | 277,275.9 | 1.782555 | 1.431393 | [cell output](evidence/V4.4.1/cell-output.txt) |
| [Qwen3 V4.5.1](../gemma_270m_colab_v5e1_V4_5_1.ipynb) **NFC** | 197,396,096 | 23,115 | 75,557,854 | 62,961.3 | 250,442.8 | 1.824585 | 1.457129 | [cell output](evidence/V4.5.1/cell-output.txt) |
| [DeepSeek-V3-Base V4.6.1](../gemma_270m_colab_v5e1_V4_6_1.ipynb) | 182,769,536 | 24,477 | 80,105,111 | 66,754.1 | 277,853.6 | 1.786081 | 1.428672 | [cell output](evidence/V4.6.1/cell-output.txt) |

Llama 3 has the lowest **reported original-byte FineWiki BPB: 1.782555**. DeepSeek follows at **1.786081**, a difference of **0.003526 BPB**. That small gap is not evidence of a repeatable winner from one run. Llama 3 processes **1.77×** as many attributed original bytes/s as the Gemma 3 baseline, with **31.96% fewer parameters**. Different vocabulary cost, discovered batches and compilation overhead contribute to the observed result.

## Batches, timing and cost

| Tokenizer | Short / long batch | Training s | Initial compile s | Compile inside training s | Whole notebook s | Training CU | Total CU |
|---|---|---|---|---|---|---|---|
| Gemma 3 | 3×1,024 / 1×2,048 | 1,200.005 | 11.11 | 66.133 | 1,517.332 | 0.97334 | 1.23072 |
| Gemma 4 | 3×1,024 / 1×2,048 | 1,200.008 | 423.82 | 132.379 | 1,958.686 | 0.97334 | 1.58871 |
| BLOOM | 3×1,024 / 1×2,048 | 1,200.093 | 252.06 | 56.619 | 2,107.601 | 0.97341 | 1.70950 |
| Llama 3 | 3×1,024 / 2×2,048 | 1,200.017 | 86.21 | 117.877 | 1,830.649 | 0.97335 | 1.48486 |
| Qwen3 | 3×1,024 / 2×2,048 | 1,200.068 | 86.29 | 108.973 | 2,054.329 | 0.97339 | 1.66629 |
| DeepSeek-V3-Base | 3×1,024 / 2×2,048 | 1,200.003 | 87.35 | 117.148 | 1,971.013 | 0.97334 | 1.59871 |

CU figures are estimates at the supplied **2.92 CU/hour**, not billing receipts or USD prices. Total CU covers the notebook through its report; connected runtime afterward adds cost.
Gemma 3 V4 and Gemma 4 V4.1.1 discover batches immediately; BLOOM and the three later variants defer discovery until update 1,000. The o200k V4.2.2 notebook also defers discovery, but its run report is missing. Earlier V4.2.1 is the fixed-batch o200k alternative.
All keep the cumulative 3M short / 1M long schedule outside discovery. The timed window includes candidate compilation; initial compilation and evaluations are outside it. Whole-notebook timings also depend on cache reuse and data preparation.

## Training memory and hardware telemetry

Sampled active memory is **measured**, but synchronized samples can miss temporary peaks and runtime buffers. The compiler column is an **estimate**, not measured HBM usage. Device idle is from a post-save mixed diagnostic, not a whole-training MXU utilization measurement.

| Tokenizer | Capacity GiB | Sampled active GiB | Active % | Compiler + other resident estimate % | Diagnostic device idle % | EST. compute % peak | EST. HBM % peak |
|---|---|---|---|---|---|---|---|
| Gemma 3 | 15.748 | 6.024 | 38.250 | 81.988 | 0.800 | 32.51 | 56.77 |
| Gemma 4 | 15.748 | 6.296 | 39.978 | 81.995 | 0.800 | 32.21 | 56.24 |
| BLOOM | 15.748 | 5.543 | 35.201 | 79.134 | 0.900 | 32.71 | 56.60 |
| Llama 3 | 15.748 | 5.127 | 32.558 | 63.665 | 1.300 | 52.09 | 58.94 |
| Qwen3 | 15.748 | 5.381 | 34.167 | 68.863 | 1.500 | 51.67 | 59.17 |
| DeepSeek-V3-Base | 15.748 | 5.122 | 32.526 | 63.720 | 1.300 | 51.99 | 58.84 |

**Measured matrix-unit utilization, measured memory-bandwidth utilization and host CPU idle are unavailable in all six supplied outputs.** The runtime did not export supported TPU utilization-counter messages. The two EST. columns use compiler FLOP/byte costs divided by observed diagnostic durations; they are not hardware-counter measurements.
All six report training/learning **PASS**, performance goal **FAIL**, profiling requirements **FAIL**, and telemetry **PARTIAL**. Successful training does not mean every requested hardware measurement was obtained.

## Operation timings

Milliseconds per optimizer update from the **post-save diagnostic**, using 16 short-context and 16 long-context updates at the final batches. Values sum HLO self-time, include forward/backward, exclude idle, and are not elapsed update wall time. Because the long batch is 1 for the first three runs and 2 for the later three, operation timings are not comparisons at identical work per update.

| Tokenizer | Attention | Cross entropy | MLP | Optimizer | Other/fused | Vocabulary projection |
|---|---|---|---|---|---|---|
| Gemma 3 | 8.469 | 7.598 | 6.482 | 10.713 | 25.402 | 16.269 |
| Gemma 4 | 8.469 | 7.598 | 6.483 | 10.710 | 26.067 | 16.269 |
| BLOOM | 8.446 | 7.155 | 6.468 | 10.237 | 24.977 | 15.438 |
| Llama 3 | 13.429 | 1.273 | 9.423 | 6.122 | 7.775 | 11.083 |
| Qwen3 | 13.433 | 1.573 | 9.437 | 6.829 | 8.114 | 13.111 |
| DeepSeek-V3-Base | 13.429 | 1.373 | 9.422 | 6.139 | 7.784 | 11.124 |

## Per-language final FineWiki BPB

Qwen3's column uses NFC text; the other columns use original text.

| Language | Gemma 3 | Gemma 4 | BLOOM | Llama 3 | Qwen3 (NFC) | DeepSeek-V3-Base |
|---|---|---|---|---|---|---|
| arb_Arab | 1.598867 | 1.624386 | 1.645917 | 1.450737 | 1.490803 | 1.459438 |
| cmn_Hani | 2.302597 | 2.309453 | 2.417828 | 2.115211 | 2.155776 | 2.127665 |
| deu_Latn | 1.898481 | 1.986761 | 2.013540 | 1.734710 | 1.910696 | 1.760218 |
| eng_Latn | 1.585709 | 1.589636 | 1.611360 | 1.478534 | 1.501451 | 1.484132 |
| fra_Latn | 2.052126 | 2.058686 | 2.041814 | 1.892778 | 1.877436 | 1.879039 |
| hin_Deva | 1.160201 | 1.149223 | 1.106526 | 1.033069 | 1.054409 | 1.038038 |
| ibo_Latn | 2.378445 | 2.382442 | 2.375571 | 2.185157 | 2.226825 | 2.199025 |
| jpn_Jpan | 1.742612 | 1.739637 | 1.801762 | 1.542546 | 1.622719 | 1.555991 |
| por_Latn | 1.970525 | 1.956128 | 1.935237 | 1.810502 | 1.845504 | 1.804885 |
| rus_Cyrl | 1.793660 | 1.821961 | 1.797843 | 1.662710 | 1.671494 | 1.675160 |
| sot_Latn | 2.141096 | 2.192906 | 2.154994 | 1.986588 | 2.024218 | 1.972348 |
| spa_Latn | 1.913336 | 1.893882 | 1.880264 | 1.736058 | 1.768860 | 1.746945 |
| swh_Latn | 1.934032 | 1.953522 | 1.879710 | 1.787653 | 1.811160 | 1.787957 |
| tsn_Latn | 2.006128 | 2.078069 | 2.007315 | 1.862686 | 1.889637 | 1.849611 |
| xho_Latn | 2.410164 | 2.422625 | 2.394424 | 2.193262 | 2.241187 | 2.199594 |
| zul_Latn | 2.222599 | 2.255053 | 2.227699 | 2.048676 | 2.101187 | 2.037257 |

## Evidence and remaining files

- **Missing:** o200k_base's completed report ZIP, for V4.2.1 or V4.2.2. No training results have been invented for that row.
- Full report ZIPs for **Gemma 3 V4, Llama 3 V4.4.1, Qwen3 V4.5.1 and DeepSeek V4.6.1** would replace the cell-output evidence with exact machine-readable metrics, full logs and identities. The supplied cell outputs are enough to record the displayed results now.
- The [earlier incomplete Gemma 4 attempt](evidence/V4.1.1-incomplete/training.jsonl) ends at **48 successful updates / 81,918 scored tokens**. It has no final evaluation or completed result; the supplied files do not establish the termination cause. The completed Gemma 4 rerun is a separate attempt and is the table row above.
- [Evidence manifest](evidence-manifest.json) hashes every imported file. Original selected machine reports, training logs and pasted cell outputs are checked in unchanged. Weights, corpus text, tokenizer assets and raw profile archives are not included. No independent training replay was performed.
- [JSON](comparison.json) retains parsed values and evidence labels; [CSV](comparison.csv) provides flat training metrics. Rebuild using `python3 experiments/gemma-270m-colab/training-results/build_comparison.py` from the repository root. The builder validates evidence hashes and uses no accelerator libraries.
