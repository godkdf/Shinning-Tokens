# V3.9 corpus: 256 MiB

All percentages are **original UTF-8 bytes**, not tokens. Quotas can finish
up to **3 bytes short per bucket** to preserve UTF-8 boundaries.

| Source/category | Share | Target MiB |
|---|---:|---:|
| FinePDFs-Edu via shuffled Smol-Data | 25% | 64.00 |
| DCLM via shuffled Smol-Data | 15% | 38.40 |
| FineWeb-Edu via shuffled Smol-Data | 10% | 25.60 |
| FineWeb2 multilingual web | 30% | 76.80 |
| FineMath 3+ | 5% | 12.80 |
| Python (CodeParrot clean) | 3% | 7.68 |
| Java (Stack-smol) | 2% | 5.12 |
| C++ (Stack-smol) | 2% | 5.12 |
| JavaScript (Stack-smol) | 1.5% | 3.84 |
| Rust (Stack-smol) | 1% | 2.56 |
| HTML (Stack-smol) | 0.5% | 1.28 |
| FineTranslations, original + English | 5% | 12.80 |
| **Total** | **100%** | **256.00** |

FineWeb2 allocation, as percentages of the entire training pool:

| Language | Share |
|---|---:|
| French | 2.5% |
| Spanish | 2.0% |
| Portuguese | 1.5% |
| Chinese | 3.5% |
| Arabic | 3.5% |
| Hindi | 3.5% |
| Swahili | 2.0% |
| Zulu | 1.0% |
| Xhosa | 1.0% |
| Setswana | 1.0% |
| Sesotho | 0.8% |
| Russian | 2.5% |
| Japanese | 2.5% |
| German | 1.7% |
| Igbo | 1.0% |
| **Multilingual total** | **30.0%** |

French was already present. Russian, Japanese, German and Igbo expand the
human-language list from **12 to 16**, including English. Translation buckets
remain French/Arabic/Hindi/Swahili at **1.25% each**, containing both the original
and English side, with educational score ≥2 and no early-stop rows. Actual
language shares include English translations and are reported. Code uses six
distinct domain labels; its `eng_Latn` language ID is an accounting placeholder,
not a claim that comments or strings are all English. These are declared
engineering allocations, not measured optimal mixture weights.

Holdout: **64 KiB ×16 FineWiki languages =1 MiB**, plus **1 MiB** in-mixture
allocated by the source percentages. FineWiki configurations are checked at the
pinned dataset revision; unavailable configurations or byte shortages stop
preparation. No fabricated examples, language replacement or quota renormalizing.

Freeze source commits once per new shared recipe. Preserve exact raw byte
hashes, provenance, deterministic group splits and global exact source/span
deduplication. Code repository names from both source schemas are grouped
together. Near duplicates and unidentified cross-repository copies are not
exhaustively removed. Spans are exact UTF-8 prefixes up to **32 KiB**. Bounded
stream-prefix selection is not a uniform sample of the full source collection.

Use at most **6 concurrent source jobs**; persist source batches in Drive.
Initial candidate target is 125% of quota, at least four spans per stratum;
continue a pinned stream if deduplication consumes that buffer. The per-job
candidate JSONL cap remains 256 MiB. Stack-smol contains a finite 10,000 samples
per language: quota fulfilment is checked, not assumed. Source access and
provenance terms are those of the upstream datasets.

Both 1,024 and 2,048 packings preserve all original text targets, independent
virtual EOS accounting, and document isolation. Each packing has its own cache
key and row order. Both reside on TPU; CPU-only byte/source prefixes are used
to account for the exact rows consumed, including repeat passes. A context
switch resumes its saved row cursor. It does not advance a single common raw
text frontier, so examples may recur across the two packings before either has
completed a full pass. Report context-specific exposure, not unique data seen.

BPB excludes virtual EOS and divides original-text negative log likelihood by
original UTF-8 bytes ×ln(2). Fixed 1,024-token evaluation scores the same held-out
text before/after training. Report 16-language macro BPB plus the original
12-language subset. Expanded recipe, potentially different source revisions,
and longer training prevent a controlled comparison with V3.5/V3.8.

References: [TPU recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training),
[MaxText setup guidance](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md),
[Shinning-Tokens](https://github.com/godkdf/Shinning-Tokens/),
[FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
[Stack-smol](https://huggingface.co/datasets/bigcode/the-stack-smol).
The TPU recipes are largely GKE/multichip setup guidance. V3.9 keeps the pinned
single-chip Colab environment; it does not claim to transplant a multichip recipe.
