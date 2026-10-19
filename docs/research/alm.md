# Approximate Likelihood Matching (ALM)

Match teacher and student likelihoods of aligned text chunks, allowing different token boundaries.

**Transfers:** Teacher behavior through chunk-level distillation; optional components handle tokenization bias and hidden states.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

If teacher tokens [ab, c] and student tokens [a, bc] cover the same bytes abc, compare the likelihood of that complete chunk, not the two first-token probabilities.

## Requirements and cost

- Teacher and student models, tokenizer byte adapters and shared raw text.
- Declared chunk alignment, divergence, loss weights and any debiasing configuration.

CPU is useful for alignment diagnostics; model-scale teacher inference and student training typically need accelerators. Yes: teacher inference and student optimization. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- Both tokenizations must represent comparable underlying bytes; expose lossy normalization and unknown-token cases.
- The likelihood objective does not require equal hidden widths; optional hidden-state loss needs compatible dimensions or a declared projection.

The student can use an independent initializer; record that initializer to isolate ALM. Use each model’s own output head, summing log probabilities over each aligned chunk. Keep each model’s tying policy; different vocabulary sizes do not imply shared parameter rows. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The revised paper studies tokenizer self-distillation, model distillation and hypernetwork training. This catalog cites v4; earlier versions use a shorter title and a narrower experimental description.

- [Universal Cross-Tokenizer Distillation via Approximate Likelihood Matching](https://arxiv.org/abs/2503.20083v4) — v4 (2025-10-24)
- [Official Tokenkit](https://github.com/bminixhofer/tokenkit) — author implementation. Includes cross-tokenizer distillation and an ALM implementation guide.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Save aligned byte offsets, token spans and excluded cases.
- Compare with equal-compute text-only training and ULD baselines.
- Report per-byte likelihood, task metrics, source-language retention and total inference/training cost.

[All methods](index.md) · [Evidence policy](../../research/README.md)
