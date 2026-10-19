# FOCUS

Use shared vocabulary as anchors; choose sparse semantic combinations in an auxiliary target-token embedding space.

**Transfers:** Copied shared rows and weighted combinations of source model anchor rows for new entries.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

With 10,000 valid shared anchors, a new piece can receive a weighted combination of only a few anchor rows; publish which rows and their exact weights.

## Requirements and cost

- Both tokenizers, source model rows and usable overlap.
- Target-language text for auxiliary token embeddings, or a declared supported pretrained auxiliary resource.

CPU-feasible for auxiliary embeddings and initialization; report memory and elapsed time. Auxiliary embedding training may be needed; continued LM training is separate. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- Overlap quality matters, including special-token and boundary handling.
- The reference evaluation centers on XLM-R; a decoder-model port requires its own evidence.

Build rows in the exact target token-ID order and preserve the source hidden width. For an untied LM head, initialize and evaluate output rows separately; retain the framework orientation. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The paper evaluates monolingual specialization of XLM-R on language modeling, NLI, QA and NER. Reported gains compare specific initialization and adaptation settings; no local reproduction is supplied.

- [FOCUS](https://aclanthology.org/2023.emnlp-main.829/)
- [Official FOCUS / deepfocus](https://github.com/konstantinjdobler/focus) — author implementation. Documents target-text input, cached auxiliary embeddings and separate output-row initialization.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Report overlap size, auxiliary corpus hash and out-of-vocabulary fallbacks.
- Save the anchor IDs and sparse weights so each initialized row is auditable.
- Compare to FVT, random and exact-copy baselines at identical adaptation compute.

[All methods](index.md) · [Evidence policy](../../research/README.md)
