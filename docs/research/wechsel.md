# WECHSEL

Use aligned bilingual static embeddings to select source subword neighbors and combine their model rows.

**Transfers:** New vocabulary rows in the existing model space while retaining the model backbone.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

If a new piece uses neighbor weights 0.6, 0.3 and 0.1, its row is 0.6e1 + 0.3e2 + 0.1e3. These are illustrative weights, not a measured result.

## Requirements and cost

- Source and target tokenizers and source model rows.
- Static embeddings covering both languages plus an alignment/dictionary policy.

Initialization is CPU-feasible; auxiliary embeddings may occupy substantial RAM. No LM training in row construction; the published adaptation results include model training. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- Bilingual-resource quality and language coverage limit the method.
- Keep the source backbone and hidden width; this is not an arbitrary model-to-model vector swap.

Build rows in the exact target token-ID order and preserve the source hidden width. For an untied LM head, initialize and evaluate output rows separately; retain the framework orientation. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The paper transfers English RoBERTa and GPT-2 to French, German, Chinese and Swahili. These four-language experiments are evidence for those settings, not every model and language pair.

- [WECHSEL](https://aclanthology.org/2022.naacl-main.293/)
- [Official WECHSEL](https://github.com/CPJKU/wechsel) — author implementation. The example applies the method separately to untied input and output embeddings.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Publish auxiliary embedding versions and the dictionary or alignment.
- Record neighbor IDs, similarity weights, missing coverage and fallbacks.
- Hold out adaptation evaluation text and include source-language retention.

[All methods](index.md) · [Evidence policy](../../research/README.md)
