# Subtoken averaging / Fast Vocabulary Transfer

Tokenize each target piece with the source tokenizer and average the corresponding source model rows.

**Transfers:** An embedding initialization built from source subpieces; it does not transfer token order or a complete model behavior.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

A target piece decomposed into 3 source IDs receives (e1 + e2 + e3) / 3. Each occurrence contributes one third; changing their order leaves this average unchanged.

## Requirements and cost

- Source tokenizer, target tokenizer and source embedding tensors.
- A declared conversion from target piece bytes to valid source-tokenizer input.

Yes for initialization if the matrices fit memory. No training for the arithmetic mean; the paper evaluates additional adaptation training. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- The output width stays equal to the source model hidden width.
- Handle byte fragments, leading-space markers, unknown tokens and added-token matching explicitly; arbitrary vocabulary strings are not always standalone text.

Build rows in the exact target token-ID order and preserve the source hidden width. For an untied LM head, initialize and evaluate output rows separately; retain the framework orientation. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The 2022 FVT paper studies domain-specific vocabulary compression with subsequent masked-language-model adaptation. Its reported compression and task results do not establish Gemma or BLOOM transfer quality.

- [Fast Vocabulary Transfer for Language Model Compression](https://aclanthology.org/2022.emnlp-industry.41/)
- [Tokenkit](https://github.com/bminixhofer/tokenkit) — later implementation. Its FVT transfer path is separate from the learned ZeTT hypernetwork in the original ZeTT repository.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Log the source decomposition and averaging weights for every new row.
- Compare shared-row copy plus random initialization against shared-row copy plus FVT.
- Measure vector norms and held-out text likelihood before and after an equal adaptation budget.

[All methods](index.md) · [Evidence policy](../../research/README.md)
