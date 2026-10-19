# Constrained Word2Vec (CW2V)

Learn new rows from target-language contexts as convex combinations of existing rows.

**Transfers:** Added input and LM-head rows while retaining the existing model parameters.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

Weights 0.5, 0.3 and 0.2 sum to 1 and keep the new row inside the existing rows’ convex hull. Their values are learned from context.

## Requirements and cost

- Expanded target tokenizer, source model rows and target-language text.
- The constrained skip-gram training configuration and a held-out evaluation split.

Small prototypes can run on CPU; full matrix and corpus costs must be measured. Yes: contextual initialization training, followed by adaptation for the paper evaluations. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- The paper addresses vocabulary expansion rather than unrestricted model replacement.
- Do not generalize the paper’s convex-hull guarantee beyond its stated assumptions or to arbitrary biased output heads.

Build rows in the exact target token-ID order and preserve the source hidden width. The paper explicitly learns separate constrained combinations for untied input and LM-head matrices. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The paper compares RoBERTa and LLaMA 2 across four languages and five tasks. Its theoretical argument concerns initialization; downstream quality is an empirical adaptation result.

- [An Empirical Comparison of Vocabulary Expansion and Initialization Approaches For Language Models](https://aclanthology.org/2024.conll-1.8/)
- [Official CW2V branch](https://github.com/AI4Bharat/VocabAdaptation_LLM/tree/CW2V) — author implementation. Includes constrained training, initialization and model-specific downstream evaluation folders.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Verify nonnegative coefficients and row sums of one within numerical tolerance.
- Separate initializer training time from subsequent LM adaptation.
- Compare simple mean and multivariate baselines; publish language regressions and generation checks.

[All methods](index.md) · [Evidence policy](../../research/README.md)
