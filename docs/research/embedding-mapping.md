# Embedding-space mapping

Fit a transformation using paired anchor vectors, then project other vectors into a target model space.

**Transfers:** Vector coordinates; no automatic transfer of attention weights, token boundaries or generation behavior.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

For 768-dimensional source vectors and 1,024-dimensional target vectors, a linear matrix has 768 × 1,024 = 786,432 coefficients, plus 1,024 if a bias is used.

## Requirements and cost

- Source and target vector matrices plus defensible paired anchors.
- Separate fitting, development and evaluation anchors with no token-family leakage.

Linear least squares or SVD is CPU-feasible at modest sizes. Yes: fitting a mapper; subsequent LM adaptation is a separate requirement to evaluate. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- Equal width permits a square orthogonal mapping; differing widths require a rectangular or other declared mapper.
- MUSE is a word-embedding alignment reference, not evidence that any two LLM embedding spaces are interchangeable.
- Train and evaluate input and output mappings separately if their spaces differ.

Document source/target dimensions, centering, normalization, objective and regularization. Fit a separate output mapper for an untied head; do not assume the input mapper is correct for output rows. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

MUSE provides supervised and unsupervised cross-lingual word-embedding alignment methods. This is a method family; those word-vector results do not validate an arbitrary LLM conversion.

- [Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087)
- [MUSE](https://github.com/facebookresearch/MUSE) — reference alignment implementation. Supports supervised Procrustes alignment; its published task is word-vector alignment.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Report held-out cosine similarity, retrieval accuracy and vector-norm errors.
- Use exact-copy and subtoken-average controls where applicable.
- Evaluate target-model likelihood and behavior; low vector error alone cannot establish model utility.
- Keep the historical o200k-to-BLOOM experiment labeled exploratory until its data, split and artifacts are imported and reproduced.

[All methods](index.md) · [Evidence policy](../../research/README.md)
