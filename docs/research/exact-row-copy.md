# Exact-row copy baseline

Reuse an existing model row only when the target entry has the same token semantics in the same model space.

**Transfers:** Existing input rows, and corresponding output rows when available. Unmatched tokens need a separately declared initializer.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

If 800 of 1,000 target entries match, copy 800 rows; explicitly initialize the other 200. Copying is not a 100% vocabulary conversion.

## Requirements and cost

- Source model embedding tensors and both ID maps.
- A verified token-byte and boundary-semantics match; special tokens require role matching.

Yes for copying; peak memory still depends on matrix size. No optimization for the copy itself; adaptation is a separate experiment. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- Use rows from the same checkpoint or a verified compatible model space. Equal dimensions across unrelated models are insufficient.
- The same displayed text can encode different leading-space or continuation semantics. Byte overlap is a candidate match, not proof of interchangeability.

Build rows in the exact target token-ID order and preserve the source hidden width. For an untied LM head, initialize and evaluate output rows separately; retain the framework orientation. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

Shared-token row copying appears inside FVT and FOCUS; this is the Lab baseline definition, not a separate paper result. No independent performance claim.

- [FVT: shared-token assignment, Equation 1](https://aclanthology.org/2022.emnlp-industry.41.pdf)
- [FOCUS: overlap copy, Equation 1](https://aclanthology.org/2023.emnlp-main.829.pdf)
- [FOCUS](https://github.com/konstantinjdobler/focus) — upstream baseline component. Contains overlap copying; not a standalone Lab implementation.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Report copied, unmatched, ambiguous and special-token counts.
- Require exact equality for copied rows before and after save/load.
- Compare to random and subtoken-mean initialization on the same unmatched rows.

[All methods](index.md) · [Evidence policy](../../research/README.md)
