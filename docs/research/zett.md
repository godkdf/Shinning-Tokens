# Zero-Shot Tokenizer Transfer (ZeTT)

A previously trained hypernetwork predicts model-compatible rows for a new tokenizer.

**Transfers:** Predicted vocabulary-dependent parameters for a compatible base model and supported tokenizer representation.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

Predicting 2,000 rows of width 1,024 yields 2,048,000 values. That count says nothing about the prior hypernetwork training cost.

## Requirements and cost

- A hypernetwork trained for the source model or a demonstrated compatible variant.
- Source model, target tokenizer, conversion metadata and the exact hypernetwork checkpoint.

Inference may be CPU-feasible if all required tensors fit; full-scale training normally needs accelerators. Hypernetwork training is required unless using a compatible released checkpoint; zero-shot applies at transfer time. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- A hypernetwork trained for one base model is not a universal mapper for unrelated checkpoints.
- The original repository describes transfer to fine-tuned variants; follow its combination rule rather than blindly reusing rows.

Build rows in the exact target token-ID order and preserve the source hidden width. Require a checkpoint/configuration that predicts every required vocabulary-dependent tensor, including an untied LM head. When weights are tied, keep one shared parameter tensor and verify that saving and reloading preserves the tie. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The paper evaluates encoder and decoder tokenizer replacement using a learned embedding-prediction hypernetwork. Training-free transfer assumes the hypernetwork already exists; optional continued training is evaluated separately.

- [Zero-Shot Tokenizer Transfer](https://papers.nips.cc/paper_files/paper/2024/hash/532ce4fcf853023c4cf2ac38cbc5d002-Abstract-Conference.html)
- [Original ZeTT](https://github.com/bminixhofer/zett) — author hypernetwork implementation. Includes hypernetwork training and transfer scripts; repository notes some evaluation setup is incomplete.
- [Tokenkit](https://github.com/bminixhofer/tokenkit) — later toolkit. Its documented transfer path uses FVT; do not describe that path as the original learned hypernetwork.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Report hypernetwork training cost separately from transfer latency.
- Measure zero-update model quality before optionally fine-tuning.
- Record compatibility assumptions, changed tensor names and tokenizer byte-conversion checks.

[All methods](index.md) · [Evidence policy](../../research/README.md)
