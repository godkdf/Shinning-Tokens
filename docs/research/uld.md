# Universal Logit Distillation (ULD)

Train a student using target-token supervision and a loss comparing teacher/student sorted probability masses.

**Transfers:** Teacher behavior through student optimization; no direct embedding-row correspondence is required.

**Local evidence:** not run. The Lab has not reproduced this method.

## Worked example

Teacher probabilities [0.6, 0.3, 0.1] and student probabilities [0.1, 0.6, 0.3] become identical after sorting. The loss alone does not know which token means what.

## Requirements and cost

- Teacher logits, student model, both tokenizers and distillation text.
- Declared prompt templates, sequence alignment/truncation and supervised-loss weights.

Toy experiments can run on CPU; useful model-scale runs usually need accelerator inference and training. Yes: teacher inference and student training. Runtime, peak memory and cost have not been measured here.

## Compatibility and embeddings

- Teacher and student hidden widths need not match for the logit loss.
- The original loss pairs positions up to the shorter sequence; positions can cover different text spans.
- Sorted probability matching does not establish semantic token-to-token matches.

The student keeps its own input parameters unless an independent initializer is used. Each model emits probabilities in its own vocabulary; logit caching must retain model revision, precision, temperature and masking. Preserve the student’s declared tying policy; the distillation loss does not determine it. Check vocabulary-sized bias and special-token configuration too.

## Published evidence and code

The paper evaluates cross-tokenizer teachers and students on summarization and question-answering tasks. Treat ULD and later span-aligned or hybrid variants as separate named configurations.

- [Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs](https://arxiv.org/abs/2402.12030v3) — v3 (2025-01-27)
- [Official ULD recipes](https://github.com/Nicolas-BZRD/llm-recipes) — author implementation. Record a commit and inspect the model-specific data/prompt pipeline before reproduction.

Availability was checked on 2026-10-19; linked implementations were not installed or executed.

## Lab evaluation

- Publish teacher inference compute and student training compute separately.
- Include a student trained on the same teacher-generated text without the ULD term.
- Report aligned/omitted text, sequence-length mismatch, task metrics and held-out per-byte likelihood.

[All methods](index.md) · [Evidence policy](../../research/README.md)
