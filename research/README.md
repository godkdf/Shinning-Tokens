# Research catalog

This is the evidence index for tokenizer transfer and adaptation research. These methods use model weights. The Lab’s intrinsic UTF-8, vocabulary, compression, speed and memory analyses remain usable without loading a language model.

`methods.json` is the machine-readable source for **9 method pages**. Sources were checked on **2026-10-19**. External repositories change; pin their commits before executing experiments.

## Evidence labels

| Label | What it proves | What it does not prove |
|---|---|---|
| Paper reported | The linked authors report a result in the stated experimental setting. | That the Lab reproduced it, or that it applies to another model. |
| Implementation available | An identified primary repository contains relevant code. | That its environment installs, its code runs, or its results match the paper. |
| Locally reproduced | A pinned local experiment provides raw results, logs and artifact hashes. | Universal effectiveness beyond that experiment. |

**No method in the current catalog has been locally reproduced.** Historical exploratory artifacts elsewhere are not treated as reproductions of these papers.

## What transfers

- **Exact row copying, FVT, WECHSEL and FOCUS:** initialize model parameters without LM optimization in the initialization step. Auxiliary resources can still cost time and memory.
- **CW2V and embedding mapping:** learn initial rows or a transformation.
- **ZeTT:** predicts parameters using a previously trained, compatible hypernetwork. Training-free transfer does not mean training-free preparation.
- **ULD and ALM:** use teacher inference and student optimization to transfer behavior.

An exact text/token converter belongs to the intrinsic tools. It decodes a complete sequence and retokenizes its text; it does not initialize model weights or transfer generation quality.

## Adding evidence

Add a method only with primary paper or author-code links. Keep published claims inside `evidence.paper_reported`; put local measurements inside `evidence.locally_reproduced` only after publishing a result bundle. A reproduction bundle must identify all model, tokenizer, code and data revisions; seed and compute configuration; input/output embedding handling; raw metrics; failure cases; logs; and artifact hashes. Results on selection/training data must remain labeled as such.

Model-space assumptions and input/output weight tying are part of each experiment. Equal vector dimensions and shared token spellings are insufficient evidence that two model rows are interchangeable.

[Read the method guide](../docs/research/index.md).
