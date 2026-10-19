# Experiment format

Each experiment must supply: a concrete hypothesis; parent and candidate identities; deterministic construction; selection/development data; evaluation corpus and its independence limits; paired counts; gains and regressions; byte fidelity; uncertainty method; runtime/capacity changes; compute and dependencies; limitations; and an evidence status.

The launch example is [Gemma V3.1](../../experiments/gemma-v31/README.md). Its **2,000 added IDs** are the intervention. Gemma4 is another comparator and is not the expansion parent. A second [historical mapping example](../../experiments/embedding-mapping/README.md) demonstrates a different question: reconstruct target input vectors from another model's embedding space.
