# Gemma Colab V3.4

## Engineering requirements

| Fixed setting | Value |
|---|---:|
| TPU / attempts | 1 × v5e-1 / 1 |
| Backbone | Native Gemma 3 270M transformer: 18 layers, width 640, 4 query heads, 1 KV head |
| Vocabulary table | Tied input/output; N tokenizer IDs plus 3 protocol IDs |
| Parameter count | 100,326,016 + 640 × (N + 3); strict total below 300,000,000 |
| Training interval | 60 seconds nominal; completed-work time and overrun reported |
| Sequence / batch / maximum dispatch | 1,024 / 4 / 16 updates |
| Muon | LR 0.02; momentum 0.95; Nesterov; 5 Newton–Schulz steps; native Optax width scaling |
| Auxiliary AdamW | LR 0.0003; β1=0.9, β2=0.999, ε=1e-8 |
| Warmup / clipping / decay | 5 updates / global norm 1.0 / matrix-weight decay 0.01 |
| Compute / master state | BF16 / FP32 |
| Seed | 42 |
| Parallel source workers / source jobs | 6 / 30 |
| Prefix batch target / per-job disk cap | 4 MiB / 256 MiB |

| ID | Requirement | Colab verification |
|---|---|---|
| V34-001 | The runner shall permit one training attempt per run folder. | Exclusive Drive start marker; no automatic retry. |
| V34-002 | The runner shall use exactly one v5e TPU. | Platform, device kind and device count. |
| V34-003 | Preparation shall resolve every dataset to an immutable commit before reading records. | Shared source lock; pinned revision on each load. |
| V34-004 | Tokenizer runs shall reuse the same frozen raw corpus for this protocol. | Raw manifest and file SHA256 verification. |
| V34-005 | Training and evaluation shall have disjoint canonical source groups and exact text hashes. | Group separation and global source/span hash deduplication. |
| V34-006 | Tokenization shall reconstruct every selected original UTF-8 byte exactly. | Encode/decode equality; no silent normalization or sample deletion. |
| V34-007 | Packing shall preserve each target once per pass and each chunk's context and positions. | Assignment, content, position and count assertions. |
| V34-008 | Fused kernels shall satisfy stated numerical tolerances before model training. | Synthetic TPU loss and local/global segmented attention forward/gradient comparisons. |
| V34-009 | Training inputs shall reside on the TPU before the timer starts. | Device ownership and synchronization barrier. |
| V34-010 | The model shall retain the Gemma backbone, one tied table, FP32 masters and fewer than 300M parameters. | Shapes, count, dtype and shared initial-backbone hash. |
| V34-011 | Muon shall update only hidden projection matrices. | Shape-aware optimizer partition: 100,270,080 Muon parameters. |
| V34-012 | The runner shall start no training dispatch at or after 60 elapsed seconds. | Monotonic dispatch timestamps; actual overrun separately reported. |
| V34-013 | A training dispatch shall complete at most 16 updates. | Requested/completed counts. |
| V34-014 | The runner shall reject nonfinite losses/gradients and stop on nonfinite final weights. | Device finite guards and final parameter reduction. |
| V34-015 | The runner shall perform zero model optimizer updates outside the timed interval. | Initial count 0; final optimizer and data cursor equal timed updates. |
| V34-016 | Evaluation shall score the same bytes before and after training and report all 12 FineWiki languages. | Fixed input hashes and byte totals; pooled and per-language BPB. |
| V34-017 | The runner shall persist one full resumable checkpoint. | Local/Drive SHA256 equality, state schema and leaf count; no restore execution claim. |
| V34-018 | Training shall change a hidden transformer weight. | Initial/final parameter probe. |
| V34-019 | Training shall complete at least 5 optimizer updates. | Final timed update count. |
| V34-020 | The report shall reconcile processed bytes by source and language. | Equal accumulated byte totals and positive exposure. |

| V34-021 | Data preparation shall run at most 6 source jobs concurrently. | Executor bound and recorded peak active source jobs. |
| V34-022 | Preparation shall persist completed source batches and row cursors independently of final corpus completion. | Atomic batch/state commits; SHA256 verification before cached-batch reuse. |
| V34-023 | Worker completion order shall not determine final sample selection. | One canonical merge order, with the 12 Wiki sources reserved first; saved order record. |
| V34-024 | Cold preparation shall finish all 30 source jobs before tokenization. | Completed-job report and frozen-corpus byte quotas; a verified warm raw corpus needs no fetch. |

Verification occurs only when you run Colab. Engineering PASS and learning PASS are separate: learning PASS means FineWiki macro BPB decreases. The 95% useful-position occupancy target is a performance goal, not a promised speedup. No throughput target is presented as measured.

Dependency requirement: setup shall retain the pinned V2 Gemma/JAX package versions, use the recorded upstream Tokamax commit that avoids the incompatible PyPI `typeguard` pin, and display/save merged stdout/stderr for failed commands. Verification is the Colab setup log, exact source commit record and full dependency lock. No `--no-deps` bypass or automatic alternate training configuration is used.

Status: generated; local checks and tests not run at the user's request.
