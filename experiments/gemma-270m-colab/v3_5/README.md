# Gemma 270M Colab V3.5 — AdamW

Select **TPU v5e-1** in Colab, then **Run all**. One fixed **60-second** attempt.
The notebook is self-contained; no source upload or configuration editing is
needed for the default Gemma tokenizer. Hugging Face access to that repository
is required. HF_TOKEN can be supplied through Colab Secrets or the password prompt.

**All-parameter AdamW:** learning rate **0.0003**, betas **0.9/0.999**,
epsilon **1e-8**, weight decay **0.01** on matrix/tensor weights, gradient clipping
**1.0**, and **5** warmup updates. This uses V2's native training approach with
V3.4's frozen data and tokenizer evaluation. V2 measured **45,559 tokens/s**;
the new notebook's result remains unmeasured.


Both new versions use native Gemma attention and dense exact softmax, **batch 2 x
1024**, seed **42**, the same V3.4 frozen raw/token caches, CPU training-byte
accounting, one combined held-out evaluator and one FP32 weights-only artifact.
Splash and Tokamax do not execute. The existing dependency environment is retained
to avoid another package-resolution experiment. New Drive outputs use
`ShiningGem/colab-v5e1/V3.5/<unique run id>`.

The FP32 weights are approximately **1.07 GB** for the default tokenizer.
**Optimizer state is deliberately not saved; exact optimizer resumption is not
supported.** Reports and tokenizer files are retained. Connected setup,
compilation, evaluation and saving consume additional CU outside the 60 seconds.
One notebook does not automatically run the other. **No local checks or tests
were run. TPU results are pending your execution.**

# Gemma Colab V3.5 engineering requirements

Fixed configuration: one v5e-1 TPU, seed 42, batch 2 x 1024, 60 seconds,
at most 16 updates per dispatch, one attempt, random initialization.

| ID | Requirement | Verification |
|---|---|---|
| V35-001 | The runner shall make one attempt with no automatic retry or search. | Exclusive training.started marker; preserved prior logs. |
| V35-002 | The runner shall use exactly one v5e TPU. | Device platform, kind and count at train startup. |
| V35-003 | All source revisions shall be pinned. | Frozen shared source lock and source set. |
| V35-004 | The raw corpus shall remain the V3 static-byte mixture. | Protocol and raw/token manifest SHA256 checks. |
| V35-005 | Training and evaluation shall use disjoint source groups and exact spans. | Corpus builder deduplication and split separation; not an exhaustive near-duplicate guarantee. |
| V35-006 | Tokenization shall reconstruct every selected original UTF-8 byte. | Round-trip and byte-count equality; no corpus modification. |
| V35-007 | Packing shall preserve targets, chunk context and reset positions. | Existing intact-packing checks and cached input hashes. |
| V35-008 | Native attention shall enforce causality, document isolation and safe padding queries. | Small mask comparison in Colab before training; no custom kernels. |
| V35-009 | Training inputs shall reside on the TPU before timing. | Device ownership and synchronization barrier. |
| V35-010 | The model shall retain the native Gemma backbone, tied table, FP32 masters and fewer than 300M parameters. | Parameter shapes/count/dtypes and shared backbone hash. |
| V35-011 | Every parameter shall use AdamW; no parameter shall use the other optimizer. | Complete per-leaf optimizer routing and parameter totals. |
| V35-012 | No training dispatch shall start at or after 60 elapsed seconds. | Monotonic dispatch timestamps; actual overrun reported. |
| V35-013 | Each dispatch shall perform between 1 and 16 updates. | Requested and completed update counts. |
| V35-014 | Nonfinite loss or gradient shall stop training; final weights shall be finite. | Finite guard, failure evidence and final weight reduction. |
| V35-015 | Zero optimizer updates shall occur outside the timed interval. | Zero-update JIT preparation and optimizer/cursor counters. |
| V35-016 | Evaluation shall score the same bytes before and after training, including all 12 FineWiki languages. | Fixed inputs, original-byte denominators and separate EOS totals. |
| V35-017 | One FP32 weights artifact shall be persisted and checksum verified. | Local/Drive SHA256 equality, params-only schema and leaf count. |
| V35-018 | Training shall change a hidden transformer weight. | Initial/final weight probe. |
| V35-019 | At least 5 timed updates shall complete. | Final update count. |
| V35-020 | Processed bytes and tokens shall reconcile with completed batches. | CPU prefix sums, device token count, source/language equality. |
| V35-021 | Source preparation shall use no more than 6 workers. | Executor bound and recorded peak concurrency. |
| V35-022 | Completed source prefixes and cursors shall persist independently of final corpus completion. | Source cache state and checksums; warm raw corpus needs no fetch. |
| V35-023 | Worker completion order shall not determine corpus selection. | Canonical merge order with Wiki reserved first. |
| V35-024 | Cold preparation shall finish all 30 source jobs. | Source completion report, or verified complete raw cache. |
| V35-025 | The training batch shall be 2 sequences of 1024 tokens. | Resolved configuration; no batch search or fallback. |
| V35-026 | One combined evaluator shall report both held-out datasets separately. | One compiled evaluation function and runtime split identity. |
| V35-027 | Per-token source/language histograms shall not execute in timed training. | CPU prefix sums; training device input excludes accounting arrays. |
| V35-028 | The artifact shall declare optimizer resumption unsupported. | Weights-only schema and explicit report flag. |

The notebook shall reuse the exact successful V3.4 dependency lock; it shall not
resolve package versions again. A matching installed environment shall be reused.
Tokamax may remain installed in that lock but shall not execute in training.
Runtime startup shall perform hardware verification once, in the train process.

The notebook shall save reports and weights in a unique V3.5 run directory.
It shall preserve previous notebooks and prior attempt evidence. The artifact
builder shall use exclusive file creation. No local checks/tests shall be run.

Performance targets, not guarantees: at least 45,559.448 useful tokens/s,
at most 60 seconds compilation, at most 300 seconds notebook time with existing
data caches, at least 95% useful occupancy, and FineWiki macro BPB no higher than
2.782397528489431 for the exact V3.4 tokenizer/corpus. Performance-goal status,
engineering status and learning status shall be reported separately. These
targets also appear in V3.6; no claim is made that all-parameter Muon meets them.

The 60-second timer excludes preparation, compilation, evaluation and saving;
all of those consume paid connected runtime. At 2.92 CU/hour, exactly 60 seconds
is 0.0486667 CU and 300 seconds is 0.2433333 CU. No dollar conversion is assumed.
No successful TPU execution or restore is claimed by artifact construction.
