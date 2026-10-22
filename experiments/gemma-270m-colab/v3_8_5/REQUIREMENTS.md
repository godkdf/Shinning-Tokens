# V3.8.5 engineering contract

Default **batch 2 × 2,048**, **180 seconds**, one v5e chip. Preserve the V3.8
native model, AdamW settings, pinned dependencies, frozen source bytes and
tokenizer. No checkpoint restoration, reload equality check or reload evaluation.
Retain local/Drive file checksums and artifact schema checks during saving.

## Context and data

- Tokenize original frozen documents into 2,048-token chunks, preserving every
  text target, original byte attribution, EOS accounting and causal segment mask.
- Cache identity includes tokenizer hash, raw manifest hash, packing version,
  schema and sequence length. Batch is excluded because grouping happens later.
- Record sequence length, packing version and cache key in tokenization metadata.
- Validate cached metadata and file hashes before reuse. Validate every loaded
  input/accounting array has the configured width before training. A stale
  1,024-token cache must fail validation, never be silently accepted or deleted.
- Training and both held-out evaluations use 2,048-token inputs. Evaluation
  reports name the actual context. Source byte denominators remain unchanged.
- V385-025 checks the actual batch/context shape against configuration, rather
  than requiring the historical batch 3. Default batch is 2; another positive
  integer batch must not fail merely because an old hard-coded batch remains.

## Timing, artifacts and telemetry

Retain one attempt, at most 16 official updates per dispatch, finite-update
checks, resident inputs, CPU prefix byte accounting, weights-only save, measured
training memory and 4 × 8 post-save diagnostic updates. Diagnostic state never
changes the official score, token count or checkpoint. Earlier notebook files
and caches remain untouched. Builders use exclusive file creation.

Maintain separate engineering, learning, performance and profiling statuses.
Missing hardware utilization counters are unavailable, not zero. Compiler
roofline estimates remain labelled separately. Host synchronization-call time
is not OS CPU idle. Classify MLP before attention and ignore the broad native
`_apply_attention` parent when identifying attention operations.

Historical 1,024-token BPB and speed are descriptive references. Context length,
chunk boundaries, packing and batch changed, so they are not controlled
comparisons. Retain compilation ≤60 seconds, notebook through report ≤480
seconds, useful occupancy ≥95%, and historical throughput reference as targets,
not guarantees. Local checks cannot establish TPU memory fit or speed.
