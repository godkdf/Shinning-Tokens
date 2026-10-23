# V3.9 implementation

Open `../gemma_270m_colab_v5e1_V3_9.ipynb` in Colab with a **TPU v5e-1**.
Run cells top to bottom. Everything required from this directory is embedded;
upload only the notebook. HF_TOKEN needs Gemma and Stack-smol access.

Default: **256 MiB**, **20 minutes**, initial **1 ×1,024**, checks every **100
official updates**, +1 short sequence when admitted. At accepted short batch ≥3,
repeat **3M short-context /1M long-context scored tokens**, long fixed **1 ×2,048**.
See `INTRO.md` and `DATA.md` for the exact schedule, allocations and access link.

`build_notebook.py` derives V3.9 from the preserved V3.8.5 artifact. It refuses
to overwrite an existing notebook or runner. `data.py` supplies the new frozen
recipe and dual-context preparation, `adaptive.py` supplies admission and device
updates, `loop.py` supplies the timed controller, and `train.py` integrates
evaluation/artifacts/reports. `runner.py` is their self-contained embedded result.
Dependency pins, native model, loss and AdamW settings are inherited unchanged.

Memory gate: exact compiler buffer accounting plus inactive inputs and fixed
evaluation buffers, at most 85% capacity; before compiling the next shape,
extrapolate conservatively with 10% margin on incremental workspace. Include
observed active allocations exceeding tracked arguments/resident inputs. This
is an admission heuristic, not proof of OOM immunity. Missing memory estimates
stop increases. Capture actual allocator samples and per-shape estimates.

Timing gate: use the next 100 real short updates to assess an increase. Require
≥2% useful dispatch tokens/s improvement; otherwise revert the shape one step
and hold. All trained weights and Adam moments remain current. No optimizer
probe runs, no restore, no failed-attempt retry, no LR reset. Candidate compilation
time is inside the 1,200-second wall budget and separately itemized. Thus fewer
than 1,200 seconds may be spent executing training kernels.

Every check and dispatch is persisted, including shape, token and row counts,
compile decisions, memory snapshots and timings. Each context uses a fixed
shuffled row order and its own persistent cursor; there is no divisibility
padding when changing batch. Revisiting the same frozen text under the other
packing is intentional and reported as repeated exposure. The crossing update
completes before switching context; no partial gradient or skipped targets.

Evaluation uses fixed 1,024 context. The 16-language macro and original 12-language
subset macro are reported separately. Code/markup BPB is split by domain. These
are expanded-corpus results and do not establish a controlled V3.8 speed/quality
improvement. Byte quotas and source revisions are frozen for later tokenizer runs.

Profile after official evaluation and weights saving: 16 final-short and 16 long
updates, discarded. Operation statistics average the mixed shapes. Request TPU
matrix-unit/HBM counters, device idle, host traces and operation timings; missing
counters remain unavailable with profiling completeness FAIL. Roofline estimates
and host synchronization-call wall time never masquerade as measured utilization
or OS CPU idle. The model artifact remains saved if telemetry is incomplete.

New reports: `batch_checks.jsonl`, `context_transitions.json`, `compilation_events.json`,
`tokenization_1024.json`, `tokenization_2048.json`; existing `result.json`,
`training_memory.json`, `profiling.json`, checkpoint hashes and source manifests
remain. Reports ZIP includes all of these, the source and raw profile exports.

CPU verification: `test_v39.py` covers recipe sums, 43 source jobs, group splitting,
cache identities, schedule boundaries, memory admission, real JAX/Adam state
continuity, nonfinite rejection, and the complete controller using a deterministic
device/clock fixture. These tests do not establish live gated-source availability,
TPU memory fit, hardware-counter exposure or training speed. The first Colab run
performs the source-access, configuration, quota and runtime checks.

References reviewed: [TPU recipes](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training),
[MaxText setup](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md),
[Shinning-Tokens](https://github.com/godkdf/Shinning-Tokens/).
The recipe setup supports pinned Python 3.12 environments; this notebook keeps
its existing single-chip Colab stack rather than adding GKE orchestration.
