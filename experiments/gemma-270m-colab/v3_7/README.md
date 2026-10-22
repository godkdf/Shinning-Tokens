# Gemma Colab V3.7

Deliverable: `../gemma_270m_colab_v5e1_V3_7.ipynb`.
Upload it to Colab, select TPU v5e-1, and Run all. No source upload or configuration
editing is required. Prior notebooks are preserved.

## Changes from V3.5

- Exact full-vocabulary head in 128-token chunks. Rematerialization bounds the
  logit workspace in backward. Default logical FP32 chunk workspace: 128 MiB,
  versus 2.00 GiB for the full 2048-position batch. This trades recomputation and
  smaller matrix multiplies for workspace; actual speed needs the TPU run.
- AdamW remains at peak LR 0.0003 after five warmup updates through the first
  half of the 60-second budget, then cools linearly toward 0.00009. The multiplier
  is a runtime input, shared by at most 16 updates per dispatch; changing it does
  not trigger recompilation. It scales decoupled decay together with the update.
- Unchanged native Gemma attention, backbone, tokenizer, source mixture, packed
  context, BF16 computation, FP32 masters, batch 2, and 1024-token sequences.
- Before timed training, synthetic FP32/BF16 loss and gradient checks include
  a partial chunk. A short real-vocabulary model probe verifies the native-block
  entry point against Gemma's standard call. These apply no optimizer updates.
- Restore the saved weights, compare full parameter digests, and re-evaluate
  all languages. The artifact remains weights-only, without optimizer resume.
- After checkpoint verification, capture 64 diagnostic updates using disposable
  in-memory state. This does not change the checkpoint or official metrics.
  Trace capture, export, and analysis are counted in total runtime and CU.

## Requested measurements

`profiling.json` and `result.json` contain matrix-unit utilization, HBM bandwidth
utilization, separate host/device idle percentages, and exclusive operation
timings. Values include unit, provenance, availability, and missing-data reasons.
Raw XPlane/Perfetto traces and XProf tool exports are included in the report ZIP.

`training_memory.json` records active and reserved memory in bytes/GiB and as a
percentage of the actual device-reported limit. Samples occur immediately before
training, at synchronized progress points (about every 2 seconds), and immediately
after training, before final evaluation. Sampling overhead stays on the training
clock and is recorded. A lifetime high-water mark is attributed to training only
if it increased strictly during that interval. Otherwise the true training peak
is explicitly not isolatable, and sampled usage is shown. Compiler buffer
estimates are reported separately; they are not presented as measured usage.

Hardware-counter percentages require a TPU `OpStats.performance_counter_result`
message. XProf can emit `0.0%` even without counters; those defaults are explicitly
rejected. Idle percentages require timeline evidence. HBM bandwidth is distinct
from memory capacity. Host waiting is not automatically a performance problem.

Operation timings use exclusive HLO self-time and preserve unclassified/fused
operations. They are aggregate device-operation time, not elapsed wall time or
the sum of nested host annotations. Fine-grained attribution can remain
unavailable if the runtime strips metadata; raw evidence is retained.

Capture and extraction failures are recorded separately from model engineering
success. Telemetry status is COMPLETE, PARTIAL, or UNAVAILABLE. Missing counters
never become invented utilization estimates or silent zeros.

## Files

- `head.py`: exact loss, elapsed-time schedule, native-model correctness probes.
- `profiling.py`: capture, XProf export, conservative metric extraction.
- `memory.py`: training-only memory samples, capacity percentages, peak scope.
- `runner.py`: complete generated program embedded in the notebook.
- `build_notebook.py`: version-exclusive assembler reading V3.5.
- `display_results.py`: Colab tables, including the requested telemetry.
- `test_v37.py`: bounded CPU correctness and profiler API checks.
- `REQUIREMENTS.md`: runtime acceptance and measurement contract.

The builder refuses to replace an existing V3.7 notebook. No prior notebook or
source file is modified. No TPU execution or improved BPB/throughput is claimed
by local validation.

## Reference documentation

- [JAX profiling and TPU performance-counter options](https://docs.jax.dev/en/latest/profiling.html)
- [XProf overview measurements](https://openxla.org/xprof/overview_page)
- [XProf OpStats hardware counter fields](https://github.com/openxla/xprof/blob/master/plugin/xprof/protobuf/op_stats.proto)
- [XProf raw trace conversion](https://github.com/openxla/xprof/blob/master/plugin/xprof/convert/raw_to_tool_data.py)
- [NanoGPT training schedule inspiration](https://github.com/KellerJordan/modded-nanogpt/blob/master/track_1_short/schedule.py)
