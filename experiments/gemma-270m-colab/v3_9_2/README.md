# V3.9.2: stop checking a locked batch ramp

Self-contained notebook: `../gemma_270m_colab_v5e1_V3_9_2.ipynb`.
Use for the next run. The current V3.9.1 run can finish; repeated hold/defer
messages do not restart compilation or discard updates.

V3.9.1 set `held=True` after reaching a memory/throughput/cap limit, but kept
entering the 100-update controller block. Long-context mode was checked before
the held condition, producing misleading `defer` messages after the ramp had
already ended. Dispatches also continued truncating at each 100-update boundary.

V3.9.2 stops the controller after the first hold or rollback:

- No further ramp checks or `hold`/`defer` messages.
- No further candidate compilation or short-window timing accumulation.
- No extra dispatch boundary solely for a 100-update check; normal maximum
  remains 16 updates per dispatch.
- Continue exact CPU/device accounting, finite-update checks, approximately
  five-second memory/progress sampling and the 3M-short/1M-long token schedule.
- Record the final batch, lock reason, lock update and check count in
  `result.json["batch_ramp"]`. The final decision has `ramp_locked=true`.

The corpus, source-lock and prefix-cache identities, 256 MiB preparation fix,
sequence caches, model math, optimizer, evaluation, profile and 1,200-second
budget remain unchanged. Existing completed caches are reusable. No changes
are applied to an active Colab process or its current model state.

The builder preserves V3.9.1 and exclusively creates V3.9.2. Validation is
limited to static source/format/invariant checks. No device tests or model
training are run, following the user's instruction. No speedup is claimed
without a completed run; the removed controller work is small relative to
the training kernels and the earlier batch-shape compilations.
