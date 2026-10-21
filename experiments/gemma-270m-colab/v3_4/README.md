# Gemma Colab V3.4

Open `../gemma_270m_colab_v5e1_V3_4.ipynb` in Colab Pro on one v5e-1 TPU and Run all.

V3.4 contains the Splash tracer-leak repair: both fixed-shape attention kernels are constructed in ordinary Python before any JAX transformation, with no lru-cached factory inside tracing. It retains the fused-loss gradient dtype repair, normal JIT dispatch, zero-update training warmup, six source workers, persistent preparation caches and attempt-log preservation.

The model, optimizer, data mixture and one 60-second training attempt remain fixed. Shared corpus, dependency, environment and compiler cache directories retain their existing names. New run outputs, source filename, report filename and requirement IDs identify V3.4.

Version policy: every revision shall create a new notebook filename. Previous notebooks shall never be overwritten. This builder uses exclusive file creation and refuses to replace an existing artifact. Assembly does not execute notebook code or run checks.

No local checks or tests were run. TPU confirmation remains pending the user's execution.
