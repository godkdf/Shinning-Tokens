# Gemma Colab V3.2

Open `../gemma_270m_colab_v5e1_V3_2.ipynb` in Colab Pro on one v5e-1 TPU and Run all.

V3.2 is the user-requested name for the corrected V3.1 notebook. It includes the fused-loss gradient dtype repair, upstream Tokamax dependency pin, six source download workers, persistent source-prefix caches and one 60-second training attempt. The scientific mixture and training settings are unchanged.

The compiled-call repair routes evaluation and training through their normal JIT wrappers, preserving closed-over kernel constants. A zero-update dispatch warms the training JIT before the timed interval. Explicit compiled objects are used only for precompilation/memory inspection. `dispatch_preparation.json` records the warmup counts; initial evaluation includes first-call JIT dispatch overhead. The uploaded failed V3.2 notebook reached initial evaluation and completed zero training updates.

The notebook title, experiment version, requirement IDs, runtime source filename, report archive name and new Drive output folder identify V3.2. Shared data, dependency, environment and compiler cache paths retain their existing names for reuse.

`build_notebook.py` assembles these artifacts from the corrected V3.1 source without executing notebook code. No local checks or tests were run. TPU confirmation remains pending the user's execution.
