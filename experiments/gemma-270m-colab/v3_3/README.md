# Gemma Colab V3.3

Open `../gemma_270m_colab_v5e1_V3_3.ipynb` in Colab Pro on one v5e-1 TPU and Run all.

V3.3 is the requested name for the corrected V3.2 build. It retains the fused-loss gradient dtype repair and JIT dispatch repair, including zero-update training warmup before timing. The model, optimizer, data mixture and single 60-second training attempt are unchanged.

Attempt evidence repair: the notebook refuses a repeated train call before opening logs. Existing results are shown without retraining. A pretraining status and a separate timed-training marker distinguish preparation from model updates in future runs. Earlier failure.json is preserved when another exception occurs. Old V3.3 start markers alone cannot prove whether an earlier attempt trained; they must not be blindly removed.

Splash construction repair: the latest supplied original traceback identified a tracer escaping from an lru-cached kernel factory during the verification JIT. Both fixed-shape kernels are now constructed once in ordinary Python before transformations; subsequent calls perform only a shape check and dictionary lookup. No transformed intermediate is stored in a cache. The first reported 222/209 mismatch may have been related, but that connection has not been independently confirmed.

The notebook title, experiment version, requirement IDs, runtime source filename, report archive name and new Drive output folder identify V3.3. Shared corpus, prefix, dependency, installed environment and compiler cache paths keep their existing names for reuse.

`build_notebook.py` constructs these artifacts from corrected V3.2 without executing notebook code. No local checks or tests were run. TPU verification remains pending the user's execution.
