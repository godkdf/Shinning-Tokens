# V3.8.5

Self-contained notebook: `../gemma_270m_colab_v5e1_V3_8_5.ipynb`.
Default batch 2, context 2,048, training budget 180 seconds.

`build_notebook.py` derives the new version from the source and configuration
embedded in the user-specified V3.8 notebook. It refuses to overwrite existing
artifacts. `context.py` supplies sequence-aware cache identity and validation.
All helpers and pinned dependencies are embedded into the notebook, so this
directory does not need uploading to Colab.

Same raw corpus and tokenizer; new chunks and packing. Both evaluation sets
use the new context. Batch validation follows configuration rather than an
obsolete batch-3 assertion. The native model, optimizer math and timing budget
are unchanged. No restoration check. The classifier fixes native MLP work being
attributed to attention through the shared `_apply_attention` parent scope.
