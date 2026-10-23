# V4.2.1

Upload `../gemma_270m_colab_v5e1_V4_2_1.ipynb` to Colab, select TPU v5e-1,
use the same Drive account and run top to bottom. Only the notebook is needed.

Based on V4.1.1, with native tiktoken o200k_base and fixed **3 ×1024 /1 ×2048**.
No discovery or ramp. Keep the **1200-second** budget and repeating **3M/1M**
scored-token context cycle. Both fixed executables compile before training.

The Gemma3_270M backbone stays unchanged. Its tied vocabulary table becomes
200003 ×640: **228327936 total parameters**. Native sparse IDs are densely
mapped; raw text uses encode_ordinary and exact token-byte lengths. Encode once,
then create both context packings. Raw data remains shared; changed tokenizer
and native byte-accounting cache identities keep encoded inputs separate.

Native package pins are appended to the previous successful lock in a separate
`.venv-V4_2_1` environment. Existing training package pins stay fixed. The
exported tokenizer is native tiktoken constructor data, with its SHA256 checked
against the existing project registry. No model weights are downloaded.

`native_tokenizer.py` and `fixed_training.py` supply embedded helper code.
`build_notebook.py` derives and exclusively creates the runner and notebook.
`build_provenance.json` records their hashes and the preserved base hash.
`INTRO.md` and `REQUIREMENTS.md` are embedded user-facing documentation.

Validation: CPU native tokenizer/encoding checks and static notebook/source
checks. No accelerator initialization, device tests or model training.
