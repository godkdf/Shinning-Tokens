# V3.8 implementation

Open `../gemma_270m_colab_v5e1_V3_8.ipynb` in Colab with TPU v5e-1.
Run all cells in order. Nothing in this directory needs uploading separately.

This version takes the native dense training graph and AdamW settings from
V3.5 and uses the user-requested batch 3 × 1,024 and 180-second run. Metadata
scopes label optimizer and cross-entropy operations without changing their math.
No checkpoint restoration helper or restoration evaluation is included.

Memory and profiling helpers are embedded in `runner.py`. Profiling uses 32
disposable post-save updates. Required measured utilization remains unavailable
when TPU counters are absent; compiler estimates are explicitly separate.
The program preserves saved training results even if profiling is incomplete.
The 480-second whole-notebook budget is a reporting target, not a timeout or
promise that cold preparation will fit it.

`build_notebook.py` constructs source and notebook from the preserved V3.5
artifact. It refuses to overwrite an existing runner or notebook. `INTRO.md`
and `REQUIREMENTS.md` are embedded documentation; `display_results.py` and
`download_reports.py` are embedded notebook cells. `test_v38.py` checks batch
accounting, math preservation, telemetry parsing and artifact integration on
CPU without conducting a Gemma training experiment.
