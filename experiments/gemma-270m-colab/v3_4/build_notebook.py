"""Create V3.4 artifacts without replacing any existing notebook or source file.

Artifact assembly only; no notebook execution, local checks or tests.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PREVIOUS = PARENT / "v3_3"


def relabel(text):
    # Shared data, dependencies, environment and compiler cache names stay fixed.
    return text.replace("V3.3", "V3.4").replace("V33-", "V34-").replace("V3_3", "V3_4")


notebook = json.loads((PARENT / "gemma_270m_colab_v5e1_V3_3.ipynb").read_text())
for cell in notebook["cells"]:
    cell["source"] = relabel("".join(cell["source"])).splitlines(True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
notebook["metadata"]["colab"] = {"name": "gemma_270m_colab_v5e1_V3_4.ipynb", "provenance": []}

outputs = {
    PARENT / "gemma_270m_colab_v5e1_V3_4.ipynb": json.dumps(notebook, indent=1) + "\n",
}
for name in ("runner.py", "parallel_data.py", "loss_dtype_fix.py", "REQUIREMENTS.md", "DATA_MIXTURE.md"):
    outputs[HERE / name] = relabel((PREVIOUS / name).read_text())
outputs[HERE / "README.md"] = """# Gemma Colab V3.4

Open `../gemma_270m_colab_v5e1_V3_4.ipynb` in Colab Pro on one v5e-1 TPU and Run all.

V3.4 contains the Splash tracer-leak repair: both fixed-shape attention kernels are constructed in ordinary Python before any JAX transformation, with no lru-cached factory inside tracing. It retains the fused-loss gradient dtype repair, normal JIT dispatch, zero-update training warmup, six source workers, persistent preparation caches and attempt-log preservation.

The model, optimizer, data mixture and one 60-second training attempt remain fixed. Shared corpus, dependency, environment and compiler cache directories retain their existing names. New run outputs, source filename, report filename and requirement IDs identify V3.4.

Version policy: every revision shall create a new notebook filename. Previous notebooks shall never be overwritten. This builder uses exclusive file creation and refuses to replace an existing artifact. Assembly does not execute notebook code or run checks.

No local checks or tests were run. TPU confirmation remains pending the user's execution.
"""
for path, text in outputs.items():
    with path.open("x") as stream:
        stream.write(text)
print("Created new V3.4 notebook and source artifacts. Previous notebooks were not modified. No local checks or tests.")
