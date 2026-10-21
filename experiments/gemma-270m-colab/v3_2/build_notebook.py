"""Assemble the corrected V3.1 artifacts under the user-requested V3.2 name.

No notebook code, local checks or tests are executed by this builder.
"""
import json
from pathlib import Path
from invocation_fix import apply_invocation_fix

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PREVIOUS = PARENT / "v3_1"


def relabel(text):
    # Keep V3-shared, V3-dependencies, .venv-V3 and v31-source-prefix-1
    # unchanged so existing data, package and compiler caches remain reusable.
    return text.replace("V3.1", "V3.2").replace("V31-", "V32-").replace("V3_1", "V3_2")


notebook = json.loads((PARENT / "gemma_270m_colab_v5e1_V3_1.ipynb").read_text())
source = apply_invocation_fix(relabel((PREVIOUS / "runner.py").read_text()))
for cell in notebook["cells"]:
    text = relabel("".join(cell["source"]))
    if text.startswith("PROGRAM_SOURCE = "):
        text = "PROGRAM_SOURCE = " + repr(source) + text[text.index("\nCONFIG = "):]
        text = text.replace('"implementation_revision": "V3.2-dtype-1"',
                            '"implementation_revision": "V3.2-jit-dispatch-1"')
        text = text.replace("'implementation_revision': 'V3.2-dtype-1'",
                            "'implementation_revision': 'V3.2-jit-dispatch-1'")
    text = text.replace('"preparation_requirements.json", "dependency_source.json"',
                        '"preparation_requirements.json", "dispatch_preparation.json", "dependency_source.json"')
    cell["source"] = text.splitlines(True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
notebook["cells"][0]["source"] += ["\n**V3.2 compiled-call repair:** execution uses the normal `jax.jit` wrappers, which forward closed-over kernel constants. The training wrapper is warmed with **0 requested updates before timing**. Compiled objects are retained only for precompilation and memory inspection. The supplied failed run reached initial evaluation, completed **0 training updates**, and reused its raw/token caches. RAM/disk exhaustion was not reported by its traceback. This repair has not been locally tested.\n"]
notebook["cells"][13]["source"] += ["\nPrecompilation also performs a zero-update JIT dispatch. Initial evaluation and all timed training dispatches use the normal JIT wrappers. Initial evaluation includes any first-call JIT dispatch/cache overhead.\n"]
notebook["metadata"]["colab"] = {"name": "gemma_270m_colab_v5e1_V3_2.ipynb", "provenance": []}
(PARENT / "gemma_270m_colab_v5e1_V3_2.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(HERE / "runner.py").write_text(source)
for name in ("parallel_data.py", "loss_dtype_fix.py", "REQUIREMENTS.md", "DATA_MIXTURE.md"):
    (HERE / name).write_text(relabel((PREVIOUS / name).read_text()))
(HERE / "README.md").write_text("""# Gemma Colab V3.2

Open `../gemma_270m_colab_v5e1_V3_2.ipynb` in Colab Pro on one v5e-1 TPU and Run all.

V3.2 is the user-requested name for the corrected V3.1 notebook. It includes the fused-loss gradient dtype repair, upstream Tokamax dependency pin, six source download workers, persistent source-prefix caches and one 60-second training attempt. The scientific mixture and training settings are unchanged.

The compiled-call repair routes evaluation and training through their normal JIT wrappers, preserving closed-over kernel constants. A zero-update dispatch warms the training JIT before the timed interval. Explicit compiled objects are used only for precompilation/memory inspection. `dispatch_preparation.json` records the warmup counts; initial evaluation includes first-call JIT dispatch overhead. The uploaded failed V3.2 notebook reached initial evaluation and completed zero training updates.

The notebook title, experiment version, requirement IDs, runtime source filename, report archive name and new Drive output folder identify V3.2. Shared data, dependency, environment and compiler cache paths retain their existing names for reuse.

`build_notebook.py` assembles these artifacts from the corrected V3.1 source without executing notebook code. No local checks or tests were run. TPU confirmation remains pending the user's execution.
""")
print("Created V3.2 notebook and source artifacts. No notebook execution, local checks or tests.")
