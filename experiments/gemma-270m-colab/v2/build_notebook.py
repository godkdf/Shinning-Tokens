"""Assemble the self-contained V2 notebook. This script does not execute or test it."""
import ast
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
original = json.loads((PARENT / "gemma_270m_colab_v5e1.ipynb").read_text())
setup = "".join(original["cells"][3]["source"])
baseline = json.loads((HERE / "baseline_v1.json").read_text())
source = (HERE / "runner.py").read_text()

# Reuse the exact dependency lock from the successfully executed V1 notebook.
setup = setup.replace("import os\n", "import os\nimport hashlib\n", 1)
install_start = setup.index('print("Installing the pinned Gemma/JAX environment..."')
install_end = setup.index("SETUP_SECONDS =", install_start)
setup = setup[:install_start] + '''LOCK_DIGEST = hashlib.sha256(DEPENDENCY_LOCK.encode()).hexdigest()
ENV_MARKER = PYTHON.parent.parent / ".v2-lock-sha256"
if not ENV_MARKER.exists() or ENV_MARKER.read_text().strip() != LOCK_DIGEST:
    print("Installing the pinned Gemma/JAX environment...", flush=True)
    subprocess.run([*uv, "pip", "install", "--quiet", "--python", str(PYTHON), "-r", str(LOCK_PATH)], env=uv_env, check=True)
    ENV_MARKER.write_text(LOCK_DIGEST + "\\n")
else:
    print("Reusing the matching installed environment.", flush=True)
''' + setup[install_end:]

config = {
    "experiment": "GEM-COLAB-V2", "version": "V2",
    "model_id": "google/gemma-3-270m",
    "model_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "dataset_id": "HuggingFaceFW/fineweb", "dataset_config": "sample-10BT",
    "dataset_revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
    "train_documents": 10000, "validation_documents": 32,
    "sequence_length": 1024, "seed": 42, "expected_devices": 1,
    "local_batch_per_chip": 2, "updates_per_dispatch": 16,
    "training_seconds": 60.0, "minimum_updates": 5,
    "learning_rate": 0.0003, "warmup_updates": 5,
    "adam_b1": 0.9, "adam_b2": 0.999, "adam_eps": 1e-8,
    "weight_decay": 0.01, "gradient_clip": 1.0,
    "compute_units_per_hour": 2.92,
}

cells = []
def add(kind, text):
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

add("markdown", """# Gemma 3 270M — V2 optimized one-minute Colab notebook

**One notebook. One v5e-1 TPU. One fixed 60-second scratch-training run.**

1. Upload this notebook to Google Colab Pro.
2. Select **Runtime → Change runtime type → TPU v5e-1**.
3. Select **Runtime → Run all**. Authorize Google Drive and supply your Hugging Face token when requested.

Accept the [Gemma repository terms](https://huggingface.co/google/gemma-3-270m) first. A Colab Secret named `HF_TOKEN` is supported. No source files need uploading and no configuration edits are required.

V2 combines intact-chunk document packing, batch size **2**, resident TPU inputs, compiled groups of up to **16 updates**, buffered metrics, and reusable data/environment/compiler caches. Weights start randomly with seed **42**. AdamW and the exact full-vocabulary loss use the known V1 learning configuration.

Results go to **My Drive / ShiningGem / colab-v5e1 / V2 / unique run folder**. The full resumable checkpoint is approximately **3.2 GB**. Keep Colab connected until saving finishes.

The timer covers **training only**. Setup, compilation, validation, caching, and checkpoint saving consume additional runtime. Compute units are estimated at your supplied **2.92 CU/hour**; 60 seconds of training is approximately **0.0487 CU**. Colab billing is authoritative.

**Local checks and tests have not been run, as requested. V2 TPU throughput and learning improvement are unmeasured until you run this notebook.**
""")

requirements = """## Fixed configuration and requirements

| Setting | Fixed value |
|---|---:|
| Device | One TPU v5e-1 |
| Architecture | Native `gm.nn.Gemma3_270M`, 268,098,176 parameters |
| Vocabulary | 262,144; tied embedding/output matrix |
| Initial weights | Random; seed 42 |
| Training / validation documents | 10,000 / 32, identical to V1 by ordered document hashes |
| Validation targets | 13,229 |
| Sequence / batch | 1,024 positions / 2 packed sequences |
| Main dispatch | Up to 16 updates; one update near the deadline |
| Compute / master weights | BF16 / FP32 |
| Optimizer | AdamW, LR 0.0003, warmup 5 updates, clip 1.0 |
| Adam b1 / b2 / epsilon | 0.9 / 0.999 / 1e-8 |
| Weight decay | 0.01 on matrix/tensor weights |
| Training interval | 60 seconds nominal; actual completed-work time reported |

| ID | Requirement | Verification in your Colab run |
|---|---|---|
| V2-001 | The runner shall execute one training attempt per run folder. | Exclusive start marker; repeated training-cell execution stops. |
| V2-002 | The runner shall use exactly one v5e TPU. | Device platform, kind, and count. |
| V2-003 | The packer shall preserve every original target once per dataset pass. | Source chunk assignments, target equality, and target totals. |
| V2-004 | The model shall isolate attention between packed segments. | Causal segment mask and an explicit mask verification before training. |
| V2-005 | The packer shall preserve the original context and positions of every chunk. | Whole-chunk packing and position equality. |
| V2-006 | The runner shall place training inputs on the TPU before starting the timer. | Device ownership and completed transfer barrier. |
| V2-007 | The runner shall limit each compiled call to 16 optimizer updates. | Completed dispatch counts. |
| V2-008 | The runner shall cease dispatching training calls at 60 seconds. | Monotonic timestamps, completed-work synchronization, and overrun report. |
| V2-009 | The runner shall reject an update with nonfinite loss or gradient. | Device-side finite checks; stop without retry. |
| V2-010 | The evaluator shall score the same V1 validation documents before and after training. | Document hashes and 13,229 target count. |
| V2-011 | The runner shall persist one full resumable checkpoint. | Local/persistent SHA256 equality, array count, and state schema. |
| V2-012 | The report shall provide separate training and overhead timings. | Phase durations and compute-unit estimates. |
| V2-013 | The model shall retain the native architecture and tied embedding/output table. | Parameter count, table shape, and unique table count. |
| V2-014 | The runner shall perform zero optimizer updates outside the training interval. | Zero initial step; final step equals timed update count. |
| V2-015 | Training shall change a transformer parameter. | Initial/final parameter probe. |
| V2-016 | The runner shall complete at least five optimizer updates. | Completed update count. |

Requirements use NASA-style numbered **shall** statements. Engineering status, learning status, and performance goals are reported separately.

| Performance goal | Measured V1 | V2 goal, not a guarantee |
|---|---:|---:|
| Useful position occupancy | 52.6% | At least 95% |
| Useful tokens/second | 11,406 | At least 18,000 |
| Useful tokens in 60 seconds | 684,589 | At least 1,080,000 |
| Final validation loss | 6.872893 | At most 6.872893 |

The comparison uses your saved V1 measurement; this notebook does not rerun V1. A higher token rate does not automatically imply lower validation loss.
"""
add("markdown", requirements)
add("markdown", "## 1. Create the run and prepare the pinned environment\nThe embedded V1 dependency lock is reused. The Colab kernel stays separate from the managed Python 3.12 environment.\n")
add("code", setup)
add("markdown", "## 2. Connect Google Drive and authenticate\nThe token is passed only to data preparation and is excluded from reports and checkpoints.\n")
add("code", '''from google.colab import drive, userdata
from getpass import getpass

drive.mount("/content/drive")
OUTPUT = Path("/content/drive/MyDrive/ShiningGem/colab-v5e1/V2") / RUN_ID
OUTPUT.mkdir(parents=True, exist_ok=False)
PERSISTENT_JAX_CACHE = OUTPUT.parent.parent / "V2-cache" / "jax"
if PERSISTENT_JAX_CACHE.exists():
    shutil.copytree(PERSISTENT_JAX_CACHE, BASE / "jax-cache-V2", dirs_exist_ok=True)
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass("Hugging Face token with access to google/gemma-3-270m: ")
if not HF_TOKEN.strip():
    raise ValueError("A Hugging Face token with Gemma repository access is required.")
print("V2 results folder:", OUTPUT)
''')
add("markdown", "## 3. Write the complete embedded V2 program\nEverything used by the experiment is embedded in this notebook. The fixed revisions and V1 document hashes prevent a silent data change.\n")
program = "PROGRAM_SOURCE = " + repr(source) + "\n\nBASELINE_V1 = " + repr(baseline)
program += "\nCONFIG = " + repr(config) + '''

CONFIG["notebook_start_monotonic"] = NOTEBOOK_START_MONOTONIC
CONFIG["setup_seconds"] = SETUP_SECONDS
PROGRAM_PATH = WORK / "pilot_V2.py"
PROGRAM_PATH.write_text(PROGRAM_SOURCE)
(WORK / "baseline_v1.json").write_text(json.dumps(BASELINE_V1, indent=2))
(WORK / "config.json").write_text(json.dumps(CONFIG, indent=2))
shutil.copyfile(PROGRAM_PATH, OUTPUT / "pilot_source.py")
shutil.copyfile(WORK / "baseline_v1.json", OUTPUT / "baseline_v1.json")
shutil.copyfile(WORK / "config.json", OUTPUT / "requested_config.json")
shutil.copyfile(LOCK_PATH, OUTPUT / "requirements.lock")

def run_phase(phase, token=None):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("HF_TOKEN", None)
    if token:
        environment["HF_TOKEN"] = token
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLCONFIGDIR"] = str(BASE / "matplotlib-cache")
    command = [str(PYTHON), str(PROGRAM_PATH), "--phase", phase,
               "--work", str(WORK), "--output", str(OUTPUT)]
    # Stream the subprocess output to the notebook. Logs are buffered to reduce Drive writes.
    with (OUTPUT / (phase + ".log")).open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"V2 {phase} failed. See {OUTPUT / (phase + '.log')} and failure.json. No automatic retry was attempted.")
'''
add("code", program)
add("markdown", "## 4. Verify the accelerator\nThis stops before dataset download if the runtime is not exactly one v5e TPU. It performs no training.\n")
add("code", 'run_phase("preflight")\n')
add("markdown", "## 5. Prepare and pack the fixed data\nBest-fit packing keeps each original document chunk intact. Positions restart for each segment. Compact segment IDs are stored; full attention masks are constructed per training batch. Cached files are reused only after checksum and document-identity verification.\n")
add("code", '''PREPARATION_START = time.monotonic()
run_phase("prepare", token=HF_TOKEN)
HF_TOKEN = None
PREPARATION_SECONDS = time.monotonic() - PREPARATION_START
preparation = json.loads((OUTPUT / "preparation.json").read_text())
print(json.dumps(preparation, indent=2))
''')
add("markdown", "## 6. Execute the single 60-second training run\nThis cell initializes, compiles, performs initial validation, trains once, performs final validation, and saves one checkpoint. Compilation performs zero optimizer updates. A dynamic counted JAX loop supports both 16-update calls and one-update calls without changing input shapes. Near the deadline it returns to one-update calls; completed-work time and any overrun are reported.\n")
add("code", 'run_phase("train")\n')
add("markdown", "## 7. Read the actual V2 result\nAll V2 values below come from your execution. V1 values are the previously measured reference.\n")
add("code", '''from IPython.display import display, Markdown
result = json.loads((OUTPUT / "result.json").read_text())
rows = [
    ("Engineering", result["engineering_status"]),
    ("Learning", result["learning_status"]),
    ("Completed training runs", result["run_count"]),
    ("Actual training seconds", f'{result["training_seconds"]:.3f}'),
    ("Training overrun seconds", f'{result["training_overrun_seconds"]:.3f}'),
    ("Completed optimizer updates", f'{result["updates"]:,}'),
    ("Host dispatches", result["host_dispatches"]),
    ("Useful training tokens", f'{result["scored_tokens"]:,}'),
    ("Useful position occupancy", f'{result["training_occupancy"]:.2%}'),
    ("Useful tokens/second", f'{result["tokens_per_second"]:,.1f}'),
    ("Throughput / measured V1", f'{result["v1_comparison"]["speed_ratio"]:.3f}x'),
    ("Initial validation loss", f'{result["initial_eval"]["loss"]:.6f}'),
    ("Final validation loss", f'{result["final_eval"]["loss"]:.6f}'),
    ("Final loss difference from V1", f'{result["v1_comparison"]["final_loss_difference"]:+.6f}'),
    ("Environment setup seconds", f'{SETUP_SECONDS:.1f}'),
    ("Preparation/cache seconds", f'{PREPARATION_SECONDS:.1f}'),
    ("Compilation seconds", f'{result["compilation_seconds"]:.1f}'),
    ("Checkpoint save/verification seconds", f'{result["checkpoint_save_verify_seconds"]:.1f}'),
    ("Checkpoint GB", f'{result["checkpoint_bytes"] / 1e9:.3f}'),
    ("Estimated training CU", f'{result["compute_units_estimate"]["training"]:.5f}'),
    ("Estimated CU through report", f'{result["compute_units_estimate"]["through_report"]:.5f}'),
]
display(Markdown("| Measurement | Result |\\n|---|---|\\n" + "\\n".join(f"| {name} | {value} |" for name, value in rows)))
print("Requirements:", json.dumps(result["requirements"], indent=2))
print("Performance goals:", json.dumps(result["performance_goals"], indent=2))
print("All artifacts:", OUTPUT)
print("CU values are estimates at 2.92 CU/hour. Colab billing includes additional connected runtime.")
''')
add("markdown", "## 8. Download the report bundle\nThe checkpoint remains in Drive. The small bundle includes the result, logs, source, dependency lock, packing record, V1 reference, and checkpoint checksum manifest.\n")
add("code", '''import zipfile
from google.colab import files

REPORT_ZIP = WORK / "gemma_colab_pilot_V2_reports.zip"
with zipfile.ZipFile(REPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as bundle:
    for filename in ["result.json", "environment.json", "preflight.json", "requested_config.json", "config.json",
                     "data_manifest.json", "packing.json", "preparation.json", "baseline_v1.json", "training.jsonl",
                     "checkpoint_manifest.json", "requirements.lock", "pilot_source.py", "preflight.log",
                     "prepare.log", "train.log", "cache_warning.json", "failure.json"]:
        path = OUTPUT / filename
        if path.is_file():
            bundle.write(path, filename)
shutil.copyfile(REPORT_ZIP, OUTPUT / REPORT_ZIP.name)
files.download(str(REPORT_ZIP))
print("Checkpoint:", OUTPUT / "checkpoint.npz")
print("Finished. Use Runtime > Disconnect and delete runtime when you are done.")
''')
add("markdown", """## Reproducibility and interpretation

- Model and dataset revisions are fixed to the commits used by your measured V1 run. The ordered document-hash identities must match V1 for both splits.
- Packing preserves original chunks, including their original context limits and EOS targets. A seeded permutation defines the training order; the final partial batch is padded rather than duplicating real sequences.
- The checkpoint includes parameters, AdamW moments/counts, PRNG state, completed-update count, batch cursor, configuration, packing version, data order hash, and source manifest hash. The embedded `restore_archive` routine reconstructs the full state from an abstract state template. The training entry point initializes randomly for this one pilot.
- Checkpoint persistence is verified by matching local and Drive SHA256 hashes plus schema inspection. V2 does not perform a second full-state download and third validation pass during the ordinary run. Restore execution has not been locally tested.
- Validation preserves the original V1 inputs and masks. The compiled validation loop accumulates totals in FP32; tiny numerical differences from V1's host accumulation are possible.
- A faster token rate and lower validation loss are separate outcomes. Performance goals are not included in the engineering PASS decision.
- JAX allocator counters and compiler memory estimates are reported separately. Neither is presented as an independently measured complete hardware-memory trace.
- Caches reduce repeated setup when valid matching artifacts exist. First-run compilation still has a cost. Compiler-cache reuse depends on matching executable signatures and runtime versions.
- A repeated training-cell click stops at the exclusive start marker. The notebook does not retry failures or select a different batch size.
- No local checks or tests were run for V2, at your request. All V2 runtime measurements remain pending your Colab execution.

Sources: [native Gemma](https://gemma-llm.readthedocs.io/en/latest/api/gm/nn/Gemma3_270M.html), [JAX control flow](https://docs.jax.dev/en/latest/control-flow.html), [JAX persistent cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html), [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), [NASA requirements guidance](https://www.nasa.gov/reference/appendix-c-how-to-write-a-good-requirement/).
""")

notebook = {"cells": cells, "metadata": dict(original["metadata"]), "nbformat": 4, "nbformat_minor": 5}
notebook["metadata"]["colab"] = {"name": "gemma_270m_colab_v5e1_V2.ipynb", "provenance": []}
notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
(PARENT / "gemma_270m_colab_v5e1_V2.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(HERE / "REQUIREMENTS.md").write_text("# Gemma Colab V2\n\n" + requirements + "\nStatus: notebook generated; local checks and tests not run at the user's request.\n")
print("Created gemma_270m_colab_v5e1_V2.ipynb. No notebook code, local checks, or tests were executed.")
