"""Assemble NEW V3.5 and V3.6 artifacts. Never execute or check pilot code.

All delivered files use exclusive creation. Earlier notebooks are read only.
The supplied successful V3.4 dependency lock is embedded, not re-resolved.
"""
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PREVIOUS = PARENT / "v3_4"
REPORTS = Path("/Users/ketsiledikobe/Downloads/gemma_colab_pilot_V3_4_reports")
OLD = (PREVIOUS / "runner.py").read_text()
LOCK = (REPORTS / "requirements.lock").read_text()
INPUT = (REPORTS / "requirements.in").read_text()
RUNTIME = (HERE / "runtime.py").read_text()
OLD_NOTEBOOK = json.loads((PARENT / "gemma_270m_colab_v5e1_V3_4.ipynb").read_text())
STATE = """@struct.dataclass
class PilotState:
    step: jax.Array
    params: object
    opt_state: object
    rng: jax.Array
    data_position: jax.Array
"""


def text_cell(kind, source):
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": source.splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def relabel(text, version, prefix):
    return text.replace("V3.4", version).replace("V34-", prefix + "-").replace(
        "V3_4", version.replace(".", "_"))


def requirements(version, prefix, optimizer):
    rows = [
        ("001", "The runner shall make one attempt with no automatic retry or search.", "Exclusive training.started marker; preserved prior logs."),
        ("002", "The runner shall use exactly one v5e TPU.", "Device platform, kind and count at train startup."),
        ("003", "All source revisions shall be pinned.", "Frozen shared source lock and source set."),
        ("004", "The raw corpus shall remain the V3 static-byte mixture.", "Protocol and raw/token manifest SHA256 checks."),
        ("005", "Training and evaluation shall use disjoint source groups and exact spans.", "Corpus builder deduplication and split separation; not an exhaustive near-duplicate guarantee."),
        ("006", "Tokenization shall reconstruct every selected original UTF-8 byte.", "Round-trip and byte-count equality; no corpus modification."),
        ("007", "Packing shall preserve targets, chunk context and reset positions.", "Existing intact-packing checks and cached input hashes."),
        ("008", "Native attention shall enforce causality, document isolation and safe padding queries.", "Small mask comparison in Colab before training; no custom kernels."),
        ("009", "Training inputs shall reside on the TPU before timing.", "Device ownership and synchronization barrier."),
        ("010", "The model shall retain the native Gemma backbone, tied table, FP32 masters and fewer than 300M parameters.", "Parameter shapes/count/dtypes and shared backbone hash."),
        ("011", f"Every parameter shall use {optimizer}; no parameter shall use the other optimizer.", "Complete per-leaf optimizer routing and parameter totals."),
        ("012", "No training dispatch shall start at or after 60 elapsed seconds.", "Monotonic dispatch timestamps; actual overrun reported."),
        ("013", "Each dispatch shall perform between 1 and 16 updates.", "Requested and completed update counts."),
        ("014", "Nonfinite loss or gradient shall stop training; final weights shall be finite.", "Finite guard, failure evidence and final weight reduction."),
        ("015", "Zero optimizer updates shall occur outside the timed interval.", "Zero-update JIT preparation and optimizer/cursor counters."),
        ("016", "Evaluation shall score the same bytes before and after training, including all 12 FineWiki languages.", "Fixed inputs, original-byte denominators and separate EOS totals."),
        ("017", "One FP32 weights artifact shall be persisted and checksum verified.", "Local/Drive SHA256 equality, params-only schema and leaf count."),
        ("018", "Training shall change a hidden transformer weight.", "Initial/final weight probe."),
        ("019", "At least 5 timed updates shall complete.", "Final update count."),
        ("020", "Processed bytes and tokens shall reconcile with completed batches.", "CPU prefix sums, device token count, source/language equality."),
        ("021", "Source preparation shall use no more than 6 workers.", "Executor bound and recorded peak concurrency."),
        ("022", "Completed source prefixes and cursors shall persist independently of final corpus completion.", "Source cache state and checksums; warm raw corpus needs no fetch."),
        ("023", "Worker completion order shall not determine corpus selection.", "Canonical merge order with Wiki reserved first."),
        ("024", "Cold preparation shall finish all 30 source jobs.", "Source completion report, or verified complete raw cache."),
        ("025", "The training batch shall be 2 sequences of 1024 tokens.", "Resolved configuration; no batch search or fallback."),
        ("026", "One combined evaluator shall report both held-out datasets separately.", "One compiled evaluation function and runtime split identity."),
        ("027", "Per-token source/language histograms shall not execute in timed training.", "CPU prefix sums; training device input excludes accounting arrays."),
        ("028", "The artifact shall declare optimizer resumption unsupported.", "Weights-only schema and explicit report flag."),
    ]
    table = "| ID | Requirement | Verification |\n|---|---|---|\n" + "\n".join(
        f"| {prefix}-{number} | {requirement} | {verification} |"
        for number, requirement, verification in rows)
    return f"""# Gemma Colab {version} engineering requirements

Fixed configuration: one v5e-1 TPU, seed 42, batch 2 x 1024, 60 seconds,
at most 16 updates per dispatch, one attempt, random initialization.

{table}

The notebook shall reuse the exact successful V3.4 dependency lock; it shall not
resolve package versions again. A matching installed environment shall be reused.
Tokamax may remain installed in that lock but shall not execute in training.
Runtime startup shall perform hardware verification once, in the train process.

The notebook shall save reports and weights in a unique {version} run directory.
It shall preserve previous notebooks and prior attempt evidence. The artifact
builder shall use exclusive file creation. No local checks/tests shall be run.

Performance targets, not guarantees: at least 45,559.448 useful tokens/s,
at most 60 seconds compilation, at most 300 seconds notebook time with existing
data caches, at least 95% useful occupancy, and FineWiki macro BPB no higher than
2.782397528489431 for the exact V3.4 tokenizer/corpus. Performance-goal status,
engineering status and learning status shall be reported separately. These
targets also appear in V3.6; no claim is made that all-parameter Muon meets them.

The 60-second timer excludes preparation, compilation, evaluation and saving;
all of those consume paid connected runtime. At 2.92 CU/hour, exactly 60 seconds
is 0.0486667 CU and 300 seconds is 0.2433333 CU. No dollar conversion is assumed.
No successful TPU execution or restore is claimed by artifact construction.
"""


SETUP = '''import os
import hashlib
import sys
import json
import time
import uuid
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

NOTEBOOK_START_MONOTONIC = time.monotonic()
BASE = Path("/content/gemma_colab_pilot")
BASE.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
WORK = BASE / RUN_ID
WORK.mkdir(exist_ok=False)
if shutil.disk_usage(BASE).free < 4 * 1024**3:
    raise RuntimeError("This weights-only pilot needs at least 4 GiB free local disk.")
from google.colab import drive
drive.mount("/content/drive")
OUTPUT = Path("/content/drive/MyDrive/ShiningGem/colab-v5e1/__VERSION__") / RUN_ID
OUTPUT.mkdir(parents=True, exist_ok=False)
DEPENDENCY_CACHE = OUTPUT.parent.parent / "V3-dependencies"
DEPENDENCY_CACHE.mkdir(parents=True, exist_ok=True)

def setup_command(command, log_name, environment=None):
    with (OUTPUT / log_name).open("x") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"Setup failed (exit {status}); error above and in {OUTPUT / log_name}.")

INPUT_PATH = WORK / "requirements.in"
LOCK_PATH = WORK / "requirements.lock"
INPUT_PATH.write_text(REQUIREMENTS_INPUT)
LOCK_PATH.write_text(RESOLVED_DEPENDENCY_LOCK)
LOCK_DIGEST = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
INPUT_DIGEST = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
TOOLING = BASE / ".bootstrap"
if not (TOOLING / "uv").is_dir():
    setup_command([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(TOOLING),
                   "uv==0.11.15"], "setup-bootstrap.log")
uv_env = dict(os.environ, PYTHONPATH=str(TOOLING))
uv = [sys.executable, "-m", "uv"]
PYTHON = BASE / ".venv-V3" / "bin" / "python"
if not PYTHON.exists():
    setup_command([*uv, "venv", "--python", "3.12.15", "--managed-python",
                   str(PYTHON.parent.parent)], "setup-python.log", uv_env)
ENV_MARKER = PYTHON.parent.parent / ".v3-lock-sha256"
ENVIRONMENT_REUSED = ENV_MARKER.exists() and ENV_MARKER.read_text().strip() == LOCK_DIGEST
if ENVIRONMENT_REUSED:
    print("Reusing matching V3.4 environment; no package resolution or install.", flush=True)
else:
    print("Installing the embedded successful V3.4 lock; no dependency resolution step.", flush=True)
    setup_command([*uv, "pip", "install", "--quiet", "--python", str(PYTHON),
                   "-r", str(LOCK_PATH)], "setup-install.log", uv_env)
    ENV_MARKER.write_text(LOCK_DIGEST + "\\n")
PERSISTENT_JAX_CACHE = OUTPUT.parent.parent / "V3-cache" / "jax"
if PERSISTENT_JAX_CACHE.exists():
    shutil.copytree(PERSISTENT_JAX_CACHE, BASE / "jax-cache-V3", dirs_exist_ok=True)
SETUP_SECONDS = time.monotonic() - NOTEBOOK_START_MONOTONIC
shutil.copyfile(INPUT_PATH, OUTPUT / "requirements.in")
shutil.copyfile(LOCK_PATH, OUTPUT / "requirements.lock")
(OUTPUT / "dependency_source.json").write_text(json.dumps({
    "requirements_input_sha256": INPUT_DIGEST, "requirements_lock_sha256": LOCK_DIGEST,
    "source": "successful user-supplied V3.4 report lock", "environment_reused": ENVIRONMENT_REUSED,
    "dependency_resolution_performed": False, "bootstrap_uv": "0.11.15",
    "tokamax_execution": False, "splash_execution": False}, indent=2))
print(f"Environment ready in {SETUP_SECONDS:.1f} seconds.")
print("Results:", OUTPUT)
'''

PROGRAM_TAIL = '''CONFIG["tokenizer_reference"] = TOKENIZER_REFERENCE.strip()
CONFIG["tokenizer_revision"] = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1" if CONFIG["tokenizer_reference"] == "google/gemma-3-270m" else "main"
CONFIG["notebook_start_monotonic"] = NOTEBOOK_START_MONOTONIC
CONFIG["setup_seconds"] = SETUP_SECONDS
PROGRAM_PATH = WORK / "pilot___SLUG__.py"
PROGRAM_PATH.write_text(PROGRAM_SOURCE)
(WORK / "config.json").write_text(json.dumps(CONFIG, indent=2))
shutil.copyfile(PROGRAM_PATH, OUTPUT / "pilot_source.py")
shutil.copyfile(WORK / "config.json", OUTPUT / "requested_config.json")

def run_phase(phase, token=None):
    if phase == "train" and (OUTPUT / "training.started").exists():
        if (OUTPUT / "result.json").exists():
            print("Existing completed attempt; displaying result without retraining.")
            print((OUTPUT / "result.json").read_text())
            return
        raise RuntimeError("This folder already owns an attempt. Prior logs preserved. Inspect attempt_status.json and train.log; do not delete the marker blindly.")
    if phase == "prepare" and (WORK / "prepared" / "resolved_config.json").exists():
        print("Preparation already completed for this run; reusing it.")
        return
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("HF_TOKEN", None)
    if token:
        environment["HF_TOKEN"] = token
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["JAX_TRACEBACK_FILTERING"] = "off"
    environment["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    environment["HF_HUB_ETAG_TIMEOUT"] = "30"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLCONFIGDIR"] = str(BASE / "matplotlib-cache")
    command = [str(PYTHON), str(PROGRAM_PATH), "--phase", phase, "--work", str(WORK), "--output", str(OUTPUT)]
    log_path = OUTPUT / (phase + ".log")
    if log_path.exists():
        raise RuntimeError(f"Preserving existing {phase} log; this phase is not retried automatically.")
    with log_path.open("x") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"__VERSION__ {phase} failed. See {log_path} and failure.json. No automatic retry occurred.")
'''

REPORT = '''from IPython.display import display, Markdown
result = json.loads((OUTPUT / "result.json").read_text())
rows = [
    ("Version", result["version"]), ("Optimizer", result["optimizer"]["name"]),
    ("Muon / AdamW parameters", f'{result["optimizer"]["muon_parameters"]:,} / {result["optimizer"]["adamw_parameters"]:,}'),
    ("Engineering / learning / target status", f'{result["engineering_status"]} / {result["learning_status"]} / {result["performance_goal_status"]}'),
    ("Training attempts", result["run_count"]), ("Model parameters", f'{result["model_parameters"]:,}'),
    ("Actual training seconds", f'{result["training_seconds"]:.3f}'),
    ("Overrun seconds", f'{result["training_overrun_seconds"]:.3f}'),
    ("Updates", f'{result["updates"]:,}'), ("Useful tokens/s", f'{result["tokens_per_second"]:,.1f}'),
    ("Attributed original bytes/s", f'{result["original_bytes_per_second_attributed"]:,.1f}'),
    ("Useful occupancy", f'{result["training_occupancy"]:.2%}'),
    ("Initial FineWiki macro BPB", f'{result["initial_eval"]["wiki"]["language_macro_bpb"]:.6f}'),
    ("Final FineWiki macro BPB", f'{result["final_eval"]["wiki"]["language_macro_bpb"]:.6f}'),
    ("Initial / final in-mixture BPB", f'{result["initial_eval"]["in_mix"]["pooled_bpb"]:.6f} / {result["final_eval"]["in_mix"]["pooled_bpb"]:.6f}'),
    ("Setup / preparation seconds", f'{SETUP_SECONDS:.1f} / {PREPARATION_SECONDS:.1f}'),
    ("Compilation seconds", f'{result["compilation_seconds"]:.1f}'),
    ("Initial / final evaluation seconds", f'{result["initial_eval_seconds"]:.1f} / {result["final_eval_seconds"]:.1f}'),
    ("Weights save/verification seconds", f'{result["checkpoint_save_verify_seconds"]:.1f}'),
    ("Weights artifact GB", f'{result["checkpoint_bytes"] / 1e9:.3f}'),
    ("Exact optimizer resume", "Not included: weights-only artifact"),
    ("Notebook seconds through report", f'{result["notebook_elapsed_seconds"]:.1f}'),
    ("Estimated training CU", f'{result["compute_units_estimate"]["training"]:.5f}'),
    ("Estimated CU through report", f'{result["compute_units_estimate"]["through_report"]:.5f}'),
]
display(Markdown("| Measurement | Result |\\n|---|---|\\n" + "\\n".join(f"| {name} | {value} |" for name, value in rows)))
language_rows = []
for language, final in result["final_eval"]["wiki"]["languages"].items():
    initial = result["initial_eval"]["wiki"]["languages"][language]["bpb"]
    language_rows.append(f"| {language} | {initial:.6f} | {final['bpb']:.6f} | {final['bpb'] - initial:+.6f} |")
display(Markdown("| FineWiki language | Initial BPB | Final BPB | Change |\\n|---|---:|---:|---:|\\n" + "\\n".join(language_rows)))
print("Timed source byte shares:", json.dumps(result["training_domain_byte_shares"], indent=2))
print("Timed language byte shares:", json.dumps(result["training_language_byte_shares"], indent=2))
print("Requirements:", json.dumps(result["requirements"], indent=2))
print("Performance targets (not promises):", json.dumps(result["performance_goals"], indent=2))
print("Artifacts:", OUTPUT)
print("2.92 CU/hour is an estimate. Disconnecting remains necessary to stop further connected-runtime use.")
'''

BUNDLE = '''import zipfile
from google.colab import files
REPORT_ZIP = WORK / "gemma_colab_pilot___SLUG___reports.zip"
with zipfile.ZipFile(REPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as bundle:
    for name in ["result.json", "environment.json", "preflight.json", "requested_config.json", "config.json",
        "data_manifest.json", "tokenization.json", "tokenizer_identity.json", "source-lock.json",
        "preparation.json", "parallel_downloads.json", "preparation_requirements.json", "dispatch_preparation.json",
        "attempt_status.json", "timed_training.started.json", "dependency_source.json", "backbone_initialization.json",
        "optimizer_partition.json", "optimizer_parameter_routing.json", "kernel_verification.json", "training_accounting.json",
        "training.jsonl", "checkpoint_manifest.json", "requirements.in", "requirements.lock", "pilot_source.py",
        "prepare.log", "train.log", "setup-install.log", "setup-bootstrap.log", "setup-python.log", "cache_warning.json", "failure.json"]:
        path = OUTPUT / name
        if path.is_file():
            bundle.write(path, name)
shutil.copyfile(REPORT_ZIP, OUTPUT / REPORT_ZIP.name)
files.download(str(REPORT_ZIP))
print("FP32 weights:", OUTPUT / "weights.npz")
print("Tokenizer files:", OUTPUT / "tokenizer")
print("Finished. Select Runtime > Disconnect and delete runtime when done.")
'''


for version, prefix, optimizer, directory in (
        ("V3.5", "V35", "AdamW", HERE),
        ("V3.6", "V36", "Muon", PARENT / "v3_6")):
    slug = version.replace(".", "_")
    source = OLD[:OLD.index("\ndef make_optimizer(config, params):")]
    start = source.index("MODEL_HELPERS = ")
    end = source.index("\n\ndef pack_chunks", start)
    source = source[:start] + "MODEL_HELPERS = " + repr(STATE) + source[end:]
    start = source.index("def load_model_helpers(")
    end = source.index("def make_model(config):", start)
    source = source[:start] + '''def load_model_helpers(cpu_test=False):
    accelerator_imports(cpu_test)
    exec(MODEL_HELPERS, globals())


''' + source[end:]
    readable = OLD[OLD.index("def readable_evaluation("):OLD.index("def install_tokamax_loss_dtype_fix(")]
    optimizer_source = (directory / ("optimizer_adamw.py" if optimizer == "AdamW" else "optimizer_muon.py")).read_text()
    source += "\nPILOT_VERSION = " + repr(version) + "\n\n" + optimizer_source
    source += "\n\n" + readable + "\n\n" + RUNTIME + "\n\n" + OLD[OLD.index("def main():"):]
    source = relabel(source, version, prefix)
    source = source.replace(version + " used different kernels/batch/optimizer",
                            "V3.4 used different kernels/batch/optimizer")
    config = {"experiment": "GEM-COLAB-" + version, "version": version,
        "implementation_revision": version + "-native-1", "requirement_prefix": prefix,
        "model_id": "google/gemma-3-270m", "tokenizer_reference": "google/gemma-3-270m",
        "tokenizer_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
        "sequence_length": 1024, "local_batch_per_chip": 2, "updates_per_dispatch": 16,
        "training_seconds": 60., "minimum_updates": 5, "seed": 42, "expected_devices": 1,
        "optimizer_label": optimizer, "learning_rate": .0003 if optimizer == "AdamW" else .0015,
        "warmup_updates": 5, "gradient_clip": 1., "weight_decay": .01,
        "compute_units_per_hour": 2.92, "checkpoint_kind": "weights_only"}
    if optimizer == "Muon":
        config.update(muon_beta=.95, muon_ns_steps=5, muon_consistent_rms=.2)
    req = requirements(version, prefix, optimizer)
    if optimizer == "Muon":
        special = """**All-parameter Muon:** every parameter, including the tied embedding/output
table and normalization vectors, goes through the public low-level Muon transform.
There is **no AdamW branch**. Vectors/scalars are reshaped to **1 x N**, transformed,
and restored to their original shape. This is an explicitly experimental extension
of matrix Muon. Learning rate **0.0015**, momentum **0.95**, Nesterov, **5**
Newton-Schulz steps and consistent-RMS factor **0.2**. The nominal learning-rate
times RMS factor is **0.0003**; actual updates need not have exactly that RMS.
Weight decay **0.01** applies to matrix/tensor weights, not normalization vectors.
The **167,774,720-parameter** tied table is included for the default tokenizer,
and its orthogonalization can increase step and compilation time substantially.
This notebook is not represented as a known speed improvement.

Muon implementation reference: [pinned Optax 0.2.8 source](https://github.com/google-deepmind/optax/blob/v0.2.8/optax/contrib/_muon.py).
"""
    else:
        special = """**All-parameter AdamW:** learning rate **0.0003**, betas **0.9/0.999**,
epsilon **1e-8**, weight decay **0.01** on matrix/tensor weights, gradient clipping
**1.0**, and **5** warmup updates. This uses V2's native training approach with
V3.4's frozen data and tokenizer evaluation. V2 measured **45,559 tokens/s**;
the new notebook's result remains unmeasured.
"""
    intro = f"""# Gemma 270M Colab {version} — {optimizer}

Select **TPU v5e-1** in Colab, then **Run all**. One fixed **60-second** attempt.
The notebook is self-contained; no source upload or configuration editing is
needed for the default Gemma tokenizer. Hugging Face access to that repository
is required. HF_TOKEN can be supplied through Colab Secrets or the password prompt.

{special}

Both new versions use native Gemma attention and dense exact softmax, **batch 2 x
1024**, seed **42**, the same V3.4 frozen raw/token caches, CPU training-byte
accounting, one combined held-out evaluator and one FP32 weights-only artifact.
Splash and Tokamax do not execute. The existing dependency environment is retained
to avoid another package-resolution experiment. New Drive outputs use
`ShiningGem/colab-v5e1/{version}/<unique run id>`.

The FP32 weights are approximately **1.07 GB** for the default tokenizer.
**Optimizer state is deliberately not saved; exact optimizer resumption is not
supported.** Reports and tokenizer files are retained. Connected setup,
compilation, evaluation and saving consume additional CU outside the 60 seconds.
One notebook does not automatically run the other. **No local checks or tests
were run. TPU results are pending your execution.**
"""
    mix = relabel((PREVIOUS / "DATA_MIXTURE.md").read_text(), version, prefix)
    auth = "".join(OLD_NOTEBOOK["cells"][6]["source"])
    setup = "REQUIREMENTS_INPUT = " + repr(INPUT) + "\nRESOLVED_DEPENDENCY_LOCK = " + repr(LOCK) + "\n" + SETUP.replace("__VERSION__", version)
    program = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + PROGRAM_TAIL.replace("__SLUG__", slug).replace("__VERSION__", version)
    prepare = '''PREPARATION_START = time.monotonic()
run_phase("prepare", token=HF_TOKEN)
HF_TOKEN = None
PREPARATION_SECONDS = time.monotonic() - PREPARATION_START
print((OUTPUT / "preparation.json").read_text())
print((OUTPUT / "parallel_downloads.json").read_text())
'''
    cells = [text_cell("markdown", intro), text_cell("markdown", req),
        text_cell("markdown", "## 1. Connect Drive and reuse the pinned environment\nNo model update or TPU initialization occurs here. A new run folder is created.\n"),
        text_cell("code", setup),
        text_cell("markdown", "## 2. Authorize the tokenizer\nThe default is fixed. TOKENIZER_REFERENCE is the only research input to change for later tokenizer comparisons.\n"),
        text_cell("code", auth), text_cell("markdown", "## 3. Write the embedded program\nThe next cell writes source and configuration; it does not execute training.\n"),
        text_cell("code", program), text_cell("markdown", "## 4. Reuse the frozen corpus and tokenize\nVerified existing raw/token caches are reused. Cold preparation retains six source workers and thirty source jobs.\n"),
        text_cell("code", prepare), text_cell("markdown", "## 5. Verify the TPU, compile, evaluate, then train once\nHardware verification occurs once in this process. Compilation, initial evaluation and zero-update JIT preparation precede timing. Final evaluation and FP32 weights saving follow the timed interval. No automatic retry.\n"),
        text_cell("code", 'run_phase("train")\n'), text_cell("markdown", "## 6. Read measured results\nEngineering, learning and performance-target results are separate. Targets are not promises.\n"),
        text_cell("code", REPORT), text_cell("markdown", "## 7. Download reports and disconnect\nThe report ZIP excludes the large weights artifact, which remains in Drive with the tokenizer.\n"),
        text_cell("code", BUNDLE.replace("__SLUG__", slug)), text_cell("markdown", mix)]
    notebook = {"cells": cells, "metadata": {"colab": {"name": f"gemma_270m_colab_v5e1_{slug}.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"}, "accelerator": "TPU"}, "nbformat": 4, "nbformat_minor": 5}
    outputs = {directory / "runner.py": source, directory / "REQUIREMENTS.md": req,
        directory / "DATA_MIXTURE.md": mix,
        directory / "README.md": intro + "\n" + req,
        PARENT / f"gemma_270m_colab_v5e1_{slug}.ipynb": json.dumps(notebook, indent=1) + "\n"}
    for path, content in outputs.items():
        with path.open("x") as stream:
            stream.write(content)
    print(f"Created {version} notebook and sources using exclusive file creation. No pilot code, local checks or tests executed.")
