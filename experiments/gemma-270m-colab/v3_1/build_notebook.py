"""Assemble V3.1 artifacts only. Do not execute notebook code or local checks."""
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
original = json.loads((PARENT / "gemma_270m_colab_v5e1_V3.ipynb").read_text())
source = (PARENT / "v3" / "runner.py").read_text()
start, end = source.index("\ndef immutable_sources("), source.index("\ndef verify_raw(")
source = source[:start] + source[end:]
start, end = source.index("\ndef build_raw_corpus("), source.index("\ndef prepare_tokenizer(")
source = source[:start] + "\n\n" + (HERE / "parallel_data.py").read_text() + "\n" + source[end:]
dtype_fix = (HERE / "loss_dtype_fix.py").read_text()
source = source.replace("\ndef verify_tpu_kernels(", "\n\n" + dtype_fix + "\n\ndef verify_tpu_kernels(", 1)
source = source.replace("    attention_kernel = install_splash_attention()",
    "    install_tokamax_loss_dtype_fix()\n    attention_kernel = install_splash_attention()", 1)
source = source.replace('        return tokamax.linear_softmax_cross_entropy_loss(x, labels, w, reduction="mean", implementation="mosaic_tpu")',
    '        return fused_loss_probe(x, w, labels)', 1)
source = source.replace('        logits = jnp.matmul(x.astype(jnp.float32), w.astype(jnp.float32))\n        return jnp.mean(jax.scipy.special.logsumexp(logits, axis=-1) - logits[jnp.arange(1024), labels])',
    '        return reference_loss_probe(x, w, labels)', 1)
source = source.replace('    report["seconds"] = time.monotonic() - started',
    '''    report["loss_gradient_dtypes"] = [str(gradient.dtype) for gradient in actual_grad]
    report["loss_composition"] = "unreduced fused CE through BF16 intermediate multiplies; input-matched cotangents"
    report["loss_gradient_boundary_fix"] = "V3.1-dtype-1"
    if any(gradient.dtype != jnp.bfloat16 for gradient in actual_grad):
        raise TypeError("Fused loss probe returned a gradient with the wrong input dtype.")
    report["seconds"] = time.monotonic() - started''', 1)
source = source.replace('"V3"', '"V3.1"').replace('"V3-0', '"V31-0')
source = source.replace("Gemma Colab V3:", "Gemma Colab V3.1:")
source = source.replace("V3: one training attempt", "V3.1: one training attempt")
source = source.replace("60-second V3 run", "60-second V3.1 run")
source = source.replace("V3 engineering requirement failed", "V3.1 engineering requirement failed")
source = source.replace('        raw_reused = False',
    '        raw_reused = bool(read_json(shared / "parallel_downloads.json").get("recovered_complete_staging", False))')
source = source.replace('    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")', '''    download_report_path = shared / "parallel_downloads.json"
    if raw_reused:
        download_report = {"version": "V3.1", "raw_cache_reused": True, "source_jobs": 0,
            "max_workers": DOWNLOAD_WORKERS, "peak_active_source_jobs": 0, "fetch_seconds": 0.,
            "note": "Verified completed raw corpus reused; no source fetch was needed."}
    else:
        download_report = read_json(download_report_path)
        download_report["raw_cache_reused"] = False
    if download_report["peak_active_source_jobs"] > DOWNLOAD_WORKERS:
        raise ValueError("Source download concurrency exceeded six jobs.")
    write_json(output / "parallel_downloads.json", download_report)
    preparation_checks = {
        "V31-021_DOWNLOAD_BOUND": download_report["peak_active_source_jobs"] <= 6,
        "V31-022_PREFIX_CACHE": raw_reused or all(Path(item["directory"]).joinpath("state.json").exists()
                                                    for item in download_report.get("jobs", {}).values()),
        "V31-023_MERGE_ORDER": raw_reused or download_report.get("merge_order", [])[:12] == ["wiki_" + lang for lang in LANGUAGES],
        "V31-024_SOURCE_COMPLETE": raw_reused or download_report.get("source_jobs") == 30 and len(download_report.get("jobs", {})) == 30,
    }
    write_json(output / "preparation_requirements.json", {key: "PASS" if passed else "FAIL" for key, passed in preparation_checks.items()})
    if not all(preparation_checks.values()):
        raise ValueError("V3.1 preparation engineering requirement failed.")
    tokenizer, mapping, identity = prepare_tokenizer(config, prepared / "tokenizer")''')
source = source.replace('    runtime_elapsed = time.monotonic() - config["notebook_start_monotonic"]', '''    preparation_requirements = read_json(output / "preparation_requirements.json")
    checks.update({key: value == "PASS" for key, value in preparation_requirements.items()})
    download_report = read_json(output / "parallel_downloads.json")
    runtime_elapsed = time.monotonic() - config["notebook_start_monotonic"]''')
source = source.replace('        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,',
'''        "device": environment["device_kind"], "model_parameters": parameter_count, "optimizer": partition,
        "source_downloads": {key: download_report[key] for key in
            ("source_jobs", "max_workers", "peak_active_source_jobs", "fetch_seconds", "raw_cache_reused")},''')
(HERE / "runner.py").write_text(source)

config = {"experiment": "GEM-COLAB-V3.1", "version": "V3.1", "model_id": "google/gemma-3-270m",
    "implementation_revision": "V3.1-dtype-1",
    "tokenizer_reference": "google/gemma-3-270m", "tokenizer_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "sequence_length": 1024, "local_batch_per_chip": 4, "updates_per_dispatch": 16,
    "training_seconds": 60., "minimum_updates": 5, "seed": 42, "expected_devices": 1,
    "muon_learning_rate": .02, "adam_learning_rate": .0003, "compute_units_per_hour": 2.92}

cells = []
for old in original["cells"]:
    cell = {"cell_type": old["cell_type"], "id": uuid.uuid4().hex[:8], "metadata": {}, "source": old["source"][:]}
    if cell["cell_type"] == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

def read(index):
    return "".join(cells[index]["source"])

def put(index, text):
    cells[index]["source"] = text.splitlines(True)

intro = read(0).replace("— V3:", "— V3.1:").replace("V3 uses", "V3.1 uses")
intro = intro.replace("colab-v5e1 / V3 /", "colab-v5e1 / V3.1 /")
intro += """
**V3.1 preparation changes:** the dependency fix is built in; **6 source workers** overlap **30 source jobs** (12 Wiki + 18 training streams); translation reads project **7 columns** instead of loading duplicate chunk lists. Completed source batches and row cursors persist in Drive, and progress appears every **5 seconds** while source futures are pending. Final source selection remains in a fixed order. Training settings and the mixture remain fixed.

Your supplied V3 notebook recorded successful patched environment setup in **43.2 seconds**, then sequential data preparation. It recorded no V3 training result. V3.1's preparation speedup and training measurements remain unmeasured. An ideal six-worker source-fetch stage has a **6× ceiling under equal-cost, unconstrained I/O assumptions**; this is not an end-to-end prediction.

**V3.1 dtype repair:** the fused loss retains BF16 inputs and FP32 internal gradient accumulation, then returns each gradient in its corresponding input dtype before native Gemma differentiation. The synthetic loss check now differentiates unreduced cross-entropy through intermediate BF16 multiplies, which exercises the previously missed boundary. The supplied compilation failure occurred before any optimizer updates. The data cache and training settings remain unchanged. This repair has not been locally tested.
"""
put(0, intro)
requirements = read(1).replace("V3-0", "V31-0")
requirements = requirements.replace("| Seed | 42 |", "| Seed | 42 |\n| Parallel source workers / source jobs | 6 / 30 |\n| Prefix batch target / per-job disk cap | 4 MiB / 256 MiB |")
requirements = requirements.replace("\nVerification occurs only", """
| V31-021 | Data preparation shall run at most 6 source jobs concurrently. | Executor bound and recorded peak active source jobs. |
| V31-022 | Preparation shall persist completed source batches and row cursors independently of final corpus completion. | Atomic batch/state commits; SHA256 verification before cached-batch reuse. |
| V31-023 | Worker completion order shall not determine final sample selection. | One canonical merge order, with the 12 Wiki sources reserved first; saved order record. |
| V31-024 | Cold preparation shall finish all 30 source jobs before tokenization. | Completed-job report and frozen-corpus byte quotas; a verified warm raw corpus needs no fetch. |

Verification occurs only""")
requirements += """
Dependency requirement: setup shall retain the pinned V2 Gemma/JAX package versions, use the recorded upstream Tokamax commit that avoids the incompatible PyPI `typeguard` pin, and display/save merged stdout/stderr for failed commands. Verification is the Colab setup log, exact source commit record and full dependency lock. No `--no-deps` bypass or automatic alternate training configuration is used.
"""
put(1, requirements)
mixture = read(2)
mixture += """
V3.1 fetches source prefixes concurrently, then merges in the same canonical source order. Its initial cache target is **125% of each byte quota**, with at least **4 candidate spans per split/source bucket**, to allow global deduplication without reopening most streams. Only the original byte quotas enter the final corpus. If the buffer is insufficient, the same pinned stream continues from its committed cursor, within the original row limit and **256 MiB per-source cached JSONL cap**. The cap bounds cached candidate bytes, not HTTP traffic or Parquet row-group read sizes.

Source batches target **4 MiB** and commit after complete input rows, so a batch can exceed that target by one row. Commit attempts also occur after **5 seconds of row processing**; a blocked network read cannot be checkpointed mid-read. An interrupted job resumes at its saved row cursor; `skip()` can reread earlier source data when the remote format cannot seek directly. Completed caches avoid refetching those jobs. The original V3 implementation did not persist individual source samples before finalizing the corpus, so its unfinished sequential progress cannot be recovered from console logs. A complete, verified V3 raw corpus remains reusable.
"""
put(2, mixture)
put(3, """## 1. Connect Drive and install the managed environment
V3.1 includes the dependency repair that succeeded in your uploaded notebook. The upstream Tokamax commit is resolved once and stored in the shared dependency directory; an existing commit pin, complete lock and matching environment are reused. Commands stream stdout/stderr into the notebook and save setup logs. Nothing trains in this cell.
""")
old_setup = read(4)
setup = old_setup[:old_setup.index("REQUIREMENTS_INPUT = ")]
setup = setup.replace('colab-v5e1/V3")', 'colab-v5e1/V3.1")')
setup += '''def setup_command(command, log_name, environment=None):
    lines = []
    with (OUTPUT / log_name).open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"Setup command failed (exit {status}); full error is above and in {OUTPUT / log_name}.")
    return "".join(lines)

TOOLING = BASE / ".bootstrap"
if not (TOOLING / "uv").is_dir():
    setup_command([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(TOOLING), "uv==0.11.15"], "setup-bootstrap.log")
uv_env = dict(os.environ, PYTHONPATH=str(TOOLING))
uv = [sys.executable, "-m", "uv"]
PYTHON = BASE / ".venv-V3" / "bin" / "python"
if not PYTHON.exists():
    setup_command([*uv, "venv", "--python", "3.12", "--managed-python", str(PYTHON.parent.parent)], "setup-python.log", uv_env)
PIN_PATH = DEPENDENCY_CACHE / "tokamax-source-commit.txt"
if PIN_PATH.exists():
    tokamax_commit = PIN_PATH.read_text().strip()
else:
    git_output = setup_command(["git", "ls-remote", "https://github.com/openxla/tokamax.git", "refs/heads/main"], "setup-tokamax-pin.log")
    tokamax_commit = git_output.split()[0]
    if len(tokamax_commit) != 40 or any(character not in "0123456789abcdef" for character in tokamax_commit):
        raise RuntimeError("Tokamax did not resolve to a 40-character Git commit.")
    PIN_PATH.write_text(tokamax_commit + "\\n")
if len(tokamax_commit) != 40 or any(character not in "0123456789abcdef" for character in tokamax_commit):
    raise RuntimeError("The stored Tokamax source pin is invalid.")
REQUIREMENTS_INPUT = BASE_DEPENDENCY_LOCK + f"\\ntokamax @ git+https://github.com/openxla/tokamax.git@{tokamax_commit}\\n"
INPUT_DIGEST = hashlib.sha256(REQUIREMENTS_INPUT.encode()).hexdigest()
INPUT_PATH = WORK / "requirements.in"
INPUT_PATH.write_text(REQUIREMENTS_INPUT)
LOCK_PATH = WORK / "requirements.lock"
PERSISTENT_LOCK = DEPENDENCY_CACHE / (INPUT_DIGEST + ".lock")
if PERSISTENT_LOCK.exists():
    shutil.copyfile(PERSISTENT_LOCK, LOCK_PATH)
    print("Reusing the corrected dependency lock.", flush=True)
else:
    print("Resolving corrected V3.1 dependencies...", flush=True)
    setup_command([*uv, "pip", "compile", str(INPUT_PATH), "--python", str(PYTHON),
                   "--output-file", str(LOCK_PATH)], "dependency-resolver.log", uv_env)
    temporary_lock = DEPENDENCY_CACHE / (INPUT_DIGEST + ".lock.partial")
    shutil.copyfile(LOCK_PATH, temporary_lock)
    temporary_lock.replace(PERSISTENT_LOCK)
LOCK_DIGEST = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
ENV_MARKER = PYTHON.parent.parent / ".v3-lock-sha256"
if not ENV_MARKER.exists() or ENV_MARKER.read_text().strip() != LOCK_DIGEST:
    print("Installing the pinned V3.1 environment...", flush=True)
    setup_command([*uv, "pip", "install", "--quiet", "--python", str(PYTHON), "-r", str(LOCK_PATH)], "setup-install.log", uv_env)
    ENV_MARKER.write_text(LOCK_DIGEST + "\\n")
else:
    print("Reusing the matching installed environment.", flush=True)
PERSISTENT_JAX_CACHE = OUTPUT.parent.parent / "V3-cache" / "jax"
if PERSISTENT_JAX_CACHE.exists():
    shutil.copytree(PERSISTENT_JAX_CACHE, BASE / "jax-cache-V3", dirs_exist_ok=True)
SETUP_SECONDS = time.monotonic() - NOTEBOOK_START_MONOTONIC
shutil.copyfile(INPUT_PATH, OUTPUT / "requirements.in")
shutil.copyfile(LOCK_PATH, OUTPUT / "requirements.lock")
(OUTPUT / "dependency_source.json").write_text(json.dumps({"tokamax_commit": tokamax_commit,
    "requirements_input_sha256": INPUT_DIGEST, "requirements_lock_sha256": LOCK_DIGEST,
    "base_environment": "V2 pinned versions", "bootstrap_uv": "0.11.15"}, indent=2))
print(f"Environment ready in {SETUP_SECONDS:.1f} seconds. Tokamax commit: {tokamax_commit}")
print("Results:", OUTPUT)
'''
put(4, setup)
put(7, read(7).replace("fixed configuration", "fixed V3.1 configuration"))
old_program = read(8)
program_tail = old_program[old_program.index('\nCONFIG["tokenizer_reference"]'):]
program_tail = program_tail.replace("pilot_V3.py", "pilot_V3_1.py").replace("V3 {phase}", "V3.1 {phase}")
program_tail = program_tail.replace('    environment["TOKENIZERS_PARALLELISM"] = "false"', '''    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["JAX_TRACEBACK_FILTERING"] = "off"
    environment["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    environment["HF_HUB_ETAG_TIMEOUT"] = "30"''')
put(8, "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + program_tail)
put(11, """## 5. Fetch 30 source jobs with 6 workers, then freeze and tokenize
Twelve Wiki streams and eighteen training streams are fetched concurrently, at most six jobs at once. The large English source and four translation sources start early. Progress appears every five seconds while source futures are pending; completed batches and row cursors persist in the shared Drive cache. A coordinator reserves Wiki and applies exact source quotas/deduplication in canonical order. Translation projection reads seven required columns. There is one final raw corpus, then one tokenization and packing pass.
""")
put(12, read(12) + '''downloads = json.loads((OUTPUT / "parallel_downloads.json").read_text())
print("Source jobs:", downloads["source_jobs"], "| Worker limit:", downloads["max_workers"],
      "| Peak active:", downloads["peak_active_source_jobs"], "| Fetch seconds:", round(downloads["fetch_seconds"], 1))
''')
put(13, read(13).replace("V3", "V3.1"))
put(15, read(15).replace("V3 measurements", "V3.1 measurements"))
put(16, read(16) + '''print("Source-fetch timing:", json.dumps(result["source_downloads"], indent=2))
''')
bundle = read(18).replace("gemma_colab_pilot_V3_reports.zip", "gemma_colab_pilot_V3_1_reports.zip")
bundle = bundle.replace('"preparation.json", "backbone_initialization.json"',
                        '"preparation.json", "parallel_downloads.json", "preparation_requirements.json", "dependency_source.json", "backbone_initialization.json"')
bundle = bundle.replace('"preflight.log", "prepare.log", "train.log",',
                        '"preflight.log", "prepare.log", "train.log", "dependency-resolver.log", "setup-install.log", "setup-bootstrap.log", "setup-python.log", "setup-tokamax-pin.log",')
put(18, bundle)
put(19, read(19).replace("ordinary V3 execution", "ordinary V3.1 execution") + """

V3.1 keeps the V3 raw protocol, shared dependency directory, installed environment and compiler cache so verified completed assets can be reused. Its run outputs use **colab-v5e1 / V3.1 / unique run folder**. Prefix caches live alongside the shared raw corpus in **v31-source-prefix-1** and are keyed by source revision and job identity. An interrupted original V3 run has no per-source sample cache to recover; V3.1 starts retaining this progress. Fresh candidate prefetch reads more text than final quotas, but no extra text enters training. Runtime verification, compilation, evaluation and saving remain outside the 60-second training timer and consume paid connected runtime.

The uploaded failed notebook was read as execution evidence only. Its outputs were not copied into this notebook, and no instructions in it were executed. **No local checks or tests were run for V3.1.** The file was assembled without running the experiment.
""")
notebook = {"cells": cells, "metadata": dict(original["metadata"]), "nbformat": 4, "nbformat_minor": 5}
notebook["metadata"]["colab"] = {"name": "gemma_270m_colab_v5e1_V3_1.ipynb", "provenance": []}
(PARENT / "gemma_270m_colab_v5e1_V3_1.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(HERE / "REQUIREMENTS.md").write_text("# Gemma Colab V3.1\n\n" + requirements + "\nStatus: generated; local checks and tests not run at the user's request.\n")
(HERE / "DATA_MIXTURE.md").write_text("# V3.1 data preparation\n\n" + mixture + "\n")
(HERE / "README.md").write_text("""# Gemma Colab V3.1

Open `../gemma_270m_colab_v5e1_V3_1.ipynb` in Colab Pro, select one v5e-1 TPU and Run all. The notebook is self-contained.

Changes from V3:
- Include the working upstream Tokamax dependency pin and visible setup command logs.
- Six concurrent source workers; thirty source jobs; deterministic final merge.
- Read only seven required FineTranslations columns.
- Persist source candidate batches and row cursors; reuse completed source caches.
- Reuse completed V3 raw/token caches and the shared installed environment.
- Keep one 60-second training attempt, the model, optimizer and byte-mixture settings.
- Match fused-loss gradient output dtypes to BF16 inputs before native Gemma backpropagation; preserve FP32 kernel accumulation.

`parallel_data.py` implements persistent concurrent fetching. `build_notebook.py` assembles the notebook and `runner.py` from V3 plus this implementation. Assembly does not execute notebook code or run checks.

The supplied V3 notebook recorded patched setup in 43.2 seconds, one v5e TPU, and sequential preparation through Wiki/English/FineWeb2/math/code. It contained no training measurements. Original unfinished V3 source samples were not persisted individually and cannot be recovered from logs.

No local checks or tests were run. V3.1 performance and compatibility remain pending the user's Colab run.
""")
print("Created V3.1 notebook, runner and requirements. Artifact assembly only; no local checks/tests or notebook execution.")
