"""Assemble V3 artifacts. This builder does not execute, import, or test the pilot."""
import ast
import json
from pathlib import Path
import uuid

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
original = json.loads((PARENT / "gemma_270m_colab_v5e1_V2.ipynb").read_text())
old_setup = "".join(original["cells"][3]["source"])
# Read an existing literal to reuse the previously measured Linux environment.
lock_text = old_setup.split("DEPENDENCY_LOCK = ", 1)[1].split("\nLOCK_PATH =", 1)[0].strip()
base_lock = ast.literal_eval(lock_text)
base = (HERE / "base.py").read_text()
# Include only the reusable I/O, accelerator, checkpoint and packing definitions.
# These are text slices for artifact construction, not Python execution/validation.
source = base[:base.index("\ndef prepare(")]
source = source.replace("Gemma Colab V2: one fixed 60-second scratch-training run on one v5e TPU.\nEmbedded verbatim in the self-contained notebook. CPU fixtures are development only.",
                        "Gemma Colab V3: one fixed multilingual tokenizer-screening pilot on one v5e TPU.")
source += base[base.index("\ndef accelerator_imports("):base.index("\ndef load_model_helpers(")]
source += base[base.index("\ndef pack_chunks("):base.index("\ndef verify_prepared(")]
source += base[base.index("\ndef packed_attention_mask("):base.index("\ndef loss_parts(")]
source += base[base.index("\ndef batch_training_arrays("):base.index("\ndef make_training_functions(")]
source += "\n\n" + (HERE / "extension.py").read_text()
(HERE / "runner.py").write_text(source)

config = {
    "experiment": "GEM-COLAB-V3", "version": "V3", "model_id": "google/gemma-3-270m",
    "tokenizer_reference": "google/gemma-3-270m",
    "tokenizer_revision": "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1",
    "sequence_length": 1024, "local_batch_per_chip": 4, "updates_per_dispatch": 16,
    "training_seconds": 60.0, "minimum_updates": 5, "seed": 42, "expected_devices": 1,
    "muon_learning_rate": .02, "adam_learning_rate": .0003, "compute_units_per_hour": 2.92,
}

cells = []
def add(kind, text):
    cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

intro = """# Gemma 3 270M — V3: one-minute JAX tokenizer pilot

**One notebook. One v5e-1 TPU. One fixed 60-second training attempt.**

1. Upload this notebook into Google Colab Pro.
2. Select **Runtime → Change runtime type → TPU v5e-1**.
3. Select **Runtime → Run all**. Authorize Drive and supply your Hugging Face token.

Accept the [Gemma repository terms](https://huggingface.co/google/gemma-3-270m) first. A Colab Secret named `HF_TOKEN` is read automatically. The default tokenizer is Gemma's; no source uploads or configuration edits are required.

V3 uses **Muon on 100,270,080 hidden-matrix parameters**, AdamW on the tied vocabulary table and norms, **Splash local/global attention**, **Tokamax fused exact full-vocabulary cross-entropy**, batch **4 × 1,024**, intact-chunk packing, resident TPU inputs and compiled groups of at most **16 updates**. FP32 master weights and optimizer states accompany BF16 model computation. All model weights start randomly at seed **42**.

The training pool contains **64 MiB of original UTF-8 text**, with **256 KiB** of separate in-mixture evaluation and **12 × 64 KiB** of FineWiki evaluation. Source weights are measured in original bytes. FineWiki never trains the model. The primary score is the mean of **12 per-language bits-per-byte values**, lower being better.

Output: **My Drive / ShiningGem / colab-v5e1 / V3 / unique run folder**. The full resumable checkpoint is approximately **2.8 GB** with the default vocabulary. Keep Colab connected until saving finishes.

**60 seconds refers to training only.** Downloading, data preparation, runtime kernel verification, compilation, evaluation and saving take additional paid runtime. At your stated **2.92 CU/hour**, exactly 60 training seconds cost **0.04867 CU**; 10 connected minutes cost **0.48667 CU**. The first preparation can take several minutes. Cached raw and tokenized data are reused on later tokenizer runs.

**No local checks or tests were run, as requested. Throughput, learning and TPU compatibility remain unmeasured until your Colab execution.** This is a screening pilot, not evidence that one tokenizer will win at full training scale.
"""
add("markdown", intro)

requirements = """## Engineering requirements

| Fixed setting | Value |
|---|---:|
| TPU / attempts | 1 × v5e-1 / 1 |
| Backbone | Native Gemma 3 270M transformer: 18 layers, width 640, 4 query heads, 1 KV head |
| Vocabulary table | Tied input/output; N tokenizer IDs plus 3 protocol IDs |
| Parameter count | 100,326,016 + 640 × (N + 3); strict total below 300,000,000 |
| Training interval | 60 seconds nominal; completed-work time and overrun reported |
| Sequence / batch / maximum dispatch | 1,024 / 4 / 16 updates |
| Muon | LR 0.02; momentum 0.95; Nesterov; 5 Newton–Schulz steps; native Optax width scaling |
| Auxiliary AdamW | LR 0.0003; β1=0.9, β2=0.999, ε=1e-8 |
| Warmup / clipping / decay | 5 updates / global norm 1.0 / matrix-weight decay 0.01 |
| Compute / master state | BF16 / FP32 |
| Seed | 42 |

| ID | Requirement | Colab verification |
|---|---|---|
| V3-001 | The runner shall permit one training attempt per run folder. | Exclusive Drive start marker; no automatic retry. |
| V3-002 | The runner shall use exactly one v5e TPU. | Platform, device kind and device count. |
| V3-003 | Preparation shall resolve every dataset to an immutable commit before reading records. | Shared source lock; pinned revision on each load. |
| V3-004 | Tokenizer runs shall reuse the same frozen raw corpus for this protocol. | Raw manifest and file SHA256 verification. |
| V3-005 | Training and evaluation shall have disjoint canonical source groups and exact text hashes. | Group separation and global source/span hash deduplication. |
| V3-006 | Tokenization shall reconstruct every selected original UTF-8 byte exactly. | Encode/decode equality; no silent normalization or sample deletion. |
| V3-007 | Packing shall preserve each target once per pass and each chunk's context and positions. | Assignment, content, position and count assertions. |
| V3-008 | Fused kernels shall satisfy stated numerical tolerances before model training. | Synthetic TPU loss and local/global segmented attention forward/gradient comparisons. |
| V3-009 | Training inputs shall reside on the TPU before the timer starts. | Device ownership and synchronization barrier. |
| V3-010 | The model shall retain the Gemma backbone, one tied table, FP32 masters and fewer than 300M parameters. | Shapes, count, dtype and shared initial-backbone hash. |
| V3-011 | Muon shall update only hidden projection matrices. | Shape-aware optimizer partition: 100,270,080 Muon parameters. |
| V3-012 | The runner shall start no training dispatch at or after 60 elapsed seconds. | Monotonic dispatch timestamps; actual overrun separately reported. |
| V3-013 | A training dispatch shall complete at most 16 updates. | Requested/completed counts. |
| V3-014 | The runner shall reject nonfinite losses/gradients and stop on nonfinite final weights. | Device finite guards and final parameter reduction. |
| V3-015 | The runner shall perform zero model optimizer updates outside the timed interval. | Initial count 0; final optimizer and data cursor equal timed updates. |
| V3-016 | Evaluation shall score the same bytes before and after training and report all 12 FineWiki languages. | Fixed input hashes and byte totals; pooled and per-language BPB. |
| V3-017 | The runner shall persist one full resumable checkpoint. | Local/Drive SHA256 equality, state schema and leaf count; no restore execution claim. |
| V3-018 | Training shall change a hidden transformer weight. | Initial/final parameter probe. |
| V3-019 | Training shall complete at least 5 optimizer updates. | Final timed update count. |
| V3-020 | The report shall reconcile processed bytes by source and language. | Equal accumulated byte totals and positive exposure. |

Verification occurs only when you run Colab. Engineering PASS and learning PASS are separate: learning PASS means FineWiki macro BPB decreases. The 95% useful-position occupancy target is a performance goal, not a promised speedup. No throughput target is presented as measured.
"""
add("markdown", requirements)

mixture = """## Fixed data mixture and scientific limits

| Training source | Original UTF-8 byte share |
|---|---:|
| FinePDFs-Edu via Smol-Data | 27.5% |
| DCLM via Smol-Data | 16.5% |
| FineWeb-Edu via Smol-Data | 11.0% |
| FineWeb2, 11 non-English languages | 30.0% |
| FineMath `finemath-3plus` | 5.0% |
| CodeParrot clean Python | 5.0% |
| FineTranslations, original text + English translation | 5.0% |
| **Total** | **100.0%** |

The first three sources use [Smol-Data's globally shuffled English mixture](https://huggingface.co/datasets/HuggingFaceFW/finepdfs_edu_50BT-dclm_30BT-fineweb_edu_20BT-shuffled). Its upstream 50/30/20 recipe uses reference-token counts; V3 reads its source labels and enforces **27.5/16.5/11.0% original-byte quotas itself**. It does not assume token shares equal byte shares.

FineWeb2's 30% is split: French **3%**, Spanish **3%**, Portuguese **2.4%**, Chinese **4.5%**, Arabic **4.5%**, Hindi **4.5%**, Swahili **2.4%**, Zulu **1.5%**, Xhosa **1.5%**, Setswana **1.5%**, Sesotho **1.2%** of total training bytes. These are declared screening priorities, including Southern African languages, not measured optimal weights.

Translations allocate **1.25%** of total bytes to each French/Arabic/Hindi/Swahili source bucket, including its English side. Sides are separate documents with one shared source-group split; neither can attend to the other. Only educational score **≥2** and non-early-stopped rows are accepted. These English sides are synthetic translations, unlike the extracted PDF source. Actual language exposure is reported rather than assumed to be 50/50.

FineWiki reserves **64 KiB per language** across English plus the 11 languages above, approximately **768 KiB total**. It is a separate domain holdout. The in-mixture set contains approximately **256 KiB**, allocated by the training source weights. UTF-8 boundaries allow up to **3 bytes shortfall per bucket**. Selected source spans are at most **32 KiB**; clipping preserves the exact source prefix, with no normalization.

Canonical URL/source groups use a deterministic hash split: **10% of groups eligible for evaluation, 90% for training**; the evaluation byte quota is much smaller than 10% of the training quota. Code groups are whole repositories. Global exact source/span hashes remove exact repeats across splits, and Wikipedia URLs are excluded from non-Wiki inputs. This does not prove absence of near duplicates or web copies without identifiable URLs.

Dataset commits are resolved **once on first preparation**, saved in the shared Drive source lock, and reused with frozen raw hashes for all tokenizer runs. Sampling uses bounded deterministic stream prefixes, followed by seeded packing/order; it is not a uniform sample of each full corpus. Realized 60-second source shares can differ from pool weights and are recorded.

The [OLMo 3 paper](https://arxiv.org/abs/2512.13961) motivates high-quality web/PDF/code/math coverage and byte-based evaluation. The [dataset-mixing article](https://huggingface.co/blog/codelion/optimal-dataset-mixing) and [Smol-Data collection](https://huggingface.co/collections/HuggingFaceFW/smol-data) inform the static English mixture. **The final multilingual weights are an engineering choice, not a reproduced optimal-mixture result.** V3 performs no mixture search, optimizer sweep or multiple pilot runs.

Primary BPB is `sum(original-text token negative log probabilities) / (original UTF-8 bytes × ln(2))`. Virtual EOS loss is reported separately. For example **69,314.7 nats over 100,000 bytes → 1 BPB**. Language macro BPB gives each of the 12 languages equal weight; pooled BPB weights them by bytes. Accumulation is FP32. All candidates score the same text, but a 1,024-token context covers different byte lengths; this is a fixed-token-context, fixed-time screen.

Vocabulary IDs are densely remapped with separate PAD/BOS/EOS protocol IDs. Total parameters change by **640 per added vocabulary row**; the backbone stays identical, and its initial parameter SHA256 must match the shared baseline. The notebook fails if a tokenizer changes selected text on decode or exceeds the 300M limit. It supports Hugging Face `tokenizer.json` artifacts without remote tokenizer code.

The timed throughput report includes tokens/s and **attributed original bytes/s**. Byte attribution uses the monotonic original-character offset frontier; each source byte is counted once per full pass. It is not a claim that every tokenizer token has an independent byte boundary. Repeated passes count as repeated processed bytes. This one-seed, one-minute result can reject obvious failures; it cannot establish a final scientific tokenizer ranking.

V2's recorded **45,559 tokens/s** used different data, batch, optimizer and vocabulary handling. V3 reports that historical number only as context. It does not claim a controlled speed ratio or compare old token loss with new BPB.
"""
add("markdown", mixture)
add("markdown", "## 1. Connect Drive and install the managed environment\nThe base Gemma/JAX versions come from your measured V2 lock. Tokamax 0.0.14 is added; Colab resolves and persists the complete dependency lock once, then installs those exact versions. This cell does no training.\n")
setup = old_setup[:old_setup.index("DEPENDENCY_LOCK = ")]
setup += '''from google.colab import drive
drive.mount("/content/drive")
OUTPUT = Path("/content/drive/MyDrive/ShiningGem/colab-v5e1/V3") / RUN_ID
OUTPUT.mkdir(parents=True, exist_ok=False)
DEPENDENCY_CACHE = OUTPUT.parent.parent / "V3-dependencies"
DEPENDENCY_CACHE.mkdir(parents=True, exist_ok=True)
'''
setup += "BASE_DEPENDENCY_LOCK = " + repr(base_lock) + "\n"
setup += '''REQUIREMENTS_INPUT = BASE_DEPENDENCY_LOCK + "\\ntokamax==0.0.14\\n"
INPUT_DIGEST = hashlib.sha256(REQUIREMENTS_INPUT.encode()).hexdigest()
INPUT_PATH = WORK / "requirements.in"
INPUT_PATH.write_text(REQUIREMENTS_INPUT)
LOCK_PATH = WORK / "requirements.lock"
PERSISTENT_LOCK = DEPENDENCY_CACHE / (INPUT_DIGEST + ".lock")
TOOLING = BASE / ".bootstrap"
if not (TOOLING / "uv").is_dir():
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--target", str(TOOLING), "uv==0.11.15"], check=True)
uv_env = dict(os.environ, PYTHONPATH=str(TOOLING))
uv = [sys.executable, "-m", "uv"]
PYTHON = BASE / ".venv-V3" / "bin" / "python"
if not PYTHON.exists():
    subprocess.run([*uv, "venv", "--python", "3.12", "--managed-python", str(PYTHON.parent.parent)], env=uv_env, check=True)
if PERSISTENT_LOCK.exists():
    shutil.copyfile(PERSISTENT_LOCK, LOCK_PATH)
else:
    print("Resolving the one V3 dependency lock...", flush=True)
    subprocess.run([*uv, "pip", "compile", str(INPUT_PATH), "--python", str(PYTHON),
                    "--output-file", str(LOCK_PATH)], env=uv_env, check=True)
    shutil.copyfile(LOCK_PATH, PERSISTENT_LOCK)
LOCK_DIGEST = hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest()
ENV_MARKER = PYTHON.parent.parent / ".v3-lock-sha256"
if not ENV_MARKER.exists() or ENV_MARKER.read_text().strip() != LOCK_DIGEST:
    print("Installing the pinned V3 environment...", flush=True)
    subprocess.run([*uv, "pip", "install", "--quiet", "--python", str(PYTHON), "-r", str(LOCK_PATH)], env=uv_env, check=True)
    ENV_MARKER.write_text(LOCK_DIGEST + "\\n")
else:
    print("Reusing the matching V3 environment.", flush=True)
PERSISTENT_JAX_CACHE = OUTPUT.parent.parent / "V3-cache" / "jax"
if PERSISTENT_JAX_CACHE.exists():
    shutil.copytree(PERSISTENT_JAX_CACHE, BASE / "jax-cache-V3", dirs_exist_ok=True)
SETUP_SECONDS = time.monotonic() - NOTEBOOK_START_MONOTONIC
shutil.copyfile(INPUT_PATH, OUTPUT / "requirements.in")
shutil.copyfile(LOCK_PATH, OUTPUT / "requirements.lock")
print(f"Environment ready in {SETUP_SECONDS:.1f} seconds. Results: {OUTPUT}")
'''
add("code", setup)
add("markdown", "## 2. Authenticate and identify the tokenizer\nThe default runs Gemma's tokenizer. For a later candidate run, this single form accepts a Hugging Face tokenizer repository or a mounted Drive directory containing `tokenizer.json`. The raw corpus and scientific protocol remain frozen. The token is excluded from artifacts.\n")
add("code", '''from google.colab import userdata
from getpass import getpass

TOKENIZER_REFERENCE = "google/gemma-3-270m" #@param {type:"string"}
try:
    HF_TOKEN = userdata.get("HF_TOKEN")
except Exception:
    HF_TOKEN = None
if not HF_TOKEN:
    HF_TOKEN = getpass("Hugging Face token with Gemma repository access: ")
if not HF_TOKEN.strip():
    raise ValueError("A Hugging Face token is required for the default gated tokenizer.")
''')
add("markdown", "## 3. Write the embedded program and fixed configuration\nThe notebook contains all experiment source. Native Gemma transformer blocks retain their parameter paths; their attention operation is replaced by verified segmented Splash attention. No pretrained model checkpoint is downloaded.\n")
program = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config)
program += '''
CONFIG["tokenizer_reference"] = TOKENIZER_REFERENCE.strip()
CONFIG["tokenizer_revision"] = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1" if TOKENIZER_REFERENCE.strip() == "google/gemma-3-270m" else "main"
CONFIG["notebook_start_monotonic"] = NOTEBOOK_START_MONOTONIC
CONFIG["setup_seconds"] = SETUP_SECONDS
PROGRAM_PATH = WORK / "pilot_V3.py"
PROGRAM_PATH.write_text(PROGRAM_SOURCE)
(WORK / "config.json").write_text(json.dumps(CONFIG, indent=2))
shutil.copyfile(PROGRAM_PATH, OUTPUT / "pilot_source.py")
shutil.copyfile(WORK / "config.json", OUTPUT / "requested_config.json")

def run_phase(phase, token=None):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("HF_TOKEN", None)
    if token:
        environment["HF_TOKEN"] = token
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["MPLCONFIGDIR"] = str(BASE / "matplotlib-cache")
    command = [str(PYTHON), str(PROGRAM_PATH), "--phase", phase, "--work", str(WORK), "--output", str(OUTPUT)]
    with (OUTPUT / (phase + ".log")).open("w") as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        status = process.wait()
    if status:
        raise RuntimeError(f"V3 {phase} failed. See {OUTPUT / (phase + '.log')} and failure.json. No automatic retry occurred.")
'''
add("code", program)
add("markdown", "## 4. Verify the accelerator\nStops before data streaming unless exactly one v5e TPU is available.\n")
add("code", 'run_phase("preflight")\n')
add("markdown", "## 5. Freeze source text, tokenize and pack\nOn the first run this reads 6 source repositories and their selected language configurations. It reserves evaluation before training, pins commits, enforces byte quotas, verifies tokenizer byte reconstruction and packs intact chunks. Future candidates reuse the same source text from the shared Drive cache. A missing source or quota fails explicitly.\n")
add("code", '''PREPARATION_START = time.monotonic()
run_phase("prepare", token=HF_TOKEN)
HF_TOKEN = None
PREPARATION_SECONDS = time.monotonic() - PREPARATION_START
print(json.dumps(json.loads((OUTPUT / "preparation.json").read_text()), indent=2))
''')
add("markdown", "## 6. Execute the single training attempt\nThis initializes the model, checks synthetic kernel accuracy on your TPU, compiles without updating model weights, scores the initial model, trains for a nominal 60 seconds, scores the final model and saves one checkpoint. A final in-flight update may finish after the deadline; the report records the overrun. Repeating this cell stops at the Drive start marker.\n")
add("code", 'run_phase("train")\n')
add("markdown", "## 7. Read the V3 measurements\nLower BPB is better. The primary score weights each of the 12 languages equally. Source and language exposure describe what actually entered the timed run.\n")
add("code", '''from IPython.display import display, Markdown
result = json.loads((OUTPUT / "result.json").read_text())
rows = [
    ("Engineering", result["engineering_status"]), ("Learning", result["learning_status"]),
    ("Training attempts", result["run_count"]), ("Model parameters", f'{result["model_parameters"]:,}'),
    ("Actual training seconds", f'{result["training_seconds"]:.3f}'),
    ("Overrun seconds", f'{result["training_overrun_seconds"]:.3f}'), ("Updates", f'{result["updates"]:,}'),
    ("Useful tokens/s", f'{result["tokens_per_second"]:,.1f}'),
    ("Attributed original bytes/s", f'{result["original_bytes_per_second_attributed"]:,.1f}'),
    ("Useful occupancy", f'{result["training_occupancy"]:.2%}'),
    ("Initial FineWiki macro BPB", f'{result["initial_eval"]["wiki"]["language_macro_bpb"]:.6f}'),
    ("Final FineWiki macro BPB", f'{result["final_eval"]["wiki"]["language_macro_bpb"]:.6f}'),
    ("Initial in-mixture BPB", f'{result["initial_eval"]["in_mix"]["pooled_bpb"]:.6f}'),
    ("Final in-mixture BPB", f'{result["final_eval"]["in_mix"]["pooled_bpb"]:.6f}'),
    ("Setup seconds", f'{SETUP_SECONDS:.1f}'), ("Preparation seconds", f'{PREPARATION_SECONDS:.1f}'),
    ("Kernel verification seconds", f'{result["kernel_verification_seconds"]:.1f}'),
    ("Compilation seconds", f'{result["compilation_seconds"]:.1f}'),
    ("Initial / final evaluation seconds", f'{result["initial_eval_seconds"]:.1f} / {result["final_eval_seconds"]:.1f}'),
    ("Checkpoint save/verification seconds", f'{result["checkpoint_save_verify_seconds"]:.1f}'),
    ("Checkpoint GB", f'{result["checkpoint_bytes"] / 1e9:.3f}'),
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
print("Artifacts:", OUTPUT)
print("CU estimates use 2.92 CU/hour; Colab billing includes additional connected runtime.")
''')
add("markdown", "## 8. Download the report bundle\nThe checkpoint, tokenizer artifact and frozen shared source text remain in Drive. The download contains measurements, manifests, hashes, source, locks and logs.\n")
add("code", '''import zipfile
from google.colab import files
REPORT_ZIP = WORK / "gemma_colab_pilot_V3_reports.zip"
with zipfile.ZipFile(REPORT_ZIP, "w", zipfile.ZIP_DEFLATED) as bundle:
    for name in ["result.json", "environment.json", "preflight.json", "requested_config.json", "config.json",
                 "data_manifest.json", "tokenization.json", "tokenizer_identity.json", "source-lock.json",
                 "preparation.json", "backbone_initialization.json", "optimizer_partition.json", "kernel_verification.json",
                 "training.jsonl", "checkpoint_manifest.json", "requirements.in", "requirements.lock", "pilot_source.py",
                 "preflight.log", "prepare.log", "train.log", "cache_warning.json", "failure.json"]:
        path = OUTPUT / name
        if path.is_file():
            bundle.write(path, name)
shutil.copyfile(REPORT_ZIP, OUTPUT / REPORT_ZIP.name)
files.download(str(REPORT_ZIP))
print("Checkpoint:", OUTPUT / "checkpoint.npz")
print("Finished. Select Runtime > Disconnect and delete runtime when done.")
''')
add("markdown", """## What the run establishes

This notebook measures one fixed 60-second configuration. It neither reruns V2 nor searches recipes, batch sizes, kernels or learning rates. Synthetic kernel checks perform no optimizer update to the pilot model. A failure stops without an automatic second attempt or an unreported fallback.

The full checkpoint stores parameters, Muon/AdamW states, optimizer step, PRNG state, data cursor, tokenizer mapping identity, source manifests and configuration. `restore_archive` is included for later controlled continuation; ordinary V3 execution does not load or resume a checkpoint. Saving is checked by SHA256/schema, not by a claimed restore execution.

The corpus is frozen per protocol in **My Drive / ShiningGem / colab-v5e1 / V3-shared**. Keep that shared directory for future tokenizer comparisons. Exact sequence packing and timed bytes differ by tokenizer, and vocabulary size affects parameter count and compute. Both effects are explicitly reported. A one-minute learning score is noisy and does not replace a later controlled confirmation.

**No local execution, local checks or local tests were performed.** Your Colab run supplies all runtime verification and measurements. Sources: [Smol-Data](https://huggingface.co/collections/HuggingFaceFW/smol-data), [OLMo 3](https://arxiv.org/abs/2512.13961), [mixing context](https://huggingface.co/blog/codelion/optimal-dataset-mixing), [Optax Muon](https://optax.readthedocs.io/en/latest/api/contrib.html#optax.contrib.muon), [Tokamax](https://github.com/openxla/tokamax), [JAX Splash attention](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu/splash_attention), [NASA requirements guidance](https://www.nasa.gov/reference/appendix-c-how-to-write-a-good-requirement/).
""")

notebook = {"cells": cells, "metadata": dict(original["metadata"]), "nbformat": 4, "nbformat_minor": 5}
notebook["metadata"]["colab"] = {"name": "gemma_270m_colab_v5e1_V3.ipynb", "provenance": []}
notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
(PARENT / "gemma_270m_colab_v5e1_V3.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(HERE / "REQUIREMENTS.md").write_text("# Gemma Colab V3\n\n" + requirements + "\nStatus: generated; local checks and tests not run at the user's request.\n")
(HERE / "DATA_MIXTURE.md").write_text("# Gemma Colab V3 data protocol\n\n" + mixture + "\n")
(HERE / "README.md").write_text("# V3 source\n\nOpen `../gemma_270m_colab_v5e1_V3.ipynb` in Google Colab Pro on one v5e-1 TPU, then Run all. The notebook is self-contained.\n\n`base.py` supplies reusable V2 definitions. `extension.py` contains the V3 implementation. `build_notebook.py` assembles `runner.py`, the notebook, and the requirements/data documents without executing notebook code or running checks.\n\nNo local checks or tests were run. V3 measurements remain pending the user's Colab run.\n")
print("Created V3 notebook, embedded runner, engineering requirements and data protocol. No notebook execution or local checks/tests were performed.")
