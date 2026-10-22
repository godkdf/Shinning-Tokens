# Offline Tokenizer Lab

A portable kit contains **one Python runtime, native packages, the selected tokenizers, compressed benchmark text, and the complete local interface**. It runs without Python, uv, an account, a model download, or an internet connection on the destination computer. Building or downloading a kit is the one online preparation step.

## Supported kits

The source explorer now includes **Realistic code**, a separate saved study with 32 programming languages and 2,419,200 observations. Rebuilt kits retain its small counts/provenance bundle without adding tokenizer assets or code-text payloads. Previously built kits are unchanged. Code-study reruns use the dedicated source-checkout runners and full dataset described in the [code benchmark guide](CODE_BENCHMARK.md); they are not selections in the historical Local runs catalog.

The source checkout and current starter packages are **0.2.3**, with the complete **54-profile catalog**, including GPT-2, original Llama 3 and **8 Kimi/Moonlight profiles** as separate selections. Historical 0.2.1 archives remain separate local artifacts. Empty download selections still install **0 tokenizer / 0 language payloads**. Each Kimi profile downloads only its rank data and recorded configuration/notices; no model weights or media payloads are needed.

| Kit | Computer | Minimum supported system |
| --- | --- | --- |
| `windows-x86_64` | Windows, Intel or AMD 64-bit | Windows 10 |
| `linux-x86_64` | Linux, Intel or AMD 64-bit | glibc 2.28; Ubuntu 20.04 and newer satisfy this |
| `macos-arm64` | Mac, Apple Silicon | macOS 13 |
| `macos-x86_64` | Mac, Intel | macOS 13 |
| `linux-arm64` (build-only target) | Linux, 64-bit ARM; no prebuilt release ZIP | glibc 2.28 |

Windows ARM and 32-bit systems are not native kit targets. The Mac kits use different native packages; choose the kit for the computer's processor. The Linux kit targets glibc distributions; Alpine's musl environment is outside this target.

The runtimes are pinned to **CPython 3.12.14, python-build-standalone release 20260901**. `configs/runtime-assets.json` records the exact archive byte count and SHA-256 for each target. Dependencies come from `uv.lock`; the builder accepts only hash-matching binary wheels for Python 3.12 and the destination platform. Each archive has a separate `.manifest.json` containing its final size and SHA-256.

## Launch

1. Extract the complete ZIP into a writable local folder. Keep the `runtime`, `vendor`, `src`, `site`, `configs`, and `benchmarks` folders together.
2. On Windows, double-click **Launch Tokenizer Lab.cmd**. On macOS, double-click **Launch Tokenizer Lab.command**. On Linux, run **launch-tokenizer-lab.sh**.
3. In the local launcher, select the installed tokenizers, datasets, and languages. Run the selected benchmark or open the saved comparison explorer.
4. Keep the launch window open while working. Close it to stop the local server.

The launcher opens a browser connected to this computer. A browser is required; the application does not contact the hosted Site. Language text stays in gzip-compressed files until the corresponding language is selected. The interface and benchmark engine share one compressed text copy. Before omitting the canonical JSONL copy, the builder verifies both source hashes and exact equality of every document record. A kit can benchmark only the tokenizer assets and corpus partitions it contains.

If a ZIP extractor discards executable permissions on macOS or Linux, open a terminal in the extracted `Tokenizer Lab` folder and run:

```sh
sh scripts/launch.sh
```

macOS may require the normal **Open** action for a downloaded unsigned launcher. The kit is a Python application rather than a signed macOS installer. A kit should be extracted to a user-owned folder because benchmark results are written locally.

## Custom tokenizers

Use the local launcher's custom-tokenizer import with a Hugging Face-compatible **tokenizer.json** and a display name. Select it alongside any installed reference tokenizers, or run it alone. Custom import currently accepts tokenizer JSON only; built-in SentencePiece and tiktoken profiles use their recorded adapters. Model weights are unnecessary. The application does not execute a model repository's custom Python code.

Select a dataset and its languages and run the benchmark. The run records the custom asset hash and uses the same document IDs, original text hashes, and fidelity checks as reference tokenizers. Results belong to that local run; a successful import alone is not a benchmark result.

UTF-16 checks compare the decoded text's UTF-16 code units against the original text. They do not reinterpret arbitrary UTF-16 bytes as UTF-8. Valid supplementary characters occupy **two UTF-16 code units**. Invalid isolated surrogates are a separate input-validity diagnostic; they must not be silently replaced.

## Build a selected kit

From a prepared source checkout with uv installed and online access, build the Apple Silicon kit containing BLOOM and Qwen3 on the existing English and Arabic FineWeb partitions:

```sh
uv run --frozen python -m tokenizer_lab.offline build macos-arm64 --tokenizers bloom qwen3 --languages eng_Latn arb_Arab --datasets fineweb-mixture-v0.1 --output artifacts/offline/tokenizer-lab-review-macos-arm64.zip
```

Use the dataset IDs listed by the local launcher or `benchmarks/catalog.json`; selections use exact IDs, while the interface also supports normal language names. `fineweb-mixture-v0.1` above is the existing FineWeb benchmark ID; a source checkout must already contain those pinned partitions. To include every prepared tokenizer and corpus partition, request that explicitly:

```sh
uv run --frozen python -m tokenizer_lab.offline build windows-x86_64 --all-prepared --output artifacts/offline/tokenizer-lab-windows-x86_64.zip
```

The builder rejects unknown selections, missing assets, corrupt input hashes, missing wheels, and existing output archives. It never silently substitutes a tokenizer, language, or operating-system build. Preparing more corpora or downloading tokenizer assets must happen before the kit is built. Agreements and notices in `licenses/`, runtime license files, and wheel package metadata are retained inside the kit.

The Python API is:

```python
from tokenizer_lab.offline import build_bundle
result = build_bundle(
    target="macos-arm64",
    tokenizers=["bloom", "qwen3"],
    languages=["eng_Latn", "arb_Arab"],
    datasets=["fineweb-mixture-v0.1"],
    output="artifacts/offline/tokenizer-lab-review-macos-arm64.zip",
)
print(result)
```

No relocated virtual environment is shipped. The launcher directly runs the included CPython and reads the included native packages from `vendor/`. A local `sitecustomize.py` blocks external Python network connections and DNS while permitting the loopback connection used by the interface.

## Verify a kit

The kit's `offline-manifest.json` contains the size and SHA-256 of every packaged file. From its extracted folder, run the following on macOS or Linux:

```sh
TOKENIZER_LAB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH=src:vendor runtime/bin/python3 -m tokenizer_lab.offline verify
```

On Windows, run:

```cmd
set TOKENIZER_LAB_OFFLINE=1
set PYTHONNOUSERSITE=1
set PYTHONPATH=src;vendor
runtime\python.exe -m tokenizer_lab.offline verify
```

Verification reports checked file count, corrupt or missing files, and whether the kit matches the current operating system and processor. A matching checksum proves file integrity; it does not prove execution on another operating system. Manual native portability checks are defined for all four required targets in `.github/workflows/portability.yml`, including relocation into a path containing spaces, actual Unicode encode/decode, UTF-16 code-unit equality, and blocked external DNS. The workflow runs only when dispatched manually. Its existence does not mean those remote jobs have already run.

Primary references: [Python Build Standalone release](https://github.com/astral-sh/python-build-standalone/releases/tag/20260901), [distribution documentation](https://github.com/astral-sh/python-build-standalone/blob/main/docs/running.rst), [uv target installation](https://docs.astral.sh/uv/reference/cli/), [GitHub native runner labels](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

For a starter kit with **zero tokenizer or benchmark downloads**, pass empty selections explicitly. This includes the selected operating system's runtime, its native dependencies, interface, and small catalogs:

```sh
uv run --frozen python -m tokenizer_lab.offline build macos-arm64 --tokenizers --languages --datasets --output artifacts/offline/tokenizer-lab-starter-macos-arm64.zip
```

The command line defaults to zero selected payloads. Use `--all-prepared` only when you intend to include every prepared item. In the Python API, `None` means all prepared items and `[]` means none. The Python equivalent is `build_bundle("macos-arm64", tokenizers=[], languages=[], datasets=[])`. The builder never fetches missing tokenizer or language inputs on your behalf. The local manager can show the exact size and integrity information before an explicit later download. Network access must be enabled specifically for that preparation action; benchmark execution remains offline.

## Selective downloads and storage

Fresh browser selections are empty. Tokenizers can be downloaded without choosing a benchmark; benchmark text requires an explicit dataset and language. Clear a list to choose none. Reviewing a plan does not fetch data. Download starts a separate worker only after pressing Download; benchmark and conversion requests use cached files only. Saved browser choices never trigger automatic downloads.

The plan distinguishes installed compressed bytes from network transfer bytes. FLORES source replay temporarily downloads the **25,585,843-byte** official archive once, installs only the selected split/language files, and deletes the archive. FineWeb uses bounded Parquet ranges; MADLAD reads a bounded prefix of a pinned gzip stream. Their network totals are unknown before transfer and are labelled accordingly. These costs are shown before Download. Removal deletes only selected known cached payloads and returns the actual bytes freed.

The app and benchmark engine share one compressed language-text copy. At most two selected language files stay unpacked in the browser. Starter archives include runtime/native dependencies and metadata; zero selected payloads does not mean a zero-byte application. The unused terminal-info database is omitted from portable runtimes to avoid case-sensitive filename collisions when preparing Linux packages on macOS.

## Publication artifacts

The four current starter ZIPs, exact sizes, source fingerprints and SHA-256 values are prepared under `artifacts/releases/0.2.3/`. `site/data/offline-index.json` records the corresponding multipart browser downloads and per-archive native verification. Packages contain no historical downloadable ZIPs recursively. Current static hosting and GitHub-release upload are separate publication steps. See [release preparation](RELEASE.md).

A small source archive is corpus-free; use the local manager to restore only selected pinned text and tokenizer inputs. The larger explorer archive contains saved vocabularies/maps/text. A starter contains only the selected OS runtime/dependencies plus the interface and aggregate metadata. Declared reference-source `.py` companions may be retained as data for catalog integrity; the lab never imports those upstream files.
