# Realistic code benchmark

The completed study measures **54 tokenizer profiles × 44,800 inputs = 2,419,200 observations**, with **0 execution errors**. **28 profiles reconstruct every input byte-for-byte**. It measures token counts and reconstruction fidelity, not code-generation quality or isolated tokenizer speed.

## Open the results

Start Tokenizer Lab from the repository root:

```sh
uv run --frozen tokenizer-lab launch
```

Open the research explorer and select **Realistic code**. The page starts with original source files, all 32 languages, and the held-out **evaluation** split. Change the workload, programming language or split; compare two profiles; filter the table to profiles that preserve every input in that selection. The CSV button exports the visible table. Separate downloads provide all language/split counts, provenance, the report and the validation ledger.

“Byte-exact on this selection” applies only to the displayed inputs. The headline **28/54** applies to all 44,800 inputs, including both splits and all six input views. A run with zero execution errors may still change bytes. Canonical-only changes are byte-exact failures.

The code study is a separate result cohort. It does not alter the historical **23,304,780-observation** intrinsic benchmark, its runtime measurements, or its frozen input catalog. The **Local runs** picker continues to use that historical catalog; the code study uses the dedicated resumable runners below. Viewing saved results requires no tokenizer assets or source payloads.

## Dataset

| Workload | Cases | Inputs per profile | Development inputs | Evaluation inputs |
|---|---:|---:|---:|---:|
| Original source files | 32,000 | 32,000 | 6,400 | 25,600 |
| Actual modified-file edit pairs, with before/after/Git diff views | 3,200 | 9,600 | 1,920 | 7,680 |
| Constructed repository contexts | 1,600 | 1,600 | 320 | 1,280 |
| Controlled stress cases | 1,600 | 1,600 | 320 | 1,280 |
| **Total** | **38,400** | **44,800** | **8,960** | **35,840** |

The source corpus contains **437,900,158 original UTF-8 bytes**, from **608 repositories**. Companion inputs add **110,078,284 bytes**, for **547,978,442 input bytes per profile**. Repeated bytes across edit views or stress transformations are intentional; these totals are input volume, not unique source bytes.

Each language contributes 1,000 source files, 100 edit pairs, 50 contexts and 50 stresses. The 32 languages are Python, JavaScript, TypeScript, Ruby, PHP, Perl, Lua, C, C++, Rust, Go, Zig, Assembly, Java, Kotlin, Scala, C#, Swift, Dart, Haskell, OCaml, Elixir, Erlang, R, Julia, MATLAB, Fortran, SQL, Bash, PowerShell, Solidity and COBOL.

Splits are **20% development / 80% evaluation**, assigned by provisional repository family. A family stays within one split; source contributions are capped at 100 files per repository. Exact and whitespace-equivalent source duplicates were removed. Related repositories and near-duplicates still require independent review.

Source files must be valid UTF-8, between **80 and 1,048,576 bytes**, inclusive. The four disjoint file-size bands are **80–1,024**, **1,025–8,192**, **8,193–65,536**, and **65,537–1,048,576 bytes**. All 32 language totals are met. **102/128 language/size groups** meet the proposed 100-file minimum; **26** do not. Exact counts and shortfalls are retained in the source manifest and the app coverage table.

Edits are real modified-existing-file pairs from pinned Git history. All **6,400** before/after file versions were verified against Git blob IDs. Context requests serialize complete real files using frozen filename delimiters; they are not captured developer conversations. Stress cases use five base files per language and ten transformations per base; those inputs are correlated and may not be syntactically valid.

## Prepare and rerun

The local full-dataset archive is `artifacts/code-realistic-v0.1-full-dataset.zip` (**126,957,629 bytes**). Extract it into the repository root; its `benchmarks/` directory supplies the frozen source and companion partitions. The results archive is `artifacts/code-realistic-v0.1-results.zip`. Each archive has a sibling `.sha256` file. These are local artifacts; their presence does not mean a GitHub release asset has been published.

Cache all 54 registered tokenizer profiles through the launcher's explicit download controls. Runs do not fetch missing assets. No model weights or training are required. From the prepared repository root:

```sh
uv sync --frozen --extra test
uv run --frozen python -m tokenizer_lab.codebench audit
uv run --frozen python -m tokenizer_lab.code_scenarios audit
uv run --frozen python -m tokenizer_lab.codebench run
uv run --frozen python -m tokenizer_lab.code_scenarios run
uv run --frozen python -m tokenizer_lab.codebench report
uv run --frozen python -m tokenizer_lab.code_scenarios report
```

Runners use cached native tokenizer adapters, preserve native normalization, and block worker network connections. Compatible checkpoints resume automatically; incompatible manifest, tokenizer or measurement fingerprints are rejected. Source payloads and raw checkpoints are excluded from ordinary Git commits. The source runner can restore its frozen inputs from recorded upstream commits with `uv run --frozen python -m tokenizer_lab.codebench restore`; that preparation step requires network access. The full dataset archive supplies the already-frozen companion payloads.

The current publication is bound to the independent audit in `reports/code-realistic-v0.1-validation.json`. A new measurement run needs its own independent reconciliation before replacing that evidence. Generating a report or rebuilding the app does not independently re-audit every raw row.

To rebuild the app from the completed audited study:

```sh
uv run --frozen tokenizer-lab build-site
```

The build validates both manifest hashes, all 54 completed profile aggregates, language/workload/split coverage, input denominators and agreement with the validation ledger. It exports a separate `site/data/code-study/` bundle. Stale or incomplete code aggregates fail the build. The build does not modify historical scientific results.

## Evidence and remaining work

The independent audit reconciled **all 2,419,200 raw rows and 10,368 checkpoint hashes**, including input identity, original byte lengths, categorical outcomes, UTF-8/UTF-16 fidelity, dependency fingerprints and aggregate counts. **89 software tests** passed for the measurement implementation before the app integration; later app verification is separate evidence.

The corpus remains provisional. Remaining work includes the 26 size-coverage gaps, near-duplicate family and dialect review, historical per-file license review, production-workload validation, isolated latency/throughput and memory trials, requirements traceability, and native platform acceptance. This is the dataset and tokenization/fidelity portion of the NASA-inspired plan; it is not NASA certification or complete plan acceptance.

Upstream licenses and notices apply to collected source. Their text and repository license metadata are retained in the manifests. The repository's Apache-2.0 license does not relicense upstream files. Original source bytes, BOMs and line endings remain preserved.
