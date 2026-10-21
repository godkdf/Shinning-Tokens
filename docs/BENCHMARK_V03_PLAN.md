# Proposed benchmark upgrade: realistic code workloads

Status: proposal, 2026-10-21. Proposed application version: 0.3.0. Proposed protocol: code-realistic-v0.1. Existing intrinsic-v0.2 results remain frozen.

## Purpose and assumptions

Help a developer determine how much real code fits in a token budget, whether code survives tokenization unchanged, and how much time and memory tokenization consumes. “Code benchmark” is interpreted here as the programming-code portion of Tokenizer Lab. The existing multilingual benchmark remains available.

Use NASA-inspired requirements discipline: identifiable, individually verifiable requirements; documented rationale; traceability; verification evidence; operational validation; controlled baselines and changes. This proposal does not establish NASA compliance, certification, flight suitability, or a NASA software classification. All numerical targets below are project proposals, not NASA-mandated thresholds.

## Current evidence

- The current code diagnostic has 14 handwritten snippets, each with 4 indentation prefixes: 56 cases across Python, JavaScript, C, SQL, JSON and HTML.
- Published runtime cohorts use 216 documents / 1,112,864 bytes of interleaved FineWeb text, one warm-up and 5 timed repeats. Runtime is collected in separate historical cohorts.
- Native normalization remains enabled; additional templates, padding and truncation are disabled.
- Exact reconstruction, canonical-only changes, other mismatches and errors are already separate outcomes. Retain this distinction.
- The source test suite passed 79 tests locally during the initial review. That is software-test evidence, not validation that the current code sample represents developer work.

## Operational scenarios

1. File ingestion: encode complete source files with original whitespace and line endings.
2. Repository context: serialize related files with a frozen, tokenizer-independent filename/separator format. Measure complete-file retention at fixed token budgets.
3. Editing: encode actual before/after files and unified diffs from pinned commits. Measure complete-request tokens and token-count changes.
4. Code completion: tokenize prefix and suffix workloads, including incomplete code. Generic text and model-specific fill-in-the-middle wrappers are separate profiles.
5. Tool interactions: tokenize issue text, stack traces, test logs, JSON tool results and source together.
6. Stress: long lines, repetitive whitespace, minified files, Unicode identifiers, combining characters, LF/CRLF, tabs and special-token-looking literals.

## Proposed corpus

Initial target: 32,000 real source files from at least 320 repository families. At most 100 primary files per family. Split by repository family before sampling: at least 64 development families and at least 256 held-out evaluation families; forks and close copies belong to one family. Allocate 200 development files and 800 held-out evaluation files per language. Each language shall span at least 10 families, including at least 8 held-out families. Multilingual families may contribute to several languages but remain entirely within one split and within the 100-file family cap. Freeze evaluation before tokenizer comparisons. Do not claim model-training decontamination.

Provisional coverage: 32 programming languages, 1,000 real source files per language. These are balanced engineering quotas, not estimates of production traffic.

| Group | Languages | Files |
|---|---|---:|
| Application and scripting | Python, JavaScript, TypeScript, Ruby, PHP, Perl, Lua | 7,000 |
| Systems and native | C, C++, Rust, Go, Zig, Assembly | 6,000 |
| JVM and .NET | Java, Kotlin, Scala, C# | 4,000 |
| Apple and cross-platform applications | Swift, Dart | 2,000 |
| Functional and concurrent | Haskell, OCaml, Elixir, Erlang | 4,000 |
| Scientific and numerical | R, Julia, MATLAB, Fortran | 4,000 |
| Databases, automation, contracts and legacy | SQL, Bash, PowerShell, Solidity, COBOL | 5,000 |
| **Total** | **32 languages** | **32,000** |

HTML, CSS, JSON, YAML, TOML, XML, Markdown, Dockerfiles and lockfiles receive separate supporting-format results; they do not inflate the 32-language count. JavaScript and TypeScript are separate strata; JSX and TSX are labelled subgroups. Record language identification evidence beyond file extensions, using pinned classifiers and manual review of ambiguous cases. Record dialect/compiler family where relevant: SQL engine/dialect, Assembly ISA/syntax, Bash version, MATLAB versus Octave provenance, and fixed/free-form Fortran or COBOL. Aggregate dialect results only with disclosed counts and weights.

Report four source-byte bands: <=1 KiB, >1–8 KiB, >8–64 KiB, >64 KiB–1 MiB. Target at least 100 files per programming-language/size cell; audit feasibility in the pilot and resolve shortfalls before freezing the corpus. Never silently fill a shortfall from another category. Generated, vendored, minified and lockfile data receive separate strata.

Add 3,200 pinned real editing pairs/diffs (100 per language) and 1,600 repository/tool-context requests (50 per primary language), split by the same family rule. Allocate 20% development and 80% held-out cases within each language for both suites. Tag every language in mixed requests; assign one primary language for quota accounting so a request is counted once. Add 1,600 controlled adversarial cases separately (50 per language); transformations are not extra independent source samples. Include language-specific hazards such as Python indentation, shell quoting, SQL quoted identifiers, COBOL columns and Fortran continuation. Record actual byte totals after collection; there is no defensible storage/runtime estimate yet.

Sampling: enumerate eligible files at pinned commits; apply declared license, encoding and size eligibility; choose with a recorded seed within strata. Preserve original valid UTF-8 bytes, BOM and line endings. Account separately for binary, unsupported-encoding, oversize and license exclusions. Detect exact duplicates by SHA-256; identify near duplicates using a versioned comparison representation and proposed >=0.90 token-shingle Jaccard threshold. Preserve raw evaluation bytes even if detection uses normalization. Audit residual cross-split matches.

## Requirements and verification matrix

Each requirement shall gain an owner, rationale, parent scenario, implementation link, verification procedure, evidence path and disposition before implementation acceptance.

| ID | Proposed requirement | Acceptance evidence |
|---|---|---|
| B03-001 | Every evaluated source artifact shall record repository URL, commit, file path, license basis, SHA-256 and byte length. | Manifest audit: 100% populated and hashes match. |
| B03-002 | Development and evaluation sets shall share zero repository families and zero exact file hashes. | Split audit; all near-duplicate alerts resolved or disclosed before freeze. |
| B03-003 | Every compared tokenizer shall receive identical bytes and request ordering within a workload profile. | Input-hash and serialization equality tests. |
| B03-004 | Every attempted input shall produce exactly one recorded outcome, including timeout, crash and OOM outcomes. | Inject failures; reconcile scheduled and recorded IDs with zero omissions. |
| B03-005 | Fidelity shall distinguish byte-exact, canonical-only, other mismatch and execution failure. | Known fixtures for all four outcomes; classify 100% of attempts. |
| B03-006 | A tokenizer shall receive the lossless label only with zero byte mismatches and zero errors on its declared completed corpus. | Full fidelity ledger; lossy tokenizers remain in diagnostic results. |
| B03-007 | Reports shall publish per-file token counts and original-byte totals by language, size and scenario. | Recompute aggregates from rows; failed inputs never become zero-token successes. |
| B03-008 | Context-capacity analysis shall use 8,192, 32,768 and 131,072 token budgets with deterministic whole-file packing. | Boundary tests; report retained files/bytes and overhead separately. Budgets are analytical, not model-support claims. |
| B03-009 | Runtime shall separately measure batch sizes 1, 8 and 32, with actual thread settings recorded. | Configuration audit and batch/scalar output comparisons. |
| B03-010 | Runtime shall use 5 fresh-process sessions per selected profile, each with 3 warm-ups and 30 timed trials, in seeded randomized tokenizer order. | Preserve all 150 timed trial records and session metadata per cell. These are not 150 independent processes. |
| B03-011 | Single-request latency profiles shall retain at least 1,000 request durations before publishing p50/p95/p99. | Quantile calculation tests and sample-count gate; display uncertainty for tail estimates. |
| B03-012 | Memory reports shall distinguish baseline RSS, post-load RSS, sampled peak, OS high-water RSS where available and artifact bytes. | Platform-specific tests; mark unavailable measurements explicitly. No claim of cold filesystem cache from a fresh process alone. |
| B03-013 | Comparisons shall report paired effects and 95% repository-cluster bootstrap intervals using 10,000 seeded resamples. | Known-effect fixtures and exact paired-input checks. Report excluded pairs and denominators. |
| B03-014 | A failed tokenizer process shall not terminate the remaining benchmark schedule. | Timeout, crash, memory-limit and interruption fault injection. |
| B03-015 | Resuming an interrupted run shall reproduce token counts, fidelity outcomes and aggregate totals of an uninterrupted run. | Kill/resume integration test; reject incompatible checkpoint hashes. Timing equality is not required. |
| B03-016 | Completed benchmark execution shall require zero network requests after explicit input preparation. | Run with networking blocked; verify no missing input is fetched. |
| B03-017 | Every advertised native starter platform shall pass its declared smoke test on that platform. | Install/start, cached conversion and tiny benchmark on Windows x86_64, Linux x86_64, macOS arm64 and macOS x86_64; unsupported platforms clearly excluded. |
| B03-018 | Every release download link shall resolve to an uploaded asset matching its manifest hash. | Post-upload download/hash verification before public release announcement. |
| B03-019 | The frozen source suite shall contain 1,000 files for each of the 32 listed languages, with 200 development and 800 held-out files per language. | Manifest audit: 32,000 files total, per-language quotas met, family caps and split boundaries satisfied. Missing quotas block the full-coverage claim. |
| B03-020 | Each listed language shall have independently visible counts, fidelity, token efficiency and size-band results. | Report/export audit: 32 language rows with sample counts, failure counts, size coverage and repository-family counts; no undisclosed pooling. |

Timeouts and memory caps are required configuration fields. Pilot starting values: 60 seconds per scalar request, 300 seconds per timed batch and 4 GiB per worker where enforceable. Revisit using observed input sizes and hardware before freezing; keep stress profiles separately configurable. Distinguish unsupported enforcement from a passed memory-limit check.

For inferential claims, predeclare primary comparisons and metrics. Use repository-cluster resampling, not individual files as independent samples. If publishing many significance decisions, declare and apply a multiple-comparison procedure. Exploratory per-language intervals must be labelled exploratory. A speed winner requires a paired session-level 95% interval excluding no change and an effect of at least 5%; otherwise report no demonstrated practical winner. The 5% threshold is a project proposal.

## Metrics users will see

- Original bytes / emitted tokens, token counts per file, tokens per 100 lines, and exact-reconstruction rates. Lines are secondary because minification changes their meaning.
- Relative token savings on identical paired inputs, with repository-cluster uncertainty and explicit failure/mismatch counts.
- Files and bytes retained at each context budget. Example: 24,000 versus 18,000 tokens for identical source is 25% fewer tokens. It is not evidence of 25% better code generation.
- Single-request encode/decode p50, p95 and p99 milliseconds; aggregate MB/s and requests/s by batch size.
- Load seconds, process memory and measured failure/timeout counts, with hardware and workload shown beside each result.
- Tokenizer-independent prompt-body results and native chat/FIM template results in separate tables. Native-template costs are model-specific and not pooled into a universal tokenizer ranking.

## Implementation sequence and gates

1. Requirements review: resolve intended users, target languages, quotas, comparison questions, owners and execution budget. Approve the traceability matrix and sampling policy. Gate: zero unresolved mandatory acceptance criteria.
2. Corpus pilot: collect 3,200 files (100 per language) from at least 64 families and 320 editing pairs (10 per language); exercise 6 materially different tokenizer implementations. Keep pilot sources in the development split; the 3,200 source files count toward the 6,400-file development allocation. Validate language identification, dialect coverage, quotas, licenses, duplicate detection and resource limits. Gate: provenance complete; sample shortfalls and legal distribution decisions resolved.
3. Measurement engine: add scenario serialization, immutable manifests, isolated workers, failures-as-rows, resume integrity and raw timing storage. Gate: deterministic fixtures and fault injection pass.
4. Statistical/reporting layer: add strata, cluster intervals, context budgets and latency distributions. Gate: independently recomputed known-effect fixtures match.
5. Full evaluation: freeze the held-out corpus and protocol, then evaluate all 54 current profiles for fidelity and token counts. Runtime begins with the 6 pilot implementations; expand only after measuring cost and reporting which profiles were timed. At 32,000 files, the file suite yields 1,728,000 tokenizer/file observations: 345,600 development and 1,382,400 held-out. Publish held-out comparisons separately from development results; other scenarios are counted separately.
6. Operational validation: replay at least 64 preselected developer workflows (2 per listed language) from at least 32 held-out families through the real app/CLI. Compare displayed counts against captured adapter outputs and replay real packing/truncation behavior. Gate: zero unexplained discrepancies; document whether users can answer the intended context/speed/fidelity questions. This validates these workflows, not all production traffic.
7. Release review: close mandatory requirement evidence, report deviations, perform platform tests, build packages and verify uploaded release assets. Gate: 100% mandatory requirements have passing evidence or an explicit approved scope change; skipped tests never count as passes.

Do not estimate elapsed days before the pilot. Record pilot wall time, CPU time, peak memory and output bytes, then extrapolate by measured workload and tokenizer class. Avoid rerunning all historical vocabulary pair comparisons merely to add this suite.

## Planned code areas and documents

Retain old corpus/protocol generation. Add a code-corpus builder and scenario module; extend metrics.py, engine.py and runtime.py with versioned outputs; extend the launcher and explorer with workload strata and evidence links. Use unambiguous record IDs and counts for files, requests, sessions and trials.

Create requirements.json, verification-matrix.csv, corpus-manifest.json, sampling-policy.md, risk-register.md and validation-report.md. Bind evidence to source commit, protocol, tokenizer artifacts, input manifests, environment and invocation hashes. Change requests describe affected requirements and invalidate stale checkpoints/evidence. No implementation changes are made by this plan.

Primary risks: curated corpora do not establish market representativeness; model training overlap may be unknowable; repository licensing may constrain redistribution; large-file and multilingual-comment coverage may be sparse; machine noise and adapter policy differences can dominate small effects. Track each with owner, mitigation and release disposition.

## NASA references

- NASA Systems Engineering Handbook, system design processes: https://www.nasa.gov/reference/4-0-system-design-processes/
- NASA software requirements validation guidance: https://swehb.nasa.gov/spaces/7150/pages/16449673/SWE-055%2B-%2BRequirements%2BValidation
- NASA software implementation verification guidance: https://swehb.nasa.gov/spaces/7150/pages/16450546/SWE-067%2B-%2BVerify%2BImplementation

NASA guidance informs the requirements/evidence process. Corpus quantities, timing repetitions, acceptance thresholds and release scope are local engineering choices requiring project review.
