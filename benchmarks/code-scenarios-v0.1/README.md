# Code companion workloads v0.1

32 programming languages. Each language contributes 100 real modified-file pairs, 50 constructed repository-context requests and 50 controlled stress cases.

Each real edit contributes three inputs: the original file, modified file and actual Git diff. The 3,200 pairs therefore produce 9,600 inputs. Contexts add 1,600 and stresses add 1,600: **12,800 inputs per tokenizer**.

Edits use actual pinned Git commits and modified existing files, not generated changes. The edit index records both Git blob IDs and SHA-256 values; the audit verifies all 6,400 original/modified file versions against their Git object IDs. Development/evaluation splits are 20/80 pairs and 10/40 contexts or stresses per language, inherited from repository families in the source corpus.

Contexts serialize complete real files with frozen filename delimiters. They are constructed requests, not captured conversations or tool outputs. Stress cases use five real base files per language and ten labelled transformations; these correlated cases do not represent independent production samples. Syntactic validity is not asserted for stress inputs.

Upstream licenses apply. Repository notices and license metadata are in the base corpus manifest; historical file-specific exceptions have not been independently audited. This is a local research dataset, not an Apache-2.0 relicensing of upstream source.

Run from the source checkout after preparing tokenizer assets:

```sh
uv run --frozen python -m tokenizer_lab.code_scenarios audit
uv run --frozen python -m tokenizer_lab.code_scenarios run
```

Results are in results/code-scenarios-v0.1/. Benchmark workers block network connections. Timing fields describe correctness collection and do not establish isolated tokenizer speed rankings.
