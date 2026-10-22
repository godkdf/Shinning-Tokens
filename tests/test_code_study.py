"""Keep published code filters tied to disjoint groups and audited totals."""
from copy import deepcopy
import csv
import gzip
import io
import re

import pytest

from tokenizer_lab import code_study, offline, publication
from tokenizer_lab.io import file_hash, read_json, write_json


@pytest.fixture
def study_root(tmp_path):
    config = {"languages": [{"id": "python", "name": "Python"}, {"id": "rust", "name": "Rust"}], "min_band_files": 100}
    write_json(tmp_path / "configs/code-realistic-v0.1.json", config)
    source = {"files": 200, "config_sha256": file_hash(tmp_path / "configs/code-realistic-v0.1.json"),
              "repositories": {"a": {}, "b": {}}, "languages": {key: {"files": 100, "splits": {"development": 20, "evaluation": 80},
              "bands": {"le1k": 50, "1k-8k": 50}} for key in ("python", "rust")}}
    ledger = {"at": "2026-10-21T00:00:00+00:00", "all_profiles_verified": 2,
              "byte_exact_profiles_all_inputs": 1, "total_observations": 2400, "execution_errors": 0,
              "checkpoint_files_verified": 24, "tests_passed": 89, "suites": [], "profiles": []}
    for suite in ("code-realistic-v0.1", "code-scenarios-v0.1"):
        manifest = source if suite == "code-realistic-v0.1" else {"documents": 1000}
        path = tmp_path / f"benchmarks/{suite}/manifest.json"
        write_json(path, manifest)
        datasets = ["code-realistic-v0.1"] if suite == "code-realistic-v0.1" else list(code_study.WORKLOADS)[1:]
        ledger["suites"].append({"suite": suite, "manifest_sha256": file_hash(path), "inputs_per_profile": 200 * len(datasets),
                                 "observations": 400 * len(datasets), "errors": 0})
        for profile in ("exact", "normalizes"):
            rows = []
            for dataset in datasets:
                for language in ("python", "rust"):
                    for split, count in (("development", 20), ("evaluation", 80)):
                        changed = int(profile == "normalizes" and dataset == "code-realistic-v0.1" and language == "python" and split == "evaluation")
                        row = {"tokenizer": profile, "dataset": dataset, "language": language, "split": split, "size_band": "all",
                               "documents": count, "utf8_bytes": count * 10, "tokens": count * 2, "exact": count - changed,
                               "canonical_only": changed, "mismatches": 0, "errors": 0, "unknown_emissions": 0,
                               "utf16le_exact": count - changed, "utf16be_exact": count - changed}
                        rows.append(row)
            # Size-band subtotals deliberately duplicate some inputs; exports must exclude them.
            rows.append({**rows[0], "size_band": "le1k"})
            write_json(tmp_path / f"results/{suite}/{profile}.json", {"tokenizer": profile, "status": "completed", "manifest_sha256": file_hash(path), "benchmarks": rows})
    for profile in ("exact", "normalizes"):
        changed = int(profile == "normalizes")
        ledger["profiles"].append({"tokenizer": profile, "source_inputs": 200, "source_tokens": 400, "source_exact": 200 - changed,
                                    "companion_inputs": 1000, "companion_tokens": 2000, "companion_exact": 1000,
                                    "canonical_only": changed, "other_mismatches": 0, "execution_errors": 0})
    write_json(tmp_path / "reports/code-realistic-v0.1-validation.json", ledger)
    report = tmp_path / "reports/code-realistic-v0.1.md"
    report.write_text("[Counts](code-realistic-v0.1-summary.csv) [Evidence](code-realistic-v0.1-validation.json)\n")
    for name in ("reports/code-realistic-v0.1-summary.csv", "results/code-realistic-v0.1/REPORT.md", "results/code-realistic-v0.1/by-language.csv", "results/code-scenarios-v0.1/by-scenario.csv", "docs/CODE_BENCHMARK.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    return tmp_path


def test_export_preserves_denominators_and_excludes_size_subtotals(study_root):
    result = code_study.build(study_root)
    assert result["observations"] == 2400
    study = read_json(study_root / "site/data/code-study/study.json.gz")
    assert len(study["rows"]) == 48
    assert sum(r["documents"] for r in study["rows"]) == 2400
    assert len(study["provenance"]["aggregates"]) == 4
    assert study["summary"]["byte_exact_profiles_all_inputs"] == 1
    assert study["coverage"] == {"groups": 4, "minimum": 100, "below_minimum": 4}
    with gzip.open(study_root / "site/data/code-study/measurements.csv.gz", "rt", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 48
    assert sum(int(r["documents"]) for r in csv_rows) == 2400
    out = study_root / "site/data/code-study"
    for target in re.findall(r"\]\(([^)]+)\)", (out / "report.md").read_text()):
        assert (out / target).is_file()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "different_bytes", "wrong_fidelity", "changed_tokens", "stale_manifest"])
def test_export_rejects_incomplete_or_changed_evidence(study_root, mutation):
    path = study_root / "results/code-realistic-v0.1/exact.json"
    result = read_json(path)
    if mutation == "missing":
        result["benchmarks"].pop(0)
    elif mutation == "duplicate":
        result["benchmarks"].append(deepcopy(result["benchmarks"][0]))
    elif mutation == "different_bytes":
        result["benchmarks"][0]["utf8_bytes"] += 1
    elif mutation == "wrong_fidelity":
        result["benchmarks"][0]["exact"] -= 1
    elif mutation == "changed_tokens":
        result["benchmarks"][0]["tokens"] += 1
    else:
        result["manifest_sha256"] = "0" * 64
    write_json(path, result)
    with pytest.raises(ValueError):
        code_study.build(study_root)
    assert not (study_root / "site/data/code-study/study.json.gz").exists()


def test_saved_code_counts_survive_empty_starter_and_source_selection(study_root, tmp_path):
    code_study.build(study_root)
    raw = study_root / "site/data/code-study/raw.jsonl.gz"
    raw.write_bytes(b"unselected source text")
    kit = tmp_path / "kit"
    offline._copy_site(study_root, kit, set(), set(), set())
    assert read_json(kit / "site/data/code-study/study.json.gz")["summary"]["total_observations"] == 2400
    assert (kit / "site/data/code-study/measurements.csv.gz").is_file()
    assert not (kit / "site/data/code-study/raw.jsonl.gz").exists()
    source_files = {p.relative_to(study_root).as_posix() for p in publication.source_paths(study_root)}
    assert "site/data/code-study/study.json.gz" in source_files
    assert "site/data/code-study/measurements.csv.gz" in source_files
    assert "site/data/code-study/raw.jsonl.gz" not in source_files


def test_unprepared_study_does_not_block_historical_site(tmp_path):
    assert code_study.build(tmp_path)["status"] == "unavailable"
