"""Publish the independently audited code study without changing historical hubs."""
from __future__ import annotations

import csv
import gzip
import io
import shutil
from pathlib import Path

from .io import ROOT, file_hash, read_json, write_json

WORKLOADS = {
    "code-realistic-v0.1": "Original source files",
    "real-edit-before-v0.1": "Real edits: before",
    "real-edit-after-v0.1": "Real edits: after",
    "real-edit-diff-v0.1": "Real edits: Git diff",
    "repository-context-v0.1": "Constructed repository contexts",
    "code-stress-v0.1": "Controlled stress cases",
}
COUNTS = ("documents", "utf8_bytes", "tokens", "exact", "canonical_only",
          "mismatches", "errors", "unknown_emissions", "utf16le_exact", "utf16be_exact")
KEYS = ("tokenizer", "dataset", "language", "split")


def build(root: Path = ROOT):
    """Reject stale/incomplete aggregates; export counts, never source payloads."""
    ledger_path = root / "reports/code-realistic-v0.1-validation.json"
    if not ledger_path.is_file():
        return {"status": "unavailable", "reason": "Code study validation ledger is missing"}
    ledger = read_json(ledger_path)
    config = read_json(root / "configs/code-realistic-v0.1.json")
    languages = {item["id"]: item["name"] for item in config["languages"]}
    profile_ids = [p["tokenizer"] for p in ledger["profiles"]]
    if len(set(profile_ids)) != len(profile_ids) or len(profile_ids) != ledger["all_profiles_verified"]:
        raise ValueError("Code study profile coverage differs from validation ledger")
    rows, provenance, manifests = [], [], {}
    for suite in ledger["suites"]:
        identifier = suite["suite"]
        if identifier not in {"code-realistic-v0.1", "code-scenarios-v0.1"}:
            raise ValueError("Unknown code study suite")
        manifest_path = root / f"benchmarks/{identifier}/manifest.json"
        if file_hash(manifest_path) != suite["manifest_sha256"]:
            raise ValueError("Code study manifest differs from validation ledger")
        manifest = read_json(manifest_path)
        manifests[identifier] = manifest
        expected_datasets = {"code-realistic-v0.1"} if identifier == "code-realistic-v0.1" else set(WORKLOADS) - {"code-realistic-v0.1"}
        expected_keys = {(dataset, language, split) for dataset in expected_datasets
                         for language in languages for split in ("development", "evaluation")}
        suite_rows = []
        for profile_id in profile_ids:
            path = root / f"results/{identifier}/{profile_id}.json"
            result = read_json(path)
            if result.get("status") != "completed" or result.get("manifest_sha256") != suite["manifest_sha256"] or result.get("tokenizer") != profile_id:
                raise ValueError("Incomplete or stale code study profile: " + profile_id)
            selected = [r for r in result["benchmarks"] if r["size_band"] == "all"]
            actual_keys = {(r["dataset"], r["language"], r["split"]) for r in selected}
            if actual_keys != expected_keys or len(selected) != len(expected_keys):
                raise ValueError("Missing or duplicate code study groups: " + profile_id)
            for row in selected:
                if row["tokenizer"] != profile_id or any(not isinstance(row[k], int) or row[k] < 0 for k in COUNTS):
                    raise ValueError("Invalid code study counts")
                if sum(row[k] for k in ("exact", "canonical_only", "mismatches", "errors")) != row["documents"]:
                    raise ValueError("Code study fidelity denominator differs")
                suite_rows.append({k: row[k] for k in (*KEYS, *COUNTS)})
            if sum(r["documents"] for r in selected) != suite["inputs_per_profile"]:
                raise ValueError("Code study input coverage differs")
            provenance.append({"path": str(path.relative_to(root)), "sha256": file_hash(path)})
        # Counts and original byte denominators must agree for every tokenizer.
        baseline = {}
        for row in suite_rows:
            key = tuple(row[k] for k in ("dataset", "language", "split"))
            denominator = (row["documents"], row["utf8_bytes"])
            if baseline.setdefault(key, denominator) != denominator:
                raise ValueError("Code study tokenizers received different input denominators")
        if sum(r["documents"] for r in suite_rows) != suite["observations"] or sum(r["errors"] for r in suite_rows) != suite["errors"]:
            raise ValueError("Code study suite totals differ from validation ledger")
        rows.extend(suite_rows)
    if set(manifests) != {"code-realistic-v0.1", "code-scenarios-v0.1"}:
        raise ValueError("Both audited code study suites are required")
    for expected in ledger["profiles"]:
        selected = [r for r in rows if r["tokenizer"] == expected["tokenizer"]]
        for prefix, source in (("source", True), ("companion", False)):
            subset = [r for r in selected if (r["dataset"] == "code-realistic-v0.1") == source]
            for field, result_key in (("documents", "inputs"), ("tokens", "tokens"), ("exact", "exact")):
                if sum(r[field] for r in subset) != expected[f"{prefix}_{result_key}"]:
                    raise ValueError("Code study profile totals differ from validation ledger")
        for field, result_key in (("canonical_only", "canonical_only"), ("mismatches", "other_mismatches"), ("errors", "execution_errors")):
            if sum(r[field] for r in selected) != expected[result_key]:
                raise ValueError("Code study fidelity totals differ from validation ledger")
    if sum(r["documents"] for r in rows) != ledger["total_observations"] or sum(r["errors"] for r in rows) != ledger["execution_errors"]:
        raise ValueError("Code study totals differ from validation ledger")
    source = manifests["code-realistic-v0.1"]
    if source["config_sha256"] != file_hash(root / "configs/code-realistic-v0.1.json"):
        raise ValueError("Code study configuration differs from frozen manifest")
    byte_exact_profiles = sum(all(r["exact"] == r["documents"] and r["errors"] == 0
                                 for r in rows if r["tokenizer"] == identifier) for identifier in profile_ids)
    if byte_exact_profiles != ledger["byte_exact_profiles_all_inputs"]:
        raise ValueError("Code study byte-exact profile count differs from validation ledger")
    coverage = [{"id": key, "name": name, **source["languages"][key]} for key, name in languages.items()]
    all_cells = [n for language in coverage for n in language["bands"].values()]
    payload = {
        "schema_version": "1.0", "protocol": "code-realistic-v0.1", "validated_at": ledger["at"],
        "summary": {key: ledger[key] for key in ("all_profiles_verified", "total_observations", "execution_errors", "byte_exact_profiles_all_inputs", "checkpoint_files_verified", "tests_passed")},
        "source_files": source["files"], "repositories": len(source["repositories"]),
        "inputs_per_profile": sum(s["inputs_per_profile"] for s in ledger["suites"]),
        "coverage": {"groups": len(all_cells), "minimum": config["min_band_files"],
                     "below_minimum": sum(n < config["min_band_files"] for n in all_cells)},
        "languages": coverage, "workloads": [{"id": key, "name": name} for key, name in WORKLOADS.items()],
        "profiles": profile_ids, "rows": rows,
        "provenance": {"validation_sha256": file_hash(ledger_path), "suites": ledger["suites"], "aggregates": provenance},
        "limitations": ["Token counts and reconstruction fidelity; isolated speed and model quality were not measured.",
                        "Context requests are constructed; stress transformations are correlated and may be syntactically invalid.",
                        "Size coverage remains incomplete. Near-duplicate families, dialects and historical file-specific license exceptions are not independently audited.",
                        "NASA-inspired engineering plan; full acceptance and NASA certification are not established."],
    }
    out = root / "site/data/code-study"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "study.json.gz", payload, compact=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[*KEYS, *COUNTS])
    writer.writeheader(); writer.writerows(rows)
    (out / "measurements.csv.gz").write_bytes(gzip.compress(buffer.getvalue().encode(), mtime=0))
    shutil.copyfile(ledger_path, out / "validation.json")
    report = (root / "reports/code-realistic-v0.1.md").read_text()
    report_links = {
        "code-realistic-v0.1-summary.csv": "summary.csv",
        "../results/code-realistic-v0.1/REPORT.md": "source-report.md",
        "../results/code-realistic-v0.1/by-language.csv": "source-by-language.csv.gz",
        "../results/code-scenarios-v0.1/by-scenario.csv": "scenarios-by-language.csv.gz",
        "code-realistic-v0.1-validation.json": "validation.json",
    }
    for original, published in report_links.items():
        report = report.replace(f"]({original})", f"]({published})")
    (out / "report.md").write_text(report)
    shutil.copyfile(root / "reports/code-realistic-v0.1-summary.csv", out / "summary.csv")
    shutil.copyfile(root / "results/code-realistic-v0.1/REPORT.md", out / "source-report.md")
    for source_path, output_name in (("results/code-realistic-v0.1/by-language.csv", "source-by-language.csv.gz"),
                                     ("results/code-scenarios-v0.1/by-scenario.csv", "scenarios-by-language.csv.gz")):
        (out / output_name).write_bytes(gzip.compress((root / source_path).read_bytes(), mtime=0))
    shutil.copyfile(root / "docs/CODE_BENCHMARK.md", out / "guide.md")
    return {"status": "completed", "profiles": len(profile_ids), "rows": len(rows),
            "observations": ledger["total_observations"], "path": str(out)}
