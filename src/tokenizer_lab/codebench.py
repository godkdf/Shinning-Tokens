"""Pinned real-source corpus collection and isolated all-tokenizer evaluation.

Collection reads public repository data; it never executes collected source code.
The benchmark worker is a separate cached-only process for each tokenizer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import random
import re
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote

from .io import ROOT, file_hash, read_json, read_jsonl, sha256, write_json, write_jsonl

CONFIG = ROOT / "configs/code-realistic-v0.1.json"
CACHE = ROOT / ".cache/code-realistic-v0.1"
CORPUS = ROOT / "benchmarks/code-realistic-v0.1"
RESULTS = ROOT / "results/code-realistic-v0.1"
SEED = 20261021
CORRECTNESS_WORKERS = 4
EXCLUDED = {"node_modules", "vendor", "vendored", ".git", "dist", "build", "third_party",
            "third-party", "thirdparty", "external", "extern", "deps", "bower_components",
            "generated", "Pods", ".venv", "venv", "__pycache__"}
LICENSES = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MPL-2.0",
            "GPL-2.0", "GPL-3.0", "LGPL-2.1", "LGPL-3.0", "AGPL-3.0", "Unlicense",
            "CC0-1.0", "EPL-1.0", "EPL-2.0", "BSL-1.0", "Zlib", "Artistic-2.0",
            "GPL-2.0-only", "GPL-3.0-only", "GPL-2.0-or-later", "GPL-3.0-or-later",
            "LGPL-2.1-only", "LGPL-3.0-only", "LGPL-2.1-or-later", "LGPL-3.0-or-later"}


def now():
    return datetime.now(timezone.utc).isoformat()


def event(kind, **fields):
    print(json.dumps({"at": now(), "event": kind, **fields}, ensure_ascii=False), flush=True)


def band(size):
    return "le1k" if size <= 1024 else "1k-8k" if size <= 8192 else "8k-64k" if size <= 65536 else "64k-1m"


def stable(value):
    return sha256(f"{SEED}:{value}".encode())


def fingerprint():
    import importlib.metadata
    code = {name: file_hash(ROOT/f"src/tokenizer_lab/{name}.py") for name in ["codebench", "metrics", "adapters", "legacy", "io", "transport"]}
    return {"code": code, "registry_sha256": file_hash(ROOT/"configs/registry.json"), "uv_lock_sha256": file_hash(ROOT/"uv.lock"),
        "packages": {name: importlib.metadata.version(name) for name in ["tokenizers", "tiktoken", "sentencepiece", "numpy"]}}


def api_get(url, *, params=None):
    import requests
    for attempt in range(8):
        response = requests.get(url, params=params, timeout=(15, 60),
                                headers={"User-Agent": "Tokenizer-Lab-code-corpus/0.1", "Accept": "application/vnd.github+json"})
        if response.status_code in (403, 429):
            delay = max(7, min(65, int(response.headers.get("X-RateLimit-Reset", time.time()+30))-int(time.time())+2))
            event("api_rate_wait", seconds=delay)
            time.sleep(delay)
            continue
        if response.status_code >= 500:
            time.sleep(min(2 ** attempt, 20))
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("GitHub metadata request exhausted retries")


def discover():
    CACHE.mkdir(parents=True, exist_ok=True)
    cfg = read_json(CONFIG)
    for language in cfg["languages"]:
        dest = CACHE / "discovery" / f"{language['id']}.json"
        if dest.exists():
            continue
        query = f'language:"{language["github_language"]}" fork:false archived:false size:50..50000 stars:>=2'
        payload = api_get("https://api.github.com/search/repositories", params={"q": query, "sort": "stars", "per_page": 100})
        repos = []
        for item in payload.get("items", []):
            license_id = (item.get("license") or {}).get("spdx_id")
            if license_id in LICENSES:
                repos.append({"repo": item["full_name"], "url": item["html_url"], "branch": item["default_branch"],
                              "license": license_id, "size_kib": item["size"], "stars": item["stargazers_count"],
                              "language": language["id"], "github_language": item.get("language")})
        write_json(dest, {"query": query, "retrieved_at": now(), "total_search_matches": payload.get("total_count"), "repositories": repos})
        event("discovered", language=language["id"], repositories=len(repos))
        time.sleep(7)


def _repository(spec, language, cfg):
    """Download a bounded archive at an exact commit, save eligible candidates."""
    import requests
    key = spec["repo"].replace("/", "--")
    dest = CACHE / "repositories" / f"{key}.json.gz"
    if dest.exists():
        return read_json(dest)
    commit = spec.get("commit")
    if commit is None:
        reference = "refs/heads/" + spec["branch"]
        output = subprocess.check_output(["git", "ls-remote", "--refs", spec["url"]+".git", reference],
                                         timeout=90, text=True, env=os.environ | {"GIT_TERMINAL_PROMPT": "0"})
        commit = output.split()[0]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Repository did not resolve to a full commit")
    archive = CACHE / "archives" / f"{key}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    exclusions = Counter()
    candidates = defaultdict(list)
    licenses = []
    total_bytes = 0
    try:
        with requests.get(f"https://codeload.github.com/{spec['repo']}/tar.gz/{commit}", stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with archive.open("wb") as target:
                for chunk in response.iter_content(1 << 20):
                    total_bytes += len(chunk)
                    if total_bytes > 64 << 20:
                        raise ValueError("Archive exceeds 64 MiB download cap")
                    target.write(chunk)
        archive_hash = file_hash(archive)
        with tarfile.open(archive, "r:gz") as tar:
            expanded = 0
            for member in tar:
                expanded += max(0, member.size)
                if expanded > 1024 << 20:
                    raise ValueError("Archive exceeds 1 GiB expanded cap")
                if not member.isfile():
                    continue
                path = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                if ".." in path.parts or path.is_absolute():
                    raise ValueError("Unsafe archive member")
                if len(path.parts) <= 2 and re.match(r"^(licen[cs]e|copying|notice|copyright)([.-]|$)", path.name, re.I) and member.size <= 262144:
                    raw = tar.extractfile(member).read()
                    licenses.append({"path": str(path), "sha256": sha256(raw), "text": raw.decode("utf-8", "replace")})
                if path.suffix not in language["extensions"]:
                    continue
                if any(part in EXCLUDED or part.startswith(".") for part in path.parts):
                    exclusions["excluded_directory"] += 1
                    continue
                if not cfg["min_file_bytes"] <= member.size <= cfg["max_file_bytes"]:
                    exclusions["file_size"] += 1
                    continue
                raw = tar.extractfile(member).read()
                try:
                    text = raw.decode("utf-8", "strict")
                except UnicodeDecodeError:
                    exclusions["not_utf8"] += 1
                    continue
                if "\x00" in text or not text.strip():
                    exclusions["binary_or_empty"] += 1
                    continue
                if re.search(r"(?i)(automatically generated|auto-generated|autogenerated|DO NOT EDIT|@generated)", text[:2000]):
                    exclusions["generated_header"] += 1
                    continue
                candidates[band(len(raw))].append({"path": str(path), "text": text, "sha256": sha256(raw), "utf8_bytes": len(raw),
                    "size_band": band(len(raw)), "normalized_sha256": sha256(re.sub(r"\s+", " ", text).encode()),
                    "source_id": spec["repo"], "language": language["id"], "domain": "real repository source", "dataset": cfg["protocol"]})
        if not licenses:
            raise ValueError("No root license/notice text found; repository not sampled")
        # Keep bounded candidates per band; choose randomly by stable path hash,
        # not directory order or file content. Rare large-file bands remain visible.
        rows = []
        for group in candidates.values():
            group.sort(key=lambda row: stable(spec["repo"]+":"+row["path"]))
            rows.extend(group[:120])
        value = {**spec, "commit": commit, "archive_sha256": archive_hash, "archive_bytes": total_bytes,
                 "retrieved_at": now(), "licenses": licenses, "exclusions": dict(exclusions),
                 "eligible_counts": {key: len(group) for key, group in candidates.items()}, "files": rows}
        write_json(dest, value, compact=True)
        return value
    finally:
        archive.unlink(missing_ok=True)


def collect():
    cfg = read_json(CONFIG)
    all_selected = {}
    manifest_repos = []
    assigned_repos = set()
    seen_hashes = set()
    seen_normalized = set()
    exclusions = Counter()
    stats = {}
    CORPUS.mkdir(parents=True, exist_ok=True)
    # A corpus is rebuilt from pinned cached repository snapshots on each resume.
    # The configuration is immutable once benchmark result hashes are recorded.
    for language in cfg["languages"]:
        lid = language["id"]
        discovery_path = CACHE / "discovery" / f"{lid}.json"
        waiting_since = time.monotonic()
        while not discovery_path.exists():
            if time.monotonic()-waiting_since > 900:
                raise FileNotFoundError("Run discovery first: "+str(discovery_path))
            time.sleep(3)
        source = read_json(discovery_path)
        repos = [r for r in source["repositories"] if r["repo"] not in assigned_repos]
        picked = []
        counts = Counter()
        split_count = Counter()
        failures = []
        for offset in range(0, len(repos), 4):
            if len(picked) >= cfg["files_per_language"]:
                break
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {pool.submit(_repository, spec, language, cfg): spec for spec in repos[offset:offset+4]}
                outcomes = {}
                for future in as_completed(futures):
                    spec = futures[future]
                    try:
                        outcomes[spec["repo"]] = future.result()
                    except Exception as exc:
                        failures.append({"repo": spec["repo"], "error": f"{type(exc).__name__}: {exc}"})
                        event("repository_failed", language=lid, repo=spec["repo"], error=str(exc)[:180])
            for spec in repos[offset:offset+4]:
                snapshot = outcomes.get(spec["repo"])
                if not snapshot or len(picked) >= cfg["files_per_language"]:
                    continue
                split = "development" if split_count["development"] < cfg["development_files_per_language"] else "evaluation"
                remaining = cfg["development_files_per_language"]-split_count[split] if split == "development" else cfg["files_per_language"]-len(picked)
                candidates = list(snapshot["files"])
                if lid == "bash":
                    # GitHub's Shell category also contains POSIX sh and zsh.
                    # Require positive Bash evidence rather than relabel them.
                    candidates = [r for r in candidates if r["path"].endswith(".bash")
                        or re.search(r"^#![^\n]*\bbash\b", r["text"])
                        or re.search(r"(?m)^#\s*shellcheck\s+shell=bash\b", r["text"][:500])]
                # Fill lower-count byte bands first, retaining deterministic ties.
                candidates.sort(key=lambda r: stable(spec["repo"]+":"+r["path"]))
                groups = {key: [r for r in candidates if r["size_band"] == key] for key in ["le1k", "1k-8k", "8k-64k", "64k-1m"]}
                chosen = []
                while len(chosen) < min(cfg["max_files_per_repository"], remaining) and any(groups.values()):
                    available = [key for key, rows in groups.items() if rows]
                    target = min(available, key=lambda k: (counts[k] >= cfg["min_band_files"], counts[k], k))
                    row = groups[target].pop(0)
                    if row["sha256"] in seen_hashes or row["normalized_sha256"] in seen_normalized:
                        exclusions["exact_or_whitespace_duplicate"] += 1
                        continue
                    seen_hashes.add(row["sha256"])
                    seen_normalized.add(row["normalized_sha256"])
                    row = row | {"id": f"{lid}:{stable(spec['repo']+':'+snapshot['commit']+':'+row['path'])[:24]}",
                        "split": split, "repository": spec["repo"], "commit": snapshot["commit"], "license": spec["license"],
                        "source_url": f"https://github.com/{spec['repo']}/blob/{snapshot['commit']}/{quote(row['path'], safe='/')}",
                        "language_evidence": {"method": "repository primary language plus extension", "github_language": spec["github_language"], "extension": Path(row["path"]).suffix},
                        "dialect_status": "not independently verified"}
                    chosen.append(row)
                    counts[target] += 1
                    split_count[split] += 1
                if chosen:
                    assigned_repos.add(spec["repo"])
                    picked.extend(chosen)
                    manifest_repos.append({key: value for key, value in snapshot.items() if key != "files"} | {"split": split, "selected_files": len(chosen)})
            event("collection_progress", language=lid, files=len(picked), target=cfg["files_per_language"], repositories=len({r["repository"] for r in picked}))
        all_selected[lid] = picked
        write_jsonl(CORPUS / f"{lid}.jsonl.gz", picked)
        stats[lid] = {"files": len(picked), "bytes": sum(r["utf8_bytes"] for r in picked), "bands": dict(counts),
            "splits": dict(split_count), "repositories": len({r["repository"] for r in picked}), "collection_failures": failures,
            "shortfall": max(0, cfg["files_per_language"]-len(picked)),
            "size_band_shortfalls": {key: max(0, cfg["min_band_files"]-counts[key]) for key in ["le1k", "1k-8k", "8k-64k", "64k-1m"]}}
        write_json(CACHE / "collection-progress.json", {"at": now(), "languages": stats})
    partitions = [{"language": lid, "path": str((CORPUS/f"{lid}.jsonl.gz").relative_to(ROOT)),
        "sha256": file_hash(CORPUS/f"{lid}.jsonl.gz"), "documents": len(rows), "utf8_bytes": sum(r["utf8_bytes"] for r in rows)} for lid, rows in all_selected.items()]
    write_jsonl(CORPUS / "file-index.jsonl.gz", ({k: v for k, v in row.items() if k != "text"} for rows in all_selected.values() for row in rows))
    manifest = {"protocol": cfg["protocol"], "created_at": now(), "config_sha256": file_hash(CONFIG), "seed": SEED,
        "file_index_sha256": file_hash(CORPUS / "file-index.jsonl.gz"),
        "files": sum(len(rows) for rows in all_selected.values()), "languages": stats, "partitions": partitions,
        "repositories": manifest_repos, "exclusions": dict(exclusions),
        "limitations": ["Curated popular GitHub repositories, not a random sample of developer traffic.",
            "Only valid UTF-8 files of 80 bytes to 1 MiB are sampled; other encodings, binary and oversized files are excluded.",
            "Repository license metadata and notices retained; file-specific licensing exceptions require review before redistribution.",
            "Language attribution uses repository metadata and suffixes; dialects not independently validated.",
            "Exact and whitespace-equivalent duplicates removed globally; near-duplicate family audit pending.",
            "Repository names are provisional families; related non-fork repositories can remain.",
            "Editing, context/tool and adversarial companion suites are not part of this source-file corpus."],
        "quota_complete": all(s["shortfall"] == 0 for s in stats.values()),
        "acceptance_status": "provisional; full NASA-inspired plan acceptance not established"}
    write_json(CORPUS / "manifest.json", manifest)
    write_corpus_readme(manifest)
    event("corpus_written", files=manifest["files"], quota_complete=manifest["quota_complete"], manifest=str(CORPUS/"manifest.json"))


def write_corpus_readme(manifest):
    cfg = read_json(CONFIG)
    rows = ["# Real-source code corpus v0.1", "", "Locally collected research corpus. Upstream source licenses apply; this is not an Apache-2.0 relicensing of the collected files.", "",
        f"{manifest['files']:,} files across {len(manifest['languages'])} languages and {len(manifest['repositories'])} repositories.", "",
        "| Language | Files | Development | Evaluation | Repositories | Original bytes |", "|---|---:|---:|---:|---:|---:|"]
    for language in cfg["languages"]:
        s = manifest["languages"][language["id"]]
        rows.append(f"| {language['name']} | {s['files']:,} | {s['splits'].get('development',0):,} | {s['splits'].get('evaluation',0):,} | {s['repositories']:,} | {s['bytes']:,} |")
    rows.extend(["", "## Integrity and reuse", "", "Each compressed language partition preserves original valid UTF-8 bytes, including BOMs and line endings. The manifest records exact commits, archive hashes, repository license metadata, collected notices, source-file hashes and exclusions. The text-free file index enables replay at frozen commits. Raw text partitions and per-file result checkpoints are excluded from ordinary Git commits.", "",
        "From the repository root, restore the frozen corpus:", "", "```sh", "uv run --frozen python -m tokenizer_lab.codebench restore", "uv run --frozen python -m tokenizer_lab.codebench audit", "```", "",
        "With all registered tokenizer assets already cached, evaluate every profile:", "", "```sh", "uv run --frozen python -m tokenizer_lab.codebench run", "```", "",
        "The run writes results/code-realistic-v0.1/REPORT.md, summary.csv and per-tokenizer JSON aggregates. Correctness workers block network connections. End-to-end collection times are not isolated tokenizer speed rankings.", "", "## Limitations", "", *["- "+text for text in manifest["limitations"]], "",
        "Size-band shortfalls and download exclusions are explicit in manifest.json. This first source-file run does not establish all acceptance criteria in docs/BENCHMARK_V03_PLAN.md."])
    (CORPUS/"README.md").write_text("\n".join(rows)+"\n")


def audit():
    manifest = read_json(CORPUS / "manifest.json")
    if manifest.get("file_index_sha256") and file_hash(CORPUS/"file-index.jsonl.gz") != manifest["file_index_sha256"]:
        raise ValueError("File index changed")
    seen = set()
    seen_ids = set()
    normalized = set()
    families = {}
    counts = Counter()
    for partition in manifest["partitions"]:
        path = ROOT / partition["path"]
        if file_hash(path) != partition["sha256"]:
            raise ValueError("Partition changed: "+str(path))
        n = 0
        for row in read_jsonl(path):
            raw = row["text"].encode("utf-8")
            if len(raw) != row["utf8_bytes"] or sha256(raw) != row["sha256"]:
                raise ValueError("Source byte integrity failed")
            if row["sha256"] in seen or row["normalized_sha256"] in normalized or row["id"] in seen_ids:
                raise ValueError("Duplicate source file")
            if row["normalized_sha256"] != sha256(re.sub(r"\s+", " ", row["text"]).encode()):
                raise ValueError("Whitespace fingerprint mismatch")
            seen.add(row["sha256"])
            seen_ids.add(row["id"])
            normalized.add(row["normalized_sha256"])
            repo = row["repository"]
            if repo in families and families[repo] != row["split"]:
                raise ValueError("Repository crosses splits")
            families[repo] = row["split"]
            counts[repo] += 1
            n += 1
        if n != partition["documents"]:
            raise ValueError("Document count mismatch")
    if counts and max(counts.values()) > read_json(CONFIG)["max_files_per_repository"]:
        raise ValueError("Repository contribution cap exceeded")
    if len(seen) != manifest["files"]:
        raise ValueError("Manifest file total mismatch")
    if manifest["quota_complete"]:
        cfg = read_json(CONFIG)
        if len(manifest["partitions"]) != len(cfg["languages"]):
            raise ValueError("Language coverage mismatch")
        for language in cfg["languages"]:
            stats = manifest["languages"][language["id"]]
            if stats["files"] != cfg["files_per_language"] or stats["splits"].get("development") != cfg["development_files_per_language"]:
                raise ValueError("Language quota or split count mismatch")
    result = {"files": len(seen), "repositories": len(families), "languages": len(manifest["partitions"]),
        "manifest_sha256": file_hash(CORPUS/"manifest.json"), "integrity_pass": True,
        "quota_complete": manifest["quota_complete"], "at": now()}
    write_json(CORPUS / "audit.json", result)
    event("audit", **result)
    return result


def forbid_network():
    import socket
    def blocked(*args, **kwargs):
        raise RuntimeError("Network disabled inside benchmark worker")
    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked


def restore():
    """Rebuild selected bytes from the frozen file index, not current branches."""
    import requests
    manifest = read_json(CORPUS/"manifest.json")
    index = CORPUS/"file-index.jsonl.gz"
    if file_hash(index) != manifest["file_index_sha256"]:
        raise ValueError("File index changed")
    groups = defaultdict(list)
    for row in read_jsonl(index):
        groups[row["repository"]].append(row)
    restored = defaultdict(list)
    for repo, records in groups.items():
        commit = records[0]["commit"]
        wanted = {r["path"]: r for r in records}
        with requests.get(f"https://codeload.github.com/{repo}/tar.gz/{commit}", stream=True, timeout=(20, 90)) as response:
            response.raise_for_status()
            with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
                for member in archive:
                    path = str(PurePosixPath(*PurePosixPath(member.name).parts[1:]))
                    if not member.isfile() or path not in wanted:
                        continue
                    row = wanted.pop(path)
                    if member.size != row["utf8_bytes"]:
                        raise ValueError("Restored file size mismatch")
                    raw = archive.extractfile(member).read()
                    if sha256(raw) != row["sha256"]:
                        raise ValueError("Restored source hash mismatch")
                    restored[row["language"]].append(row | {"text": raw.decode("utf-8", "strict")})
        if wanted:
            raise ValueError("Missing pinned files: "+repo)
        event("repository_restored", repo=repo, files=len(records))
    # Original index ordering, preserved independently from archive member order.
    ordered = defaultdict(list)
    by_id = {r["id"]: r for rows in restored.values() for r in rows}
    for row in read_jsonl(index):
        ordered[row["language"]].append(by_id[row["id"]])
    # JSON object order also affects gzip hashes; write rows in the original
    # source record key order, using the known creation order of the collector.
    for lid, rows in ordered.items():
        canonical = []
        for row in rows:
            value = {}
            for key in ("path", "text", "sha256", "utf8_bytes", "size_band", "normalized_sha256", "source_id", "language", "domain", "dataset", "id", "split", "repository", "commit", "license", "source_url", "language_evidence", "dialect_status"):
                value[key] = row[key]
            canonical.append(value)
        write_jsonl(CORPUS/f"{lid}.jsonl.gz", canonical)
    audit()


def worker(tokenizer):
    import importlib.metadata
    import platform
    import psutil
    from .adapters import load_adapter
    from .metrics import aggregate, measure
    forbid_network()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    manifest_hash = file_hash(CORPUS/"manifest.json")
    manifest = read_json(CORPUS/"manifest.json")
    output = RESULTS / "checkpoints" / tokenizer
    output.mkdir(parents=True, exist_ok=True)
    run_fingerprint = fingerprint()
    code_hash = run_fingerprint["code"]
    baseline = psutil.Process().memory_info().rss
    start = time.perf_counter()
    adapter = load_adapter(tokenizer, download=False)
    loaded = time.perf_counter()-start
    totals = []
    for part in manifest["partitions"]:
        checkpoint = output / f"{part['language']}.jsonl.gz"
        metadata = checkpoint.with_suffix(".meta.json")
        key = {"manifest_sha256": manifest_hash, "partition_sha256": part["sha256"], "asset_sha256": adapter.asset_sha,
            "configuration_sha256": sha256(json.dumps(adapter.spec, sort_keys=True).encode()), "run_fingerprint": run_fingerprint}
        old = read_json(metadata) if metadata.exists() else {}
        if checkpoint.exists() and all(old.get(k) == v for k, v in key.items()) and old.get("sha256") == file_hash(checkpoint):
            rows = list(read_jsonl(checkpoint))
            elapsed = old["measurement_seconds"]
        else:
            path = ROOT/part["path"]
            if file_hash(path) != part["sha256"]:
                raise ValueError("Corpus changed")
            docs = list(read_jsonl(path))
            rows = []
            start = time.perf_counter()
            # Small batches limit memory, especially for long legacy sequences.
            for offset in range(0, len(docs), 8):
                chunk = docs[offset:offset+8]
                measured = measure(adapter, chunk)
                if len(measured) != len(chunk) or [r["id"] for r in measured] != [r["id"] for r in chunk]:
                    raise ValueError("Adapter returned incomplete input coverage")
                for row, doc in zip(measured, chunk):
                    row.update(split=doc["split"], repository=doc["repository"], size_band=doc["size_band"], lines=doc["text"].count("\n")+int(not doc["text"].endswith("\n")))
                rows.extend(measured)
                if offset % 200 == 0:
                    write_json(output/"progress.json", {"language": part["language"], "completed": offset+len(chunk), "total": len(docs), "at": now()})
            elapsed = time.perf_counter()-start
            write_jsonl(checkpoint, rows)
            write_json(metadata, key | {"sha256": file_hash(checkpoint), "documents": len(rows), "measurement_seconds": elapsed})
        if len(rows) != part["documents"]:
            raise ValueError("Checkpoint has missing inputs")
        groups = []
        for split in ("development", "evaluation"):
            for size_band in (None, "le1k", "1k-8k", "8k-64k", "64k-1m"):
                selected = [r for r in rows if r["split"] == split and (size_band is None or r["size_band"] == size_band)]
                for group in aggregate(selected):
                    groups.append(group | {"split": split, "size_band": size_band or "all"})
        totals.extend(groups)
        event("partition_complete", tokenizer=tokenizer, language=part["language"], documents=len(rows), seconds=round(elapsed, 3))
        write_json(RESULTS/f"{tokenizer}.json", {"tokenizer": tokenizer, "status": "running", "manifest_sha256": manifest_hash, "benchmarks": totals})
    write_json(RESULTS/f"{tokenizer}.json", {"tokenizer": tokenizer, "status": "completed", "manifest_sha256": manifest_hash,
        "asset_sha256": adapter.asset_sha, "measurement_code": code_hash, "load_seconds": loaded,
        "configuration_sha256": sha256(json.dumps(adapter.spec, sort_keys=True).encode()), "run_fingerprint": run_fingerprint,
        "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "packages": {name: importlib.metadata.version(name) for name in ["tokenizers", "tiktoken", "sentencepiece", "numpy"]},
            "uv_lock_sha256": file_hash(ROOT/"uv.lock"), "tokenizers_parallelism": False, "correctness_workers": CORRECTNESS_WORKERS},
        "baseline_rss_bytes": baseline, "ending_rss_bytes": psutil.Process().memory_info().rss,
        "benchmarks": totals, "at": now(), "network_downloads": 0,
        "timing_scope": "End-to-end correctness collection; not isolated tokenizer throughput or latency benchmark."})


def run_all():
    from .adapters import registry
    audit()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ids = [s["id"] for s in registry()]
    manifest_hash = file_hash(CORPUS/"manifest.json")
    if not read_json(CORPUS/"manifest.json")["quota_complete"]:
        raise ValueError("Complete all 32 language quotas before the full run; inspect collection-progress.json")
    run_fingerprint = fingerprint()
    random.Random(SEED).shuffle(ids)
    states = {}
    pending = []
    for identifier in ids:
        result_path = RESULTS/f"{identifier}.json"
        if result_path.exists():
            value = read_json(result_path)
            if value.get("status") == "completed" and value.get("manifest_sha256") == manifest_hash and value.get("run_fingerprint") == run_fingerprint:
                states[identifier] = {"status": "completed", "resumed": True}
                continue
        pending.append(identifier)

    def evaluate(identifier):
        event("tokenizer_start", tokenizer=identifier, total=len(ids))
        log = RESULTS / "checkpoints" / f"{identifier}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        with log.open("a") as stream:
            process = subprocess.Popen([sys.executable, "-m", "tokenizer_lab.codebench", "worker", "--tokenizer", identifier],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, env=os.environ | {"TOKENIZER_LAB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
            try:
                code = process.wait(timeout=7200)
                outcome = {"status": "completed" if code == 0 else "failed", "exit_code": code, "seconds": round(time.monotonic()-start, 2)}
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                outcome = {"status": "timeout", "seconds": round(time.monotonic()-start, 2)}
        return outcome

    # Bounded correctness workers limit wall time without the memory cost of one
    # worker per CPU. These concurrent timings are not speed-ranking evidence.
    with ThreadPoolExecutor(max_workers=CORRECTNESS_WORKERS) as pool:
        futures = {pool.submit(evaluate, identifier): identifier for identifier in pending}
        for future in as_completed(futures):
            identifier = futures[future]
            try:
                states[identifier] = future.result()
            except Exception as exc:
                states[identifier] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            write_json(RESULTS/"run-status.json", {"at": now(), "manifest_sha256": manifest_hash, "registered_profiles": len(ids), "profiles": states})
            event("tokenizer_finished", tokenizer=identifier, **states[identifier])
    report()


def report():
    import csv
    from .adapters import registry
    manifest = read_json(CORPUS/"manifest.json")
    digest = file_hash(CORPUS/"manifest.json")
    current_fingerprint = fingerprint()
    profiles = []
    language_rows = []
    for spec in registry():
        path = RESULTS/f"{spec['id']}.json"
        if not path.exists():
            profiles.append({"tokenizer": spec["id"], "status": "not_completed"})
            continue
        value = read_json(path)
        if value.get("manifest_sha256") != digest or (value.get("status") == "completed" and value.get("run_fingerprint") != current_fingerprint):
            profiles.append({"tokenizer": spec["id"], "status": "stale"})
            continue
        if value.get("status") == "completed":
            language_rows.extend(r for r in value["benchmarks"] if r["size_band"] == "all")
        for split in ("development", "evaluation"):
            rows = [r for r in value["benchmarks"] if r["split"] == split and r["size_band"] == "all"]
            totals = {key: sum(r[key] for r in rows) for key in ["documents", "utf8_bytes", "tokens", "exact", "canonical_only", "mismatches", "errors", "unknown_emissions"]}
            totals.update(tokenizer=spec["id"], split=split, status=value["status"], languages=len(rows))
            totals["bytes_per_token"] = totals["utf8_bytes"]/totals["tokens"] if totals["tokens"] and not totals["errors"] else None
            profiles.append(totals)
    write_json(RESULTS/"summary.json", {"manifest_sha256": digest, "corpus_files": manifest["files"], "quota_complete": manifest["quota_complete"],
        "profiles": profiles, "limitations": manifest["limitations"], "at": now()})
    columns = ["tokenizer", "status", "split", "languages", "documents", "utf8_bytes", "tokens", "bytes_per_token", "exact", "canonical_only", "mismatches", "errors", "unknown_emissions"]
    with (RESULTS/"summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in profiles)
    language_columns = ["tokenizer", "language", "split", "documents", "utf8_bytes", "tokens", "bytes_per_token", "exact", "canonical_only", "mismatches", "errors", "unknown_emissions", "strict_lossless", "token_p90", "token_max"]
    with (RESULTS/"by-language.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=language_columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in language_columns} for row in language_rows)
    evaluated = [r for r in profiles if r.get("split") == "evaluation" and r["status"] == "completed"]
    lines = ["# Real-source code tokenizer results", "", f"Corpus: {manifest['files']:,} files; {len(manifest['partitions'])} languages. Completed profiles: {len(evaluated)}/{len(registry())}.", "",
             "Held-out evaluation only. Compression values do not establish code-generation quality. Ratios with fidelity failures are diagnostic.", "",
             "| Tokenizer | Files | Tokens | Bytes/token | Exact | Canonical-only | Other mismatch | Errors |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(evaluated, key=lambda r: -(r["bytes_per_token"] or 0)):
        ratio = f"{row['bytes_per_token']:.4f}" if row["bytes_per_token"] is not None else "unavailable"
        lines.append(f"| {row['tokenizer']} | {row['documents']:,} | {row['tokens']:,} | {ratio} | {row['exact']:,} | {row['canonical_only']:,} | {row['mismatches']:,} | {row['errors']:,} |")
    lines.extend(["", "## Per-language observed token counts", "", "Minimum measured held-out token count among profiles reconstructing every file exactly. These are descriptive results for this frozen corpus, not significance tests or model-quality rankings. Equal-token profiles are ties; complete per-profile language results are in by-language.csv.", "",
        "| Language | Files | Lowest-count exact profile(s) | Tokens | Bytes/token |", "|---|---:|---|---:|---:|"])
    for language in read_json(CONFIG)["languages"]:
        group = [r for r in language_rows if r["language"] == language["id"] and r["split"] == "evaluation" and r["strict_lossless"]]
        if not group:
            lines.append(f"| {language['name']} | — | No completed exact profile | — | — |")
            continue
        minimum = min(r["tokens"] for r in group)
        winners = sorted(r["tokenizer"] for r in group if r["tokens"] == minimum)
        sample = next(r for r in group if r["tokens"] == minimum)
        lines.append(f"| {language['name']} | {sample['documents']:,} | {', '.join(winners)} | {minimum:,} | {sample['bytes_per_token']:.4f} |")
    lines.extend(["", "## Collection limitations", "", *["- "+v for v in manifest["limitations"]]])
    (RESULTS/"REPORT.md").write_text("\n".join(lines)+"\n")
    event("report_written", completed_profiles=len(evaluated), path=str(RESULTS/"REPORT.md"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["discover", "collect", "restore", "audit", "run", "worker", "report"])
    parser.add_argument("--tokenizer")
    args = parser.parse_args()
    if args.command == "worker":
        if not args.tokenizer:
            parser.error("--tokenizer is required")
        worker(args.tokenizer)
    else:
        {"discover": discover, "collect": collect, "restore": restore, "audit": audit, "run": run_all, "report": report}[args.command]()


if __name__ == "__main__":
    main()
