"""Build relocatable, self-contained offline kits; no installer runs at launch.

The build machine needs uv and network access once. Each kit contains its own
hash-pinned CPython, wheel-installed native dependencies, selected input assets,
compressed benchmark partitions, and the browser UI. No model weights are used.
"""
from __future__ import annotations

import argparse
from email.parser import Parser
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import urllib.request
import urllib.parse
import zipfile
from pathlib import Path, PurePosixPath

from .io import ROOT, file_hash, read_json, read_jsonl, write_json


def host_target():
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x86_64" if machine in ("amd64", "x86_64") else machine
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system(), platform.system().lower())
    return f"{system}-{arch}"


def _path(root, relative):
    """Only root-relative paths are permitted in distribution manifests."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe distribution path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Distribution path leaves project: {relative}")
    return resolved


def _copy(root, kit, relative, *, sha256=None):
    source = _path(root, relative)
    if not source.is_file():
        raise FileNotFoundError(f"Offline input is missing: {relative}")
    actual = file_hash(source)
    if sha256 and actual != sha256:
        raise ValueError(f"Offline input hash mismatch: {relative}")
    dest = kit / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def _selected(partition, languages, datasets):
    return (languages is None or partition.get("language") in languages) and (datasets is None or partition.get("dataset") in datasets)


def _preflight(root, tokenizers, languages, datasets):
    registry_path = root / "configs/registry.json"
    registry = read_json(registry_path)
    entries = registry.get("tokenizers", [])
    ids = {p["id"] for p in entries}
    selected_ids = ids if tokenizers is None else set(tokenizers)
    unknown = selected_ids - ids
    if unknown:
        raise ValueError(f"Unknown tokenizer IDs: {', '.join(sorted(unknown))}")
    entries = [p for p in entries if p["id"] in selected_ids]
    catalog_path = root / "benchmarks/catalog.json"
    if not catalog_path.is_file():
        if languages == [] or datasets == []:
            catalog = {"schema_version": "1.0", "partitions": []}
        else:
            raise FileNotFoundError("Prepare benchmark partitions first: benchmarks/catalog.json is missing")
    else:
        catalog = read_json(catalog_path)
    partitions = catalog.get("partitions", [])
    available_languages = {p["language"] for p in partitions}
    available_datasets = {p["dataset"] for p in partitions}
    for values, available, label in ((languages, available_languages, "language"), (datasets, available_datasets, "dataset")):
        unknown = set(values or ()) - available
        if unknown:
            raise ValueError(f"Unknown {label} IDs: {', '.join(sorted(unknown))}")
    partitions = [p for p in partitions if _selected(p, None if languages is None else set(languages), None if datasets is None else set(datasets))]
    files = []
    for spec in entries:
        if spec["kind"] == "tiktoken":
            relative = spec.get("offline_file", f"artifacts/tiktoken/{spec['id']}.json")
            # The adapter verifies the canonical rank hash after loading this export.
            files.append((relative, spec.get("offline_sha256")))
        else:
            if spec["kind"] == "local":
                relative = spec["file"]
            else:
                filename = Path(spec.get("file", "tokenizer.json")).name
                relative = f".cache/assets/{spec['id']}/{filename}"
            files.append((relative, spec.get("sha256")))
            asset_dir = root / ".cache/assets" / spec["id"]
            # Declared reference .py files are data, never imported or executed.
            # Catalog readiness requires these companions, including their hashes.
            for companion in spec.get("companions", []):
                name=companion['file']
                path=(asset_dir/name).relative_to(root).as_posix()
                if path!=relative:files.append((path,companion['sha256']))
            for name in ['provenance.json','ported-tokenizer.json']:
                path=asset_dir/name
                if path.is_file() and not any(path.relative_to(root).as_posix()==p for p,_ in files):
                    files.append((path.relative_to(root).as_posix(),None))
    for partition in partitions:
        if not partition.get("sha256"):
            raise ValueError(f"Partition has no integrity hash: {partition.get('path')}")
        files.append((partition["path"], partition["sha256"]))
    missing = [relative for relative, _ in files if not _path(root, relative).is_file()]
    if missing:
        raise FileNotFoundError("Offline inputs are missing: " + ", ".join(missing))
    for relative, expected in files:
        if expected and file_hash(_path(root, relative)) != expected:
            raise ValueError(f"Offline input hash mismatch: {relative}")
    return registry, entries, catalog, partitions, files


def _runtime(root, target):
    config = read_json(root / "configs/runtime-assets.json")
    if target not in config["targets"]:
        raise ValueError(f"Unsupported target {target}; choose {', '.join(config['targets'])}")
    spec = config["targets"][target]
    cache = root / ".cache/offline/runtimes" / spec["filename"]
    if cache.is_file() and file_hash(cache) == spec["sha256"]:
        return config, spec, cache
    if os.environ.get("TOKENIZER_LAB_OFFLINE") == "1":
        raise FileNotFoundError(f"Pinned runtime is not cached: {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    partial = cache.with_suffix(".part")
    url = spec.get("url") or f"https://github.com/astral-sh/python-build-standalone/releases/download/{config['release']}/{urllib.parse.quote(spec['filename'])}"
    request = urllib.request.Request(url, headers={"User-Agent": "Tokenizer-Lab-offline-builder/0.2"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, 1024 * 1024)
        if partial.stat().st_size != spec["bytes"] or file_hash(partial) != spec["sha256"]:
            raise ValueError(f"Runtime integrity verification failed for {target}")
        partial.replace(cache)
    finally:
        partial.unlink(missing_ok=True)
    return config, spec, cache


def _extract_runtime(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != "python" or ".." in parts or member.isdev():
                raise ValueError(f"Unexpected runtime archive member: {member.name}")
        # Linux terminfo includes names differing only by case and symlinks
        # that collide on default macOS filesystems during cross-packaging.
        # The HTTP/benchmark application does not use a terminal database.
        members=[m for m in bundle.getmembers() if not m.name.startswith("python/share/terminfo/") and m.name!="python/share/terminfo"]
        bundle.extractall(destination,members=members,filter="data")
    (destination / "python").rename(destination / "runtime")


def _uv(root):
    executable = shutil.which("uv")
    if not executable:
        raise RuntimeError("Building offline kits requires uv on the build machine; launching a kit does not require uv")
    return executable


def _dependencies(root, kit, spec):
    uv = _uv(root)
    requirements = kit / "requirements.lock.txt"
    cache = Path(os.environ.get("UV_CACHE_DIR", str(root / ".cache/offline/uv")))
    subprocess.run([uv, "export", "--frozen", "--no-dev", "--no-editable", "--no-emit-project", "--no-header", "--format", "requirements.txt", "--cache-dir",str(cache),"--output-file", str(requirements)], cwd=root, check=True, stdout=subprocess.DEVNULL)
    command = [uv, "pip", "install", "--target", str(kit / "vendor"), "--python-version", "3.12", "--python-platform", spec["python_platform"], "--only-binary", ":all:", "--require-hashes", "--link-mode", "copy", "--cache-dir", str(cache), "--requirements", str(requirements), "--no-python-downloads"]
    if os.environ.get("TOKENIZER_LAB_OFFLINE") == "1":
        command.append("--offline")
    subprocess.run(command, cwd=root, check=True)
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    wheels = []
    for metadata in sorted((kit / "vendor").glob("*.dist-info/METADATA")):
        package = Parser().parsestr(metadata.read_text(encoding="utf-8"))
        tags = {line[5:] for line in (metadata.parent / "WHEEL").read_text(encoding="utf-8").splitlines() if line.startswith("Tag: ")}
        locked = next((p for p in lock["package"] if p["name"].lower().replace("_", "-") == package["Name"].lower().replace("_", "-") and p["version"] == package["Version"]), None)
        if not locked:
            raise ValueError(f"Installed package is not locked: {package['Name']}")
        candidates = [w for w in locked.get("wheels", []) if any(urllib.parse.unquote(w["url"]).endswith(f"-{tag}.whl") for tag in tags)]
        if len(candidates) != 1:
            # Compressed tag sets in wheel filenames expand into several WHEEL
            # records. Match each dot-separated tag component explicitly.
            candidates = []
            for wheel in locked.get("wheels", []):
                filename = urllib.parse.unquote(wheel["url"].rsplit("/", 1)[-1])
                py, abi, plat = filename[:-4].rsplit("-", 3)[-3:]
                expanded = {f"{p}-{a}-{t}" for p in py.split(".") for a in abi.split(".") for t in plat.split(".")}
                if tags & expanded:
                    candidates.append(wheel)
        if len(candidates) != 1:
            raise ValueError(f"Could not identify one locked native wheel for {package['Name']}: {tags}")
        wheels.append({"name": package["Name"], "version": package["Version"], **{key: candidates[0][key] for key in ("url", "hash", "size")}})
    # A sitecustomize in PYTHONPATH is loaded before the CLI. This prevents an
    # accidental dependency download even when a cache or configuration is wrong.
    (kit / "vendor/sitecustomize.py").write_text('''"""Offline kits permit only local network connections."""
import ipaddress, os, socket
if os.environ.get("TOKENIZER_LAB_OFFLINE") == "1":
    _connect = socket.socket.connect
    _connect_ex = socket.socket.connect_ex
    _getaddrinfo = socket.getaddrinfo
    def _local(host):
        if host in ("localhost", "localhost.", None, ""):
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except (ValueError, TypeError):
            return False
    def _check(address):
        if isinstance(address, tuple) and not _local(address[0]):
            raise OSError("Tokenizer Lab offline mode blocks external network connections")
    def _guarded_connect(self, address):
        _check(address)
        return _connect(self, address)
    def _guarded_connect_ex(self, address):
        _check(address)
        return _connect_ex(self, address)
    def _guarded_dns(host, *args, **kwargs):
        if not _local(host):
            raise OSError("Tokenizer Lab offline mode blocks external DNS")
        return _getaddrinfo(host, *args, **kwargs)
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.getaddrinfo = _guarded_dns
''', encoding="utf-8")
    return wheels


def _copy_site(root, kit, selected_ids, language_ids, dataset_ids):
    def ignore(directory, names):
        relative = Path(directory).relative_to(root / "site")
        excluded = {"__pycache__", ".DS_Store"} & set(names)
        if relative.parts and relative.parts[0] == "data":
            saved_aggregates = {"study.json.gz", "measurements.csv.gz", "source-by-language.csv.gz", "scenarios-by-language.csv.gz"} if relative.as_posix() == "data/code-study" else {"hub.json.gz"}
            excluded |= {name for name in names if name.endswith((".gz", ".zip", ".parquet", ".jsonl", ".tar", ".zst")) and name not in saved_aggregates}
        if str(relative) == "data":
            excluded |= {"vocab", "mappings", "texts", "text", "encodings", "offline", "tokenizers", "gemma-v31-inputs.tar.gz", "observations.jsonl.gz", "offline-index.json", "source-release.json", "release-validation.json"} & set(names)
        return excluded
    shutil.copytree(root / "site", kit / "site", ignore=ignore)
    examples_path = kit / "site/data/examples.json"
    if examples_path.is_file():
        examples = read_json(examples_path)
        examples["examples"] = [item for item in examples.get("examples", []) if item.get("language") in language_ids]
        for item in examples["examples"]:
            if isinstance(item.get("encodings"), dict):
                item["encodings"] = {key: value for key, value in item["encodings"].items() if key in selected_ids}
        write_json(examples_path, examples)
    for name in selected_ids:
        relative = f"site/data/vocab/{name}.json.gz"
        if (root / relative).is_file():
            _copy(root, kit, relative)
    maps = root / "site/data/mappings"
    if maps.is_dir():
        for path in maps.glob("*.json.gz"):
            # Map naming is source--target; older releases used source-to-target.
            stem = path.name[:-8]
            if any(stem in (f"{a}--{b}", f"{a}-to-{b}", f"{a}__{b}") for a in selected_ids for b in selected_ids):
                _copy(root, kit, path.relative_to(root).as_posix())
    index_path = root / "site/data/text-index.json"
    if not index_path.is_file():
        return
    index = read_json(index_path)
    partitions = index.get("partitions", [])
    selected = [p for p in partitions if _selected(p, language_ids, dataset_ids)]
    index["offline_selection"] = {"languages": sorted(language_ids), "datasets": sorted(dataset_ids), "included_partitions": len(selected)}
    write_json(kit / "site/data/text-index.json", index)
    for item in selected:
        relative = item.get("path") or item.get("file")
        if not relative:
            raise ValueError("Text index partition has no compressed path")
        # UI paths may be relative to site/, to site/data/, or project root.
        candidates = [relative, f"site/{relative}", f"site/data/{relative}"]
        source = next((p for p in candidates if _path(root, p).is_file()), None)
        if source is None:
            raise FileNotFoundError(f"Compressed language text is missing: {relative}")
        _copy(root, kit, source, sha256=item.get("sha256"))
        if item.get("examples_path") and (root/"site"/item["examples_path"]).is_file():
            _copy(root,kit,"site/"+item["examples_path"],sha256=item.get("examples_sha256"))


def _deduplicated_text(root, partitions):
    """Use one compressed text copy only after full document equality checks."""
    index_path = root / "site/data/text-index.json"
    if not index_path.is_file():
        return {}
    index = {(p["dataset"], p["language"]): p for p in read_json(index_path).get("partitions", [])}
    replacements = {}
    for partition in partitions:
        alternative = index.get((partition["dataset"], partition["language"]))
        if not alternative:
            continue
        relative = alternative.get("path")
        if not relative:
            continue
        source = _path(root, f"site/{relative}")
        if not source.is_file():
            continue
        if not alternative.get("sha256") or file_hash(source) != alternative["sha256"]:
            raise ValueError(f"Compressed UI text integrity failure: {source}")
        canonical = list(read_jsonl(_path(root, partition["path"])))
        documents = read_json(source).get("documents")
        if documents != canonical:
            raise ValueError(f"Compressed UI text differs from canonical benchmark: {partition['dataset']}/{partition['language']}")
        replacements[partition["path"]] = {"offline_text_path": f"site/{relative}", "offline_text_sha256": alternative["sha256"], "offline_jsonl_omitted": True}
    return replacements


def _write_entrypoints(kit, target):
    # Root shortcuts keep the documented launch action independent of cwd.
    if target.startswith("windows"):
        (kit / "Launch Tokenizer Lab.cmd").write_bytes(b'@echo off\r\ncall "%~dp0scripts\\launch.cmd" %*\r\n')
    else:
        name = "Launch Tokenizer Lab.command" if target.startswith("macos") else "launch-tokenizer-lab.sh"
        path = kit / name
        path.write_text('#!/bin/sh\nset -eu\nLAB_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\nexec /bin/sh "$LAB_ROOT/scripts/launch.sh" "$@"\n', encoding="utf-8")
        path.chmod(0o755)


def _manifest(kit, target, runtime_config, spec, entries, partitions, wheels):
    members = {}
    for path in sorted(kit.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            members[path.relative_to(kit).as_posix()] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    manifest = {
        "schema_version": "1.0", "target": target, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": {"version": runtime_config["python_version"], "release": runtime_config["release"], "archive_sha256": spec["sha256"], "minimum_os": spec["minimum_os"], "executable": spec["python"]},
        "tokenizers": [p["id"] for p in entries], "languages": sorted({p["language"] for p in partitions}),
        "datasets": sorted({p["dataset"] for p in partitions}), "partitions": len(partitions),
        "documents": sum(p.get("documents", 0) for p in partitions), "utf8_bytes": sum(p.get("utf8_bytes", 0) for p in partitions),
        "files": members, "installed_bytes": sum(p["bytes"] for p in members.values()),
        "payload_sizes": {"runtime_archive_bytes": spec["bytes"], "runtime_installed_bytes": sum(p["bytes"] for name, p in members.items() if name.startswith("runtime/")), "wheel_download_bytes": sum(p["size"] for p in wheels), "dependencies_installed_bytes": sum(p["bytes"] for name, p in members.items() if name.startswith("vendor/")), "tokenizer_input_bytes": sum(p["bytes"] for name, p in members.items() if name.startswith((".cache/assets/", "artifacts/tiktoken/", "artifacts/gemma_v31/"))), "compressed_text_bytes": sum(p["bytes"] for name, p in members.items() if name.startswith(("site/data/text/", "benchmarks/partitions/"))), "vocabulary_and_map_bytes": sum(p["bytes"] for name, p in members.items() if name.startswith(("site/data/vocab/", "site/data/mappings/")))},
        "wheels": wheels,
        "validation": {"build_host": host_target(), "native_execution_verified": False, "note": "Cross-platform wheel selection and hash verification do not establish native execution on another OS."},
    }
    write_json(kit / "offline-manifest.json", manifest)
    return manifest


def build_bundle(target, tokenizers=None, languages=None, datasets=None, output=None):
    """Build one platform kit and return its exact path, size, hash and scope.

    Inputs are IDs from configs/registry.json and benchmarks/catalog.json.
    None selects all prepared items; an explicit [] selects none. A starter kit
    with tokenizers=[], languages=[], datasets=[] contains no benchmark payload.
    Selection is strict: unknown IDs, missing input files and corrupt hashes fail.
    Existing archives are never overwritten. Output defaults to artifacts/offline.
    """
    root = ROOT
    runtime_config = read_json(root / "configs/runtime-assets.json")
    if target not in runtime_config["targets"]:
        raise ValueError(f"Unsupported offline target: {target}")
    registry, entries, catalog, partitions, files = _preflight(root, tokenizers, languages, datasets)
    replacements = _deduplicated_text(root, partitions)
    archive_path = Path(output).expanduser().resolve() if output else root / f"artifacts/offline/tokenizer-lab-{target}.zip"
    if archive_path.exists():
        raise FileExistsError(f"Offline archive already exists: {archive_path}")
    if archive_path.suffix != ".zip":
        raise ValueError("Offline archive output must end in .zip")
    runtime_config, spec, runtime_archive = _runtime(root, target)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tokenizer-lab-kit-", dir=archive_path.parent) as work:
        kit = Path(work) / "Tokenizer Lab"
        kit.mkdir()
        _extract_runtime(runtime_archive, kit)
        for directory in ("src", "docs", "licenses", "research", "scripts"):
            if (root / directory).is_dir():
                shutil.copytree(root / directory, kit / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        for name in ("pyproject.toml", "uv.lock", "README.md", "LICENSE", "THIRD_PARTY.md", "CITATION.cff", "CHANGELOG.md", "BENCHMARK.md", "CONTRIBUTING.md", "SECURITY.md"):
            if (root / name).is_file():
                _copy(root, kit, name)
        shutil.copytree(root / "configs", kit / "configs",ignore=shutil.ignore_patterns('custom-tokenizers.json','.DS_Store','*.tmp'))
        selected_ids = {p["id"] for p in entries}
        if "gemma_v31" in selected_ids and (root / "experiments/gemma-v31").is_dir():
            shutil.copytree(root / "experiments/gemma-v31", kit / "experiments/gemma-v31")
        selection = {"tokenizers": sorted(selected_ids), "languages": sorted({p["language"] for p in partitions}), "datasets": sorted({p["dataset"] for p in partitions})}
        # Keep the complete small catalogs for explicit later downloads. Presence
        # of a catalog row never means its tokenizer or language file is bundled.
        write_json(kit / "configs/registry.json", {**registry, "offline_selection": selection})
        kit_catalog = {**catalog, "partitions": [{**p, **replacements.get(p["path"], {})} for p in catalog.get("partitions", [])], "offline_selection": selection}
        write_json(kit / "benchmarks/catalog.json", kit_catalog)
        for relative in ["benchmarks/languages.json","benchmarks/fineweb-v0.1/manifest.json","benchmarks/fineweb-v0.1/source-lock.json","benchmarks/madlad-v0.2/manifest.json"]:
            if (root/relative).is_file():_copy(root,kit,relative)
        for relative, expected in files:
            if relative not in replacements:
                _copy(root, kit, relative, sha256=expected)
        if (root / "benchmarks/unicode").is_dir():
            shutil.copytree(root / "benchmarks/unicode", kit / "benchmarks/unicode")
        _copy_site(root, kit, selected_ids, {p["language"] for p in partitions}, {p["dataset"] for p in partitions})
        wheels = _dependencies(root, kit, spec)
        _write_entrypoints(kit, target)
        manifest = _manifest(kit, target, runtime_config, spec, entries, partitions, wheels)
        partial = archive_path.with_suffix(".zip.part")
        if partial.exists():
            raise FileExistsError(f"Offline archive is already being built: {partial}")
        try:
            with zipfile.ZipFile(partial, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
                for path in sorted(kit.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        # Dereference runtime symlinks so Windows ZIP extraction
                        # produces ordinary usable files; retain executable mode.
                        archive.write(path, Path("Tokenizer Lab") / path.relative_to(kit))
            if archive_path.exists():
                raise FileExistsError(f"Offline archive output appeared during the build: {archive_path}")
            partial.replace(archive_path)
        finally:
            partial.unlink(missing_ok=True)
        result = {"target": target, "archive": str(archive_path), "sha256": file_hash(archive_path), "bytes": archive_path.stat().st_size, "installed_bytes": manifest["installed_bytes"], "payload_sizes": manifest["payload_sizes"], "tokenizers": manifest["tokenizers"], "languages": manifest["languages"], "datasets": manifest["datasets"], "documents": manifest["documents"], "deduplicated_corpus_partitions": len(replacements), "native_execution_verified": False}
        write_json(archive_path.with_suffix(".manifest.json"), {**result, "runtime": manifest["python"]})
        return result


def verify_bundle(root=None):
    """Verify packaged members before an offline benchmark or support handoff."""
    root = Path(root or ROOT).resolve()
    manifest = read_json(root / "offline-manifest.json")
    failures = []
    for relative, expected in manifest["files"].items():
        path = _path(root, relative)
        if not path.is_file():
            failures.append({"path": relative, "error": "missing"})
        elif path.stat().st_size != expected["bytes"] or file_hash(path) != expected["sha256"]:
            failures.append({"path": relative, "error": "hash mismatch"})
    return {"target": manifest["target"], "host": host_target(), "native_target_match": manifest["target"] == host_target(), "files_checked": len(manifest["files"]), "passed": not failures, "failures": failures}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("target")
    for field in ("tokenizers", "languages", "datasets"):
        build.add_argument(f"--{field}", nargs="*", default=[])
    build.add_argument("--all-prepared", action="store_true", help="Explicitly include every prepared tokenizer and corpus partition")
    build.add_argument("--output")
    sub.add_parser("verify")
    args = parser.parse_args(argv)
    if args.command == "verify":
        result = verify_bundle()
    else:
        if args.all_prepared:
            if args.tokenizers or args.languages or args.datasets:
                parser.error("--all-prepared cannot be combined with selections")
            args.tokenizers = args.languages = args.datasets = None
        result = build_bundle(args.target, args.tokenizers, args.languages, args.datasets, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    sys.exit(main())
