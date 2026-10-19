"""Local-only explorer, explicit cache management, and offline benchmark runs.

The HTTP service binds only to loopback. Tokenization and benchmark requests never
fetch missing assets; downloads use separate, explicit endpoints.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import re
from threading import Event, Lock, Thread
from urllib.parse import unquote, urlsplit
from uuid import uuid4
import webbrowser

from .io import ROOT

MAX_BODY_BYTES = 1_000_000
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
SELECTION_KEYS = {"tokenizers", "datasets", "languages", "custom_tokenizers"}


@lru_cache(maxsize=2)
def adapter(identifier):
    from .adapters import load_adapter
    return load_adapter(identifier, download=False)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _selection(payload, *, custom=False, require_all=False):
    """Keep missing/empty selections explicit; neither means 'all'."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    allowed = SELECTION_KEYS if custom else SELECTION_KEYS - {"custom_tokenizers"}
    if unknown := set(payload) - allowed:
        raise ValueError(f"Unknown selection fields: {', '.join(sorted(unknown))}")
    result = {}
    for key in ("tokenizers", "datasets", "languages"):
        value = payload.get(key, [])
        limit = 5000 if key == "languages" else 2000
        if not isinstance(value, list) or len(value) > limit:
            raise ValueError(f"{key} must be an array of at most {limit} IDs")
        if any(not isinstance(item, str) or not item.strip() or len(item) > 200
               or any(ord(char) < 32 for char in item) for item in value):
            raise ValueError(f"{key} must contain nonempty string IDs of at most 200 characters")
        result[key] = list(dict.fromkeys(value))
    if custom:
        paths = payload.get("custom_tokenizers", [])
        if not isinstance(paths, list) or len(paths) > 100:
            raise ValueError("custom_tokenizers must be an array of at most 100 local paths")
        if any(not isinstance(path, str) or len(path) > 4096 or "\x00" in path
               or not Path(path).is_absolute() for path in paths):
            raise ValueError("custom_tokenizers must contain absolute local file paths")
        result["custom_tokenizers"] = list(dict.fromkeys(paths))
    if require_all:
        if not result["tokenizers"] and not result.get("custom_tokenizers"):
            raise ValueError("Select at least one tokenizer or a local custom tokenizer")
        for key in ("datasets", "languages"):
            if not result[key]:
                raise ValueError(f"Select at least one {key[:-1] if key.endswith('s') else key}")
    return result


class RunConflict(ValueError):
    pass


class RunManager:
    """One background benchmark at a time, with bounded status history."""
    def __init__(self):
        self.lock = Lock()
        self.jobs = {}
        self.cancel_events = {}
        self.active = None

    def start(self, selection):
        with self.lock:
            if self.active is not None:
                raise RunConflict("A benchmark is already running; wait for it to finish")
            identifier = uuid4().hex
            cancel = Event()
            job = {"id": identifier, "status": "queued", "progress": None,
                   "result": None, "error": None, "created_at": _now()}
            self.jobs[identifier] = job
            self.cancel_events[identifier] = cancel
            self.active = identifier
            while len(self.jobs) > 64:
                oldest = next(key for key in self.jobs if key != self.active)
                del self.jobs[oldest]
                self.cancel_events.pop(oldest, None)
            snapshot = deepcopy(job)
        Thread(target=self._work, args=(identifier, deepcopy(selection), cancel),
               name=f"tokenizer-benchmark-{identifier[:8]}", daemon=True).start()
        return snapshot

    def _work(self, identifier, selection, cancel):
        def progress(value=None, **fields):
            if value is None:
                value = fields
            elif fields:
                value = {**value, **fields} if isinstance(value, dict) else {"message": str(value), **fields}
            # Keep status snapshots independent of mutable engine state.
            value = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
            with self.lock:
                self.jobs[identifier]["progress"] = value
        with self.lock:
            self.jobs[identifier].update(status="running", started_at=_now())
        try:
            from .engine import run_selection
            result = run_selection(selection, progress=progress, cancel=cancel, vocabulary=True)
            result = json.loads(json.dumps(result, ensure_ascii=False, allow_nan=False))
            with self.lock:
                self.jobs[identifier].update(status="cancelled" if cancel.is_set() else "completed",
                                             result=result, finished_at=_now())
        except Exception as exc:
            with self.lock:
                self.jobs[identifier].update(status="cancelled" if cancel.is_set() else "failed",
                                             error=f"{type(exc).__name__}: {exc}", finished_at=_now())
        finally:
            with self.lock:
                if self.active == identifier:
                    self.active = None

    def get(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError("Run not found")
            return deepcopy(self.jobs[identifier])

    def cancel(self, identifier):
        with self.lock:
            if identifier not in self.jobs:
                raise KeyError("Run not found")
            if self.jobs[identifier]["status"] in {"queued", "running"}:
                self.cancel_events[identifier].set()
            return deepcopy(self.jobs[identifier])

    def close(self):
        with self.lock:
            for event in self.cancel_events.values():
                event.set()


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, handler=None, *, site_directory=None):
        self.site_directory = Path(site_directory or ROOT / "site").resolve()
        self.runs = RunManager()
        super().__init__(address, handler or Handler)

    def server_close(self):
        self.runs.close()
        super().server_close()


class Handler(SimpleHTTPRequestHandler):
    server_version = "TokenizerLab/0.2"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        # The final positional argument is the HTTP server supplied by socketserver.
        server = args[2]
        super().__init__(*args, directory=str(server.site_directory), **kwargs)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _authority(self, value):
        if not value or any(char.isspace() for char in value) or "," in value:
            raise ValueError("Invalid Host")
        parsed = urlsplit("//" + value)
        if parsed.username is not None or parsed.password is not None or parsed.path or parsed.query or parsed.fragment:
            raise ValueError("Invalid Host")
        host = (parsed.hostname or "").lower()
        if host not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("The Host must be localhost or a loopback IP")
        port = parsed.port if parsed.port is not None else 80
        if port != self.server.server_port:
            raise ValueError("Host port does not match the local service")
        return host, port

    def _trusted(self):
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                raise ValueError("Only loopback clients are allowed")
            hosts = self.headers.get_all("Host", [])
            if len(hosts) != 1:
                raise ValueError("Exactly one loopback Host header is required")
            authority = self._authority(hosts[0])
            origins = self.headers.get_all("Origin", [])
            if len(origins) > 1:
                raise ValueError("Multiple Origin headers are not allowed")
            if origins:
                origin = urlsplit(origins[0])
                if origin.scheme != "http" or origin.path or origin.query or origin.fragment:
                    raise ValueError("Cross-origin requests are disabled")
                if self._authority(origin.netloc) != authority:
                    raise ValueError("Cross-origin requests are disabled")
            if self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
                raise ValueError("Cross-site requests are disabled")
            return True
        except (ValueError, UnicodeError) as exc:
            self.json({"error": str(exc)}, 403)
            return False

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            raise ValueError("Chunked request bodies are not supported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1:
            raise ValueError("Exactly one Content-Length header is required")
        length = int(lengths[0])
        if not 0 < length <= MAX_BODY_BYTES:
            raise ValueError(f"Request must be between 1 and {MAX_BODY_BYTES:,} bytes")
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
            raise ValueError("Content-Type must be application/json")
        body = self.rfile.read(length)
        if len(body) != length:
            raise ValueError("Incomplete request body")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        return payload

    def _job_id(self, value):
        if not JOB_ID.fullmatch(value):
            raise ValueError("Invalid job ID")
        return value

    def _static_path(self):
        raw = unquote(urlsplit(self.path).path, errors="strict")
        if "\x00" in raw or "\\" in raw:
            raise ValueError("Invalid static path")
        parts = raw.split("/")
        if any(part in {"..", "."} or part.startswith(".") for part in parts if part):
            raise ValueError("Hidden files and parent paths are not served")
        root = self.server.site_directory
        target = (root / raw.lstrip("/")).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Static path leaves the explorer directory")
        if target.is_dir():
            target = target / "index.html"
            if not target.is_file():
                raise FileNotFoundError("Directory listing is disabled")
            # The index itself may be a symlink; check the resolved destination too.
            if not target.resolve().is_relative_to(root):
                raise ValueError("Static path leaves the explorer directory")
        return target

    def translate_path(self, path):
        return str(self._static_path())

    def _error(self, exc):
        if isinstance(exc, RunConflict):
            return self.json({"error": str(exc)}, 409)
        if isinstance(exc, (KeyError, FileNotFoundError)):
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            return self.json({"error": str(message)}, 404)
        if isinstance(exc, (ValueError, TypeError, UnicodeError)):
            return self.json({"error": str(exc)}, 400)
        return self.json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    def do_GET(self):
        if not self._trusted():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/health":
                return self.json({"ok": True, "service": "tokenizer-lab",
                                  "offline": os.environ.get("TOKENIZER_LAB_OFFLINE") == "1",
                                  "offline_ready": True})
            if path == "/api/catalog":
                from . import downloads
                return self.json(downloads.catalog())
            if path.startswith("/api/runs/") and path.count("/") == 3:
                return self.json(self.server.runs.get(self._job_id(path.rsplit("/", 1)[-1])))
            if path.startswith("/api/downloads/") and path.count("/") == 3:
                from . import downloads
                return self.json(downloads.get(self._job_id(path.rsplit("/", 1)[-1])))
            if path.startswith("/api/"):
                return self.json({"error": "Unknown endpoint"}, 404)
            self._static_path()
            return super().do_GET()
        except Exception as exc:
            return self._error(exc)

    def do_HEAD(self):
        if not self._trusted():
            return
        if urlsplit(self.path).path.startswith("/api/"):
            return self.json({"error": "HEAD is not supported for API endpoints"}, 405)
        try:
            self._static_path()
            return super().do_HEAD()
        except Exception as exc:
            return self._error(exc)

    def do_POST(self):
        if not self._trusted():
            return
        path = urlsplit(self.path).path
        try:
            payload = self._body()
            if path == "/api/convert":
                from .conversion import convert
                if set(payload) - {"source", "target", "text", "ids"}:
                    raise ValueError("Unknown conversion fields")
                source, target = payload.get("source"), payload.get("target")
                if not isinstance(source, str) or not source or not isinstance(target, str) or not target:
                    raise ValueError("Source and target tokenizer IDs are required")
                if ("text" in payload) == ("ids" in payload):
                    raise ValueError("Supply exactly one of text or source token IDs")
                if "text" in payload and not isinstance(payload["text"], str):
                    raise ValueError("text must be a string")
                if "ids" in payload and (not isinstance(payload["ids"], list)
                    or not payload["ids"] or any(type(item) is not int or item < 0 for item in payload["ids"])):
                    raise ValueError("ids must be a nonempty array of nonnegative integers")
                return self.json(convert(adapter(source), adapter(target), text=payload.get("text"), ids=payload.get("ids")))
            if path == "/api/custom-tokenizers":
                if set(payload) - {"path", "name"}:
                    raise ValueError("Unknown custom tokenizer fields")
                path_value, name = payload.get("path"), payload.get("name")
                if not isinstance(path_value, str) or len(path_value) > 4096 or "\x00" in path_value or not Path(path_value).is_absolute():
                    raise ValueError("Use an absolute path to a local tokenizer.json")
                if not isinstance(name, str) or not name.strip() or len(name) > 100 or any(ord(char) < 32 for char in name):
                    raise ValueError("Supply a display name of 1–100 characters")
                from .adapters import register_custom
                result = register_custom(path_value, name.strip())
                adapter.cache_clear()
                return self.json(result, 201)
            if path == "/api/runs":
                selection = _selection(payload, custom=True, require_all=True)
                return self.json(self.server.runs.start(selection), 202)
            if path == "/api/downloads/plan":
                from . import downloads
                return self.json(downloads.plan(_selection(payload)))
            if path == "/api/downloads":
                from . import downloads
                return self.json(downloads.start(_selection(payload)), 202)
            match = re.fullmatch(r"/api/(downloads|runs)/([^/]+)/cancel", path)
            if match:
                if payload:
                    raise ValueError("Cancellation body must be an empty JSON object")
                identifier = self._job_id(match[2])
                if match[1] == "downloads":
                    from . import downloads
                    return self.json(downloads.cancel(identifier))
                return self.json(self.server.runs.cancel(identifier))
            return self.json({"error": "Unknown endpoint"}, 404)
        except Exception as exc:
            return self._error(exc)

    def do_DELETE(self):
        if not self._trusted():
            return
        try:
            if urlsplit(self.path).path != "/api/cache":
                return self.json({"error": "Unknown endpoint"}, 404)
            from . import downloads
            selection = _selection(self._body())
            with self.server.runs.lock:
                if self.server.runs.active is not None:
                    raise RunConflict('Wait for the active benchmark before removing its cached inputs')
                result = downloads.remove(selection)
            adapter.cache_clear()
            return self.json(result)
        except Exception as exc:
            return self._error(exc)


def serve(port=8765, launch=False):
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer between 0 and 65535")
    server = LabServer(("127.0.0.1", port))
    address = f"http://localhost:{server.server_port}"
    print(f"Tokenizer Lab: {address}", flush=True)
    if launch:
        webbrowser.open(f"{address}/launcher.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
