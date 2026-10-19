"""Local API boundaries: no implicit selections, no downloads during runs."""
from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace

import pytest

from tokenizer_lab import server


@pytest.fixture
def service(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("Tokenizer Lab fixture", encoding="utf-8")
    (site / "launcher.html").write_text("Local launcher fixture", encoding="utf-8")
    calls = []
    downloads = SimpleNamespace(
        catalog=lambda: {"tokenizers": [{"id": "cached", "cached": True}], "languages": [], "datasets": []},
        plan=lambda selection: calls.append(("plan", selection)) or {
            "files": [], "download_bytes": 0, "cached_bytes": 0, "total_bytes": 0, "missing": []},
        start=lambda selection: calls.append(("download", selection)) or {"id": "download_1", "status": "queued"},
        get=lambda identifier: calls.append(("get", identifier)) or {"id": identifier, "status": "running"},
        cancel=lambda identifier: calls.append(("cancel", identifier)) or {"id": identifier, "status": "cancelled"},
        remove=lambda selection: calls.append(("remove", selection)) or {"removed_files": [], "freed_bytes": 0},
    )
    import tokenizer_lab
    monkeypatch.setitem(sys.modules, "tokenizer_lab.downloads", downloads)
    monkeypatch.setattr(tokenizer_lab, "downloads", downloads, raising=False)
    httpd = server.LabServer(("127.0.0.1", 0), site_directory=site)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    address = f"127.0.0.1:{httpd.server_port}"

    def request(method, path, payload=None, *, headers=None, raw=None):
        connection = HTTPConnection("127.0.0.1", httpd.server_port, timeout=5)
        request_headers = {"Host": address}
        if headers:
            request_headers.update(headers)
        body = raw
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            request_headers.setdefault("Content-Type", "application/json")
        connection.request(method, path, body=body, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        result = json.loads(data) if response.getheader("Content-Type", "").startswith("application/json") and data else data
        status = response.status
        connection.close()
        return status, result

    yield SimpleNamespace(server=httpd, request=request, calls=calls, site=site, root=tmp_path,
                          address=address, downloads=downloads)
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=3)
    server.adapter.cache_clear()


def install_engine(monkeypatch, function):
    import tokenizer_lab
    module = SimpleNamespace(run_selection=function)
    monkeypatch.setitem(sys.modules, "tokenizer_lab.engine", module)
    monkeypatch.setattr(tokenizer_lab, "engine", module, raising=False)


def wait_status(service, identifier, terminal=("completed", "failed", "cancelled")):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status, job = service.request("GET", f"/api/runs/{identifier}")
        assert status == 200
        if job["status"] in terminal:
            return job
        time.sleep(.01)
    pytest.fail("Run did not reach its expected status within 5 seconds")


def test_health_reports_offline_mode_and_readiness(service, monkeypatch):
    monkeypatch.delenv("TOKENIZER_LAB_OFFLINE", raising=False)
    assert service.request("GET", "/api/health") == (200, {
        "ok": True, "service": "tokenizer-lab", "offline": False, "offline_ready": True})
    monkeypatch.setenv("TOKENIZER_LAB_OFFLINE", "1")
    assert service.request("GET", "/api/health")[1]["offline"] is True


@pytest.mark.parametrize("headers", [
    {"Host": "attacker.example:8765"},
    {"Host": "127.0.0.1:1"},
    {"Origin": "https://attacker.example"},
    {"Origin": "null"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_rejects_dns_rebinding_and_cross_origin_requests(service, headers):
    assert service.request("GET", "/api/health", headers=headers)[0] == 403
    assert service.request("POST", "/api/downloads", {"tokenizers": ["cached"]}, headers=headers)[0] == 403
    assert not service.calls


def test_same_origin_accepts_only_the_exact_local_authority(service):
    assert service.request("GET", "/api/health", headers={"Origin": f"http://{service.address}"})[0] == 200
    # Localhost and 127.0.0.1 are separate browser origins despite both being loopback.
    port = service.server.server_port
    assert service.request("GET", "/api/health", headers={"Origin": f"http://localhost:{port}"})[0] == 403


@pytest.mark.parametrize("payload", [[], {"tokenizers": "all"}, {"tokenizers": [True]},
                                     {"languages": [""]}, {"all": True}])
def test_invalid_selection_never_expands_to_all(service, payload):
    assert service.request("POST", "/api/downloads/plan", payload)[0] == 400
    assert not service.calls


def test_download_plan_supports_partial_and_empty_selections(service):
    for payload, expected in [
        ({"tokenizers": ["cached"]}, {"tokenizers": ["cached"], "datasets": [], "languages": []}),
        ({"datasets": ["flores"], "languages": ["eng_Latn"]},
         {"tokenizers": [], "datasets": ["flores"], "languages": ["eng_Latn"]}),
        ({}, {"tokenizers": [], "datasets": [], "languages": []}),
    ]:
        assert service.request("POST", "/api/downloads/plan", payload)[0] == 200
        assert service.calls[-1] == ("plan", expected)


def test_download_jobs_and_quantified_removal_delegate_exact_selections(service):
    payload = {"tokenizers": ["cached"], "datasets": [], "languages": []}
    assert service.request("POST", "/api/downloads", payload) == (202, {"id": "download_1", "status": "queued"})
    assert service.calls[-1] == ("download", payload)
    assert service.request("GET", "/api/downloads/download_1")[1]["id"] == "download_1"
    assert service.request("POST", "/api/downloads/download_1/cancel", {})[1]["status"] == "cancelled"
    assert service.request("DELETE", "/api/cache", payload) == (200, {"removed_files": [], "freed_bytes": 0})
    assert service.calls[-1] == ("remove", payload)


def test_bounded_json_body_is_required_before_cache_mutation(service):
    assert service.request("POST", "/api/downloads", raw=b"{}", headers={"Content-Type": "text/plain"})[0] == 400
    assert service.request("POST", "/api/downloads", raw=b"{" ,headers={"Content-Type": "application/json"})[0] == 400
    assert service.request("POST", "/api/downloads", raw=b"", headers={"Content-Type": "application/json", "Content-Length": "1000001"})[0] == 400
    assert not service.calls


def test_static_server_does_not_expose_outside_or_hidden_files(service):
    assert service.request("GET", "/")[1] == b"Tokenizer Lab fixture"
    assert service.request("GET", "/launcher.html")[0] == 200
    secret = service.root / "secret.txt"
    secret.write_text("not served", encoding="utf-8")
    (service.site / "outside.txt").symlink_to(secret)
    (service.site / ".hidden").write_text("not served", encoding="utf-8")
    (service.site / "directory").mkdir()
    for path in ("/outside.txt", "/.hidden", "/%2e%2e/secret.txt", "/..%5csecret.txt"):
        assert service.request("GET", path)[0] == 400
    assert service.request("GET", "/directory/")[0] == 404
    assert service.request("GET", "/api/not-real")[0] == 404


def test_conversion_only_loads_cached_adapters(service, monkeypatch):
    from tokenizer_lab import adapters, conversion
    loaded = []
    monkeypatch.setattr(adapters, "load_adapter", lambda identifier, **kwargs: loaded.append((identifier, kwargs)) or identifier)
    monkeypatch.setattr(conversion, "convert", lambda source, target, **kwargs: {"source": source, "target": target, **kwargs})
    server.adapter.cache_clear()
    status, result = service.request("POST", "/api/convert", {"source": "one", "target": "two", "text": "🧪 e\u0301"})
    assert status == 200 and result["text"] == "🧪 e\u0301"
    assert loaded == [("one", {"download": False}), ("two", {"download": False})]
    assert not service.calls


def test_custom_registration_requires_absolute_path_and_name(service, monkeypatch):
    from tokenizer_lab import adapters
    calls = []
    monkeypatch.setattr(adapters, "register_custom", lambda path, name: calls.append((path, name)) or {"id": "custom_1", "name": name}, raising=False)
    assert service.request("POST", "/api/custom-tokenizers", {"path": "relative.json", "name": "Mine"})[0] == 400
    path = str(service.root / "tokenizer.json")
    assert service.request("POST", "/api/custom-tokenizers", {"path": path, "name": "  Mine  "}) == (201, {"id": "custom_1", "name": "Mine"})
    assert calls == [(path, "Mine")]


def test_only_one_run_is_active_and_progress_is_observable(service, monkeypatch):
    entered, release = Event(), Event()
    received = []
    def engine(selection, *, progress, cancel, vocabulary, runtime):
        assert vocabulary is True
        assert runtime is True
        received.append(selection)
        progress({"completed": 1, "total": 2})
        entered.set()
        assert release.wait(4)
        return {"documents": 2, "exact": 2}
    install_engine(monkeypatch, engine)
    selection = {"tokenizers": ["cached"], "datasets": ["flores"], "languages": ["eng_Latn"]}
    try:
        status, job = service.request("POST", "/api/runs", selection)
        assert status == 202 and entered.wait(2)
        assert service.request("POST", "/api/runs", selection)[0] == 409
        current = service.request("GET", f"/api/runs/{job['id']}")[1]
        assert current["status"] == "running"
        assert current["progress"] == {"completed": 1, "total": 2}
        assert service.request("DELETE", "/api/cache", selection)[0] == 409
    finally:
        release.set()
    done = wait_status(service, job["id"])
    assert done["result"] == {"documents": 2, "exact": 2}
    assert received == [{**selection, "custom_tokenizers": []}]
    assert not service.calls  # A run never invokes a download.


def test_custom_only_run_is_valid_but_missing_datasets_are_not(service, monkeypatch):
    install_engine(monkeypatch, lambda selection, **kwargs: {"custom": selection["custom_tokenizers"]})
    selection = {"tokenizers": [], "custom_tokenizers": [str(service.root / "mine.json")],
                 "datasets": ["flores"], "languages": ["eng_Latn"]}
    status, job = service.request("POST", "/api/runs", selection)
    assert status == 202
    assert wait_status(service, job["id"])["result"]["custom"] == selection["custom_tokenizers"]
    assert service.request("POST", "/api/runs", {**selection, "datasets": []})[0] == 400
    assert service.request("POST", "/api/runs", {**selection, "custom_tokenizers": []})[0] == 400


def test_failed_run_releases_slot_and_reports_error(service, monkeypatch):
    def broken(selection, **kwargs):
        raise FileNotFoundError("Selected corpus is not cached")
    install_engine(monkeypatch, broken)
    selection = {"tokenizers": ["cached"], "datasets": ["flores"], "languages": ["eng_Latn"]}
    _, job = service.request("POST", "/api/runs", selection)
    done = wait_status(service, job["id"])
    assert done["status"] == "failed"
    assert done["error"] == "FileNotFoundError: Selected corpus is not cached"
    assert service.request("POST", "/api/runs", selection)[0] == 202


def test_serve_opens_launcher_after_successful_binding(monkeypatch, capsys):
    calls = []
    class FakeServer:
        server_port = 12345
        def __init__(self, address):
            calls.append(("bind", address))
        def serve_forever(self):
            calls.append(("serve",))
            raise KeyboardInterrupt
        def server_close(self):
            calls.append(("close",))
    monkeypatch.setattr(server, "LabServer", FakeServer)
    monkeypatch.setattr(server.webbrowser, "open", lambda url: calls.append(("browser", url)))
    server.serve(0, launch=True)
    assert calls == [("bind", ("127.0.0.1", 0)), ("browser", "http://localhost:12345/launcher.html"), ("serve",), ("close",)]
    assert "http://localhost:12345" in capsys.readouterr().out
