"""Source-byte integrity, duplicate leakage and corpus accounting checks."""
import re
import io
import tarfile
from contextlib import contextmanager

import pytest

from tokenizer_lab import codebench
from tokenizer_lab.io import file_hash, sha256, write_json, write_jsonl


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    monkeypatch.setattr(codebench, "ROOT", tmp_path)
    monkeypatch.setattr(codebench, "CORPUS", tmp_path / "corpus")
    monkeypatch.setattr(codebench, "CONFIG", tmp_path / "config.json")
    write_json(tmp_path / "config.json", {"max_files_per_repository": 100})

    def build(texts, splits=None, repo_names=None, total=None):
        rows = []
        for index, text in enumerate(texts):
            raw = text.encode("utf-8")
            rows.append({"id": str(index), "text": text, "sha256": sha256(raw), "utf8_bytes": len(raw),
                "normalized_sha256": sha256(re.sub(r"\s+", " ", text).encode()),
                "repository": repo_names[index] if repo_names else f"owner/repo{index}",
                "split": splits[index] if splits else "evaluation"})
        path = tmp_path / "corpus/python.jsonl.gz"
        write_jsonl(path, rows)
        write_json(tmp_path / "corpus/manifest.json", {"files": len(rows) if total is None else total,
            "quota_complete": False, "partitions": [{"path": "corpus/python.jsonl.gz", "sha256": file_hash(path), "documents": len(rows)}]})
        return path
    return build


def test_source_bytes_preserve_crlf_bom_and_unicode(corpus):
    corpus(["\ufeff# café\r\nx = 'e\u0301'\r\n", "print('你好')\n"])
    result = codebench.audit()
    assert result["files"] == 2
    assert result["integrity_pass"]
    assert not result["quota_complete"]


def test_whitespace_duplicate_cannot_cross_split(corpus):
    corpus(["x = 1\n", "x = 1\r\n"], ["development", "evaluation"])
    with pytest.raises(ValueError, match="Duplicate"):
        codebench.audit()


def test_repository_cannot_cross_split(corpus):
    corpus(["x = 1\n", "y = 2\n"], ["development", "evaluation"], ["a/repo", "a/repo"])
    with pytest.raises(ValueError, match="crosses splits"):
        codebench.audit()


def test_truncated_manifest_is_rejected(corpus):
    corpus(["x = 1\n"], total=2)
    with pytest.raises(ValueError, match="total mismatch"):
        codebench.audit()


def test_changed_partition_is_rejected(corpus):
    path = corpus(["x = 1\n"])
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Partition changed"):
        codebench.audit()


def test_frozen_restore_reproduces_original_partition_bytes(tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(codebench, "ROOT", tmp_path)
    monkeypatch.setattr(codebench, "CORPUS", tmp_path/"corpus")
    monkeypatch.setattr(codebench, "CONFIG", tmp_path/"config.json")
    write_json(tmp_path/"config.json", {"max_files_per_repository": 100})
    text = "\ufeff# café\r\nx = 'e\u0301'\r\n"
    raw = text.encode("utf-8")
    row = {"path": "source/example.py", "text": text, "sha256": sha256(raw), "utf8_bytes": len(raw),
        "size_band": "le1k", "normalized_sha256": sha256(re.sub(r"\s+", " ", text).encode()),
        "source_id": "owner/project", "language": "python", "domain": "real repository source", "dataset": "code-realistic-v0.1",
        "id": "python:example", "split": "evaluation", "repository": "owner/project", "commit": "a"*40,
        "license": "MIT", "source_url": "https://github.com/owner/project/blob/"+"a"*40+"/source/example.py",
        "language_evidence": {"method": "test"}, "dialect_status": "not independently verified"}
    partition = tmp_path/"corpus/python.jsonl.gz"
    write_jsonl(partition, [row])
    original = partition.read_bytes()
    index = tmp_path/"corpus/file-index.jsonl.gz"
    write_jsonl(index, [{k: v for k, v in row.items() if k != "text"}])
    write_json(tmp_path/"corpus/manifest.json", {"files": 1, "quota_complete": False, "file_index_sha256": file_hash(index),
        "partitions": [{"path": "corpus/python.jsonl.gz", "sha256": file_hash(partition), "documents": 1}]})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("project-commit/source/example.py")
        info.size = len(raw)
        archive.addfile(info, io.BytesIO(raw))

    @contextmanager
    def fake_get(url, **kwargs):
        assert url.endswith("/"+"a"*40)
        class Response:
            def raise_for_status(self):
                pass
            raw = io.BytesIO(buffer.getvalue())
        yield Response()

    monkeypatch.setattr(requests, "get", fake_get)
    partition.unlink()
    codebench.restore()
    assert partition.read_bytes() == original
