"""Native adapters. IDs, vocabulary pieces and decoded bytes are distinct."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import re
from .io import ROOT, file_hash, read_json, sha256

def byte_alphabet():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}

BYTE_DECODER = byte_alphabet()

def has_component(value, kind):
    if isinstance(value, dict):
        return value.get("type") == kind or any(has_component(v, kind) for v in value.values())
    if isinstance(value, list):
        return any(has_component(v, kind) for v in value)
    return False

class HFAdapter:
    def __init__(self, spec, path):
        from tokenizers import Tokenizer
        self.spec = spec; self.id = spec["id"]; self.path = Path(path)
        self.raw = read_json(self.path)
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.no_padding(); self.tokenizer.no_truncation()
        self.vocab = self.tokenizer.get_vocab(with_added_tokens=True)
        self.by_id = {i:s for s,i in self.vocab.items()}
        self.added = {x["id"]:x for x in self.raw.get("added_tokens", [])}
        self.special_ids = {i for i,a in self.added.items() if a.get("special")}
        model = self.raw["model"]
        self.unknown_id = model.get("unk_id", self.vocab.get(model.get("unk_token")))
        self.byte_level = has_component(self.raw.get("decoder"), "ByteLevel")
        self.byte_fallback = has_component(self.raw.get("decoder"), "ByteFallback")
        self.metaspace = "▁" in json.dumps(self.raw.get("decoder"), ensure_ascii=False)
        self.model_vocab = model["vocab"]
        self.asset_sha = file_hash(self.path)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return [x.ids for x in self.tokenizer.encode_batch(texts, add_special_tokens=False)]

    def decode_bytes(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8")

    def pieces(self, ids):
        return [self.by_id.get(i, f"<unknown-id:{i}>") for i in ids]

    def token_bytes(self, i):
        if i in self.special_ids:
            return None
        piece = self.by_id[i]
        # AddedToken strings outside the model vocab are literal text, not byte alphabet.
        in_model = piece in self.model_vocab if isinstance(self.model_vocab, dict) else i < len(self.model_vocab)
        if i in self.added and not in_model:
            return piece.encode("utf-8")
        if self.byte_fallback and re.fullmatch(r"<0x[0-9A-Fa-f]{2}>", piece):
            return bytes([int(piece[3:5], 16)])
        if self.byte_level:
            try:
                return bytes(BYTE_DECODER[c] for c in piece)
            except KeyError:
                return None
        if self.metaspace:
            piece = piece.replace("▁", " ")
        return piece.encode("utf-8")

    def components(self):
        return {k:self.raw.get(k) for k in ["normalizer","pre_tokenizer","post_processor","decoder"]} | {"model":self.raw["model"]["type"]}

class TikAdapter:
    def __init__(self, spec):
        import tiktoken
        self.spec = spec; self.id = spec["id"]
        self.tokenizer = tiktoken.get_encoding(spec.get("encoding", "o200k_base"))
        self.by_id = {i:b.decode("utf-8", errors="backslashreplace") for b,i in self.tokenizer._mergeable_ranks.items()}
        self.byte_vocab = {i:b for b,i in self.tokenizer._mergeable_ranks.items()}
        self.by_id.update({i:s for s,i in self.tokenizer._special_tokens.items()})
        self.special_ids = set(self.tokenizer._special_tokens.values())
        self.unknown_id = None
        canonical = b"".join(i.to_bytes(4,"big") + len(b).to_bytes(4,"big") + b for i,b in sorted(self.byte_vocab.items()))
        self.asset_sha = sha256(canonical + json.dumps(self.tokenizer._special_tokens, sort_keys=True).encode())
        self.path = None

    def encode(self, text):
        return self.tokenizer.encode_ordinary(text)

    def encode_batch(self, texts):
        return self.tokenizer.encode_ordinary_batch(texts, num_threads=1)

    def decode_bytes(self, ids):
        return self.tokenizer.decode_bytes(ids)

    def pieces(self, ids):
        return [self.by_id.get(i, f"<unknown-id:{i}>") for i in ids]

    def token_bytes(self, i):
        return self.byte_vocab.get(i)

    def components(self):
        return {"model":"byte-level BPE", "normalizer":None,
                "pre_tokenizer":{"type":"regex","pattern":self.tokenizer._pat_str},
                "decoder":"byte concatenation", "implementation":"tiktoken"}

def registry(path=None):
    return read_json(path or ROOT / "configs/registry.json")["tokenizers"]

def resolve_asset(spec, *, download=True):
    local = ROOT / (spec["file"] if spec["kind"] == "local" else f".cache/assets/{spec['id']}/tokenizer.json")
    if local.exists():
        if spec.get("sha256") and file_hash(local)!=spec["sha256"]:raise ValueError(f"Tokenizer artifact hash mismatch: {spec['id']}")
        if spec.get("structural_sha256"):
            from .expansion import structural_hash
            if structural_hash(read_json(local))!=spec["structural_sha256"]:raise ValueError(f"Tokenizer structure mismatch: {spec['id']}")
        return local
    if spec["kind"] == "local":
        raise FileNotFoundError(f"Missing {local}. See docs/REPRODUCE.md for V3.1 artifact installation.")
    if not download:
        raise FileNotFoundError(f"Asset not cached for {spec['id']}")
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(spec["repo"], spec["file"], revision=spec["revision"], cache_dir=ROOT/".cache/huggingface")
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(Path(src).read_bytes())
    return resolve_asset(spec,download=False)

def load_adapter(tokenizer_id, *, download=True):
    spec = next((s for s in registry() if s["id"] == tokenizer_id), None)
    if spec is None:
        raise ValueError(f"Unknown tokenizer {tokenizer_id}")
    return TikAdapter(spec) if spec["kind"] == "tiktoken" else HFAdapter(spec, resolve_asset(spec, download=download))
