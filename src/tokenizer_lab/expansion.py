"""Rebuild the expansion from explicit AddedToken records, preserving parent IDs."""
import json
from pathlib import Path
from .io import ROOT,read_json,read_jsonl,sha256

def structural_hash(value):
    return sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())

def build(parent,output):
    raw=read_json(parent);expected=read_json(ROOT/"experiments/gemma-v31/identity.json")
    if structural_hash(raw)!=expected["parent_structural_sha256"]:raise ValueError("Parent structure differs from the frozen control; inspect the difference before running this experiment")
    additions=list(read_jsonl(ROOT/"experiments/gemma-v31/additions.jsonl"))
    used={a["id"] for a in raw["added_tokens"]}|set(raw["model"]["vocab"].values())
    for a in additions:
        if a["id"] in used:raise ValueError("Expansion would overwrite a parent ID")
        if a["text"].encode().hex()!=a["bytes_hex"]:raise ValueError("Corrupt addition bytes")
        raw["added_tokens"].append({"id":a["id"],"content":a["text"],**a["flags"]});used.add(a["id"])
    if structural_hash(raw)!=expected["child_structural_sha256"]:raise ValueError("Built expansion differs from frozen candidate")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(raw,ensure_ascii=False,indent=2),encoding="utf-8")
    return {"additions":len(additions),"output":str(output),"structural_sha256":structural_hash(raw)}
