"""Versioned vocabulary properties. Categories describe pieces, not abilities."""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter
import re
import unicodedata as ud
from .io import ROOT, read_json, write_json

KEYWORDS = set("if else elif for while return def class import from as try except finally with yield async await lambda pass break continue True False None null true false function const let var new this public private static void int char float double struct enum typedef sizeof switch case default SELECT FROM WHERE JOIN INSERT UPDATE DELETE CREATE TABLE AND OR NOT NULL SELECT DISTINCT GROUP ORDER BY LIMIT print console require export interface extends implements package include namespace using std".split())
NUMBER = re.compile(r"^[+−-]?(?:\d+(?:[.,]\d*)?|[.,]\d+)(?:[eE][+-]?\d+)?$")

class Scripts:
    def __init__(self, path=None):
        path = path or ROOT / "benchmarks/unicode/Scripts-15.1.0.txt"
        self.starts=[]; self.ranges=[]
        for line in path.read_text(encoding="utf-8").splitlines():
            line=line.split("#",1)[0].strip()
            if not line: continue
            span,name=map(str.strip,line.split(";")); ends=span.split("..")
            self.ranges.append((int(ends[0],16),int(ends[-1],16),name))
        self.ranges.sort(); self.starts=[x[0] for x in self.ranges]

    def char(self,c):
        n=ord(c); i=bisect_right(self.starts,n)-1
        return self.ranges[i][2] if i>=0 and n<=self.ranges[i][1] else "Unknown"

    def text(self,text):
        all_scripts={self.char(c) for c in text}
        strong=all_scripts-{"Common","Inherited"}
        if len(strong)>1: return "Mixed"
        if strong: return next(iter(strong))
        return "Inherited" if all_scripts=={"Inherited"} else "Common"

def classify(raw, scripts, *, special=False):
    if special: return "Special",["special"]
    if raw is None: return "Unavailable",["unavailable_bytes"]
    try: text=raw.decode("utf-8",errors="strict")
    except UnicodeDecodeError: return "Byte fragment",["byte_fragment"]
    t=[]; stripped=text.strip()
    if not text: t.append("empty")
    if text.isascii() and text.isdecimal(): t.append("ascii_digits")
    if text.isdecimal(): t.append("decimal_digits")
    if text.isnumeric(): t.append("unicode_numeric")
    if any(c.isdecimal() for c in text): t.append("contains_digits")
    if stripped and NUMBER.fullmatch(stripped): t.append("numeric_expression")
    if re.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+",stripped): t.append("hexadecimal")
    if text.isspace(): t.append("whitespace")
    if text.startswith(" "): t.append("leading_space")
    if text.endswith(" "): t.append("trailing_space")
    if "\n" in text: t.append("newline")
    if "\t" in text: t.append("tab")
    if any(ud.category(c).startswith("M") for c in text): t.append("combining_marks")
    if any(ud.category(c)=="Cc" for c in text): t.append("control")
    if any(0x1F300<=ord(c)<=0x1FAFF or 0x2600<=ord(c)<=0x27BF for c in text): t.append("emoji_candidate")
    if stripped in KEYWORDS: t.append("code_keyword")
    if stripped and all(c in "+-*/%=<>!&|^~?:.,;()[]{}" for c in stripped): t.append("code_operator_or_punctuation")
    if stripped.isidentifier(): t.append("identifier")
    if "<" in text or ">" in text: t.append("markup_candidate")
    if re.search(r"\\[ntruxU\\\"']",text): t.append("escape_sequence")
    if any(x in t for x in ["code_keyword","code_operator_or_punctuation","markup_candidate","escape_sequence"]): t.append("code_associated")
    if any(c.isalpha() for c in text): t.append("contains_letters")
    return scripts.text(text), sorted(t)

def inspect_vocabulary(adapter, *, output=None):
    scripts=Scripts(); rows=[]; script_counts=Counter(); type_counts=Counter(); lengths=Counter(); pieces=set(); unavailable=0
    for i,surface in sorted(adapter.by_id.items()):
        b=adapter.token_bytes(i); special=i in adapter.special_ids
        script,types=classify(b,scripts,special=special)
        script_counts[script]+=1; type_counts.update(types)
        if b is not None:
            pieces.add(b); lengths[len(b)]+=1
        elif not special: unavailable+=1
        rows.append([i,surface,b.hex() if b is not None else None,script,types,len(b) if b is not None else None])
    spec=adapter.spec
    profile={"id":adapter.id,"name":spec["name"],"role":spec["role"],"repo":spec.get("repo"),"revision":spec.get("revision"),
             "artifact_sha256":adapter.asset_sha,"terms":spec.get("terms"),"components":adapter.components(),
             "vocabulary":{"total":len(rows),"ordinary":len(rows)-len(adapter.special_ids),"special":len(adapter.special_ids),
               "id_space":max(adapter.by_id)+1,"id_gaps":max(adapter.by_id)+1-len(rows),"unique_byte_pieces":len(pieces),
               "unavailable_ordinary_bytes":unavailable,"script_counts":dict(script_counts),"type_counts":dict(type_counts),
               "token_length_histogram":dict(sorted(lengths.items())),"file_bytes":adapter.path.stat().st_size if adapter.path else None},
             "provenance":{"classification":"unicode-script-15.1.0","category_unicode_version":ud.unidata_version,
               "boundary_note":"Piece byte overlap is descriptive. Equal pieces do not imply equal in-context segmentation.",
               "types_note":"Categories overlap. Code-associated and emoji-candidate labels are heuristics, not capability scores."}}
    profile.update({k:spec[k] for k in ["family","families","aliases","catalog_structural_sha256"] if k in spec})
    metadata=ROOT/".cache/assets"/adapter.id/"provenance.json"
    if metadata.exists(): profile["provenance"].update(read_json(metadata))
    payload={"schema_version":"1.0","tokenizer":adapter.id,"artifact_sha256":adapter.asset_sha,
             "fields":["id","surface","bytes_hex","script","types","byte_length"],"rows":rows}
    if output: write_json(output,payload,compact=True)
    return profile,payload

def vocabulary_sets(payload):
    by_bytes={}; by_category={}
    for i,surface,h,script,types,length in payload["rows"]:
        if h is None: continue
        by_bytes.setdefault(h,[]).append(i)
        for category in ["script:"+script,*types]: by_category.setdefault(category,set()).add(h)
    return set(by_bytes),by_category,by_bytes

def compare_vocabularies(left,right):
    a,ac,_=vocabulary_sets(left); b,bc,_=vocabulary_sets(right); shared=a&b; union=a|b
    categories={k:{"left":len(ac.get(k,set())),"right":len(bc.get(k,set())),"shared":len(ac.get(k,set())&bc.get(k,set()))}
                for k in sorted(ac.keys()|bc.keys())}
    return {"left":left["tokenizer"],"right":right["tokenizer"],"shared":len(shared),"union":len(union),
            "jaccard":len(shared)/len(union) if union else None,
            "coverage_left":len(shared)/len(a) if a else None,"coverage_right":len(shared)/len(b) if b else None,
            "unique_left":len(a-b),"unique_right":len(b-a),"category_overlap":categories,
            "definition":"Unique ordinary byte pieces. Special IDs excluded; no Unicode normalization. Boundary behavior is not equated."}
