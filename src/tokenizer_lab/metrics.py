"""Direct measurements from identical original text; never remove failed text."""
from __future__ import annotations
from collections import Counter
import unicodedata as ud
import numpy as np
from .io import sha256,validate_text_record
from .transport import transport_sizes,fidelity

def measure(adapter, documents):
    output=[]
    for start in range(0,len(documents),256):
        chunk=documents[start:start+256]
        try: encoded=adapter.encode_batch([x["text"] for x in chunk])
        except Exception: encoded=[None]*len(chunk)
        for doc,ids in zip(chunk,encoded):
            original=validate_text_record(doc)
            row={k:doc.get(k) for k in ["id","dataset","language","source_id","domain","sha256"]}
            row.update(tokenizer=adapter.id,utf8_bytes=len(original),tokens=0,exact=False,canonical_only=False,error=None,unknown_emissions=0)
            row.update(transport_sizes(doc["text"]),utf16le_exact=False,utf16be_exact=False)
            try:
                if ids is None: ids=adapter.encode(doc["text"])
                decoded=adapter.decode_bytes(ids)
                exact=decoded==original
                canonical=ud.normalize("NFC",decoded.decode("utf-8",errors="strict"))==ud.normalize("NFC",doc["text"])
                row.update(tokens=len(ids),exact=exact,canonical_only=bool(not exact and canonical),decoded_sha256=sha256(decoded),
                           unknown_emissions=sum(i==adapter.unknown_id for i in ids) if adapter.unknown_id is not None else 0)
                row.update(fidelity(doc["text"],decoded))
                if not exact: row["mismatch_example"]={"original":doc["text"][:500],"decoded":decoded.decode("utf-8",errors="backslashreplace")[:500]}
            except Exception as exc: row["error"]=f"{type(exc).__name__}: {exc}"[:400]
            output.append(row)
    return output

def aggregate(rows):
    groups={}
    for row in rows: groups.setdefault((row["dataset"],row["language"],row["tokenizer"]),[]).append(row)
    output=[]
    for (dataset,language,tokenizer),group in sorted(groups.items()):
        n=len(group); tokens=sum(x["tokens"] for x in group); size=sum(x["utf8_bytes"] for x in group)
        errors=sum(x["error"] is not None for x in group); exact=sum(x["exact"] for x in group); canonical=sum(x["canonical_only"] for x in group)
        counts=[x["tokens"] for x in group if x["error"] is None]
        output.append({"dataset":dataset,"language":language,"tokenizer":tokenizer,"documents":n,"utf8_bytes":size,"tokens":tokens,
                       "utf16le_bytes":sum(x.get("utf16le_bytes",0) for x in group),"utf16be_bytes":sum(x.get("utf16be_bytes",0) for x in group),
                       "utf16_code_units":sum(x.get("utf16_code_units",0) for x in group),"utf16le_exact":sum(x.get("utf16le_exact",False) for x in group),"utf16be_exact":sum(x.get("utf16be_exact",False) for x in group),
                       "bytes_per_token":size/tokens if tokens and not errors else None,"exact":exact,"canonical_only":canonical,
                       "mismatches":n-exact-canonical-errors,"errors":errors,"unknown_emissions":sum(x["unknown_emissions"] for x in group),
                       "strict_lossless":exact==n and errors==0,"token_p90":float(np.percentile(counts,90)) if counts else None,"token_max":max(counts) if counts else None,
                       "count_scope":"All original documents. Failures remain visible; compression is diagnostic when fidelity fails."})
    return output

def paired_bootstrap(parent_rows,child_rows,*,repeats=1000,seed=20261019):
    a={x["id"]:x for x in parent_rows}; b={x["id"]:x for x in child_rows}
    if a.keys()!=b.keys(): raise ValueError("Paired document IDs differ")
    groups={}
    for key in sorted(a):
        x,y=a[key],b[key]
        if x["sha256"]!=y["sha256"]: raise ValueError("Paired input bytes differ")
        # Fixed, disclosed exact-pair scope, not silently changing the main benchmark.
        if not (x["exact"] and y["exact"] and x["error"] is None and y["error"] is None): continue
        groups.setdefault((x["dataset"],x["language"]),[]).append((x,y))
    out=[];rng=np.random.default_rng(seed)
    for (dataset,language),pairs in sorted(groups.items()):
        clusters={}
        for x,y in pairs:
            g=x.get("source_id") or x["id"]; p,c=clusters.get(g,(0,0));clusters[g]=(p+x["tokens"],c+y["tokens"])
        values=np.array(list(clusters.values()),dtype=np.int64)
        if not len(values) or values[:,0].sum()==0: continue
        draws=[]
        for _ in range(repeats):
            v=values[rng.integers(0,len(values),size=len(values))].sum(axis=0)
            if v[0]: draws.append(100*(1-v[1]/v[0]))
        out.append({"dataset":dataset,"language":language,"documents":len(pairs),"source_groups":len(values),
                    "reduction_percent":100*(1-values[:,1].sum()/values[:,0].sum()),"ci95":np.percentile(draws,[2.5,97.5]).tolist(),
                    "replicates":repeats,"scope":"exact paired inputs; source-group resampling; no multiple-comparison correction"})
    return out
