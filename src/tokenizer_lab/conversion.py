"""Verified full-sequence conversion and context-labelled vocabulary maps."""
from __future__ import annotations
from .io import sha256

def convert(source,target,*,text=None,ids=None):
    if (text is None)==(ids is None): raise ValueError("Supply exactly one of text or source IDs")
    if text is not None and not isinstance(text,str):raise ValueError("Text must be a string")
    if ids is not None and not isinstance(ids,list):raise ValueError("IDs must be an array")
    if ids is None: ids=source.encode(text)
    if any(not isinstance(i,int) or isinstance(i,bool) or i not in source.by_id for i in ids):
        raise ValueError("Source sequence contains an invalid token ID")
    special=[i for i in ids if i in source.special_ids]
    if special: raise ValueError("Special/control IDs require an explicit model-specific policy; ordinary-text converter rejects them")
    # Native HF decoders can replace an incomplete byte fragment with U+FFFD.
    # Reject such ID streams before that information can disappear.
    pieces=[source.token_bytes(i) for i in ids]
    if all(p is not None for p in pieces):b"".join(pieces).decode("utf-8",errors="strict")
    raw=source.decode_bytes(ids)
    decoded=raw.decode("utf-8",errors="strict")
    target_ids=target.encode(decoded); reconstructed=target.decode_bytes(target_ids)
    return {"source":source.id,"target":target.id,"input_text":text,"source_ids":ids,"source_pieces":source.pieces(ids),
            "source_decoded":decoded,"source_roundtrip_exact":raw==text.encode() if text is not None else None,
            "target_ids":target_ids,"target_pieces":target.pieces(target_ids),"target_decoded":reconstructed.decode("utf-8",errors="backslashreplace"),
            "text":decoded,"source_encoding":{"ids":ids,"pieces":source.pieces(ids),"exact":raw==text.encode() if text is not None else None},
            "target_encoding":{"ids":target_ids,"pieces":target.pieces(target_ids),"exact":raw==reconstructed},
            "exact":raw==reconstructed,"source_sha256":sha256(raw),"target_sha256":sha256(reconstructed),
            "note":"Exact refers to preservation of source-decoded bytes. Source normalization cannot be undone. Target IDs are not model-weight conversion."}

def map_vocabulary(source,target,payload):
    target_pieces={}
    for i in target.by_id:
        raw=target.token_bytes(i)
        if raw is not None: target_pieces.setdefault(raw.hex(),[]).append(i)
    results=[]; texts=[]; positions=[]
    for row in payload["rows"]:
        i,surface,h,script,types,length=row
        item={"source_id":i,"source_surface":surface,"bytes_hex":h}
        if h is None: item.update(status="special_or_unavailable",target_ids=None)
        else:
            raw=bytes.fromhex(h)
            if h in target_pieces:
                item.update(status="exact_byte_piece",matching_piece_ids=target_pieces[h])
            try:
                text=raw.decode("utf-8",errors="strict")
                texts.append(text); positions.append(len(results))
            except UnicodeDecodeError: item.update(status="byte_fragment_context_required",target_ids=None)
        results.append(item)
    for start in range(0,len(texts),2048):
        chunk=texts[start:start+2048]
        encodings=target.encode_batch(chunk)
        for j,ids in enumerate(encodings,start):
            item=results[positions[j]]; exact=target.decode_bytes(ids)==chunk[j-start].encode()
            item.update(target_ids=ids,standalone_exact=exact,context_required=True)
            if item.get("status")!="exact_byte_piece": item["status"]="standalone_one_to_many" if exact else "standalone_changes_bytes"
    return {"schema_version":"1.0","source":source.id,"target":target.id,"source_sha256":source.asset_sha,"target_sha256":target.asset_sha,
            "warning":"Standalone token diagnostics are not a universal token-ID substitution table. Convert complete sequences to preserve boundaries.","rows":results}
