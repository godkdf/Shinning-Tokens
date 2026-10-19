"""Explicit Unicode transports. Tokenizers consume text, not UTF-16 code units."""
def transport_sizes(text):
    le=text.encode("utf-16-le",errors="strict");be=text.encode("utf-16-be",errors="strict")
    return {"utf16le_bytes":len(le),"utf16be_bytes":len(be),"utf16_code_units":len(le)//2}

def fidelity(text,decoded_bytes):
    decoded=decoded_bytes.decode("utf-8",errors="strict")
    result={}
    for encoding,key in [("utf-16-le","utf16le"),("utf-16-be","utf16be")]:
        original=text.encode(encoding,errors="strict")
        transported=original.decode(encoding,errors="strict")
        assert transported==text
        result[key+"_exact"]=decoded.encode(encoding,errors="strict")==original
    return result

def decode_input(raw,encoding="utf-8"):
    allowed={"utf-8","utf-8-sig","utf-16","utf-16-le","utf-16-be"}
    if encoding not in allowed:raise ValueError("Input encoding must be UTF-8, UTF-16 (BOM), UTF-16LE or UTF-16BE")
    return raw.decode(encoding,errors="strict")

def probe(adapter):
    """Malformed transports must fail before tokenization, never become U+FFFD."""
    cases=[("BMP","Aéالعربية"),("supplementary","你好 👩🏽‍💻 𝄞"),("combining","e\u0301"),("NUL","a\x00b"),("empty","")]
    rows=[]
    for label,text in cases:
        for encoding in ["utf-16-le","utf-16-be"]:
            raw=text.encode(encoding);value=decode_input(raw,encoding);ids=adapter.encode(value);decoded=adapter.decode_bytes(ids).decode("utf-8",errors="strict")
            rows.append({"case":label,"encoding":encoding,"bytes":len(raw),"expected":"valid","exact":decoded.encode(encoding)==raw})
    for encoding in ["utf-16-le","utf-16-be"]:
        for label,raw in [("odd byte",b"a"),("unpaired high surrogate",b"\x00\xd8" if encoding.endswith("le") else b"\xd8\x00"),("unpaired low surrogate",b"\x00\xdc" if encoding.endswith("le") else b"\xdc\x00")]:
            try:decode_input(raw,encoding);rejected=False
            except UnicodeDecodeError:rejected=True
            rows.append({"case":label,"encoding":encoding,"expected":"reject","rejected":rejected})
    return rows
