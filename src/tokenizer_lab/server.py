"""Loopback-only explorer and whole-sequence conversion service."""
from __future__ import annotations
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler,ThreadingHTTPServer
import json
from urllib.parse import urlsplit
from .io import ROOT

@lru_cache(maxsize=2)
def adapter(id):
    from .adapters import load_adapter
    return load_adapter(id)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kwargs):super().__init__(*args,directory=str(ROOT/"site"),**kwargs)
    def json(self,value,status=200):
        body=json.dumps(value,ensure_ascii=False).encode();self.send_response(status);self.send_header("Content-Type","application/json; charset=utf-8");self.send_header("Content-Length",str(len(body)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if urlsplit(self.path).path=="/api/health":return self.json({"ok":True,"service":"tokenizer-lab"})
        super().do_GET()
    def do_POST(self):
        if urlsplit(self.path).path!="/api/convert":return self.json({"error":"Unknown endpoint"},404)
        if self.headers.get("Origin") and urlsplit(self.headers["Origin"]).netloc!=self.headers.get("Host"):return self.json({"error":"Cross-origin conversion is disabled"},403)
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=1_000_000:raise ValueError("Request must be between 1 and 1,000,000 bytes")
            payload=json.loads(self.rfile.read(length))
            from .conversion import convert
            result=convert(adapter(payload["source"]),adapter(payload["target"]),text=payload.get("text"),ids=payload.get("ids"))
            self.json(result)
        except (ValueError,KeyError,TypeError,UnicodeError) as exc:self.json({"error":str(exc)},400)
        except Exception as exc:self.json({"error":f"{type(exc).__name__}: {exc}"},500)

def serve(port=8765):
    server=ThreadingHTTPServer(("127.0.0.1",port),Handler)
    print(f"Tokenizer Lab: http://localhost:{port}",flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
