"""Data-only ports of pinned legacy tokenizer settings; no remote Python."""
from __future__ import annotations
import json
import re
from .io import file_hash,sha256

class SentencePieceAdapter:
    def __init__(self,spec,path):
        import sentencepiece as spm
        self.spec=spec;self.id=spec["id"];self.path=path;self.native=spec["native_spec"]
        self.tokenizer=spm.SentencePieceProcessor(model_file=str(path));self.offset=self.native.get("id_offset",0)
        self.by_id={i+self.offset:self.tokenizer.id_to_piece(i) for i in range(self.tokenizer.vocab_size())}
        self.appended={i:s for s,i in self.native.get("appended_special_tokens",{}).items()};self.by_id.update(self.appended)
        self.added={int(i):a for i,a in self.native.get("config_added_tokens_decoder",{}).items()}
        self.special_ids={i+self.offset for i in range(self.tokenizer.vocab_size()) if self.tokenizer.is_control(i) or self.tokenizer.is_unknown(i)}|set(self.appended)
        self.special_ids-={i for i,a in self.added.items() if not a.get("special",False)}
        self.unknown_id=self.tokenizer.unk_id()+self.offset
        self.intercept={a["content"]:i for i,a in self.added.items()}
        if self.id=="glm_original_zh":self.intercept.update({s:i for i,s in self.appended.items()})
        self.pattern=re.compile("("+"|".join(re.escape(s) for s in sorted(self.intercept,key=len,reverse=True))+")") if self.intercept else None
        self.asset_sha=sha256(path.read_bytes()+json.dumps(self.native,sort_keys=True).encode())

    def _transform(self,text):
        if self.native.get("remove_space"):text=" ".join(text.strip().split())
        if self.native.get("do_lower_case"):text=text.lower()
        if self.native.get("transforms"):
            text=text.replace("\n","<n>").replace("\t","<|tab|>")
            for n in range(80,1,-1):text=text.replace(" "*n,f"<|blank_{n}|>")
        return text

    def _inverse(self,text):
        if self.native.get("transforms"):
            text=text.replace("<n>","\n").replace("<|tab|>","\t")
            for n in range(2,81):text=text.replace(f"<|blank_{n}|>"," "*n)
        return text

    def encode(self,text):
        chunks=self.pattern.split(text) if self.pattern else [text];out=[]
        for chunk in chunks:
            if chunk in self.intercept:out.append(self.intercept[chunk])
            elif chunk:out.extend(i+self.offset for i in self.tokenizer.encode(self._transform(chunk),out_type=int))
        return out

    def encode_batch(self,texts):return [self.encode(text) for text in texts]

    def decode_bytes(self,ids):
        chunks=[];buffer=[]
        def flush():
            if buffer:chunks.append(self._inverse(self.tokenizer.decode(buffer)));buffer.clear()
        for i in ids:
            if i in self.appended:
                flush()
                if self.native.get("decode_appended_tokens")!="empty":chunks.append(self.appended[i])
            elif i in self.by_id:buffer.append(i-self.offset)
            else:raise ValueError(f"Unknown token ID {i}")
        flush()
        # Transformers' slow added-token wrapper joins decoded segments with
        # spaces by default. ChatGLM3's four ordinary config additions invoke it.
        text=" ".join(chunks) if self.added and len(chunks)>1 else "".join(chunks)
        return text.encode("utf-8")

    def pieces(self,ids):return [self.by_id[i] for i in ids]
    def token_bytes(self,i):
        if i in self.special_ids:return None
        piece=self.by_id[i]
        if re.fullmatch(r"<0x[0-9A-Fa-f]{2}>",piece):return bytes([int(piece[3:5],16)])
        return self._inverse(piece.replace("▁"," ")).encode("utf-8")
    def components(self):
        return {"model":"SentencePiece","native_spec":self.native,"reference_wrapper_parity":"not established; data-only port of recorded pinned settings","automatic_bos_eos":False}

class GPT2Adapter:
    def __new__(cls,spec,path):
        from tokenizers import Tokenizer,models,pre_tokenizers,decoders,AddedToken
        from .adapters import HFAdapter
        native=spec["native_spec"];merges=path.parent/native["merges_file"]
        tokenizer=Tokenizer(models.BPE.from_file(str(path),str(merges)))
        tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=native.get("add_prefix_space",False));tokenizer.decoder=decoders.ByteLevel()
        special=native.get("configured_special_tokens",{});tokens=[]
        for value in special.values():tokens.extend(value if isinstance(value,list) else [value])
        for piece,id in sorted(native.get("added_tokens",{}).items(),key=lambda x:x[1]):
            tokenizer.add_special_tokens([AddedToken(piece,special=True,normalized=False)])
            if tokenizer.token_to_id(piece)!=id:raise ValueError("Declared GLM added-token IDs differ from native BPE")
        for piece in tokens:
            if tokenizer.token_to_id(piece) is not None:tokenizer.add_special_tokens([AddedToken(piece,special=True,normalized=False)])
        generated=path.parent/"ported-tokenizer.json";generated.write_text(tokenizer.to_str(),encoding="utf-8")
        adapter=HFAdapter(spec,generated);adapter.path=path
        adapter.asset_sha=sha256(path.read_bytes()+merges.read_bytes()+json.dumps(native,sort_keys=True).encode())
        components=adapter.components
        adapter.components=lambda:components()|{"reference_wrapper_parity":"not established; asset-bounded data-only port","missing_configured_specials":[s for s in tokens if tokenizer.token_to_id(s) is None]}
        return adapter
