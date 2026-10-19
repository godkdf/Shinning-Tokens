import io
import unicodedata
import numpy as np
import pytest
from tokenizer_lab.io import write_jsonl,read_jsonl,file_hash,validate_text_record
from tokenizer_lab.corpora import document,HTTPRangeFile
from tokenizer_lab.metrics import measure,aggregate,paired_bootstrap
from tokenizer_lab.conversion import convert
from tokenizer_lab.vocabulary import compare_vocabularies,Scripts,classify
from tokenizer_lab.transfer import subtoken_average,ridge_mapping,apply_mapping,mapping_metrics

class BytesAdapter:
    id="toy";special_ids={256};unknown_id=None
    by_id={i:str(i) for i in range(257)}
    def encode(self,text):return list(text.encode())
    def encode_batch(self,texts):return [self.encode(t) for t in texts]
    def decode_bytes(self,ids):return bytes(ids)
    def token_bytes(self,i):return bytes([i]) if i<256 else None
    def pieces(self,ids):return [str(i) for i in ids]

def test_original_bytes_and_gzip_are_stable(tmp_path):
    rows=[document("e\u0301\x00=Function",dataset="test",language="mixed",index=0)]
    a=tmp_path/"a.jsonl.gz";b=tmp_path/"b.jsonl.gz"
    write_jsonl(a,rows);write_jsonl(b,rows)
    assert file_hash(a)==file_hash(b)
    assert list(read_jsonl(a))==rows
    rows[0]["text"]=unicodedata.normalize("NFC",rows[0]["text"])
    with pytest.raises(ValueError,match="hash"):validate_text_record(rows[0])

def test_fidelity_counts_normalization_as_failure():
    class Normalizing(BytesAdapter):
        def encode(self,text):return super().encode(unicodedata.normalize("NFC",text))
    rows=measure(Normalizing(),[document(t,dataset="d",language="l",index=i) for i,t in enumerate(["e\u0301","é","plain"])])
    result=aggregate(rows)[0]
    assert result["documents"]==3 and result["exact"]==2 and result["canonical_only"]==1
    assert not result["strict_lossless"]
    assert result["bytes_per_token"]==sum(r["utf8_bytes"] for r in rows)/sum(r["tokens"] for r in rows)

def test_failures_are_retained():
    class Failing(BytesAdapter):
        def encode(self,text):raise ValueError("failure")
    result=aggregate(measure(Failing(),[document("x",dataset="d",language="l",index=0)]))[0]
    assert result["documents"]==1 and result["errors"]==1 and result["bytes_per_token"] is None

def test_full_sequence_conversion_rejects_incomplete_bytes_and_specials():
    a=BytesAdapter();ids=a.encode("你好 é")
    assert convert(a,a,ids=ids)["exact"]
    with pytest.raises(UnicodeDecodeError):convert(a,a,ids=[0xe4])
    with pytest.raises(ValueError,match="Special"):convert(a,a,ids=[256])
    with pytest.raises(ValueError):convert(a,a,ids=[True])

def test_overlap_uses_unique_bytes_and_directional_denominators():
    def payload(id,values):return {"tokenizer":id,"rows":[[i,h,h,"Latin",[],1] for i,h in enumerate(values)]}
    result=compare_vocabularies(payload("a",["61","61","62"]),payload("b",["62","63","64"]))
    assert result["shared"]==1 and result["union"]==4
    assert result["coverage_left"]==.5 and result["coverage_right"]==1/3

def test_unicode_script_and_number_properties():
    scripts=Scripts()
    assert classify("العربية".encode(),scripts)[0]=="Arabic"
    assert classify("a中".encode(),scripts)[0]=="Mixed"
    assert "ascii_digits" in classify(b"123",scripts)[1]
    assert "ascii_digits" not in classify("١٢٣".encode(),scripts)[1]
    assert classify(b"\xe4",scripts)[0]=="Byte fragment"

def test_paired_input_mismatch_is_fatal():
    row={"id":"a","sha256":"x","tokens":3,"exact":True,"error":None,"dataset":"d","language":"l"}
    with pytest.raises(ValueError,match="bytes"):paired_bootstrap([row],[row|{"sha256":"y"}])
    out=paired_bootstrap([row],[row|{"tokens":2}],repeats=20)
    assert out[0]["reduction_percent"]==pytest.approx(100/3)

def test_embedding_baselines_have_real_numeric_checks():
    matrix=np.array([[1,2],[3,4],[8,9]],dtype=np.float32)
    np.testing.assert_allclose(subtoken_average(matrix,[[0,1]]),[[2,3]])
    with pytest.raises(ValueError):subtoken_average(matrix,[[]])
    rng=np.random.default_rng(7);x=rng.normal(size=(200,5));w=rng.normal(size=(5,3));y=x@w+2
    model=ridge_mapping(x[:150],y[:150],regularization=1e-6)
    metrics=mapping_metrics(apply_mapping(x[150:],model),y[150:])
    assert metrics["coordinate_mse"]<1e-10 and metrics["mean_cosine"]>.99999

def test_range_reader_refuses_unbounded_response():
    class Response:
        status_code=200;raw=io.BytesIO(b"abcdefghijk")
        def raise_for_status(self):pass
        def close(self):pass
    class Session:
        def get(self,*a,**kw):return Response()
    with pytest.raises(IOError):HTTPRangeFile("https://example.test",100,Session(),block_size=4).read(4)
