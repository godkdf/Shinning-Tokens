"""Reference IDs, independent encoders, transport and selective offline loading."""
import base64
import pytest
from tokenizer_lab import adapters,downloads
from tokenizer_lab.io import ROOT,read_json,write_json,sha256
from tokenizer_lab.conversion import convert
from tokenizer_lab.transport import probe

@pytest.fixture(scope="module")
def references():
    # Public CI can run unit tests without implicitly downloading licensed assets.
    specs={s['id']:s for s in adapters.registry()}
    if any(not adapters.asset_path(specs[i]).exists() for i in ['gpt2','llama3']):
        pytest.skip('Explicitly cache GPT-2 and Llama 3 to run reference integration checks')
    return {i:adapters.load_adapter(i,download=False) for i in ['gpt2','llama3']}

def test_published_reference_ids_and_capacity(references):
    g,l=references['gpt2'],references['llama3']
    assert g.encode('Hello world!')==[15496,995,0]
    assert l.encode('Hello world!')==[9906,1917,0]
    assert (len(g.by_id),len(g.special_ids))==(50257,1)
    assert (len(l.by_id),len(l.special_ids))==(128256,256)
    assert l.tokenizer._special_tokens['<|begin_of_text|>']==128000
    assert l.tokenizer._special_tokens['<|eot_id|>']==128009

@pytest.mark.parametrize('text',['','你好 العربية नमस्ते','e\u0301 é\x00\r\n\t','👩🏽‍💻\U0010ffff','<|endoftext|> <|begin_of_text|> <|eot_id|>','000123456789 +1.2e-3'])
def test_reference_roundtrip_and_bidirectional_conversion(references,text):
    for source,target in [('gpt2','llama3'),('llama3','gpt2')]:
        a,b=references[source],references[target]
        ids=a.encode(text)
        assert not set(ids)&a.special_ids
        assert a.decode_bytes(ids)==text.encode('utf-8')
        assert a.encode_batch([text])[0]==ids
        result=convert(a,b,text=text)
        assert result['exact'] and result['source_roundtrip_exact']
        for encoding in ['utf-16-le','utf-16-be']:
            assert a.decode_bytes(ids).decode().encode(encoding)==text.encode(encoding)

def test_llama_independent_huggingface_encoder_parity(references):
    from tokenizers import Tokenizer
    l=references['llama3'];raw=read_json(l.path)
    # Remove model-specific automatic special handling to compare ordinary text.
    raw['added_tokens']=[];raw['post_processor']=None
    independent=Tokenizer.from_str(__import__('json').dumps(raw))
    for text in ['Hello world!','e\u0301 你好 العربية नमस्ते','\x00\r\n\t👩🏽‍💻','1234567890 <|eot_id|>']:
        assert independent.encode(text,add_special_tokens=False).ids==l.encode(text)

def test_reference_long_input_boundaries(references):
    l=references['llama3'];encode=l.tokenizer.encode_ordinary
    for text,segments in [
        ('a'*25001,['a'*25000,'a']),
        (' '*25001,[' '*25000,' ']),
        ('a '*200001,['a '*200000,'a ']),
        ('x'*25000+' '+'y'*25001,['x'*25000+' '+'y'*25000,'y']),
    ]:
        assert l.encode(text)==[i for part in segments for i in encode(part)]
        assert l.decode_bytes(l.encode(text))==text.encode()

def test_reference_malformed_transport_and_special_id_rejection(references):
    for a in references.values():
        cases=probe(a)
        assert len(cases)==16 and sum(p.get('rejected',False) for p in cases)==6
        with pytest.raises(ValueError,match='Special/control'):
            convert(a,a,ids=[min(a.special_ids)])

def test_json_rank_identity_is_checked(tmp_path):
    path=tmp_path/'tokenizer.json'
    write_json(path,{'model':{'vocab':{'a':0}},'added_tokens':[]})
    native={'asset_format':'huggingface-bytelevel-json','reference_ranks_sha256':sha256(base64.b64encode(b'a')+b' 0\n')}
    assert adapters.native_ranks(path,native)=={b'a':0}
    write_json(path,{'model':{'vocab':{'b':0}},'added_tokens':[]})
    with pytest.raises(ValueError,match='reference artifact'):adapters.native_ranks(path,native)

def test_reference_selection_downloads_only_requested_tokenizer():
    assert downloads.plan({})['files']==[]
    for id in ['gpt2','llama3']:
        plan=downloads.plan({'tokenizers':[id]})
        assert plan['missing']==[]
        assert all(f.get('tokenizer',id)==id and f['kind']!='corpus' for f in plan['files'])
        assert all(id in f['path'] for f in plan['files'])

def test_cached_reference_loading_never_calls_network(references,monkeypatch):
    import tiktoken
    import huggingface_hub
    def forbidden(*args,**kwargs):pytest.fail('Cached reference attempted a download')
    monkeypatch.setattr(tiktoken,'get_encoding',forbidden)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',forbidden)
    monkeypatch.setenv('TOKENIZER_LAB_OFFLINE','1')
    for id in references:assert adapters.load_adapter(id,download=False).encode('Hello world!')==references[id].encode('Hello world!')

def test_site_rebuild_keeps_latest_results_and_updates_csv(tmp_path,monkeypatch):
    from tokenizer_lab import cli
    import csv,gzip,io
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    write_json(tmp_path/'research/methods.json',{})
    for version,id in [('v0.1','old'),('v0.3','llama3'),('v0.4','kimi_k3')]:
        write_json(tmp_path/f'results/{version}/hub.json',{'profiles':[{'id':id}],
            'benchmarks':[{'tokenizer':id,'documents':1}],'checks':{'observations':1}})
    write_json(tmp_path/'site/data/hub.json',{'profiles':[{'id':'stale'}]})
    cli.main(['build-site'])
    assert read_json(tmp_path/'site/data/hub.json.gz')['profiles']==[{'id':'kimi_k3'}]
    assert not (tmp_path/'site/data/hub.json').exists()
    rows=list(csv.DictReader(io.StringIO(gzip.decompress((tmp_path/'site/data/benchmarks.csv.gz').read_bytes()).decode())))
    assert rows[0]['tokenizer']=='kimi_k3'
