import gzip
import io
import tarfile
from pathlib import Path
import pytest
from tokenizer_lab import downloads,engine,adapters
from tokenizer_lab.io import write_json,file_hash,sha256
from tokenizer_lab.transport import decode_input,transport_sizes,probe

@pytest.fixture
def selection_root(tmp_path,monkeypatch):
    for module in (downloads,engine,adapters):monkeypatch.setattr(module,'ROOT',tmp_path)
    write_json(tmp_path/'configs/registry.json',{'tokenizers':[{'id':'one','name':'One','kind':'local','file':'artifacts/one.json','role':'custom','sha256':sha256(b'one'),'bytes':3},{'id':'two','name':'Two','kind':'local','file':'artifacts/two.json','role':'custom','sha256':sha256(b'two'),'bytes':3}]})
    (tmp_path/'artifacts').mkdir();(tmp_path/'artifacts/one.json').write_bytes(b'one');(tmp_path/'artifacts/two.json').write_bytes(b'two')
    write_json(tmp_path/'benchmarks/catalog.json',{'datasets':[],'partitions':[]})
    return tmp_path

def test_empty_selection_has_zero_payloads(selection_root):
    p=downloads.plan({});assert p['files']==[];assert p['download_bytes']==p['total_bytes']==0

def test_partial_tokenizer_plan_and_removal_only_touch_selection(selection_root):
    p=downloads.plan({'tokenizers':['one']});assert len(p['files'])==1;assert p['cached_bytes']==3
    r=downloads.remove({'tokenizers':['one']});assert r['freed_bytes']==3
    assert (selection_root/'artifacts/two.json').read_bytes()==b'two'
    assert not (selection_root/'artifacts/one.json').exists()

def test_corrupt_payload_is_not_cached(selection_root):
    (selection_root/'artifacts/one.json').write_bytes(b'bad')
    assert not downloads.plan({'tokenizers':['one']})['files'][0]['cached']

@pytest.mark.parametrize('value',[{'tokenizers':['absent']},{'tokenizers':'one'},{'languages':['absent']},{'datasets':['absent']}])
def test_bad_selections_are_rejected(selection_root,value):
    with pytest.raises(ValueError):downloads.plan(value)

@pytest.mark.parametrize('encoding',['utf-16-le','utf-16-be'])
def test_utf16_supplementary_units_and_rejected_surrogates(encoding):
    text='A😀';assert transport_sizes(text)['utf16_code_units']==3
    assert decode_input(text.encode(encoding),encoding)==text
    with pytest.raises(UnicodeDecodeError):decode_input(b'x',encoding)
    with pytest.raises(UnicodeDecodeError):decode_input(('\ud800').encode(encoding,errors='surrogatepass'),encoding)

def test_cached_only_native_encoding_never_calls_remote(monkeypatch,tmp_path):
    import tiktoken
    monkeypatch.setattr(adapters,'ROOT',tmp_path)
    monkeypatch.setattr(tiktoken,'get_encoding',lambda _:pytest.fail('Unexpected network-capable fallback'))
    with pytest.raises(FileNotFoundError):adapters.TikAdapter({'id':'missing','kind':'tiktoken'},download=False)

def test_restoration_checks_hash_before_installing_and_leaves_unselected(selection_root,monkeypatch):
    target=selection_root/'site/data/text/test/eng.json.gz'
    item={'id':'test:eng','kind':'corpus','dataset':'test','language':'eng','path':str(target.relative_to(selection_root)),'sha256':sha256(b'wrong'),'cached':False}
    plan=selection_root/'job/plan.json';write_json(plan,{'files':[item]})
    monkeypatch.setattr(downloads,'_replay_corpus',lambda *args:[{'text':'selected'}])
    downloads.worker(plan,plan.parent/'status.json')
    assert not target.exists();assert (selection_root/'artifacts/two.json').exists()
    assert __import__('json').loads((plan.parent/'status.json').read_text())['status']=='failed'

def test_utf16_transport_probe_counts_all_cases():
    class Identity:
        encode=lambda self,text:list(text.encode())
        decode_bytes=lambda self,ids:bytes(ids)
    rows=probe(Identity());assert len(rows)==16
    assert sum(r.get('exact',False) for r in rows)==10
    assert sum(r.get('rejected',False) for r in rows)==6
