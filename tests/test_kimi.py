"""Pinned Kimi identities, native policies and text-only offline integration."""
import ast
import pytest
from tokenizer_lab import adapters,downloads
from tokenizer_lab.io import ROOT,read_json,file_hash
from tokenizer_lab.conversion import convert
from tokenizer_lab.transport import probe

IDS=read_json(ROOT/'configs/kimi-selection.json')['tokenizers']

@pytest.mark.parametrize('id',IDS)
def test_kimi_native_capacity_policy_and_transport(id,monkeypatch):
    spec=next(s for s in adapters.registry() if s['id']==id)
    if not adapters.asset_path(spec).exists():pytest.skip('Explicitly cache Kimi assets for integration tests')
    monkeypatch.setenv('TOKENIZER_LAB_OFFLINE','1')
    a=adapters.load_adapter(id,download=False)
    assert len(a.by_id)==spec['vocabulary_size']
    assert len(a.by_id)-len(a.special_ids)==163584
    assert len(a.special_ids)==(258 if id in ['moonlight','kimi_vl','kimi_k2_base','kimi_linear'] else 256)
    assert a.encode('Hello world!')==[19180,2695,0]
    assert a.encode('')==[]
    for text in ['Hello 你好 العربية नमस्ते e\u0301 👩🏽‍💻\x00\r\n','[BOS] <|im_end|> <think> <|media_begin|> <|end_of_msg|>','a'*25001,' '*25001]:
        ids=a.encode(text)
        assert a.decode_bytes(ids)==text.encode('utf-8')
        assert a.encode_batch([text])[0]==ids
    if id=='kimi_k2_base':assert not set(a.encode('[BOS]'))&a.special_ids
    else:
        assert a.encode('[BOS]')==[163584]
        with pytest.raises(ValueError,match='Special/control'):convert(a,a,text='[BOS]')
    ordinary='Hello 你好 العربية e\u0301 👩🏽‍💻'
    assert convert(a,a,text=ordinary)['exact']
    cases=probe(a)
    assert sum(p.get('exact',False) for p in cases)==10
    assert sum(p.get('rejected',False) for p in cases)==6

def test_kimi_selection_defaults_and_storage_scope():
    assert downloads.plan({})['files']==[]
    for id in IDS:
        plan=downloads.plan({'tokenizers':[id]})
        assert not plan['missing']
        assert all(f['kind']!='corpus' and f.get('tokenizer',id)==id for f in plan['files'])
        assert not any(f['path'].endswith(('.safetensors','.bin')) for f in plan['files'])

def test_kimi_aliases_require_equal_native_pipeline():
    catalog=read_json(ROOT/'configs/kimi-candidates.json')
    assert len(catalog['repositories'])==19
    assert len(catalog['included'])==16 and len(catalog['excluded'])==3
    specs={s['id']:s for s in adapters.registry()}
    for alias in catalog['included']:
        spec=specs[alias['profile']]
        assert alias['file_sha256']==spec['sha256']
        if alias['profile']!='qwen25':assert alias['native_pipeline_sha256']==spec['catalog_structural_sha256']
    assert len(specs['kimi_k25']['aliases'])==3
    assert len(specs['kimi_k2']['aliases'])==2
    assert any(a['repo']=='moonshotai/Kimi-Dev-72B' for a in specs['qwen25']['aliases'])

def test_kimi_recorded_regex_and_range_match_inspected_sources():
    specs={s['id']:s for s in adapters.registry()}
    for alias in read_json(ROOT/'configs/kimi-candidates.json')['included']:
        if alias['profile']=='qwen25':continue
        folder=ROOT/'.cache/kimi/assets'/alias['repo'].replace('/','--')
        source=next(folder.glob('tokenization_*.py'),None)
        if source is None:pytest.skip('Source inspection cache is absent')
        assert file_hash(source)==alias['wrapper_source_sha256']
        tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        pattern=next(n.value for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pat_str' for t in n.targets))
        spec=specs[alias['profile']]
        assert '|'.join(ast.literal_eval(pattern.args[0]))==spec['native_spec']['pat_str']
        encode=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='encode')
        assert ('all' if any(a.arg=='allow_special_tokens' for a in encode.args.args) else 'ordinary text')==spec['native_spec']['allowed_special']
        config=read_json(folder/'tokenizer_config.json')
        for id,token in config['added_tokens_decoder'].items():
            assert spec['native_spec']['special_tokens'][token['content']]==int(id)
