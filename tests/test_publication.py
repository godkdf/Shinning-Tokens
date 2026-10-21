"""Distribution correctness: explicit scope, companions, hashes and defaults."""
import json
import tarfile
from pathlib import Path
import pytest
from tokenizer_lab import offline,publication
from tokenizer_lab.io import write_json,file_hash,sha256

def kit_inputs(root):
    assets=root/'.cache/assets/selected';assets.mkdir(parents=True)
    (assets/'ranks.model').write_bytes(b'YQ== 0\n')
    # Reference source is inert data, including in a selected portable kit.
    (assets/'tokenization.py').write_text('raise RuntimeError("Never execute upstream reference source")\n')
    spec={'id':'selected','kind':'tiktoken_asset','file':'ranks.model','sha256':file_hash(assets/'ranks.model'),
        'companions':[{'file':'tokenization.py','sha256':file_hash(assets/'tokenization.py')}]}
    write_json(root/'configs/registry.json',{'tokenizers':[spec]})
    write_json(root/'benchmarks/catalog.json',{'partitions':[]})
    return assets

def test_selected_kit_preserves_declared_reference_data_without_execution(tmp_path):
    assets=kit_inputs(tmp_path)
    *_,files=offline._preflight(tmp_path,['selected'],[],[])
    assert ('.cache/assets/selected/tokenization.py',file_hash(assets/'tokenization.py')) in files

def test_selected_kit_rejects_corrupt_required_companion(tmp_path):
    assets=kit_inputs(tmp_path);(assets/'tokenization.py').write_text('changed')
    with pytest.raises(ValueError,match='hash mismatch'):offline._preflight(tmp_path,['selected'],[],[])

def test_starter_selection_does_not_include_available_assets(tmp_path):
    kit_inputs(tmp_path)
    _,entries,_,parts,files=offline._preflight(tmp_path,[],[],[])
    assert entries==parts==files==[]

def test_portable_site_omits_recursive_and_stale_distribution_indexes(tmp_path):
    root=tmp_path/'source';kit=tmp_path/'kit';(root/'site/data').mkdir(parents=True)
    (root/'site/index.html').write_text('Local app')
    for name in ['offline-index.json','source-release.json','release-validation.json']:write_json(root/'site/data'/name,{'old':True})
    write_json(root/'site/data/hub.json.gz',{'profiles':[]})
    offline._copy_site(root,kit,set(),set(),set())
    assert (kit/'site/data/hub.json.gz').is_file()
    assert not (kit/'site/data/offline-index.json').exists()
    assert not (kit/'site/data/source-release.json').exists()

def source_fixture(root):
    (root/'site/data').mkdir(parents=True)
    (root/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0.2.3"\n')
    (root/'README.md').write_text('[Guide](docs/guide.md)\n')
    (root/'docs').mkdir();(root/'docs/guide.md').write_text('[Home](../README.md)\n')
    (root/'src').mkdir();(root/'src/main.py').write_text('print("public")\n')
    (root/'site/index.html').write_text('Current explorer')
    return root

def test_source_archive_excludes_payloads_private_state_and_unrelated_work(tmp_path):
    root=source_fixture(tmp_path)
    for name in ['.env','configs/custom-tokenizers.json','experiments/gemma-270m-tpu-pilot/private.md',
                 'benchmarks/fineweb/corpus.jsonl.gz','results/v0.4/runtime-input.json','results/v0.4/checkpoints/private.json']:
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('local data')
    result=publication.build_source(root)
    with tarfile.open(root/'site/data/source.tar.gz') as archive:
        names=archive.getnames();manifest=json.load(archive.extractfile('SOURCE-MANIFEST.json'))
        assert 'src/main.py' in names and 'docs/guide.md' in names
        assert not any('custom-tokenizers' in n or 'gemma-270m' in n or 'corpus.jsonl' in n or 'runtime-input' in n or 'checkpoints' in n for n in names)
        assert manifest['src/main.py']['sha256']==sha256(archive.extractfile('src/main.py').read())
    assert result['release']=='0.2.3' and result['bytes']<publication.STATIC_BUDGET

def test_local_markdown_links_fail_when_target_is_missing(tmp_path):
    root=source_fixture(tmp_path);assert publication.markdown_errors(root)==[]
    (root/'docs/guide.md').write_text('[Missing](absent.md)\n[Remote](https://example.com)\n')
    assert publication.markdown_errors(root)==[{'file':'docs/guide.md','target':'absent.md'}]

def test_starter_source_fingerprint_ignores_its_download_index_but_detects_docs(tmp_path):
    root=source_fixture(tmp_path)
    write_json(root/'site/data/offline-index.json',{'release':'old'})
    first=publication.kit_fingerprint(root)
    write_json(root/'site/data/offline-index.json',{'release':'new'})
    assert publication.kit_fingerprint(root)==first
    (root/'docs/guide.md').write_text('Changed launch instructions\n')
    assert publication.kit_fingerprint(root)!=first

def test_doctor_does_not_claim_an_uncached_native_encoding_is_installed(monkeypatch,capsys):
    from tokenizer_lab import cli,downloads
    import tiktoken
    monkeypatch.setattr(downloads,'catalog',lambda:{'tokenizers':[{'id':'uncached','kind':'tiktoken','cached':False}]})
    monkeypatch.setattr(tiktoken,'get_encoding',lambda *a:pytest.fail('doctor attempted a download-capable load'))
    cli.main(['doctor']);data=json.loads(capsys.readouterr().out)
    assert data['assets']==[{'id':'uncached','status':'missing or corrupt','sha256':None}]

def test_release_verification_detects_an_artifact_changed_after_packaging(tmp_path):
    root=source_fixture(tmp_path);folder=publication.destination(root);folder.mkdir(parents=True)
    path=folder/'selected.zip';path.write_bytes(b'original archive')
    write_json(folder/'release-manifest.json',{'release':'0.2.3','artifacts':[{'file':path.name,'bytes':path.stat().st_size,'sha256':file_hash(path)}]})
    assert publication.verify_release(root)['passed']
    path.write_bytes(b'changed archive')
    assert publication.verify_release(root)['failures']==['selected.zip']
