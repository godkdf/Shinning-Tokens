"""Exercise an actual native byte-level decoder without downloading assets."""
import json
import pytest
from tokenizers import Tokenizer,models,pre_tokenizers,decoders,normalizers
from tokenizer_lab.adapters import HFAdapter,BYTE_DECODER
from tokenizer_lab.conversion import convert

def native(tmp_path,normalize=False):
    vocab={char:i for i,char in enumerate(BYTE_DECODER)}
    tok=Tokenizer(models.BPE(vocab=vocab,merges=[]))
    tok.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=False)
    tok.decoder=decoders.ByteLevel()
    if normalize:tok.normalizer=normalizers.NFC()
    path=tmp_path/"tokenizer.json";tok.save(str(path))
    return HFAdapter({"id":"native","name":"Native fixture","role":"control"},path)

def test_native_whole_sequence_preserves_fragments_controls_and_utf8(tmp_path):
    adapter=native(tmp_path)
    text="你好 e\u0301 👩🏽‍💻\x00\n=Function"
    ids=adapter.encode(text)
    assert adapter.decode_bytes(ids)==text.encode()
    assert b"".join(adapter.token_bytes(i) for i in ids)==text.encode()
    assert convert(adapter,adapter,ids=ids)["exact"]
    partial=next(i for i in ids if adapter.token_bytes(i)==b"\xe4")
    with pytest.raises(UnicodeDecodeError):convert(adapter,adapter,ids=[partial])

def test_source_normalization_cannot_be_undone(tmp_path):
    adapter=native(tmp_path,normalize=True)
    result=convert(adapter,adapter,text="e\u0301")
    assert not result["source_roundtrip_exact"]
    assert result["source_decoded"]=="é"
    assert result["exact"]
