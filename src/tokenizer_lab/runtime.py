"""Isolated native-tokenizer runtime measurement, separate from inventory analysis."""
from __future__ import annotations
import gc
import importlib.metadata
import os
import platform
import statistics
import threading
import time
from .adapters import registry, resolve_asset

def hardware():
    return {"system":platform.system(),"release":platform.release(),"machine":platform.machine(),"processor":platform.processor(),
            "logical_cpus":os.cpu_count(),"python":platform.python_version(),"threads":1,
            "timing_note":"Same host, native tokenizer only; one warm-up then timed repeats. No model inference."}

def measure_runtime(tokenizer_id,texts,*,repeats=5):
    import psutil
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    process=psutil.Process();baseline=process.memory_info().rss;spec=next(x for x in registry() if x["id"]==tokenizer_id)
    start=time.perf_counter()
    if spec["kind"]=="tiktoken":
        import tiktoken
        tok=tiktoken.get_encoding(spec["encoding"])
        encode=lambda values:tok.encode_ordinary_batch(values,num_threads=1)
        decode=lambda values:[tok.decode_bytes(ids) for ids in values]
    else:
        from tokenizers import Tokenizer
        tok=Tokenizer.from_file(str(resolve_asset(spec)));tok.no_padding();tok.no_truncation()
        encode=lambda values:[x.ids for x in tok.encode_batch(values,add_special_tokens=False)]
        decode=lambda values:[x.encode("utf-8") for x in tok.decode_batch(values,skip_special_tokens=False)]
    load=time.perf_counter()-start;gc.collect();loaded=process.memory_info().rss
    peak=[loaded];stop=threading.Event()
    def sample():
        while not stop.wait(.005):peak[0]=max(peak[0],process.memory_info().rss)
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    try:
        ids=encode(texts);decoded=decode(ids)
        durations_encode=[];durations_decode=[]
        for _ in range(repeats):
            start=time.perf_counter();ids=encode(texts);durations_encode.append(time.perf_counter()-start)
            start=time.perf_counter();decoded=decode(ids);durations_decode.append(time.perf_counter()-start)
        peak[0]=max(peak[0],process.memory_info().rss)
    finally:stop.set();thread.join()
    size=sum(len(x.encode()) for x in texts);decoded_size=sum(len(x) for x in decoded)
    return {"tokenizer":tokenizer_id,"load_seconds":load,"baseline_rss_bytes":baseline,"loaded_rss_bytes":loaded,
            "incremental_loaded_rss_bytes":loaded-baseline,"peak_rss_bytes":peak[0],"encode_seconds":durations_encode,"decode_seconds":durations_decode,
            "encode_mbps":size/1e6/statistics.median(durations_encode),"decode_mbps":decoded_size/1e6/statistics.median(durations_decode),
            "utf8_bytes":size,"documents":len(texts),"repeats":repeats,"roundtrip_exact_documents":sum(a.encode()==b for a,b in zip(texts,decoded)),
            "hardware":hardware(),"memory_scope":"Fresh-process native tokenizer, excluding vocabulary analysis; process RSS and incremental RSS both reported. Peak sampled every 5 ms."}
