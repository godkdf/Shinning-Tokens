"""CPU correctness checks; no full Gemma training or TPU performance claim."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v39_builder", HERE / "build_notebook.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
NOTEBOOK, SOURCE = builder.build_notebook()
module = SimpleNamespace(__name__="v39_fixture")
sys.modules["v39_fixture"] = module
exec(compile(SOURCE, "V39", "exec"), module.__dict__)
R = module


def test_notebook_contract_and_embedding():
    values = builder.literals(NOTEBOOK)
    assert values["PROGRAM_SOURCE"] == SOURCE
    config = values["CONFIG"]
    R.validate_training_config(config)
    assert config["training_seconds"] == 1200
    assert config["sequence_length"] == 1024 and config["long_sequence_length"] == 2048
    assert config["local_batch_per_chip"] == 1 and config["batch_check_updates"] == 100
    assert config["memory_budget_fraction"] == .85
    assert config["learning_rate"] == .0003 and config["warmup_updates"] == 5
    assert "restore_archive" not in SOURCE
    for cell in NOTEBOOK["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert "pilot_V3_9.py" in "".join(NOTEBOOK["cells"][7]["source"])
    assert "colab-v5e1/V3.9" in "".join(NOTEBOOK["cells"][3]["source"])
    assert "gemma_colab_pilot_V3_9_reports.zip" in "".join(NOTEBOOK["cells"][15]["source"])


def test_expanded_recipe_and_independent_code_labels():
    recipe = R.recipe()
    assert recipe["training_bytes_target"] == 268435456
    assert len(R.LANGUAGES) == len(R.WIKI_CODES) == 16
    assert sum(b["bp"] for b in recipe["buckets"]) == 10000
    code = {b["domain"]: b["bp"] for b in recipe["buckets"] if b["domain"].startswith("code_")}
    assert code == dict(code_python=300, code_java=200, code_cpp=200, code_javascript=150, code_rust=100, code_html=50)
    assert sum(b["bp"] for b in recipe["buckets"] if b["domain"] == "multilingual_web") == 3000
    assert sum(b["bp"] for b in recipe["buckets"] if b["repo"] == R.SMOL) == 5000
    old = builder.literals(json.loads(builder.BASE.read_text()))["PROGRAM_SOURCE"]
    def dump(source, name):
        return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name))
    for name in ("make_model", "make_optimizer", "token_losses", "pack_chunks", "tokenize_split"):
        assert dump(SOURCE, name) == dump(old, name)


@pytest.mark.parametrize("tokens,mode", [(0,"short"),(2999999,"short"),(3000000,"long"),
    (3999999,"long"),(4000000,"short"),(6999999,"short"),(7000000,"long"),
    (8000000,"short"),(11000000,"long")])
def test_repeating_token_schedule(tokens, mode):
    assert R.context_phase(tokens, 3) == mode
    assert R.context_phase(tokens, 8) == mode
    assert R.context_phase(tokens, 2) == "short"
    assert R.context_phase(tokens, 1) == "short"


def test_switch_on_crossing_update_without_dropping_targets():
    prefix = np.arange(6, dtype=np.int64) * 1024
    assert R.bounded_dispatch(prefix, 0, 3, 16, 2995000, 3) == 2
    assert R.bounded_dispatch(prefix, 0, 3, 16, 3000000, 3) == 16
    assert R.bounded_dispatch(prefix, 0, 3, 16, 3999999, 3) == 1
    assert R.bounded_dispatch(prefix, 0, 2, 16, 2999999, 2) == 16
    # Nonfull masks use actual scored counts, not the nominal batch*length.
    nonfull = np.array([0, 8, 15, 20])
    assert R.bounded_dispatch(nonfull, 2, 2, 16, 2999990, 3) == 1


def test_memory_admission_and_throughput_hold():
    decide = R.batch_decision
    assert decide(1, 20000, None, 849, 1000, .85, 32)["action"] == "compile_candidate"
    assert decide(2, 25000, 20000, 851, 1000, .85, 32)["action"] == "hold"
    rollback = decide(3, 25200, 25000, 700, 1000, .85, 32)
    assert rollback["action"] == "rollback_and_hold" and rollback["batch"] == 2
    for projected, limit in ((None,1000),(800,None)):
        assert decide(1, 20000, None, projected, limit, .85, 32)["action"] == "hold"
    assert decide(32, 60000, None, 800, 1000, .85, 32)["action"] == "hold"
    analyses = {1: dict(argument_size_in_bytes=100, output_size_in_bytes=60,
        alias_size_in_bytes=60, temp_size_in_bytes=200)}
    assert R.compiler_peak(analyses[1]) == 300
    assert R.projected_candidate_peak(analyses, 2, 40) == 560


def test_code_repository_groups_cross_schema_and_provenance():
    assert R.canonical_group({"repo_name":"owner/repo"}, "x") == R.canonical_group({"repository_name":"owner/repo"}, "y")
    bucket = next(b for b in R.recipe()["buckets"] if b["key"] == "code_java")
    job = dict(kind="training", repo=R.STACK, buckets=[bucket])
    records = R.candidate_records(job, dict(content="class Main {}", repository_name="owner/repo",
        path="Main.java", lang="Java", licenses=["MIT"]))
    assert records[0]["text"] == "class Main {}"
    assert records[0]["domain"] == "code_java"
    assert records[0]["source"]["licenses"] == ["MIT"]
    assert records[0]["source_group"] == "code-repo:owner/repo"
    other = R.candidate_records(job, dict(content="class Second {}", repository_name="owner/repo", path="Second.java"))
    assert records[0]["split"] == other[0]["split"]


def test_row_order_and_prefix_are_batch_independent():
    rows = 7
    arrays = {key: np.zeros((rows, 4), dtype=np.int32) for key in R.INPUT_KEYS}
    arrays["tokens"][:,0] = np.arange(rows)
    arrays["mask"][:] = 1
    arrays["byte_weights"][:] = 2
    shuffled, digest = R.row_training_arrays(arrays, 42)
    assert sorted(shuffled["tokens"][:,0]) == list(range(rows))
    assert shuffled["tokens"].shape == (7,4)
    assert digest == R.row_training_arrays(arrays,42)[1]
    prefix = R.row_prefix_counts(shuffled)
    assert int(R.completed_prefix(prefix["tokens"], 10)) == 40
    assert int(R.completed_prefix(prefix["domain_bytes"], 10).sum()) == 80
    assert R.tokenization_cache_key("a","b",1024) != R.tokenization_cache_key("a","b",2048)


def test_real_jax_state_and_adam_continuity_across_batch_and_context(monkeypatch):
    R.load_model_helpers(cpu_test=True)
    jax, jnp, optax = R.jax, R.jnp, R.optax
    monkeypatch.setattr(R, "token_losses", lambda model, params, batch: (params["w"] - batch["targets"])**2)
    optimizer = optax.adam(.01)
    params = {"w": jnp.array(0.)}
    state = R.PilotState(jnp.array(0,jnp.int32),params,optimizer.init(params),jax.random.PRNGKey(42),
        jnp.array(0,jnp.int32),jnp.array(0,jnp.int32),jnp.array(0,jnp.int32))
    short = dict(tokens=jnp.zeros((5,2),jnp.int32), targets=jnp.arange(10,dtype=jnp.float32).reshape(5,2),mask=jnp.ones((5,2),bool))
    long = dict(tokens=jnp.zeros((3,4),jnp.int32), targets=jnp.arange(12,dtype=jnp.float32).reshape(3,4),mask=jnp.ones((3,4),bool))
    # Includes short wraparound, a long interval and resuming the short cursor.
    expected_params, expected_opt = params, optimizer.init(params)
    short_cursor = long_cursor = 0
    for batch, mode, count in ((1,"short",2),(3,"short",2),(1,"long",2),(2,"short",2)):
        data = short if mode == "short" else long
        cursor = short_cursor if mode == "short" else long_cursor
        for _ in range(count):
            indices=(np.arange(batch)+cursor)%data["tokens"].shape[0]
            target=data["targets"][indices]
            grad=jax.grad(lambda p:jnp.mean((p["w"]-target)**2))(expected_params)
            delta,expected_opt=optimizer.update(grad,expected_opt,expected_params)
            expected_params=optax.apply_updates(expected_params,delta)
            cursor+=batch
        if mode=="short": short_cursor=cursor
        else: long_cursor=cursor
        _,group=R.make_training_functions(None,optimizer,batch,mode)
        state,metrics=jax.block_until_ready(group(state,data,jnp.array(count,jnp.int32)))
        assert int(metrics["updates"]) == count
        np.testing.assert_allclose(state.params["w"],expected_params["w"],rtol=1e-6)
        assert int(state.short_rows)==short_cursor and int(state.long_rows)==long_cursor
        assert int(state.opt_state[0].count)==int(state.step)
    assert int(state.step)==8 and int(state.data_position)==8
    assert int(state.short_rows)==12 and int(state.long_rows)==2
    # Rejected nonfinite update changes neither weights, optimizer nor cursor.
    bad=dict(short,targets=jnp.full((5,2),jnp.nan))
    _,group=R.make_training_functions(None,optimizer,1,"short")
    state,metrics=jax.block_until_ready(group(state,bad,jnp.array(1,jnp.int32)))
    assert not bool(metrics["finite"]) and int(metrics["updates"])==0
    assert int(state.step)==8 and int(state.short_rows)==12


def test_exclusive_creation_and_dependency_lock_unchanged(tmp_path,monkeypatch):
    existing=tmp_path/"delivered.ipynb"
    existing.write_text("preserve")
    monkeypatch.setattr(builder,"TARGET",existing)
    with pytest.raises(FileExistsError): builder.main()
    assert existing.read_text()=="preserve"
    old=json.loads(builder.BASE.read_text())
    def dependencies(nb):
        return {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse("".join(nb["cells"][3]["source"])).body
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)
            and node.targets[0].id in ("REQUIREMENTS_INPUT","RESOLVED_DEPENDENCY_LOCK")}
    assert dependencies(old)==dependencies(NOTEBOOK)


def test_full_controller_with_deterministic_device_and_clock(tmp_path,monkeypatch):
    class Clock:
        now=0.
        def monotonic(self): return self.now
    clock=Clock()
    class Device:
        def memory_stats(self):
            return dict(bytes_limit=1000000000,bytes_in_use=10000000,peak_bytes_in_use=10000000)
    fake_jax=SimpleNamespace(devices=lambda:[Device()],block_until_ready=lambda value:value)
    monkeypatch.setattr(R,"time",clock)
    monkeypatch.setattr(R,"jax",fake_jax,raising=False)
    monkeypatch.setattr(R,"jnp",np,raising=False)
    config=dict(training_seconds=600.,updates_per_dispatch=16,memory_budget_fraction=.85,
        safety_batch_cap=5,other_resident_bytes_short=1000)
    state=SimpleNamespace(step=0,data_position=0,short_rows=0,long_rows=0)
    training={mode:{"tokens":np.zeros((7,length),np.int32)} for mode,length in (("short",1024),("long",2048))}
    prefixes={mode:{"tokens":np.arange(8,dtype=np.int64)*length} for mode,length in (("short",1024),("long",2048))}
    functions,analyses={},{}
    compile_records=[]
    def compile_shape(mode,batch,state,scope):
        if scope=="inside_training_budget":
            clock.now+=7.
            compile_records.append((int(state.step),batch,clock.now))
        def group(state,data,requested):
            steps=int(requested)
            length=data["tokens"].shape[1]
            clock.now+=steps*(.04+.01*batch)
            updated=SimpleNamespace(**vars(state))
            updated.step+=steps
            updated.data_position+=steps
            field="short_rows" if mode=="short" else "long_rows"
            setattr(updated,field,getattr(updated,field)+batch*steps)
            tokens=steps*batch*length
            return updated,dict(updates=steps,nll_sum=tokens*2.,tokens=tokens,max_gradient_norm=1.,finite=True)
        functions[(mode,batch)]=group
        analysis=dict(argument_size_in_bytes=100000,output_size_in_bytes=10000,
            alias_size_in_bytes=10000,temp_size_in_bytes=batch*200000)
        if mode=="short": analyses[batch]=analysis
        return analysis
    compile_shape("short",1,state,"before_training")
    compile_shape("long",1,state,"before_training")
    state,result=R.adaptive_train(state,training,prefixes,compile_shape,functions,analyses,config,tmp_path)
    rows=result["rows"]
    assert all(row["batch"]==1 and row["mode"]=="short" for row in rows if row["update"]<=100)
    assert all(check["at_update"]%100==0 for check in result["checks"])
    assert result["tokens"]==sum(row["tokens"] for row in rows)
    assert state.step==result["updates"]==sum(row["updates"] for row in rows)
    assert state.short_rows==result["rows_used"]["short"]
    assert state.long_rows==result["rows_used"]["long"]
    assert all(1<=row["updates"]<=16 for row in rows)
    assert all(row["start_seconds"]<600 for row in rows)
    assert all(row["mode"]==R.context_phase(row["total_tokens"]-row["tokens"],row["short_batch"]) for row in rows)
    assert len(result["transitions"])>=5
    assert result["short_batch"]==5
    assert [record[0] for record in compile_records]==[100,200,300,400]
    assert [record[1] for record in compile_records]==[2,3,4,5]
    assert result["training_seconds"]>=600
    assert sum(row["dispatch_seconds"] for row in rows)+7*len(compile_records)==pytest.approx(result["training_seconds"])
    for transition in result["transitions"][1:]:
        position=transition["tokens"]%4000000
        overshoot=position-(3000000 if transition["mode"]=="long" else 0)
        assert 0<=overshoot<5*1024
    assert (tmp_path/"batch_checks.jsonl").exists()


def test_source_job_coverage_uses_all_16_languages(monkeypatch):
    fake=SimpleNamespace(get_dataset_config_names=lambda repo,revision:[code+"wiki" for code in R.WIKI_CODES])
    monkeypatch.setitem(sys.modules,"datasets",fake)
    jobs=R.source_jobs({repo:"a"*40 for repo in R.SOURCE_REPOS})
    assert len(jobs)==43
    assert [job["language"] for job in jobs[:16]]==R.LANGUAGES
    assert len({job["key"] for job in jobs})==43
    assert sum(sum(v for k,v in job["targets"].items() if k.startswith("train:")) for job in jobs)==pytest.approx(256*1024**2,abs=20)


def test_gated_source_preflight_fails_before_download(monkeypatch):
    class API:
        def list_repo_files(self,*args,**kwargs):
            return [f"data/{language}/data.json" for language in ("java","c++","javascript","rust","html")]
    def denied(*args,**kwargs): raise PermissionError("gated")
    fake=SimpleNamespace(HfApi=API,get_hf_file_metadata=denied,hf_hub_url=lambda *args,**kwargs:"https://example.test/file")
    monkeypatch.setitem(sys.modules,"huggingface_hub",fake)
    with pytest.raises(RuntimeError,match="Accept its access conditions"):
        R.check_source_access({R.STACK:"a"*40})
