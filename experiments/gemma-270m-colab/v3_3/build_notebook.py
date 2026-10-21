"""Assemble the corrected V3.2 artifacts under the requested V3.3 name.

Artifact construction only; no notebook execution, local checks or tests.
"""
import json
from pathlib import Path
from splash_construction_fix import fix_splash_construction

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PREVIOUS = PARENT / "v3_2"


def relabel(text):
    # Preserve shared corpus, dependency, environment and compiler-cache names.
    return text.replace("V3.2", "V3.3").replace("V32-", "V33-").replace("V3_2", "V3_3")


def preserve_attempt_evidence(source):
    source = source.replace('    environment = dict(os.environ)\n', '''    if phase == "train" and (OUTPUT / "training.started").exists():
        result_path = OUTPUT / "result.json"
        if result_path.exists():
            print("This attempt already produced result.json; showing it without retraining.")
            print(result_path.read_text())
            return
        raise RuntimeError("A previous attempt owns this run folder. Original logs were preserved. Inspect result.json, attempt_status.json and train.log before another paid attempt; do not delete training.started blindly.")
    environment = dict(os.environ)
''', 1)
    source = source.replace('    environment = preflight()\n    load_model_helpers()', '''    write_json(output / "attempt_status.json", {"revision": "V3.3-attempt-evidence-1",
        "stage": "pretraining", "optimizer_updates": 0, "timed_training_started": False})
    environment = preflight()
    load_model_helpers()''', 1)
    source = source.replace('    print(f"Initial FineWiki macro BPB=', '''    write_json(output / "timed_training.started.json", {"revision": "V3.3-attempt-evidence-1",
        "stage": "timed training", "training_seconds": config["training_seconds"]})
    write_json(output / "attempt_status.json", {"revision": "V3.3-attempt-evidence-1",
        "stage": "timed training", "timed_training_started": True, "optimizer_updates": None})
    print(f"Initial FineWiki macro BPB=''', 1)
    source = source.replace('    write_json(output / "result.json", result)', '''    write_json(output / "result.json", result)
    write_json(output / "attempt_status.json", {"revision": "V3.3-attempt-evidence-1",
        "stage": "result saved", "timed_training_started": True, "optimizer_updates": updates})''', 1)
    source = source.replace('    except Exception:\n        write_json(args.output / "failure.json",', '''    except Exception:
        if (args.output / "failure.json").exists():
            write_json(args.output / ("failure-secondary-" + str(time.time_ns()) + ".json"),
                {"phase": args.phase, "error": traceback.format_exc(), "automatic_retry": False})
            raise
        write_json(args.output / "failure.json",''', 1)
    source = source.replace('"dispatch_preparation.json", "dependency_source.json"',
        '"dispatch_preparation.json", "attempt_status.json", "timed_training.started.json", "dependency_source.json"')
    return source


notebook = json.loads((PARENT / "gemma_270m_colab_v5e1_V3_2.ipynb").read_text())
runner = preserve_attempt_evidence(fix_splash_construction(relabel((PREVIOUS / "runner.py").read_text())))
for cell in notebook["cells"]:
    text = preserve_attempt_evidence(relabel("".join(cell["source"])))
    if text.startswith("PROGRAM_SOURCE = "):
        text = "PROGRAM_SOURCE = " + repr(runner) + text[text.index("\nCONFIG = "):]
        text = text.replace("'implementation_revision': 'V3.3-jit-dispatch-1'",
                            "'implementation_revision': 'V3.3-splash-construction-1'")
    cell["source"] = text.splitlines(True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
notebook["cells"][0]["source"] += ["\n**Splash construction repair:** both fixed-shape attention kernels (global and local, 1,024 positions, 4 heads) are constructed in ordinary Python before verification, compilation or training. The factory no longer uses `lru_cache` inside JAX tracing. This addresses the supplied `UnexpectedTracerError`, whose traceback showed a cached intermediate escaping the verification transformation. The model, data, optimizer, training budget and shared caches remain unchanged. No local checks/tests were run; TPU verification remains pending.\n"]
notebook["metadata"]["colab"] = {"name": "gemma_270m_colab_v5e1_V3_3.ipynb", "provenance": []}
(PARENT / "gemma_270m_colab_v5e1_V3_3.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
(HERE / "runner.py").write_text(runner)
for name in ("parallel_data.py", "loss_dtype_fix.py", "REQUIREMENTS.md", "DATA_MIXTURE.md"):
    (HERE / name).write_text(relabel((PREVIOUS / name).read_text()))
(HERE / "README.md").write_text("""# Gemma Colab V3.3

Open `../gemma_270m_colab_v5e1_V3_3.ipynb` in Colab Pro on one v5e-1 TPU and Run all.

V3.3 is the requested name for the corrected V3.2 build. It retains the fused-loss gradient dtype repair and JIT dispatch repair, including zero-update training warmup before timing. The model, optimizer, data mixture and single 60-second training attempt are unchanged.

Attempt evidence repair: the notebook refuses a repeated train call before opening logs. Existing results are shown without retraining. A pretraining status and a separate timed-training marker distinguish preparation from model updates in future runs. Earlier failure.json is preserved when another exception occurs. Old V3.3 start markers alone cannot prove whether an earlier attempt trained; they must not be blindly removed.

Splash construction repair: the latest supplied original traceback identified a tracer escaping from an lru-cached kernel factory during the verification JIT. Both fixed-shape kernels are now constructed once in ordinary Python before transformations; subsequent calls perform only a shape check and dictionary lookup. No transformed intermediate is stored in a cache. The first reported 222/209 mismatch may have been related, but that connection has not been independently confirmed.

The notebook title, experiment version, requirement IDs, runtime source filename, report archive name and new Drive output folder identify V3.3. Shared corpus, prefix, dependency, installed environment and compiler cache paths keep their existing names for reuse.

`build_notebook.py` constructs these artifacts from corrected V3.2 without executing notebook code. No local checks or tests were run. TPU verification remains pending the user's execution.
""")
print("Created V3.3 notebook and source artifacts. No notebook execution, local checks or tests.")
