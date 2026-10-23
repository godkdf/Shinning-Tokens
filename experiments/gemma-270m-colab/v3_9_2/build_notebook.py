"""End the batch controller permanently on hold; retain token context switches."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "gemma_270m_colab_v5e1_V3_9_1.ipynb"
TARGET = HERE.parent / "gemma_270m_colab_v5e1_V3_9_2.ipynb"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one anchor; found {source.count(old)}: {old[:100]}")
    return source.replace(old, new, 1)


def embedded_literals(notebook):
    cell = next(c for c in notebook["cells"] if "PROGRAM_SOURCE =" in "".join(c["source"]))
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse("".join(cell["source"])).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("PROGRAM_SOURCE", "CONFIG")}


def build_source(original):
    source = original.replace("V3.9.1", "V3.9.2")
    source = once(source,
        '                requested = min(config["updates_per_dispatch"], 100 - updates % 100)',
        '                requested = config["updates_per_dispatch"] if held else min(\n'
        '                    config["updates_per_dispatch"], 100 - updates % 100)')
    source = once(source,
        '                if mode == "short":\n                    window["updates"] += completed',
        '                if mode == "short" and not held:\n                    window["updates"] += completed')
    source = once(source, '                if updates % 100 == 0:',
        '                if not held and updates % 100 == 0:')
    source = once(source,
        '                    elif held:\n'
        '                        decision = {"action": "hold", "batch": batch, "reason": "Ramp stopped at a prior memory/throughput/cap boundary; continue real training."}\n', '')
    source = once(source, '                    check.update(decision)',
        '                    check.update(decision)\n                    check["ramp_locked"] = held')
    source = once(source,
        '        "short_batch": batch, "training_seconds": seconds, "last_start": last_start,',
        '        "short_batch": batch, "training_seconds": seconds, "last_start": last_start,\n'
        '        "ramp": {"locked": held, "final_short_batch": batch, "checks_recorded": len(decisions),\n'
        '            "lock_update": next((check["at_update"] for check in decisions if check["ramp_locked"]), None),\n'
        '            "lock_reason": next((check["reason"] for check in decisions if check["ramp_locked"]), None),\n'
        '            "policy": "After hold or rollback: zero further ramp checks, candidate compiles, timing windows or 100-update dispatch boundaries."},')
    source = once(source,
        '        "context_rows_consumed": adaptive["rows_used"], "batch_checks": adaptive["checks"],',
        '        "context_rows_consumed": adaptive["rows_used"], "batch_checks": adaptive["checks"],\n'
        '        "batch_ramp": adaptive["ramp"],')
    ast.parse(source)
    return source


def markdown(text):
    return {"cell_type": "markdown", "id": hashlib.sha256(text.encode()).hexdigest()[:8],
        "metadata": {}, "source": text.splitlines(True)}


def build_notebook():
    notebook = json.loads(BASE.read_text())
    values = embedded_literals(notebook)
    source = build_source(values["PROGRAM_SOURCE"])
    config = values["CONFIG"]
    config.update(version="V3.9.2", experiment="GEM-COLAB-V3.9.2",
        implementation_revision="V3.9.2-stop-locked-batch-controller-1")
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if "PROGRAM_SOURCE =" in text:
            tail = text[text.index('CONFIG["tokenizer_reference"]'):]
            text = "PROGRAM_SOURCE = " + repr(source) + "\nCONFIG = " + repr(config) + "\n" + tail
        text = text.replace("V3.9.1", "V3.9.2").replace("V3_9_1", "V3_9_2")
        cell["source"] = text.splitlines(True)
        if cell["cell_type"] == "code":
            ast.parse(text)
            cell.update(outputs=[], execution_count=None)
    notebook["metadata"]["colab"]["name"] = TARGET.name
    intro = "".join(notebook["cells"][0]["source"])
    intro = once(intro, 'Start at **1 × 1,024 for the first 100 updates**. At every **100 official\nupdates**, inspect timing and memory.',
        'Start at **1 × 1,024 for the first 100 updates**. While the ramp is active,\ninspect timing and memory every **100 official updates**.')
    intro = once(intro, 'Checks continue every 100 updates after holding. Long-context checks defer\nshort-batch decisions until enough short-context updates are available.',
        'The first **hold or rollback permanently stops the ramp controller**. No\n'
        'further batch checks, `hold`/`defer` messages, candidate compilations or\n'
        '100-update dispatch truncations occur. While the ramp is still active,\n'
        'long-context checks defer decisions until enough short updates are available.\n'
        'Memory sampling, finite checks and token-based context switching remain active.')
    notebook["cells"][0] = markdown(intro)
    contract = "".join(notebook["cells"][1]["source"])
    contract = once(contract, '- First 100 updates: 1 × 1,024; checks land exactly on multiples of 100 updates.',
        '- First 100 updates: 1 × 1,024; active-ramp checks land on multiples of 100.\n'
        '- First hold or rollback locks the short batch and ends ramp checks/windows,\n'
        '  candidate compilation and 100-update dispatch boundaries for the rest of the run.\n'
        '  Token context switches and regular memory/finite checks continue.')
    notebook["cells"][1] = markdown(contract)
    train = "".join(notebook["cells"][10]["source"])
    train = once(train, 'Check every 100 official updates;', 'While the ramp is active, check every 100 official updates;')
    train += '\nAfter the first hold or rollback, lock the short batch for the rest of the run and stop batch checks entirely. Continue 3M/1M context switching and normal telemetry.\n'
    notebook["cells"][10] = markdown(train)
    display = "".join(notebook["cells"][13]["source"])
    display += '\nprint("Batch ramp final state:", json.dumps(result["batch_ramp"], indent=2))\n'
    notebook["cells"][13]["source"] = display.splitlines(True)
    return notebook, source


def main():
    if TARGET.exists() or (HERE / "runner.py").exists():
        raise FileExistsError("Preserving delivered notebooks; create another version.")
    notebook, source = build_notebook()
    with (HERE / "runner.py").open("x") as stream:
        stream.write(source)
    with TARGET.open("x") as stream:
        json.dump(notebook, stream, indent=1, ensure_ascii=False)
        stream.write("\n")
    print(TARGET)


if __name__ == "__main__":
    main()
