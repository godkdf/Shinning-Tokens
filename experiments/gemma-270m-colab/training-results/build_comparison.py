"""Rebuild training tables from checked-in evidence; never starts training."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
CATALOG = [
    ("Gemma 3", "V4", 262145, 268100736, "original", "cell_output"),
    ("Gemma 4", "V4.1.1", 262144, 268100096, "original", "machine_report"),
    ("o200k_base", "V4.2.2", 200000, 228327936, "original", "missing"),
    ("BLOOM", "V4.3.1", 250680, 260763136, "original", "machine_report"),
    ("Llama 3", "V4.4.1", 128256, 182411776, "original", "cell_output"),
    ("Qwen3", "V4.5.1", 151669, 197396096, "NFC", "cell_output"),
    ("DeepSeek-V3-Base", "V4.6.1", 128815, 182769536, "original", "cell_output"),
]


def number(text):
    return float(text.replace(",", "").rstrip("%"))


def cell_record(path):
    text = path.read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    def value(label):
        return lines[next(i for i, line in enumerate(lines) if line.startswith(label)) + 1]

    def pair(label):
        return [number(part.strip().split(" ")[0]) for part in value(label).split("/")]

    def bpb(label):
        return [number(part.strip()) for part in value(label).split("→")]

    version, parameters = value("Version / model parameters").split(" / ")
    batches = value("Final batches:" if "Final batches: short / long" in lines else "Locked batches:")
    short, long = [int(part.strip().split(" ×")[0]) for part in batches.split("/")]
    updates, tokens = pair("Updates / useful tokens")
    initial_wiki, final_wiki = bpb("Initial → final FineWiki macro BPB")
    initial_mix, final_mix = bpb("Initial → final in-mixture pooled BPB")
    active_gib, active_percent = pair("Sampled maximum active GiB")
    train_cu, total_cu = pair("Estimated training / total CU")
    compute_tflops, compute_percent = pair("ESTIMATED compute TFLOP/s")
    hbm_gib, hbm_percent = pair("ESTIMATED HBM GiB/s")
    prefix = "Compilation seconds inside training" if "Compilation seconds inside training" in lines else "Adaptive compilation seconds inside training"
    source = "NFC-normalized scoring bytes/s" if version == "V4.5.1" else "Attributed original bytes/s"
    record = {
        "version": version, "model_parameters": int(number(parameters)),
        "training_seconds": number(value("Actual training seconds")),
        "updates": int(updates), "scored_tokens": int(tokens),
        "tokens_per_second": number(value("Useful tokens/s")),
        "scoring_bytes_per_second": number(value(source)),
        "final_short_batch": short, "final_long_batch": long,
        "initial_finewiki_macro_bpb": initial_wiki, "final_finewiki_macro_bpb": final_wiki,
        "initial_in_mix_pooled_bpb": initial_mix, "final_in_mix_pooled_bpb": final_mix,
        "sampled_peak_active_gib": active_gib, "sampled_peak_active_percent": active_percent,
        "device_capacity_gib": number(value("Reported device capacity GiB")),
        "compiler_plus_other_resident_estimate_percent": number(value("Compiler + inactive").split("%")[0]),
        "training_occupancy_percent": number(value("Useful occupancy")),
        "adaptive_compilation_seconds_inside_training": number(value(prefix)),
        "initial_compilation_seconds": pair("Compilation / initial eval / final eval seconds")[0],
        "notebook_elapsed_seconds": number(value("Whole notebook through report seconds")),
        "training_cu_estimate": train_cu, "total_cu_estimate": total_cu,
        "device_idle_percent": number(value("device_idle_time")),
        "estimated_compute_tflops_per_second": compute_tflops,
        "estimated_compute_percent_of_peak": compute_percent,
        "estimated_hbm_gib_per_second": hbm_gib, "estimated_hbm_percent_of_peak": hbm_percent,
        "engineering_status": value("Training / learning / performance").split(" / ")[0],
        "learning_status": value("Training / learning / performance").split(" / ")[1],
        "performance_goal_status": value("Training / learning / performance").split(" / ")[2],
        "profiling_status": value("Profiling requirements / telemetry").split(" / ")[0],
        "telemetry_status": value("Profiling requirements / telemetry").split(" / ")[1],
        "operation_ms_per_update": {name: number(value(name)) for name in
            ("attention", "cross_entropy", "mlp", "optimizer_updates", "other_or_fused", "vocabulary_projection")},
    }
    # Parse data blocks only; no attached prose is executed as instructions.
    if "Short / long scored tokens" in lines:
        record["tokens_by_context"] = ast.literal_eval(value("Short / long scored tokens"))
    record["finewiki_languages"] = {}
    for i, line in enumerate(lines):
        if len(line) == 8 and line[3] == "_" and lines[i + 1].replace(".", "").isdigit():
            record["finewiki_languages"][line] = {"initial": number(lines[i+1]), "final": number(lines[i+2])}
    return record


def machine_record(path):
    raw = json.loads(path.read_text())
    memory, profile = raw["training_memory"], raw["profiling"]
    record = {key: raw[key] for key in ("version", "model_parameters", "training_seconds", "updates",
        "scored_tokens", "tokens_per_second", "final_short_batch", "final_long_batch",
        "adaptive_compilation_seconds_inside_training", "notebook_elapsed_seconds", "engineering_status",
        "learning_status", "performance_goal_status", "profiling_status", "telemetry_status", "tokens_by_context",
        "data_manifest_sha256", "initial_backbone_sha256")}
    record.update(scoring_bytes_per_second=raw["original_bytes_per_second_attributed"],
        sampled_peak_active_gib=memory["sampled_peak_active_gib"],
        sampled_peak_active_percent=memory["sampled_peak_active_percent"],
        device_capacity_gib=memory["device_limit_gib"],
        compiler_plus_other_resident_estimate_percent=memory["compiled_peak_plus_other_resident_percent"],
        training_occupancy_percent=100 * raw["training_occupancy"],
        initial_compilation_seconds=raw["compilation_seconds"],
        training_cu_estimate=raw["compute_units_estimate"]["training"],
        total_cu_estimate=raw["compute_units_estimate"]["through_report"],
        device_idle_percent=profile["measurements"]["device_idle_time"]["value"],
        operation_ms_per_update={key: val["milliseconds_per_update"] for key, val in
            profile["measurements"]["operation_timings"]["categories"].items()})
    for key, val in profile["estimated_utilization"].items():
        if key.startswith("estimated_") and isinstance(val, (int, float)):
            record[key] = val
    for phase in ("initial", "final"):
        record[phase + "_finewiki_macro_bpb"] = raw[phase + "_eval"]["wiki"]["language_macro_bpb"]
        record[phase + "_in_mix_pooled_bpb"] = raw[phase + "_eval"]["in_mix"]["pooled_bpb"]
    record["finewiki_languages"] = {lang: {"initial": raw["initial_eval"]["wiki"]["languages"][lang]["bpb"],
        "final": row["bpb"]} for lang, row in raw["final_eval"]["wiki"]["languages"].items()}
    return record


def records():
    manifest = json.loads((HERE / "evidence-manifest.json").read_text())
    for entry in manifest["files"]:
        assert hashlib.sha256((HERE / "evidence" / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    rows = []
    for label, version, ids, parameters, basis, kind in CATALOG:
        relative = version + "/cell-output.txt"
        if kind == "machine_report":
            relative = ("V4.1.1-complete" if version == "V4.1.1" else version) + "/result.json"
        row = {} if kind == "missing" else (machine_record if kind == "machine_report" else cell_record)(HERE / "evidence" / relative)
        if row:
            assert row["version"] == version and row["model_parameters"] == parameters
            assert sum(row["tokens_by_context"].values()) == row["scored_tokens"]
            assert 1200 <= row["training_seconds"] < 1201
            assert abs(row["scored_tokens"] / row["training_seconds"] - row["tokens_per_second"]) < .1
            assert len(row["finewiki_languages"]) == 16
        row.update(tokenizer=label, version=version, native_ids=ids, model_rows=ids+3,
            model_parameters=parameters, scoring_basis=basis, evidence_kind=kind,
            status="report_missing" if kind == "missing" else "completed",
            evidence_path=None if kind == "missing" else "evidence/" + relative,
            notebook="../gemma_270m_colab_v5e1_" + version.replace(".", "_") + ".ipynb",
            host_idle_percent=None, matrix_unit_utilization_percent=None, memory_bandwidth_utilization_percent=None)
        assert parameters == 100326016 + (ids + 3) * 640
        rows.append(row)
    return rows


def fmt(value, decimals=0):
    return "—" if value is None else f"{value:,.{decimals}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
        ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n\n"


def render(rows):
    done = [r for r in rows if r["status"] == "completed"]
    text = """# Tokenizer training comparison — V4 series

Updated **23 October 2026**. **6 completed runs, 7 tokenizer variants, 1 missing result**.
Each completed run used approximately **1,200 seconds / 20 minutes on one TPU v5e-1**.
These are exploratory training results from one run per completed tokenizer, without independent replay.
Machine reports exist for Gemma 4 and BLOOM; the other four completed rows come from user-supplied cell outputs and retain their displayed precision.

All variants keep the Gemma3_270M transformer: **100,326,016 backbone parameters**, width **640**, fresh random initialization and AdamW.
The tied vocabulary table changes size, so total parameters differ. This measures tokenizer plus vocabulary cost under a time budget; it does not isolate segmentation at fixed parameter count.
The frozen source corpus is approximately **256 MiB**, with 16 human languages and 6 programming/markup domains.
Evaluation uses the existing **16-language FineWiki macro BPB at 1,024-token context**, plus the in-mixture holdout.
These pilots are separate from the proposed competition's 30-minute budget and hidden FineWeb 2 evaluation.

## Training and quality

BPB is lower-is-better. Bytes/s includes repeated corpus exposure and is not unique data consumed.
**Qwen3 uses NFC-normalized scoring bytes and is listed separately from original-byte results.** Its BPB and bytes/s cannot be treated as measurements on identical original bytes without a common-text rescore.

"""
    text += table(["Tokenizer / notebook", "Parameters", "Updates", "Scored tokens", "Tokens/s", "Scoring bytes/s", "FineWiki BPB", "In-mixture BPB", "Evidence"],
        [[f'[{r["tokenizer"]} {r["version"]}]({r["notebook"]})' + (" **NFC**" if r["scoring_basis"] == "NFC" else ""),
          fmt(r["model_parameters"]), fmt(r.get("updates")), fmt(r.get("scored_tokens")), fmt(r.get("tokens_per_second"),1),
          fmt(r.get("scoring_bytes_per_second"),1), fmt(r.get("final_finewiki_macro_bpb"),6), fmt(r.get("final_in_mix_pooled_bpb"),6),
          "Missing report" if r["evidence_kind"] == "missing" else f'[{r["evidence_kind"].replace("_", " ")}]({r["evidence_path"]})'] for r in rows])
    llama = next(r for r in rows if r["tokenizer"] == "Llama 3")
    deep = next(r for r in rows if r["tokenizer"] == "DeepSeek-V3-Base")
    gemma = rows[0]
    text += (f'Llama 3 has the lowest **reported original-byte FineWiki BPB: {llama["final_finewiki_macro_bpb"]:.6f}**. '
        f'DeepSeek follows at **{deep["final_finewiki_macro_bpb"]:.6f}**, a difference of **{deep["final_finewiki_macro_bpb"]-llama["final_finewiki_macro_bpb"]:.6f} BPB**. '
        'That small gap is not evidence of a repeatable winner from one run. '
        f'Llama 3 processes **{llama["scoring_bytes_per_second"]/gemma["scoring_bytes_per_second"]:.2f}×** as many attributed original bytes/s as the Gemma 3 baseline, '
        f'with **{100*(1-llama["model_parameters"]/gemma["model_parameters"]):.2f}% fewer parameters**. '
        'Different vocabulary cost, discovered batches and compilation overhead contribute to the observed result.\n\n')
    text += "## Batches, timing and cost\n\n"
    text += table(["Tokenizer", "Short / long batch", "Training s", "Initial compile s", "Compile inside training s", "Whole notebook s", "Training CU", "Total CU"],
        [[r["tokenizer"],f'{r["final_short_batch"]}×1,024 / {r["final_long_batch"]}×2,048',fmt(r["training_seconds"],3),
          fmt(r["initial_compilation_seconds"],2),fmt(r["adaptive_compilation_seconds_inside_training"],3),fmt(r["notebook_elapsed_seconds"],3),
          fmt(r["training_cu_estimate"],5),fmt(r["total_cu_estimate"],5)] for r in done])
    text += """CU figures are estimates at the supplied **2.92 CU/hour**, not billing receipts or USD prices. Total CU covers the notebook through its report; connected runtime afterward adds cost.
Gemma 3 V4 and Gemma 4 V4.1.1 discover batches immediately; BLOOM and the three later variants defer discovery until update 1,000. The o200k V4.2.2 notebook also defers discovery, but its run report is missing. Earlier V4.2.1 is the fixed-batch o200k alternative.
All keep the cumulative 3M short / 1M long schedule outside discovery. The timed window includes candidate compilation; initial compilation and evaluations are outside it. Whole-notebook timings also depend on cache reuse and data preparation.

## Training memory and hardware telemetry

Sampled active memory is **measured**, but synchronized samples can miss temporary peaks and runtime buffers. The compiler column is an **estimate**, not measured HBM usage. Device idle is from a post-save mixed diagnostic, not a whole-training MXU utilization measurement.

"""
    text += table(["Tokenizer", "Capacity GiB", "Sampled active GiB", "Active %", "Compiler + other resident estimate %", "Diagnostic device idle %", "EST. compute % peak", "EST. HBM % peak"],
        [[r["tokenizer"],fmt(r["device_capacity_gib"],3),fmt(r["sampled_peak_active_gib"],3),fmt(r["sampled_peak_active_percent"],3),
          fmt(r["compiler_plus_other_resident_estimate_percent"],3),fmt(r["device_idle_percent"],3),
          fmt(r["estimated_compute_percent_of_peak"],2),fmt(r["estimated_hbm_percent_of_peak"],2)] for r in done])
    text += """**Measured matrix-unit utilization, measured memory-bandwidth utilization and host CPU idle are unavailable in all six supplied outputs.** The runtime did not export supported TPU utilization-counter messages. The two EST. columns use compiler FLOP/byte costs divided by observed diagnostic durations; they are not hardware-counter measurements.
All six report training/learning **PASS**, performance goal **FAIL**, profiling requirements **FAIL**, and telemetry **PARTIAL**. Successful training does not mean every requested hardware measurement was obtained.

## Operation timings

Milliseconds per optimizer update from the **post-save diagnostic**, using 16 short-context and 16 long-context updates at the final batches. Values sum HLO self-time, include forward/backward, exclude idle, and are not elapsed update wall time. Because the long batch is 1 for the first three runs and 2 for the later three, operation timings are not comparisons at identical work per update.

"""
    ops = ("attention", "cross_entropy", "mlp", "optimizer_updates", "other_or_fused", "vocabulary_projection")
    text += table(["Tokenizer", "Attention", "Cross entropy", "MLP", "Optimizer", "Other/fused", "Vocabulary projection"],
        [[r["tokenizer"]] + [fmt(r["operation_ms_per_update"][key],3) for key in ops] for r in done])
    text += "## Per-language final FineWiki BPB\n\nQwen3's column uses NFC text; the other columns use original text.\n\n"
    text += table(["Language"] + [r["tokenizer"] + (" (NFC)" if r["scoring_basis"] == "NFC" else "") for r in done],
        [[language] + [fmt(r["finewiki_languages"][language]["final"],6) for r in done]
          for language in sorted(gemma["finewiki_languages"])])
    text += """## Evidence and remaining files

- **Missing:** o200k_base's completed report ZIP, for V4.2.1 or V4.2.2. No training results have been invented for that row.
- Full report ZIPs for **Gemma 3 V4, Llama 3 V4.4.1, Qwen3 V4.5.1 and DeepSeek V4.6.1** would replace the cell-output evidence with exact machine-readable metrics, full logs and identities. The supplied cell outputs are enough to record the displayed results now.
- The [earlier incomplete Gemma 4 attempt](evidence/V4.1.1-incomplete/training.jsonl) ends at **48 successful updates / 81,918 scored tokens**. It has no final evaluation or completed result; the supplied files do not establish the termination cause. The completed Gemma 4 rerun is a separate attempt and is the table row above.
- [Evidence manifest](evidence-manifest.json) hashes every imported file. Original selected machine reports, training logs and pasted cell outputs are checked in unchanged. Weights, corpus text, tokenizer assets and raw profile archives are not included. No independent training replay was performed.
- [JSON](comparison.json) retains parsed values and evidence labels; [CSV](comparison.csv) provides flat training metrics. Rebuild using `python3 experiments/gemma-270m-colab/training-results/build_comparison.py` from the repository root. The builder validates evidence hashes and uses no accelerator libraries.
"""
    return text


def build():
    rows = records()
    outputs = {"comparison.json": json.dumps(rows, indent=2, ensure_ascii=False) + "\n", "README.md": render(rows)}
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    csv_text = io.StringIO(newline="")
    writer = csv.DictWriter(csv_text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flat)
    outputs["comparison.csv"] = csv_text.getvalue()
    return outputs


if __name__ == "__main__":
    for name, content in build().items():
        (HERE / name).write_text(content)
    print("Built 7 tokenizer rows from 6 completed runs; no training or device calls.")
