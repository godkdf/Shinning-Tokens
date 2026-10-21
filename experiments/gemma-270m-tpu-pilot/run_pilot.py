#!/usr/bin/env python3
"""Run the native Gemma pilot on all TPU workers, with separately timed compilation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from pilot_common import (assess_report, read_json, save_checkpoint,
                          sha256_file, upload_file, write_json)


def create_cpu_fixture(directory, config):
    import hashlib
    import numpy as np
    from pilot_common import chunk_document
    directory.mkdir(parents=True)
    splits = {}
    for split, offset, count in (("train", 10, 64), ("validation", 80, 32)):
        rows = []
        documents = []
        for i in range(count):
            # Distinct document identities, highly repetitive token patterns.
            ids = [3 + (i + offset) % 7] * 23
            rows.extend((*chunk, i) for chunk in chunk_document(ids, 16, 2, 1))
            documents.append({"id": f"{split}-{i}", "sha256": hashlib.sha256(f"{split}-{i}".encode()).hexdigest()})
        x, y, m, indices = zip(*rows)
        np.savez(directory / f"{split}.npz", tokens=np.stack(x), targets=np.stack(y), mask=np.stack(m),
                 document_index=np.array(indices, np.int32))
        splits[split] = {"documents": documents, "blocks": len(rows)}
    fixture_config = {**config, "sequence_length": 16, "training_seconds": 1.0, "minimum_updates": 5}
    manifest = {"splits": splits, "cpu_fixture": True, "files": {}}
    for filename in ("train.npz", "validation.npz"):
        manifest["files"][filename] = sha256_file(directory / filename)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "resolved_config.json", fixture_config)


def run(args):
    # No backend discovery or Gemma import may precede distributed initialization.
    distributed_started = time.monotonic()
    if args.cpu_smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=16"
    else:
        os.environ["JAX_PLATFORMS"] = "tpu"
    import jax
    if not args.cpu_smoke:
        jax.distributed.initialize(initialization_timeout=300, heartbeat_timeout_seconds=120)
    from jax.experimental import multihost_utils
    import numpy as np
    from flax import serialization
    from pilot_core import (build_steps, check_global_loss, evaluate_all, host_value, initial_state,
                            make_batch, make_mesh, make_model, state_parameter_info, transformer_probe)
    from verify_pilot import verify_reload

    rank = jax.process_index()
    local_dir = args.run_dir / f"worker-{rank}"
    local_dir.mkdir(parents=True, exist_ok=False)
    if args.cpu_smoke:
        args.data_dir = local_dir / "fixture"
        create_cpu_fixture(args.data_dir, read_json(args.config))
    config = read_json(args.data_dir / "resolved_config.json")
    manifest = read_json(args.data_dir / "manifest.json")
    manifest_hash = sha256_file(args.data_dir / "manifest.json")
    for filename, digest in manifest["files"].items():
        if sha256_file(args.data_dir / filename) != digest:
            raise ValueError(f"Prepared input changed: {filename}")
    train_hashes = {doc["sha256"] for doc in manifest["splits"]["train"]["documents"]}
    eval_hashes = {doc["sha256"] for doc in manifest["splits"]["validation"]["documents"]}
    if train_hashes & eval_hashes:
        raise ValueError("Training and validation documents overlap.")
    device_count = jax.device_count()
    if device_count != config["expected_devices"]:
        raise RuntimeError(f"Expected {config['expected_devices']} chips, found {device_count}.")
    if not args.cpu_smoke and any(not any(label in device.device_kind.lower().replace(" ", "")
                                         for label in ("v5lite", "v5e")) for device in jax.devices()):
        raise RuntimeError("Allocation is not TPU v5e hardware.")
    mesh, replicated, data, local_indices = make_mesh()
    if config["local_batch_per_chip"] != 1:
        raise ValueError("This pilot fixes one sequence per chip.")
    global_batch = device_count
    train = dict(np.load(args.data_dir / "train.npz", allow_pickle=False))
    validation = dict(np.load(args.data_dir / "validation.npz", allow_pickle=False))
    if len(train["tokens"]) < global_batch:
        raise ValueError("Not enough distinct training blocks for 16 chips.")
    order = np.random.default_rng(config["seed"]).permutation(len(train["tokens"]))
    global_loss_pass = check_global_loss(mesh, replicated)
    model = make_model(args.cpu_smoke)
    state, optimizer, schedule = initial_state(model, config, replicated)
    parameter_count, tied_embedding, probe_path = state_parameter_info(state, model)
    if not args.cpu_smoke and parameter_count != 268_098_176:
        raise ValueError(f"Unexpected Gemma 270M parameter count: {parameter_count}")
    initial_probe = transformer_probe(state, probe_path)
    train_jit, eval_jit = build_steps(model, optimizer, schedule, replicated, data, global_batch)
    dummy = make_batch(train, order[:global_batch], local_indices, data)
    compile_started = time.monotonic()
    train_executable = train_jit.lower(state, dummy).compile()
    eval_executable = eval_jit.lower(state.params, dummy).compile()
    compile_seconds = time.monotonic() - compile_started
    hlo = train_executable.as_text()
    gradient_collective = bool(re.search(r"all[-_]reduce", hlo, re.IGNORECASE))
    if rank == 0:
        (local_dir / "training_hlo.txt").write_text(hlo)
    eval_started = time.monotonic()
    before = evaluate_all(state.params, validation, eval_executable, local_indices, data, global_batch, manifest_hash)
    initial_eval_seconds = time.monotonic() - eval_started
    versions = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    write_json(local_dir / "environment.json", {"packages": versions, "python": sys.version,
        "devices": [{"id": d.id, "host": d.process_index, "kind": d.device_kind} for d in jax.devices()],
        "process_index": rank, "process_count": jax.process_count(), "data_manifest_hash": manifest_hash})
    multihost_utils.sync_global_devices("pilot-training-start")
    jax.block_until_ready(state)
    started = time.monotonic()
    updates = 0
    scored_tokens = 0
    nonfinite = 0
    last_start = 0.0
    distinct = True
    with (local_dir / "training.jsonl").open("w") as log:
        while True:
            decision_elapsed = time.monotonic() - started
            stop = np.array(rank == 0 and decision_elapsed >= config["training_seconds"])
            stop = bool(multihost_utils.broadcast_one_to_all(stop, is_source=rank == 0))
            if stop:
                break
            # The update includes this coordinator decision, input transfer, and logging.
            last_start = decision_elapsed
            position = updates * global_batch
            indices = order[(np.arange(position, position + global_batch) % len(order))]
            distinct = distinct and len(np.unique(indices)) == global_batch
            batch = make_batch(train, indices, local_indices, data)
            state, metrics = jax.block_until_ready(train_executable(state, batch))
            values = {key: host_value(value).item() for key, value in metrics.items()}
            if not values["finite"]:
                nonfinite += 1
                raise FloatingPointError("Nonfinite loss/gradient detected; update was not applied.")
            updates += 1
            scored_tokens += int(values["tokens"])
            values.update(update=updates, start_seconds=last_start, end_seconds=time.monotonic() - started)
            log.write(json.dumps(values, allow_nan=False) + "\n")
            log.flush()
            if rank == 0:
                print(f"update={updates} loss={values['loss']:.5f} elapsed={values['end_seconds']:.2f}s", flush=True)
    jax.block_until_ready(state)
    training_seconds = time.monotonic() - started
    worker_updates = multihost_utils.process_allgather(np.array(updates, np.int32))
    workers_agree = bool(np.all(np.asarray(worker_updates) == updates))
    memory = []
    for device in jax.local_devices():
        memory.append({"device_id": device.id, "stats": device.memory_stats()})
    final_eval_started = time.monotonic()
    after = evaluate_all(state.params, validation, eval_executable, local_indices, data, global_batch, manifest_hash)
    final_eval_seconds = time.monotonic() - final_eval_started
    changed = not np.array_equal(initial_probe, transformer_probe(state, probe_path))
    checkpoint_started = time.monotonic()
    if rank == 0:
        save_checkpoint(local_dir / "checkpoint.npz", serialization.to_state_dict(jax.tree.map(host_value, state)))
        metadata = {"checkpoint_sha256": sha256_file(local_dir / "checkpoint.npz"),
            "tree_sha256": sha256_file(local_dir / "checkpoint.tree.json"),
            "data_manifest_hash": manifest_hash, "config": config,
            "optimizer_updates": updates, "data_position": int(host_value(state.data_position)),
            "initialization": "random", "cpu_fixture": args.cpu_smoke}
        write_json(local_dir / "checkpoint_meta.json", metadata)
        if args.output_uri:
            for filename in ("checkpoint.npz", "checkpoint.tree.json", "checkpoint_meta.json"):
                upload_file(local_dir / filename, f"{args.output_uri}/{filename}")
    multihost_utils.sync_global_devices("checkpoint-upload-complete")
    checkpoint_seconds = time.monotonic() - checkpoint_started
    restored = verify_reload(state, local_dir, args.output_uri, eval_executable, validation,
                              local_indices, data, global_batch, manifest_hash)
    local_checks = np.array([restored["state_exact"], changed, workers_agree], np.int32)
    all_checks = np.asarray(multihost_utils.process_allgather(local_checks)).reshape(-1, 3).all(axis=0)
    evidence = {"distributed_initialized_before_devices": not args.cpu_smoke,
        "native_architecture": not args.cpu_smoke, "tied_embedding": tied_embedding,
        "distinct_batch_indices": distinct, "global_loss_reference_pass": global_loss_pass,
        "gradient_collective_present": gradient_collective, "workers_agree_on_updates": bool(all_checks[2]),
        "synchronized_timing": True, "checkpoint_state_exact": bool(all_checks[0]),
        "persistent_checkpoint_verified": restored["persistent_verified"],
        "transformer_parameters_changed": bool(all_checks[1]), "no_shared_document_hashes": True}
    result = {"experiment": config["experiment"], "config": config, "cpu_fixture": args.cpu_smoke,
        "platform": jax.devices()[0].platform, "device_count": device_count, "process_count": jax.process_count(),
        "model_parameters": parameter_count, "initialization": "random", "updates": updates,
        "training_blocks": len(order), "block_exposures": updates * global_batch,
        "completed_data_passes": updates * global_batch // len(order),
        "scored_tokens": scored_tokens, "tokens_per_second": scored_tokens / training_seconds,
        "compilation_seconds": compile_seconds, "compilation_finished_before_training": True,
        "initial_eval_seconds": initial_eval_seconds, "final_eval_seconds": final_eval_seconds,
        "checkpoint_save_upload_seconds": checkpoint_seconds, "checkpoint_reload_seconds": restored["seconds"],
        "training_seconds": training_seconds, "training_overrun_seconds": training_seconds - config["training_seconds"],
        "last_update_start_seconds": last_start, "untimed_updates": 0, "nonfinite_updates": nonfinite,
        "initial_eval": before, "final_eval": after, "reloaded_eval": restored["evaluation"],
        "reload_loss_error": abs(after["loss"] - restored["evaluation"]["loss"]),
        "evidence": evidence, "output_uri": args.output_uri,
        "memory_scope": "device lifetime peak; includes initialization, compilation and evaluation",
        "end_to_end_worker_seconds": time.monotonic() - distributed_started,
        "estimated_total_bill_usd": None}
    write_json(local_dir / "memory.json", memory)
    write_json(local_dir / "result.json", result)
    report = assess_report(result)
    write_json(local_dir / "verification.json", report)
    if args.output_uri:
        for filename in ("environment.json", "memory.json", "training.jsonl"):
            upload_file(local_dir / filename, f"{args.output_uri}/worker-{rank}/{filename}")
        if rank == 0:
            for filename in ("result.json", "verification.json", "training_hlo.txt"):
                upload_file(local_dir / filename, f"{args.output_uri}/{filename}")
    if rank == 0:
        print(json.dumps({"report": report, "tokens_per_second": result["tokens_per_second"],
                          "initial_loss": before["loss"], "final_loss": after["loss"]}, indent=2), flush=True)
    multihost_utils.sync_global_devices("pilot-complete")
    if not args.cpu_smoke:
        jax.distributed.shutdown()
    if args.cpu_smoke:
        fixture_checks = [v for k, v in report["requirements"].items()
                          if k not in {"TPU-001", "TPU-002", "TPU-003", "TPU-013"}]
        return 0 if all(value == "PASS" for value in fixture_checks) and restored["state_exact"] else 1
    return 0 if report["engineering_status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).with_name("prepared"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-uri")
    parser.add_argument("--cpu-smoke", action="store_true", help="Tiny, synthetic 16-CPU-device fixture; never a TPU result.")
    args = parser.parse_args()
    if not args.cpu_smoke and not args.output_uri:
        parser.error("The TPU pilot requires --output-uri gs://bucket/prefix for durable checkpoints.")
    try:
        status = run(args)
    except Exception:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        error = {"status": "FAIL", "error": traceback.format_exc()}
        failure = args.run_dir / f"failure-{os.getpid()}.json"
        write_json(failure, error)
        if args.output_uri:
            try:
                upload_file(failure, f"{args.output_uri}/{failure.name}")
            except Exception:
                pass
        raise
    raise SystemExit(status)


if __name__ == "__main__":
    main()
