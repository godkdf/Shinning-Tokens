# Gemma 3 270M: 60-second pilot on 16 TPU v5e chips

Start with **`gemma_270m_tpu_pilot.ipynb`** and run its cells in order. The notebook controls an **existing Google Cloud `v5litepod-16` training slice**, checks its identity, and launches every worker. It does not create or delete cloud resources.

You need an authenticated Google Cloud CLI on the notebook's machine, SSH access to the TPU workers, an existing Cloud Storage bucket, and permission for the TPU service account to read/write that bucket. Accept Gemma's Hugging Face repository terms before preparation. The notebook prompts for the project, zone, TPU name, bucket prefix, actual total slice hourly rate, and Hugging Face token. No code edits are required. The token is passed only to the preparation subprocess; it is not packaged or sent to workers.

The notebook creates its own Python 3.12 controller environment. Each worker also receives an isolated Python 3.12 environment. Ubuntu 22.04 or later is required on workers by the locked TPU dependencies. The controller does not need an attached TPU.

## Files

| File | Purpose |
|---|---|
| `gemma_270m_tpu_pilot.ipynb` | Install controller dependencies, collect real resource details, prepare data, launch, display results |
| `prepare_pilot.py` | Pin model/data revisions, freeze 10,000 training and 32 validation documents, deduplicate, tokenize |
| `launch_pilot.py` | Verify v5e-16, package inputs, upload once, launch every worker, collect reports |
| `bootstrap_worker.py` | Download the bundle using the worker service account; install identical locked dependencies |
| `run_pilot.py` | Compile, evaluate, train for 60 seconds, save, reload, and report |
| `verify_pilot.py` | Perform distributed checkpoint verification; regenerate the report from recorded evidence |
| `pilot_core.py` | Native model, optimizer, weighted loss, and JAX sharding |
| `pilot_common.py` | Data chunking, file hashes, array checkpoint format, and acceptance rules |
| `config.json` | Fixed baseline settings |
| `requirements-controller.lock` | Exact controller package versions |
| `requirements-tpu.lock` | Exact Linux TPU package versions |
| `REQUIREMENTS.md` | Engineering requirements and evidence mapping |
| `tests/test_pilot.py` | CPU checks of loss, distributed gradients, nonfinite handling, chunking, checkpoint, and hardware selection |

## What runs

- Native Gemma 3 270M architecture from `gemma==4.0.1`; **random initial weights**, no pretrained checkpoint.
- Native tokenizer from `google/gemma-3-270m`. Preparation resolves `main` to a full revision and saves the artifacts; workers never fetch a moving model revision.
- English FineWeb `sample-10BT`. The first 32 unique nonempty documents form validation; the next 10,000 form training. SHA-256 text hashes prevent exact duplicates across these sets. This is an engineering sample, not an independently certified competition test corpus.
- One document per chunk; successive chunks overlap by one input token. Every next-token target is scored once, including EOS. Padding targets are masked. Long documents are not truncated.
- Sequence length 1,024; one sequence per chip; 16 sequences per update. Training cycles through a seeded block permutation if the sample is exhausted; exposures and completed passes are reported.
- Data parallelism with full FP32 model/optimizer replicas, BF16 forward computation, FP32 cross-entropy, and AdamW. This baseline uses the native full-vocabulary output head.
- Learning rate 0.0003 with five update warmup, then constant; gradient norm clipping at 1.0.

The model's context length and tokenizer are not modified. Native parameter-shape inspection confirms **268,098,176 parameters**, including the shared **262,144 × 640** embedding table. The runner verifies this exact count and tied embedding structure.

## Timing and costs

Compilation is explicit and separately timed. The training executable is compiled without applying an optimizer update. The 60-second timer starts after initial validation and a worker barrier. It includes coordinator decisions, input transfers, forward/backward computation, optimizer updates, and training logging. Device execution is awaited; queued work is not counted as completed training.

Worker 0 issues a shared stop decision before every update. An update begun before 60 seconds may finish afterward. The report records the overrun. No worker independently exits the collective training loop.

Checkpoint saving, upload, download, reload, and repeated validation are outside the training timer. A full FP32 model plus two AdamW moment arrays is approximately **3.2 GB before metadata**, so the checkpoint stage is materially longer than saving weights alone.

`launch.json` reports `slice_hourly_usd × observed_job_seconds / 3600`. This estimate covers the interval observed by the launcher, including worker installation/compilation/verification. It excludes any allocation time before launch and after return. It is **not an invoice total**. The existing TPU remains allocated when the notebook finishes.

## Results and acceptance

Each launch creates a unique `runs/<run-id>/` locally and `<bucket-prefix>/<run-id>/` in Cloud Storage. Primary artifacts:

- `result.json`: loss, throughput, updates, timing, parameter count, reload error, and evidence.
- `verification.json`: engineering PASS/FAIL and learning PASS/FAIL, with requirement IDs.
- `checkpoint.npz`, `checkpoint.tree.json`, `checkpoint_meta.json`: parameters, optimizer moments/counts, PRNG state, data position, config, and hashes.
- `worker-N/training.jsonl`, `worker-N/environment.json`, `worker-N/memory.json`: per-worker measurements and exact installed packages.
- `training_hlo.txt`: compiled computation used to inspect gradient collectives.
- `bundle.zip`, `launch.json`, `resource.json`, `launcher.log`: source, frozen data, resource description, orchestration, and cost scope.

Every worker downloads the persistent checkpoint and checks its file hashes. The complete restored state is compared exactly against the saved state, then the restored parameters are evaluated. Reloaded validation loss must match within **0.001 nats/scored token**.

Engineering PASS requires every TPU requirement, at least five completed updates, finite numerical results, changed transformer parameters, and exact checkpoint-state restoration. Learning PASS separately requires a lower final validation loss.

This pilot reports per-token cross-entropy, which is valid for this one fixed tokenizer. It does not claim tokenizer-independent BPB: special-token accounting, reversible byte alignment, and the multilingual evaluator need their own verification before tokenizer comparisons. Tokenizer round-trip counts are recorded without silently rewriting source text.

## Local verification

The test suite requires Python 3.12 with JAX, Flax, Optax, NumPy, and pytest. The end-to-end CPU fixture also requires the pinned Gemma package. The commands below run from this directory in that environment:

```sh
python -m pytest tests -q
python run_pilot.py --cpu-smoke --run-dir /tmp/gemma-270m-cpu-check
```

The fixture uses **16 virtual CPU devices, a reduced native Gemma configuration, synthetic tokens, and a one-second timer**. It tests execution, loss, optimizer updates, saving, and restoration. Its report deliberately fails TPU/hardware/persistent-cloud requirements. A successful fixture exit is never represented as a real TPU pilot PASS. The fixture's macOS NumPy extended-dtype compatibility alias is not used on TPU workers.

The standalone `verify_pilot.py --run-dir <worker-result-directory>` regenerates acceptance from recorded distributed reload evidence and checks any locally present checkpoint against its hash. It does not silently rerun accelerator training.

## References

- [Native Gemma 3 270M JAX API](https://gemma-llm.readthedocs.io/en/latest/api/gm/nn/Gemma3_270M.html)
- [Google TPU v5e configuration documentation](https://docs.cloud.google.com/tpu/docs/v5e)
- [JAX multi-process execution](https://docs.jax.dev/en/latest/multi_process.html)
- [Native tokenizer/model configuration repository](https://huggingface.co/google/gemma-3-270m)
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- [NASA requirement writing guidance](https://www.nasa.gov/reference/appendix-c-how-to-write-a-good-requirement/)
