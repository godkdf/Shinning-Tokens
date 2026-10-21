# GEM-PILOT-001 — Verification matrix

Revision 3: executable implementation of the 60-second, 16-chip TPU v5e pilot. “Shall” indicates a mandatory requirement. The CPU fixture is a development test and cannot satisfy the TPU pilot's acceptance requirements.

| ID | Requirement | Evidence |
|---|---|---|
| TPU-001 | The runner shall discover exactly 16 TPU devices on the selected v5e-16 slice. | Launcher checks cloud resource type; runner checks global device count, backend, and device kinds. |
| TPU-002 | The runner shall initialize distributed execution before discovering devices. | Initialization order in `run_pilot.py`; per-worker device inventory. |
| TPU-003 | The runner shall initialize the pinned native Gemma 3 270M architecture using random weights. | Pinned Gemma dependency, model/config checks, exact parameter count, initialization record. |
| TPU-004 | The runner shall preserve the native shared input/output embedding table. | Native model implementation and parameter-shape inspection. |
| TPU-005 | Each global training batch shall contain 16 distinct block indices. | Seeded global permutation; index uniqueness check before every update. |
| TPU-006 | The training objective shall equal summed token negative log-likelihood divided by summed valid-token count across all 16 devices. | Sharded numerical reference test; CPU gradient parity test with unequal padding. |
| TPU-007 | The runner shall synchronize parameter gradients across all 16 devices before applying an update. | Replicated parameter/output sharding, collective in compiled HLO, CPU gradient parity test. |
| TPU-008 | The runner shall finish training-step compilation before the training timer starts. | Explicit `.lower(...).compile()` before timing; separate compilation duration. |
| TPU-009 | The runner shall apply zero optimizer updates outside the timed training interval. | Training executable is invoked only inside the timed loop. |
| TPU-010 | All workers shall obey the same stop decision at an optimizer-update boundary. | Worker-0 decision broadcast; gathered update counts agree. |
| TPU-011 | The runner shall await completed device execution when measuring training duration. | `jax.block_until_ready` on every update and before the final duration measurement. |
| TPU-012 | Initial and final evaluation shall use identical validation inputs. | Frozen validation archive and identical manifest SHA-256. |
| TPU-013 | The runner shall save and exactly restore a resumable checkpoint from persistent storage. | Every worker downloads/checks SHA-256, restores arrays, and compares all state values. |
| TPU-014 | Reloaded validation loss shall differ from saved-model validation loss by at most 0.001 nats per scored token. | Recorded absolute difference. |
| DATA | Training and validation sets shall share zero exact document-text hashes. | Document hash intersection and input archive hashes. |
| NUMERICS | The runner shall stop before applying any update whose loss or gradient norm is nonfinite. | Conditional update guard; failure artifact; CPU injected-NaN test. |
| UPDATES | The timed run shall complete at least five optimizer updates. | Update log and result counter. |
| WEIGHTS | At least one transformer parameter shall change during training. | Initial/final transformer probe comparison on every worker. |
| TIME | Training shall continue until the first update boundary at or after 60 seconds, with no new update decision at or after 60 seconds. | Synchronized duration and final update's coordinator decision timestamp. |

An update includes the coordinator stop-decision broadcast, input transfer, computation, and log write. Duration can exceed 60 seconds by the final update/control overhead; the overrun is reported rather than hidden.

Learning PASS is a separate result: final validation cross-entropy is lower than initial validation cross-entropy on the same sample. No minimum improvement magnitude is claimed for this one-minute engineering pilot.

The baseline configuration is `config.json`. Preparation resolves model and dataset revision names into immutable revisions in `resolved_config.json`. Source and dependency versions are included in the run artifacts.
