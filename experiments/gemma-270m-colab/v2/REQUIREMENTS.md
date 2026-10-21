# Gemma Colab V2

## Fixed configuration and requirements

| Setting | Fixed value |
|---|---:|
| Device | One TPU v5e-1 |
| Architecture | Native `gm.nn.Gemma3_270M`, 268,098,176 parameters |
| Vocabulary | 262,144; tied embedding/output matrix |
| Initial weights | Random; seed 42 |
| Training / validation documents | 10,000 / 32, identical to V1 by ordered document hashes |
| Validation targets | 13,229 |
| Sequence / batch | 1,024 positions / 2 packed sequences |
| Main dispatch | Up to 16 updates; one update near the deadline |
| Compute / master weights | BF16 / FP32 |
| Optimizer | AdamW, LR 0.0003, warmup 5 updates, clip 1.0 |
| Adam b1 / b2 / epsilon | 0.9 / 0.999 / 1e-8 |
| Weight decay | 0.01 on matrix/tensor weights |
| Training interval | 60 seconds nominal; actual completed-work time reported |

| ID | Requirement | Verification in your Colab run |
|---|---|---|
| V2-001 | The runner shall execute one training attempt per run folder. | Exclusive start marker; repeated training-cell execution stops. |
| V2-002 | The runner shall use exactly one v5e TPU. | Device platform, kind, and count. |
| V2-003 | The packer shall preserve every original target once per dataset pass. | Source chunk assignments, target equality, and target totals. |
| V2-004 | The model shall isolate attention between packed segments. | Causal segment mask and an explicit mask verification before training. |
| V2-005 | The packer shall preserve the original context and positions of every chunk. | Whole-chunk packing and position equality. |
| V2-006 | The runner shall place training inputs on the TPU before starting the timer. | Device ownership and completed transfer barrier. |
| V2-007 | The runner shall limit each compiled call to 16 optimizer updates. | Completed dispatch counts. |
| V2-008 | The runner shall cease dispatching training calls at 60 seconds. | Monotonic timestamps, completed-work synchronization, and overrun report. |
| V2-009 | The runner shall reject an update with nonfinite loss or gradient. | Device-side finite checks; stop without retry. |
| V2-010 | The evaluator shall score the same V1 validation documents before and after training. | Document hashes and 13,229 target count. |
| V2-011 | The runner shall persist one full resumable checkpoint. | Local/persistent SHA256 equality, array count, and state schema. |
| V2-012 | The report shall provide separate training and overhead timings. | Phase durations and compute-unit estimates. |
| V2-013 | The model shall retain the native architecture and tied embedding/output table. | Parameter count, table shape, and unique table count. |
| V2-014 | The runner shall perform zero optimizer updates outside the training interval. | Zero initial step; final step equals timed update count. |
| V2-015 | Training shall change a transformer parameter. | Initial/final parameter probe. |
| V2-016 | The runner shall complete at least five optimizer updates. | Completed update count. |

Requirements use NASA-style numbered **shall** statements. Engineering status, learning status, and performance goals are reported separately.

| Performance goal | Measured V1 | V2 goal, not a guarantee |
|---|---:|---:|
| Useful position occupancy | 52.6% | At least 95% |
| Useful tokens/second | 11,406 | At least 18,000 |
| Useful tokens in 60 seconds | 684,589 | At least 1,080,000 |
| Final validation loss | 6.872893 | At most 6.872893 |

The comparison uses your saved V1 measurement; this notebook does not rerun V1. A higher token rate does not automatically imply lower validation loss.

Status: notebook generated; local checks and tests not run at the user's request.
