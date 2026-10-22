# Gemma 3 270M — V3.7 efficient exact training and TPU measurements

**One v5e-1 TPU. One 60-second training run. AdamW. Random initialization.**

1. Upload this notebook to Google Colab and select **TPU v5e-1**.
2. Select **Run all**; authorize Drive and the Gemma tokenizer download.
3. Keep the runtime connected through checkpoint verification, profiling, and report download.

V3.7 retains V3.5's Gemma backbone, tokenizer, frozen multilingual corpus,
batch **2 × 1,024**, BF16 computation, FP32 master weights, and AdamW settings.
It computes the **exact full-vocabulary loss in 128-token chunks**, with
rematerialization in backward. For the default vocabulary, the logical FP32
logits workspace is **128 MiB per chunk**, versus **2.00 GiB** for the full batch.
These workspace figures are not claims about compiler peak memory or speed.

After 5 warmup updates, the learning rate peaks at **0.0003**. It remains at
that rate through **30 seconds**, then decreases linearly toward **0.00009**
at 60 seconds. The time multiplier is constant within each dispatch of at most
16 updates. Both gradient and decoupled weight-decay updates follow it.

The reference is measured V3.5: **45,734.30 scored tokens/s** and **2.9347027
FineWiki macro BPB**. No V3.7 TPU speed or quality is preclaimed.

The notebook records **matrix-unit utilization, HBM bandwidth utilization,
host/device idle percentages, operation timings, and the raw XProf trace**.
It also records active and reserved training memory in **GiB and percent of the
TPU's reported memory limit**, using synchronized boundary/progress samples.
Compiler memory estimates and lifetime peak counters are labeled separately.
It captures a bounded **64-update diagnostic after the official checkpoint is
saved, restored, and checked**. Diagnostic state is discarded and cannot alter
the saved model, its quality score, or official training counts. Diagnostic
time and estimated compute units are recorded separately and included in the
whole-notebook cost. Missing counters are **unavailable**, never invented.

Results, tokenizer artifacts, verified weights, and profiler files are saved
under **My Drive / ShiningGem / colab-v5e1 / V3.7 / unique run folder**.
The downloaded report ZIP includes profiler evidence but excludes the large
weights artifact. Exact optimizer resumption remains unsupported.
