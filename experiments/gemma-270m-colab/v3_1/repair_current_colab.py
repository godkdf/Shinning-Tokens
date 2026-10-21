# Paste this entire file into ONE new cell in the existing failed Colab session.
# It repairs the recorded pre-training dtype failure and reuses prepared inputs.
import json
import shutil
import uuid

OLD_OUTPUT = OUTPUT
failure = json.loads((OLD_OUTPUT / "failure.json").read_text())
error = failure.get("error", "")
if failure.get("phase") != "train" or "lax.mul requires arguments to have the same dtypes" not in error or "group_jit.lower" not in error:
    raise RuntimeError("Stopped: this is not the recorded pre-training dtype compilation failure.")
if any((OLD_OUTPUT / name).exists() for name in ("training.jsonl", "result.json", "checkpoint.npz", "checkpoint.npz.partial")):
    raise RuntimeError("Stopped: this folder contains training artifacts; refusing to repeat a training attempt.")
if "Starting ONE 60-second" in (OLD_OUTPUT / "train.log").read_text():
    raise RuntimeError("Stopped: the previous run reached the training-start message.")

FIX = '''
def install_tokamax_loss_dtype_fix():
    from tokamax._src.ops.linear_softmax_cross_entropy_loss import pallas_mosaic_tpu as loss_impl
    cls = loss_impl.PallasMosaicTpuLinearSoftmaxCrossEntropyLossVjp
    original = cls._fwd
    if getattr(original, "_gemma_v31_dtype_fix", False):
        return
    @functools.wraps(original)
    def fixed(self, residuals, out, dout, x, labels, w, *, reduction="sum", config, return_residuals):
        (dx, dlabels, dw), saved = original(self, residuals, out, dout, x, labels, w,
            reduction=reduction, config=config, return_residuals=return_residuals)
        return (dx.astype(x.dtype), dlabels, dw.astype(w.dtype)), saved
    fixed._gemma_v31_dtype_fix = True
    cls._fwd = fixed
'''

source = PROGRAM_PATH.read_text()
entry = '\nif __name__ == "__main__":'
target = '    attention_kernel = install_splash_attention()'
if source.count(entry) != 1 or source.count(target) != 1 or "def install_tokamax_loss_dtype_fix" in source:
    raise RuntimeError("Stopped: unexpected or already patched program source.")
source = source.replace(entry, "\n" + FIX + entry, 1)
source = source.replace(target, '    install_tokamax_loss_dtype_fix()\n' + target, 1)

# Keep the failed folder and its exclusive start marker as an audit record.
OUTPUT = OLD_OUTPUT.parent / (OLD_OUTPUT.name + "-dtypefix-" + uuid.uuid4().hex[:8])
OUTPUT.mkdir(parents=True, exist_ok=False)
for name in ("preflight.json", "requested_config.json", "config.json", "data_manifest.json",
             "tokenization.json", "tokenizer_identity.json", "source-lock.json", "preparation.json",
             "parallel_downloads.json", "preparation_requirements.json", "requirements.in",
             "requirements.lock", "dependency_source.json"):
    if (OLD_OUTPUT / name).is_file():
        shutil.copyfile(OLD_OUTPUT / name, OUTPUT / name)
shutil.copytree(WORK / "prepared" / "tokenizer", OUTPUT / "tokenizer")
PROGRAM_PATH.write_text(source)
PROGRAM_SOURCE = source
shutil.copyfile(PROGRAM_PATH, OUTPUT / "pilot_source.py")
(OUTPUT / "repair.json").write_text(json.dumps({"repair": "V3.1-dtype-1",
    "prior_failed_folder": str(OLD_OUTPUT), "prior_optimizer_updates": 0,
    "data_preparation_repeated": False, "automatic_retry": False}, indent=2))
print("Patched fused-loss gradient dtypes. Reusing prepared data; no new source downloads.")
print("Results:", OUTPUT)
run_phase("train")
