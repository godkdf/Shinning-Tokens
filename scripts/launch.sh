#!/bin/sh
set -eu
LAB_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$LAB_ROOT"
export TOKENIZER_LAB_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$LAB_ROOT/src:$LAB_ROOT/vendor"
if [ ! -f "$LAB_ROOT/runtime/bin/python3" ]; then
  printf '%s\n' 'The bundled Python runtime is missing. Extract the complete offline kit for this computer.' >&2
  exit 1
fi
# Some ZIP extractors discard Unix mode bits. The interpreter is already covered
# by the kit manifest; restore only its executable bit in this user-owned folder.
if [ ! -x "$LAB_ROOT/runtime/bin/python3" ]; then chmod u+x "$LAB_ROOT/runtime/bin/python3"; fi
exec "$LAB_ROOT/runtime/bin/python3" -m tokenizer_lab.cli launch "$@"
