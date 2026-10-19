#!/bin/sh
set -eu
LAB_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec /bin/sh "$LAB_ROOT/scripts/launch.sh" "$@"
