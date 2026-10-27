#!/bin/bash
set -euo pipefail
patch_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${SGLANG_PATCH_SOURCE_DIR:-/sgl-workspace/sglang}"
python3 "${patch_dir}/check-source.py" --source-dir "${source_dir}"
exec python3 "${patch_dir}/apply-stack.py" --source-dir "${source_dir}" --patch-dir "${patch_dir}"
