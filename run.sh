#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ -n "${JLJ_GMR_PYTHON:-}" ]]; then
    task_python="$JLJ_GMR_PYTHON"
elif [[ -x "$HOME/miniconda3/envs/gmr/bin/python" ]]; then
    task_python="$HOME/miniconda3/envs/gmr/bin/python"
else
    task_python=python3
fi
exec "$task_python" -m jlj_gmr "$@"
