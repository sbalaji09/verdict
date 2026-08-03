#!/usr/bin/env bash
# One-time bootstrap: turn each case's directory into its own standalone
# git repo, same as every other examples/*/setup.sh, so `verdict
# ground-truth --dataset examples/ground_truth_dataset` has something real
# to worktree-isolate for each case. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

for case_dir in */; do
  case_dir="${case_dir%/}"
  if [ ! -f "${case_dir}/pytest.ini" ] && [ ! -f "${case_dir}/README.md" ]; then
    continue
  fi
  if [ -d "${case_dir}/.git" ]; then
    echo "${case_dir} is already a git repo."
    continue
  fi
  (
    cd "$case_dir"
    git init -q
    git add -A
    git commit -q -m "ground_truth_dataset/${case_dir}: seed case"
  )
  echo "Initialized ${case_dir} as a standalone git repo."
done
