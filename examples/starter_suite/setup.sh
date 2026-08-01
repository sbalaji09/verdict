#!/usr/bin/env bash
# One-time bootstrap: turn each task's repo/ into its own standalone git
# repo, same as every other examples/*/setup.sh, so `verdict bench --suite
# examples/starter_suite` has something real to worktree-isolate for each
# task. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

for task_dir in */; do
  repo="${task_dir}repo"
  if [ ! -d "$repo" ]; then
    continue
  fi
  if [ -d "$repo/.git" ]; then
    echo "${repo} is already a git repo."
    continue
  fi
  (
    cd "$repo"
    git init -q
    git add -A
    git commit -q -m "starter_suite/${task_dir%/}: seed task"
  )
  echo "Initialized ${repo} as a standalone git repo."
done
