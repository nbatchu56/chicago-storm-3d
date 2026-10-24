#!/usr/bin/env bash
# Check out (or start) the `learn` branch in a directory, or commit and push it with retries.
#   tools/learn_branch.sh checkout DIR      tools/learn_branch.sh push DIR "message"
# Needs GH_TOKEN and GITHUB_REPOSITORY. Other jobs may push to the branch in between: rebase and retry
# (they touch different files: backfill/correction from the Learn workflow, scorecard/ from snapshots).
set -euo pipefail
cmd=$1; dir=$2
url="https://x-access-token:${GH_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
if [ "$cmd" = checkout ]; then
  if git ls-remote --exit-code "$url" refs/heads/learn >/dev/null 2>&1; then
    git clone -q --depth 20 -b learn "$url" "$dir"
  else
    mkdir -p "$dir" && git -C "$dir" init -q -b learn && git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" config user.name "github-actions[bot]"
  git -C "$dir" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  exit 0
fi
cd "$dir"
git add -A
git diff --cached --quiet && { echo "learn: nothing to commit"; exit 0; }
git commit -q -m "$3"
for i in 1 2 3 4; do
  git push -q origin HEAD:learn && exit 0
  sleep $((i * 5))
  git fetch -q --depth 20 origin learn && git rebase -q origin/learn
done
echo "learn: push failed" >&2; exit 1
