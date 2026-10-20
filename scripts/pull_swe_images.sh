#!/usr/bin/env bash
# Pull the SWE-bench instance images of a split (Docker Hub, swebench namespace). Runs never pull (--pull=never), so
# every episode of an instance uses the image pulled here; each result row records its image ID.
# usage: scripts/pull_swe_images.sh [split]    split: eval95 (default) | rest385 | dev20 | eval100 | a file of ids
set -euo pipefail
cd "$(dirname "$0")/.."
SPLIT=${1:-eval95}
LIST=configs/splits/swe_verified_${SPLIT}.txt
[ -f "$SPLIT" ] && LIST=$SPLIT
while read -r iid; do
  [ -z "$iid" ] && continue
  name=$(echo "docker.io/swebench/sweb.eval.x86_64.${iid//__/_1776_}:latest" | tr '[:upper:]' '[:lower:]')
  docker image inspect "$name" >/dev/null 2>&1 || docker pull "$name"
done < "$LIST"
