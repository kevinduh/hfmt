#!/usr/bin/env bash
#
# delete_checkpoints.sh -- free disk space by deleting the Trainer's checkpoint-* folders
# in every run folder below a directory (default: the current one).
#
# Usage:
#   bash tools/delete_checkpoints.sh egs/mmtc/fr-en/outputs    # lists them, then asks before deleting
#
# Only finished runs are cleaned: a checkpoint is deleted only if its run folder has model/.
# An unfinished run (still running, or cancelled) keeps its checkpoints, since resuming it
# (launch.sh sft <run folder>/run_config.yaml) needs them.
set -euo pipefail

root="${1:-.}"

# 1. Find every checkpoint-* folder; keep only those of finished runs.
to_delete=()
while IFS= read -r checkpoint; do
    run_dir=$(dirname "$checkpoint")
    if [ -d "$run_dir/model" ]; then
        to_delete+=("$checkpoint")
    else
        echo "keep (run not finished, needed to resume): $checkpoint"
    fi
done < <(find "$root" -type d -name 'checkpoint-*' -prune)

if [ ${#to_delete[@]} -eq 0 ]; then
    echo "No checkpoints to delete under $root."
    exit 0
fi

# 2. Show what would be deleted and the total size.
du -sch "${to_delete[@]}"

# 3. Ask, then delete.
read -r -p "Delete these ${#to_delete[@]} folders? [y/N] " answer
if [ "$answer" = y ]; then
    rm -rf "${to_delete[@]}"
    echo "Deleted."
else
    echo "Nothing deleted."
fi
