#!/usr/bin/env bash
# List processes by command prefix, excluding this helper and its ancestors.
set -uo pipefail
PAT="${1:?Provide an exact command prefix}"
EXCLUDE=" $$ "
p=$$
while [ "$p" != "1" ] && [ -n "$p" ]; do
  p=$(awk '{print $4}' /proc/$p/stat 2>/dev/null) || break
  [ -n "$p" ] && EXCLUDE="$EXCLUDE $p "
done
for d in /proc/[0-9]*; do
  pid=${d#/proc/}
  case "$EXCLUDE" in *" $pid "*) continue;; esac
  c=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null) || continue
  case "$c" in "$PAT"*) echo "$pid";; esac
done
