#!/bin/bash
# Track one Kaggle kernel run: log status changes, progress lines and errors with the time, warn when it stays
# queued, and download the output when it ends.
#   kaggle/track_kernel.sh <owner/slug> <output dir> <log file>
K="$1"; OUT="$2"; LOG="$3"
KG=~/.venvs/urbeeg/bin/kaggle
SEEN="$LOG.seen"; : > "$SEEN"
last=""; queued_since=""; warned=0
stamp() { date '+%d %b %H:%M'; }
echo "$(stamp) TRACK start $K" >> "$LOG"
while true; do
  s=$($KG kernels status "$K" 2>&1 | tail -1 | sed -E 's/.*KernelWorkerStatus\.([A-Z_]+).*/\1/')
  case "$s" in QUEUED|RUNNING|COMPLETE|ERROR|CANCEL_REQUESTED|CANCEL_ACKNOWLEDGED|NEW_SCRIPT) ;; *) s="UNKNOWN($s)";; esac
  if [ "$s" != "$last" ]; then echo "$(stamp) STATUS $s" >> "$LOG"; last="$s"; fi
  if [ "$s" = "QUEUED" ]; then
    [ -z "$queued_since" ] && queued_since=$(date +%s)
    mins=$(( ($(date +%s) - queued_since) / 60 ))
    if [ $mins -ge $(( (warned + 1) * 30 )) ]; then warned=$((warned + 1)); echo "$(stamp) ALERT still queued after $mins min" >> "$LOG"; fi
  else
    queued_since=""; warned=0
  fi
  if [ "$s" = "RUNNING" ] || [ "$s" = "COMPLETE" ] || [ "$s" = "ERROR" ]; then
    $KG kernels logs "$K" 2>/dev/null | grep -E "rep [0-9]+:|fits done|Traceback|Error|error|assert|Killed|saved|verdict|exit code" \
      | grep -v -x -F -f "$SEEN" | while IFS= read -r line; do
          echo "$line" >> "$SEEN"; echo "$(stamp) LOG ${line:0:300}" >> "$LOG"; done
  fi
  case "$s" in
    COMPLETE|ERROR|CANCEL_*)
      mkdir -p "$OUT"
      $KG kernels output "$K" -p "$OUT" >> "$LOG.download" 2>&1
      echo "$(stamp) DONE $s; output in $OUT ($(find "$OUT" -type f | wc -l | tr -d ' ') files)" >> "$LOG"
      exit 0;;
  esac
  sleep 60
done
