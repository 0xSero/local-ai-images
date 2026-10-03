#!/bin/bash
# Host memory guard for GB10 unified memory: if MemAvailable < $1 GiB, kill the st-ds41-* containers (a GPU OOM
# otherwise wedges the host until the SBSA watchdog reboots it). Logs to ~/dsv41-st/memguard.log.
FLOOR=${1:-3}; LOG=~/dsv41-st/memguard.log; PID=~/dsv41-st/memguard.pid
[ "$1" = stop ] && { [ -f $PID ] && kill $(cat $PID) 2>/dev/null; rm -f $PID; exit 0; }
[ -f $PID ] && kill $(cat $PID) 2>/dev/null   # replace a previous instance
echo $$ > $PID
echo "$(date -Is) memguard start floor ${FLOOR}G pid $$" >> $LOG
n=0
while sleep 1; do
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  n=$((n+1))
  if [ "$a" -lt 10 ] && [ $((n % 10)) = 0 ]; then   # low-memory trace: what holds host RSS
    echo "$(date -Is) avail ${a}G top-rss: $(ps -eo rss,comm --sort=-rss | sed -n 2,4p | awk '{printf "%s=%.1fG ", $2, $1/1048576}')" >> $LOG
  fi
  if [ "$a" -lt "$FLOOR" ]; then
    ids=$(docker ps -q --filter name=st-ds41)
    [ -n "$ids" ] && { echo "$(date -Is) MemAvailable ${a}G < ${FLOOR}G: killing $ids" >> $LOG; docker kill $ids >> $LOG 2>&1; }
  fi
done
