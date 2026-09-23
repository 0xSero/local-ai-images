#!/usr/bin/env bash
# MoE front: rsync THIS checkout (the worktree the script lives in) to the MoE engineer's own copy on the H200 box and
# start the kernel build there in tmux (CPU only, niced). Never touches /scratch/aikido/exl3 (dense front).
#   csrc/sync_build_moe.sh [full|small|nobuild]      then poll: csrc/sync_build_moe.sh status
MODE=${1:-full}
DIR=${AIKIDO_BOX_DIR:-/scratch/aikido/exl3-moe}
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SSH="ssh -i $HOME/.ssh/aikido -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3"
if [ "$MODE" = status ]; then
  $SSH ubuntu@66.201.5.200 "grep -E 'BUILD_OK|BUILD_FAILED' $DIR/csrc/build.log 2>/dev/null || echo STILL_BUILDING; grep -n -E 'rror:' $DIR/csrc/build.log | head -${2:-25}" 2>/dev/null
  exit 0
fi
rsync -a --timeout=60 -e "$SSH" --exclude .git --exclude .claude --exclude __pycache__ --exclude build "$HERE/" ubuntu@66.201.5.200:$DIR/ 2>/dev/null || { echo RSYNC_FAILED; exit 1; }
[ "$MODE" = nobuild ] && { echo SYNCED; exit 0; }
$SSH ubuntu@66.201.5.200 "rm -f $DIR/csrc/build.log; tmux kill-session -t kbuild-moe 2>/dev/null; tmux new-session -d -s kbuild-moe '$DIR/csrc/build.sh $MODE'" 2>/dev/null
echo "SYNCED, build started in tmux kbuild-moe ($MODE)"
