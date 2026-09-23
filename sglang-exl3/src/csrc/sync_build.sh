#!/usr/bin/env bash
# From the Mac: rsync the package to the H200 box, build the kernels in tmux, wait, print the verdict.
# AIKIDO_BOX_DIR=/scratch/aikido/exl3-dev builds a private copy (own python sources + build/lib) so running gates /
# servers that import /scratch/aikido/exl3 are not disturbed; promote by running again without it.
MODE=${1:-small}
DIR=${AIKIDO_BOX_DIR:-/scratch/aikido/exl3}
SSH="ssh -i $HOME/.ssh/aikido -o IdentitiesOnly=yes"
rsync -a -e "$SSH" --exclude .git --exclude __pycache__ --exclude build $HOME/aikido/exl3/ ubuntu@66.201.5.200:$DIR/ 2>/dev/null
$SSH ubuntu@66.201.5.200 "rm -f $DIR/csrc/build.log; tmux kill-session -t kbuild-$(basename $DIR) 2>/dev/null; tmux new-session -d -s kbuild-$(basename $DIR) 'AIKIDO_JOBS=${AIKIDO_JOBS:-12} $DIR/csrc/build.sh $MODE'" 2>/dev/null
for i in $(seq 1 110); do
  sleep 5
  R=$($SSH ubuntu@66.201.5.200 "grep -E 'BUILD_OK|BUILD_FAILED' $DIR/csrc/build.log 2>/dev/null" 2>/dev/null)
  [ -n "$R" ] && break
done
echo "result: ${R:-STILL_BUILDING}"
$SSH ubuntu@66.201.5.200 "grep -n -E 'error|rror:' $DIR/csrc/build.log | head -${2:-25}" 2>/dev/null
