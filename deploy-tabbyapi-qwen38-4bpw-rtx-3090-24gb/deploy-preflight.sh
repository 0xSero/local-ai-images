#!/bin/sh
# Deploy entrypoint. Canonical interface (Main-approved): env OMP_SSH_FIRST + /opt/omp-acquire.
#  OMP_SSH_FIRST=1 : exec "$@" when the runner supplied a command (its CONTAINER_STARTUP sshd
#                    bootstrap) so diagnostic SSH is reachable WITHOUT waiting on any model fetch.
#                    The image declares no CMD, so a rent that sets OMP_SSH_FIRST=1 without
#                    passing dockerArgs used to fall through with an empty "$@", exit 2, and
#                    restart-loop: the container never came up and no port was ever mapped.
#                    That case now starts sshd from inside the image instead of failing.
#  unset (default) : blocking pinned acquisition (/opt/omp-acquire, fail-closed) THEN exec serve.
set -e
if [ "${OMP_SSH_FIRST:-0}" = "1" ]; then
  if [ "$#" -ge 1 ]; then
    echo "[deploy-preflight] OMP_SSH_FIRST=1: exec bootstrap now; run /opt/omp-acquire over SSH before serving"
    exec "$@"
  fi
  echo "[deploy-preflight] OMP_SSH_FIRST=1 with no command: starting sshd in-image"
  exec /opt/omp-sshd
fi
/opt/omp-acquire
cd /app
exec python3 main.py
