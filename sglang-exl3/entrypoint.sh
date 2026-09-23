#!/bin/sh
# One image, two callers (same contract as ./sglang): a registry recipe passes the server argv as separate
# arguments (exec'd directly, PID 1); the campaign's validation bootstrap passes one shell string (/bin/sh -c).
set -e
if [ "$#" -eq 0 ]; then
    exec python3 -m sglang.launch_server --help
fi
if [ "$#" -eq 1 ]; then
    exec /bin/sh -c "$1"
fi
exec "$@"
