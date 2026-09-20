#!/bin/sh
set -e
D="/opt/models/Ternary-Bonsai-2-27B"; mkdir -p "$D"
g(){ echo "$2  $D/$1" | sha256sum -c - >/dev/null 2>&1 || { echo "[deploy-preflight] fetch $1"; curl -fL --max-time 1800 "https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/6ed5e12bf84b7a63069882c91dd9e9218647d17b/$1" -o "$D/$1"; echo "$2  $D/$1" | sha256sum -c -; }; }
g Ternary-Bonsai-2-27B-PTQ1_0.gguf 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
g Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf 6807ede61d570bb86ba34b756a0fa109edc33668604de867c6ea6d8f1d631903
touch "$D/.omp-ready"
exec /opt/llama/entrypoint.sh "$@"
