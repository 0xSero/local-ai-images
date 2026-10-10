#!/bin/bash
# Omarchy Local AI passes an Intel GPU as its render node only (--device /dev/dri/renderDNNN). oneCCL, which vLLM's XPU
# worker initialises even for one rank, opens /dev/dri/by-path and fails with "opendir failed: could not open device
# directory" when it is missing. Link every passed render node under /dev/dri/by-path/pci-<bdf>-render (the name udev
# gives it), then run the base image's own entrypoint: oneAPI + oneCCL environment, then `vllm serve "$@"`.
set -e
mkdir -p /dev/dri/by-path
for r in /dev/dri/renderD*; do
    [ -e "$r" ] || continue
    ln -sf "../${r##*/}" "/dev/dri/by-path/pci-$(basename "$(readlink -f "/sys/class/drm/${r##*/}/device")")-render"
done
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force
exec vllm serve "$@"
