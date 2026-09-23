// One instantiation unit: AIKIDO_INST_MB=<0..4>, AIKIDO_INST_CB=<0..2>, AIKIDO_INST_KB=<3|4|5|6> (see setup.py; 3 = K3, 5 = K5).
#define MARLIN_NAMESPACE_NAME aikido_exl3_marlin
#define AIKIDO_KERNEL_DEFINED
#include "exl3_hopper_template.h"
#include "exl3_hopper_kernels.h"

namespace MARLIN_NAMESPACE_NAME {
#define AIKIDO_INST(threads, tn, tk, mb, cb, kb) \
  template __global__ void AIKIDO_KERNEL(threads, tn, tk, mb, cb, kb)(AIKIDO_KERNEL_PARAMS);
AIKIDO_THREAD_CFGS(AIKIDO_INST, AIKIDO_INST_MB, AIKIDO_INST_CB, AIKIDO_INST_KB)
}  // namespace MARLIN_NAMESPACE_NAME
