// One MoE instantiation unit: AIKIDO_INST_MB=<0..4> (moe block family), AIKIDO_INST_CB=<0..2>, AIKIDO_INST_BITS=<3|4> (see setup.py).
#define MARLIN_NAMESPACE_NAME aikido_exl3_marlin_moe
#define AIKIDO_MOE_KERNEL_DEFINED
#include "exl3_hopper_moe_template.h"
#include "exl3_hopper_moe_kernels.h"

namespace MARLIN_NAMESPACE_NAME {
#define AIKIDO_MOE_INST(threads, tn, tk, mb, cb, bits) \
  template __global__ void AIKIDO_MOE_KERNEL(threads, tn, tk, mb, cb, bits)(AIKIDO_MOE_KERNEL_PARAMS);
AIKIDO_MOE_THREAD_CFGS(AIKIDO_MOE_INST, AIKIDO_INST_MB, AIKIDO_INST_CB, AIKIDO_INST_BITS)
}  // namespace MARLIN_NAMESPACE_NAME
