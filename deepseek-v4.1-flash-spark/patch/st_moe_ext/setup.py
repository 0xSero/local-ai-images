# SPDX-License-Identifier: MIT
# Build: EXL3_EXT_DIR=<exllamav3 1.5.3>/exllamav3/exllamav3_ext TORCH_CUDA_ARCH_LIST=12.1 python3 setup.py build_ext --inplace
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_dir = os.environ["EXL3_EXT_DIR"]
setup(
    name="st_moe_ext",
    ext_modules=[CUDAExtension(
        "st_moe_ext", ["st_moe_ext.cu"],
        include_dirs=[ext_dir],
        extra_compile_args={"cxx": ["-O3", "-std=c++20"],
                            "nvcc": ["-O3", "--use_fast_math", "-std=c++20", "--expt-relaxed-constexpr",
                                     "-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                                     "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__",
                                     "-Xptxas", "-v"]})],
    cmdclass={"build_ext": BuildExtension},
)
