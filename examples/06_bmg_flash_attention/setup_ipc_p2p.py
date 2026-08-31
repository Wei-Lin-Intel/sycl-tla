"""
Build the sycl_tla_ipc_p2p extension:

    python setup_ipc_p2p.py build_ext --inplace

Requires a oneAPI DPC++ compiler/runtime that implements
sycl_ext_oneapi_inter_process_communication.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension

# We deliberately build with icpx directly rather than the CUDAExtension path.
from torch.utils.cpp_extension import CppExtension

CXX = os.environ.get("CXX", "icpx")
os.environ["CXX"] = CXX
os.environ["CC"] = os.environ.get("CC", "icx")

module = CppExtension(
    name="sycl_tla_ipc_p2p",
    sources=["py_ipc_p2p_module.cpp"],
    extra_compile_args=[
        "-fsycl",
        "-fsycl-targets=spir64",
        "-O3",
        "-std=c++17",
    ],
    extra_link_args=[
        "-fsycl",
    ],
)

setup(
    name="sycl_tla_ipc_p2p",
    ext_modules=[module],
    cmdclass={"build_ext": BuildExtension},
)
