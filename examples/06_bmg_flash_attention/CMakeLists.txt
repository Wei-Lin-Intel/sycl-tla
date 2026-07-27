# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# Copyright (C) 2025 - 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


set(CUTLASS_APPLICATIONS_DIR ${CUTLASS_DIR}/applications)

set(TEST_NO_PAGED "")
set(TEST_PAGED "--use_paged_kv")

option(SYCL_TLA_BUILD_PYTHON_FMHA "Build the sycl_tla_fmha Python extension" OFF)

if(SYCL_TLA_BUILD_PYTHON_FMHA)
  find_package(Python3 3.5 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent.as_posix())"
    RESULT_VARIABLE _torch_query_result
    OUTPUT_VARIABLE _torch_package_dir
    ERROR_VARIABLE _torch_query_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(NOT _torch_query_result EQUAL 0)
    message(FATAL_ERROR
      "Cannot import torch from ${Python3_EXECUTABLE}: ${_torch_query_error}")
  endif()

  set(TORCH_INCLUDE_DIRS
    "${_torch_package_dir}/include"
    "${_torch_package_dir}/include/torch/csrc/api/include")
  set(TORCH_LIB_DIR "${_torch_package_dir}/lib")

  find_library(
    TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT TORCH_PYTHON_LIBRARY)
    message(FATAL_ERROR "Cannot find torch_python library required for pybind Tensor casters")
  endif()

  find_library(
    TORCH_LIBRARY
    NAMES torch
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
  )
  find_library(
    TORCH_CPU_LIBRARY
    NAMES torch_cpu
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
  )
  find_library(
    C10_LIBRARY
    NAMES c10
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
  )
  find_library(
    C10_XPU_LIBRARY
    NAMES c10_xpu
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
  )
  find_library(
    TORCH_XPU_LIBRARY
    NAMES torch_xpu
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH
  )
  if(NOT TORCH_LIBRARY OR NOT TORCH_CPU_LIBRARY OR NOT C10_LIBRARY OR NOT C10_XPU_LIBRARY OR NOT TORCH_XPU_LIBRARY)
    message(FATAL_ERROR
      "Cannot find required Torch libraries under ${TORCH_LIB_DIR}")
  endif()

  add_library(sycl_tla_fmha MODULE py_fmha_module.cpp)
  cutlass_apply_standard_compile_options(sycl_tla_fmha)
  target_compile_features(sycl_tla_fmha PRIVATE cxx_std_17)

  # ---- Ring-Attention P2P deps: Intel MPI + Level-Zero loader ----
  find_path(IMPI_INCLUDE_DIR mpi.h
    HINTS ENV I_MPI_ROOT PATH_SUFFIXES include)
  find_library(IMPI_LIBRARY NAMES mpi impi
    HINTS ENV I_MPI_ROOT PATH_SUFFIXES lib lib/release)
  find_library(ZE_LIBRARY NAMES ze_loader
    HINTS ENV ONEAPI_ROOT ENV LD_LIBRARY_PATH PATH_SUFFIXES lib lib64)
  if(NOT IMPI_INCLUDE_DIR OR NOT IMPI_LIBRARY)
    message(FATAL_ERROR
      "sycl_tla_fmha Ring-Attention requires Intel MPI. "
      "Source oneAPI setvars.sh so I_MPI_ROOT is available.")
  endif()
  if(NOT ZE_LIBRARY)
    message(FATAL_ERROR
      "sycl_tla_fmha Ring-Attention requires the Level-Zero loader (ze_loader).")
  endif()

  target_include_directories(
    sycl_tla_fmha
    PRIVATE
    ${CUTLASS_EXAMPLES_COMMON_SOURCE_DIR}
    ${CUTLASS_EXAMPLES_UTILS_DIR}
    ${CUTLASS_APPLICATIONS_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${IMPI_INCLUDE_DIR}
    # comm headers are reachable as "flash_attention_v2/comm/ring_symm.hpp"
    # via CUTLASS_APPLICATIONS_DIR above; add the folder too so bare
    # "ring_symm.hpp" includes also resolve.
    ${CUTLASS_APPLICATIONS_DIR}/flash_attention_v2/comm
  )
  target_link_libraries(
    sycl_tla_fmha
    PRIVATE
    Python3::Module
    cutlass_tools_util_includes
    ${TORCH_PYTHON_LIBRARY}
    ${TORCH_LIBRARY}
    ${TORCH_CPU_LIBRARY}
    ${C10_LIBRARY}
    ${C10_XPU_LIBRARY}
    ${TORCH_XPU_LIBRARY}
    ${IMPI_LIBRARY}
    ${ZE_LIBRARY}
  )

  if(CUTLASS_ENABLE_SYCL)
    add_onemkl_to_target(TARGET sycl_tla_fmha)
    add_sycl_to_target(TARGET sycl_tla_fmha)
  endif()

  set_target_properties(
    sycl_tla_fmha
    PROPERTIES
    BUILD_RPATH "${TORCH_LIB_DIR}"
    INSTALL_RPATH "${TORCH_LIB_DIR}"
    PREFIX ""
    OUTPUT_NAME "sycl_tla_fmha"
  )
endif()

foreach(HEAD_DIM 64 96 128 192)
  foreach(INPUT_TYPE bfloat16_t float_e5m2_t float_e4m3_t)
    cutlass_example_add_executable(
      06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM}
      06_xe_fmha_fwd.cpp
    )

    cutlass_example_add_executable(
      06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM}
      06_xe_fmha_fwd.cpp
    )

    if (NOT HEAD_DIM STREQUAL 192)
      # specific test for persistent kernel
      cutlass_example_add_executable(
        06_xe_fmha_fwd_decode_persistent_${INPUT_TYPE}_hdim${HEAD_DIM}
        06_xe_fmha_fwd.cpp
      )
    endif()

    if(INPUT_TYPE STREQUAL "bfloat16_t")
      set(INPUT_MACRO "IS_BFLOAT16")
    elseif(INPUT_TYPE STREQUAL "float_e5m2_t")
      set(INPUT_MACRO "IS_FLOAT_E5M2")
    elseif(INPUT_TYPE STREQUAL "float_e4m3_t")
      set(INPUT_MACRO "IS_FLOAT_E4M3")
    endif()

    target_compile_definitions(06_xe_fmha_fwd_prefill_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} PREFILL SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    target_compile_definitions(06_xe_fmha_fwd_decode_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    if (NOT HEAD_DIM STREQUAL 192)
      target_compile_definitions(06_xe_fmha_fwd_decode_persistent_${INPUT_TYPE}_hdim${HEAD_DIM} PRIVATE HEAD_DIM=${HEAD_DIM} DECODE PERSISTENT SHOW_DIFF=1 INPUT_TYPE=${INPUT_TYPE} ${INPUT_MACRO})
    endif()
  endforeach()
endforeach()
