# Copyright (C) 2025 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Example 14: Xe FMHA pipelined QK–softmax overlap
#
# This directory contains documentation, annotated pseudocode, and a benchmark
# scaffold for the 2-slot block pipeline added to xe_fmha_fwd_mainloop.hpp.
# There is no standalone binary in this example; the pipeline is exercised via
# the existing 06_bmg_flash_attention example.
#
# To validate the pipelined mainloop:
#
#   ninja 06_xe_fmha_fwd
#   ./examples/sycl/06_bmg_flash_attention/06_xe_fmha_fwd \
#       --seq_len_qo=1024 --seq_len_kv=1024 \
#       --head_size_vo=128 --head_size_qk=128 \
#       --verify=true
#
# See README.md in this directory for profiling and ISA inspection guidance.
message(STATUS "Example 14: Xe FMHA pipelined QK-softmax (documentation only)")
