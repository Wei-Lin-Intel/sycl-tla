/***************************************************************************************************
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <torch/extension.h>

#include <tuple>

at::Tensor streaming_ring_bf16(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &k_workspace, const at::Tensor &v_workspace,
    const at::Tensor &signal_pad, const at::Tensor &peer_k_workspace_ptrs,
    const at::Tensor &peer_v_workspace_ptrs,
    const at::Tensor &peer_signal_ptrs, int64_t rank, int64_t world_size,
    int64_t iteration, bool is_causal = false, int64_t work_groups = 8);

std::tuple<at::Tensor, at::Tensor> streaming_ring_bf16_lse(
    const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
    const at::Tensor &k_workspace, const at::Tensor &v_workspace,
    const at::Tensor &signal_pad, const at::Tensor &peer_k_workspace_ptrs,
    const at::Tensor &peer_v_workspace_ptrs,
    const at::Tensor &peer_signal_ptrs, int64_t rank, int64_t world_size,
    int64_t iteration, bool is_causal = false, int64_t work_groups = 8);
