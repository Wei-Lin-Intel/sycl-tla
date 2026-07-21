# Copyright (C) 2026 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import os
import unittest

import torch


def online_merge(outputs, lses):
    output = outputs[0].float()
    lse = lses[0].float()
    for partial_output, partial_lse in zip(outputs[1:], lses[1:]):
        maximum = torch.maximum(lse, partial_lse)
        old_weight = torch.exp(lse - maximum)
        new_weight = torch.exp(partial_lse - maximum)
        denominator = old_weight + new_weight
        output = (
            old_weight.div(denominator).unsqueeze(-1) * output
            + new_weight.div(denominator).unsqueeze(-1)
            * partial_output.float()
        )
        lse = maximum + torch.log(denominator)
    return output, lse


class OnlineMergeTest(unittest.TestCase):
    def test_matches_concatenated_attention(self):
        torch.manual_seed(7)
        q = torch.randn(2, 3, 11, 16)
        keys = [torch.randn(2, 3, 5, 16) for _ in range(4)]
        values = [torch.randn(2, 3, 5, 12) for _ in range(4)]
        outputs, lses = [], []
        for key, value in zip(keys, values):
            scores = torch.matmul(q, key.transpose(-1, -2)) / math.sqrt(16)
            lses.append(torch.logsumexp(scores, dim=-1))
            outputs.append(torch.matmul(torch.softmax(scores, dim=-1), value))

        merged, merged_lse = online_merge(outputs, lses)
        scores = torch.matmul(
            q, torch.cat(keys, dim=2).transpose(-1, -2)
        ) / math.sqrt(16)
        expected_lse = torch.logsumexp(scores, dim=-1)
        expected = torch.matmul(
            torch.softmax(scores, dim=-1), torch.cat(values, dim=2)
        )
        torch.testing.assert_close(merged, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(
            merged_lse, expected_lse, rtol=2e-5, atol=2e-5
        )


def _load_extension():
    try:
        import sycl_tla_fmha

        return sycl_tla_fmha
    except ImportError:
        return None


def _reference(q, keys, values):
    scores = torch.matmul(
        q.float(), torch.cat(keys, dim=2).float().transpose(-1, -2)
    ) / math.sqrt(q.size(-1))
    return (
        torch.matmul(torch.softmax(scores, dim=-1), torch.cat(values, dim=2).float()),
        torch.logsumexp(scores, dim=-1),
    )


@unittest.skipUnless(torch.xpu.is_available(), "requires XPU")
class StreamingRingXPUTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.extension = _load_extension()
        if cls.extension is None:
            raise unittest.SkipTest("sycl_tla_fmha is not importable")

    def _world_one_metadata(self, q, v, groups=4):
        k_workspace = torch.empty(
            (0, *q.shape), dtype=q.dtype, device=q.device
        )
        v_workspace = torch.empty(
            (0, *v.shape), dtype=v.dtype, device=v.device
        )
        signals = torch.empty((0, groups), dtype=torch.int32, device=q.device)
        k_ptrs = torch.zeros(1, dtype=torch.int64, device=q.device)
        v_ptrs = torch.zeros(1, dtype=torch.int64, device=q.device)
        signal_ptrs = torch.zeros(1, dtype=torch.int64, device=q.device)
        return (
            k_workspace,
            v_workspace,
            signals,
            k_ptrs,
            v_ptrs,
            signal_ptrs,
        )

    def test_world_one_supported_dimensions_and_existing_api(self):
        device = torch.device("xpu", 0)
        for dimension in (64, 96, 128, 192):
            q = torch.randn(
                1, 2, 65, dimension, dtype=torch.bfloat16, device=device
            )
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            metadata = self._world_one_metadata(q, v)
            output, lse = self.extension.streaming_ring_bf16_lse(
                q, k, v, *metadata, 0, 1, dimension, False, 4
            )
            expected, expected_lse = _reference(q, [k], [v])
            torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
            torch.testing.assert_close(
                lse, expected_lse, rtol=3e-2, atol=3e-2
            )

        legacy = self.extension.prefill_bf16_tensor(q, k, v)
        torch.testing.assert_close(legacy, expected, rtol=3e-2, atol=3e-2)

    def test_validation(self):
        device = torch.device("xpu", 0)
        q = torch.randn(1, 2, 17, 64, dtype=torch.bfloat16, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        metadata = self._world_one_metadata(q, v)
        call = self.extension.streaming_ring_bf16
        with self.assertRaisesRegex(RuntimeError, "causal"):
            call(q, k, v, *metadata, 0, 1, 1, True, 4)
        with self.assertRaisesRegex(RuntimeError, "sequence"):
            call(q, k[:, :, :-1], v, *metadata, 0, 1, 1, False, 4)
        with self.assertRaisesRegex(RuntimeError, "head counts"):
            call(q, k[:, :1], v[:, :1], *metadata, 0, 1, 1, False, 4)
        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
            call(q.float(), k, v, *metadata, 0, 1, 1, False, 4)
        with self.assertRaisesRegex(RuntimeError, "XPU"):
            call(q.cpu(), k, v, *metadata, 0, 1, 1, False, 4)
        bad_ptrs = metadata[3].to(torch.int32)
        with self.assertRaisesRegex(RuntimeError, "torch.int64"):
            call(
                q,
                k,
                v,
                *metadata[:3],
                bad_ptrs,
                *metadata[4:],
                0,
                1,
                1,
                False,
                4,
            )
        with self.assertRaisesRegex(RuntimeError, "world_size entries"):
            call(
                q,
                k,
                v,
                *metadata[:3],
                metadata[3][:0],
                *metadata[4:],
                0,
                1,
                1,
                False,
                4,
            )
        bad_signals = torch.empty(
            (0, 3), dtype=torch.int32, device=device
        )
        with self.assertRaisesRegex(RuntimeError, "signal_pad"):
            call(
                q,
                k,
                v,
                metadata[0],
                metadata[1],
                bad_signals,
                *metadata[3:],
                0,
                1,
                1,
                False,
                4,
            )
        with self.assertRaisesRegex(RuntimeError, "positive"):
            call(q, k, v, *metadata, 0, 1, 0, False, 4)


def run_distributed():
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    extension = _load_extension()
    if extension is None:
        raise RuntimeError("sycl_tla_fmha is not importable")
    dist.init_process_group(backend=os.environ.get("TORCH_DIST_BACKEND", "ccl"))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world not in (2, 4):
        raise RuntimeError("distributed test requires world size 2 or 4")
    device = torch.device("xpu", rank)
    torch.xpu.set_device(device)
    groups = 8

    def symmetric_tensor(shape, dtype):
        numel = math.prod(shape)
        flat = symm_mem.empty(max(1, numel), dtype=dtype, device=device)
        handle = symm_mem.rendezvous(flat, dist.group.WORLD)
        return flat[:numel].view(shape), handle

    torch.manual_seed(100 + rank)
    for dimension, sequence in ((64, 65), (128, 257)):
        q = torch.randn(
            1, 2, sequence, dimension, dtype=torch.bfloat16, device=device
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        k_workspace, kh = symmetric_tensor(
            (world - 1, *k.shape), torch.bfloat16
        )
        v_workspace, vh = symmetric_tensor(
            (world - 1, *v.shape), torch.bfloat16
        )
        signals, sh = symmetric_tensor((world - 1, groups), torch.int32)
        signals.zero_()
        k_ptrs = torch.tensor(kh.buffer_ptrs, dtype=torch.int64, device=device)
        v_ptrs = torch.tensor(vh.buffer_ptrs, dtype=torch.int64, device=device)
        signal_ptrs = torch.tensor(
            sh.buffer_ptrs, dtype=torch.int64, device=device
        )
        dist.barrier()

        gathered_k = [torch.empty_like(k) for _ in range(world)]
        gathered_v = [torch.empty_like(v) for _ in range(world)]
        dist.all_gather(gathered_k, k)
        dist.all_gather(gathered_v, v)
        expected, expected_lse = _reference(q, gathered_k, gathered_v)
        for iteration in (1, 2):
            output, lse = extension.streaming_ring_bf16_lse(
                q,
                k,
                v,
                k_workspace,
                v_workspace,
                signals,
                k_ptrs,
                v_ptrs,
                signal_ptrs,
                rank,
                world,
                iteration,
                False,
                groups,
            )
            torch.testing.assert_close(
                output, expected, rtol=3e-2, atol=3e-2
            )
            torch.testing.assert_close(
                lse, expected_lse, rtol=3e-2, atol=3e-2
            )
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run_distributed()
    else:
        unittest.main()
