#!/usr/bin/env python3
"""Validation-derived, unmasked int8 scatter Store on Ascend950PR.

Source: simt_store_none_f1, full_store_i8_2x256_w8_f1_none_k512_sink16_g1,
from the frozen IndirectGatherMemory validation/guard panel (2026-10-21).
Dependencies: numpy, torch, torch_npu, Triton-Ascend with PR2528 route support.
E=512, shape=(2,256), W=8, int32 unique indices in [0,32768).
Use fresh, unfilled target storage for every timed launch; no timing correction,
forced route, private compiler patch, or single-AIV command override is applied.
"""

import argparse
import json
from pathlib import Path
import random
import statistics

import numpy as np
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver

from profile_utils import profile_kernel


@triton.jit
def indirect_store_i8_kernel(X, Index, Values, E: tl.constexpr, SHAPE: tl.constexpr):
    lane = tl.arange(0, E)
    logical = tl.program_id(0) * E + lane
    offset = tl.reshape(tl.load(Index + logical), SHAPE)
    value = tl.reshape(tl.load(Values + logical), SHAPE)
    tl.store(X + offset, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--route', choices=('auto', 'simd', 'simt_only'), required=True)
    parser.add_argument('--factor', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--seed', type=int, default=90521000)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error('warmup must be nonnegative; repeats must be positive')

    e, domain, programs = 512, 32768, 1
    indices = np.array(random.Random(args.seed).sample(range(domain), e), dtype=np.int32)
    values = (np.arange(e) % 31 + 2).astype(np.int8)
    source = (np.arange(domain + 64) % 97 + 1).astype(np.int8)
    index = torch.from_numpy(indices).npu()
    payload = torch.from_numpy(values).npu()
    target = torch.from_numpy(source.copy()).npu()
    cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())['num_vectorcore'])
    options = dict(num_warps=8, superblock_factor=args.factor,
                   compile_mode='simd_simt' if args.route == 'auto' else args.route,
                   auto_simt_scope_mode='auto' if args.route == 'auto' else 'off',
                   auto_simt_scope_dump=str(args.report) if args.route == 'auto' else '',
                   logical_program_count_hint=programs, physical_vector_core_count_hint=cores,
                   compile_on_910_95=True, multibuffer=False, enable_dynamic_cv_pipeline=False)

    def run(x):
        indirect_store_i8_kernel[(programs,)](x, index, payload, E=e, SHAPE=(2, 256), **options)

    run(target[32:-32])
    torch.npu.synchronize()
    expected = source.copy()
    expected[32 + indices] = values
    np.testing.assert_array_equal(target.cpu().numpy(), expected)

    # The shared profiler makes 28 calls. Fresh targets preserve the measured
    # first-store contract rather than silently timing repeated writes in place.
    slots = args.warmup + args.repeats + 32
    arena = torch.empty((slots, domain + 64), dtype=torch.int8, device='npu')
    arena[:, :32].fill_(23)
    arena[:, -32:].fill_(23)
    calls = 0

    def launch():
        nonlocal calls
        if calls >= slots:
            raise RuntimeError('fresh target arena exhausted')
        run(arena[calls, 32:-32])
        calls += 1

    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        begin.record()
        launch()
        end.record()
        torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000)
    profiler = profile_kernel(launch, args.profile_dir, 'indirect_store_i8_kernel') if args.profile_dir else None
    torch.npu.synchronize()
    observed = arena[:calls].cpu().numpy()
    assert np.all(observed[:, :32] == 23) and np.all(observed[:, -32:] == 23), 'canary changed'
    np.testing.assert_array_equal(observed[:, 32 + indices], np.broadcast_to(values, (calls, e)))
    print(json.dumps(dict(case='maxh2018_indirect_store_i8_2x256_w8', correct=True,
                          elapsed_us=dict(median=statistics.median(samples)), profiler=profiler,
                          seed=args.seed, target_state='fresh_no_fill'), indent=2))


if __name__ == '__main__':
    main()
