# Random-prior indirect load calibration

This change replaces the **load resource**, not the entire Stage, for a narrow
calibrated domain on `Ascend950PR/dav-c310`. Both models estimate real-device
payload-minus-ALU-baseline increments in SYS_CNT ticks. One tick is one
CostModel `system_cycle`; no compute-clock conversion is applied.

## Profile switches and accounting

Optional profile fields select versioned models:

```json
"simt_indirect_load_model": "random_i32_six_term_20261007",
"simd_indirect_load_model": "random_dtype_matched_ab_20261008"
```

Absent switches and unsupported workloads retain legacy pricing. JSON reports
`indirect_load_pricing` per implementation. Unknown model identifiers are
rejected. Coefficients reside in `StageCostModels.cpp` under the versioned model.

For a matched load, replace the transaction-rate term and suppress its old
`indirectDependencyLatencyCycles`. Preserve independent scalar, compute, store,
control, synchronization, issue-floor and setup charges. Scope-transition
pricing is unchanged. SIMT predictions already include warp parallelism:
multiply only this resource term by the framework's parallelism factor before
the common division. SIMD has no such division. Loop iteration count still
applies normally.

## SIMT equation

Let E be the number of payload elements, W the compile-time warp count,
q=E/(32W), and C the last extent for rank two (C=32 for rank one). Define:

```text
H = max(4*q/C - 2, 0)
S = 1 if E <= 128 else 0
D = max(log2(32/C), 0)
F = 1 for rank one, otherwise 0
T = a + b*E + c*H + d*S + e*E*D + f*E*F
```

| Coefficient | SYS_CNT-domain value |
|---|---:|
| a | 62.930686069640686 |
| b | 1.5502588407166296 |
| c | 141.51064147799983 |
| d | 47.159408679724294 |
| e | 0.05304437217224805 |
| f | 0.19338028618547126 |

Domain: one INT32 `tt.load` owned by an `IndirectGatherMemory` Stage;
W in {1,2,4,8,16,32,64}, q in {1,2,4}; rank one with a loaded-index axis,
or rank two with both axes loaded-index-derived, or fixed outer stride 4096
and loaded inner index. Rank-two C is in {4,8,16,32,64,128}, with at least two
rows. Computed-nonaffine `outer_square` is excluded.

The six-term simplification retained 357/360 historical test inputs within
20% (MAPE 5.06%, maximum 25.79%). It was examined after test-set comparison;
this is not a fresh blind-test claim for the simplified equation.

## SIMD dtype/rank equation (2026-10-08)

The David profile now selects `random_dtype_matched_ab_20261008`.
With E = payload tensor elements, b = bytes per element and r = tensor rank:

```text
T = beta_small32 * E    if b == 4 and r == 1 and E <= 16
T = beta_regular * E    otherwise within the guarded domain
beta_small32 = 49.57621548794853
beta_regular = 81.94869549595556
```

Values are effective SYS_CNT/system_cycle increments per element, not
isolated opcode latencies. No warp division or runtime-index feature is used.
INT32 E16 instruction traces show unrolled scalar GM loads with overlapping
execution intervals; INT32 E32 uses a loop. INT8 E16 also unrolls but lacks
the same overlap. The tiny-32-bit coefficient therefore represents measured
grouping/scheduling effects, not a universal hardware threshold.

Guarded domain: one mask-free loaded-index `tt.load` in
`IndirectGatherMemory`, rank 1--5, power-of-two non-unit extents, E=4--2048.
The Stage may also own the load's address/shape producers; it need not contain
only one operation. Those helpers retain their normal resource charges.
Rank two additionally requires E>=32 and C=4--128. Supported TTIR result
types are i8/i16/i32/i64, f16/bf16/f32, f8E4M3FN and f8E5M2. Triton bool
loads normalize to i8; raw i1 is not admitted. The float8e4b15 experiment
uses i8 TTIR plus a raw-byte host-pointer ABI, not a stock FP8 launcher.
FP64 and other unmeasured FP8 formats retain legacy pricing.

The whole pointer must depend on a loaded index and every summarized extent
must match the type. An axis may be opaque/opaque_loaded, or a positive proven
fixed stride on an outer axis. Reshape-erased per-axis provenance (`opaque`)
does not invalidate the independently proven whole-pointer dependency.
A structured tail or computed_nonaffine axis is not forced into this fit.
These are bounded generalization guards, not a claim that every combination
in the Cartesian product has been measured. Rank 6--8 remains excluded;
the existing classifier's rank gate is unchanged.

The main matrix planned 514 configurations: 493 measured and 21 compiler
failures. Main splits: 283 train, 48 development, 90 test, 48 warp controls,
24 high-rank controls; eight random inputs/configuration. Another 45 tiny
boundary configurations were collected. Payload addresses sample random
unique elements without sorting; matrix, fixed-outer/loaded-inner and
loaded-outer/loaded-inner templates are represented. This is an explicit
random-scattered prior, not a compiler proof or worst-case bound.

A and B derive from the same real Triton lowering. B replaces only the
payload GM load by a fold of its complete address; it retains index reads,
address dependencies, loop structure and same-width UB output. The timer
surrounds the outer payload loop, after index materialization and before
output DMA. A-B is an effective increment: extra baseline ALU and changed
scheduling cannot be perfectly cancelled. Earlier FP32 baseline sensitivity
was at most 5.37% of the primary label.

After freezing both coefficients, new random seeds on 90 existing
configurations produced **720/720 inputs within 20%**, MAPE 4.03%, P95 11.45%,
maximum 16.48%. This is new-input validation, not unseen-shape validation.
Retrospective old-test results are 720/720, MAPE 3.77%, maximum 18.90%;
five old-training inputs remain slightly above 20% (maximum 20.14%).
No failing input was removed for model accuracy. On the shared device a
measurement-only >5% cross-round spread rule triggered whole-configuration
repeats; median aggregation across nine rounds retained all raw slow samples
and did not change the frozen coefficients.

The prior `random_f32_matched_ab_20261007` switch retains its original
83.56748010753823*E behavior and narrow domain for profile reproducibility.
It is no longer the David default. Historical wide-column counterexamples
are not erased or combined with the new measurement protocol.

## Shared limits and TTIR information

Both integrations require SuperBlock=1, no mask, no estimated spill, no partial
continuous load and no indirect store in the target Stage. Unsupported cases
fall back rather than extrapolate. The campaigns used grid=1; concurrency and
cache interference across programs are not newly calibrated here.

Axis summaries preserve extent, proven affine stride, loaded-value provenance
and opaque/non-affine status. They use original TTIR expressions and coarse
pointer-axis analysis on a detached clone. They do not inspect runtime index
values or final ISA. Random-scattered addresses remain an explicit prior, not
a compiler proof or a worst-case bound. No Stage classification is changed.

## Validation status

The current PR's pricing, profile-loader and partitioner translation units
were compiled and linked against existing generated/dependency objects.
All 58 repository `SimdSimtCostModel` GTests passed, including old-model
compatibility, dtype/rank rates, resource preservation, duplicate-dependency
suppression, iteration accounting, warp independence and guarded fallback.
The previous independent 137 SIMT / 7 rank-one SIMD / 44 rank-two checks also
remain passing.

End-to-end source-level replay of 530 archived real TTIR programs using the
new David profile admitted all 506 rank-1--5 payload loads and left all 24
rank-6--8 controls on legacy pricing. Every admitted real Stage owns address
helpers as well as the single target load; the integration deliberately
accepts this structure rather than requiring a one-operation Stage.
A profile with only the new switch disabled confirmed unchanged Stage
workloads, all non-load resources, serial/iteration accounting and all SIMT
implementation costs. C++ load values matched the frozen two-rate equation.
This is a real classification/profile/pricing replay, not a complete compiler
rebuild or installed-wheel deployment.

Raw timing, binaries and simulator files are retained in the calibration
workspace rather than committed into this source PR. No compiler wheel was
installed during integration.
