# Random-prior indirect load calibration

This change replaces the **load resource**, not the entire Stage, for a narrow
calibrated domain on `Ascend950PR/dav-c310`. Both models estimate real-device
payload-minus-ALU-baseline increments in SYS_CNT ticks. One tick is one
CostModel `system_cycle`; no compute-clock conversion is applied.

## Profile switches and accounting

Optional profile fields select versioned models:

```json
"simt_indirect_load_model": "random_i32_six_term_20261007",
"simd_indirect_load_model": "random_f32_matched_ab_20261007"
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

## SIMD equation and baseline audit

```text
T = 83.56748010753823 * N
```

Domain: one FP32 rank-one loaded-index `tt.load`, N in
{8,16,32,64,128,256,512}. This is **not** the older whole-scalar-loop equation
`61.72 + 96.52*N`.

The baseline preserves UB index loads, address computation, UB result stores
and loop structure by writing a fold of the full address instead of loading
the GM value. Actual N32 instruction traces confirm 32 index loads and 32 UB
stores in both versions, with 32 GM loads only in the payload. The baseline
also emits extra address/bitwise ALU instructions. A second minimal baseline
on N8/N32/N512 changes the load difference by at most 5.37% of the primary
label. Thus this is an effective incremental cost with measured baseline
sensitivity, not an isolated opcode latency.

840 random inputs, 40320 raw samples; 288 training, 144 development, 408 test.
Inputs sample N unique indices without replacement from pools of
`max(32768,256*N) * {1,4,16}`. Each pool uses seeds 2026100800..2026100839.
N128 is held out entirely; other sizes use the first 16 seeds for training,
next 8 for development and last 16 for testing. For each A/B input, take the
median of six retained launches after two warmups, then the median of three
rounds. Subtract baseline from payload. Alternate A/B order.

Relative-error weighted least squares fits training data only. Development
selects between slope-only and intercept-plus-slope, preferring the simpler
model when its maximum error is at most 10%. Test: **408/408 within 20%**,
MAPE 1.56%, P95 4.45%, maximum 11.04%. These are per-input differential-label
errors, not complete-Stage or arbitrary-address guarantees.

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

Independent C++ execution checked 137 SIMT shape/warp configurations and seven
SIMD sizes, including replacement, iteration accounting, other-resource
preservation, duplicate-dependency suppression and out-of-domain fallback.
Repository regression tests cover both mode integrations and TTIR axis-summary
provenance. Full backend/GTest execution remains blocked in the calibration
environment by existing external NPU-IR header/build configuration issues;
do not interpret independent tests as a full compiler rebuild or deployment.

Raw timing, binaries and simulator files are retained in the calibration
workspace rather than committed into this source PR. No compiler wheel was
installed during integration.
