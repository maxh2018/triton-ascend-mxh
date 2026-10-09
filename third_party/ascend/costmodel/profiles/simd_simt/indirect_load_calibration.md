# Random-prior indirect load calibration

This change replaces the **load resource**, not the entire Stage, for a narrow
calibrated domain on `Ascend950PR/dav-c310`. Both models estimate real-device
payload-minus-ALU-baseline increments in SYS_CNT ticks. One tick is one
CostModel `system_cycle`; no compute-clock conversion is applied.

## Profile switches and accounting

Optional profile fields select versioned models:

```json
"simt_indirect_load_model": "random_dtype_six_term_20261008",
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
q=E/(32W), and C the last extent for rank >=2 (C=32 for rank one). Define:

```text
H = max(4*q/C - 2, 0)
S = 1 if E <= 128 else 0
D = max(log2(32/C), 0)
F = 1 for rank one, otherwise 0
T = alpha + beta*E + sigma*S + phi*E*F + gamma*H + nu*E*D
```

| Storage bytes | alpha | beta | sigma | phi | gamma | nu |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 7.617911783542814 | 1.5780971099377332 | 92.18952965944916 | 0.19387737354356369 | 0 | 0.049471659398567715 |
| 2 | 12.456944150614481 | 1.6732132663368477 | 95.13859585355112 | 0.11214618741568584 | 5.634318857353958 | 0.026588037490546692 |
| 4 | 50.71936539506111 | 1.693396365504775 | 60.38681262899275 | 0.09265123974451413 | 82.91193957489877 | 0.026814982781878355 |
| 8 | 96.09572807125711 | 1.937402636239715 | 24.962585867214496 | 0.20099293498091206 | 41.8669299352394 | 0.03855485734655757 |

INT32, UINT32 and FP32 use the same 4-byte row; integer signedness and floating
names do not select separate coefficients. Supported result types are
i8/i16/i32/i64, f16/bf16/f32/f64, f8E4M3FN and f8E5M2. Raw i1 and unmeasured
FP8 formats are not admitted; language-normalized bool loads use i8.

Domain: exactly one target `tt.load` owned by an `IndirectGatherMemory` Stage;
independent address/shape helpers are allowed and retain their existing costs.
The target load may be nested in an owned explicit SIMT scope; inspect the
owned regions rather than assuming every owned operation is itself a load.
W in {1,2,4,8,16,32,64}, q in {1,2,4}, rank 1--5, static power-of-two extents
each >=2; C in {2,4,8,16,32,64,128} for rank >=2 and H<=2. C=2,q=4 gives H=6,
has measured counterexamples and remains on legacy pricing. The whole pointer
must depend on loaded indices; its tail must be opaque/opaque_loaded. Prefix
axes may be opaque/opaque_loaded or proven positive fixed strides. Prefix
structure does not create an independent inner coefficient group. Computed
non-affine axes, structured tails and multiple target loads are not admitted.

This is a frozen pure-random prior fit, not an assertion that the real indices
are random. H and E*D are baseline-mismatch compensations, not proven GM row
or short-axis hardware penalties. The 128-element small-load threshold is
empirical within q=1/2/4, not a universal hardware discontinuity.

The retained independent dtype/rank test data give 583/584 inputs within 20%,
MAPE 4.22%, maximum 21.09%. A separate 244-configuration coverage matrix gives
1948/1952 within 20%, MAPE 3.92%, maximum 28.80%. These remain measured A/B
proxy labels, not independently timed whole Stages. The guarded production
domain excludes four C=2,q=4 configurations, not their archived observations.

The prior `random_i32_six_term_20261007` profile identifier retains its original
INT32 rank-1/2 coefficients and single-operation guard solely for reproducible
old profiles. It is no longer the David default or a dtype-specific exception
inside the four-width model.

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

The PR's pricing, profile-loader and partitioner translation units were
compiled and linked against existing generated/dependency objects. All 65
`SimdSimtCostModel` GTests passed, including 1155 dtype/W/q/rank combinations
inside the new SIMT test, owned-region loads, old-profile compatibility,
resource preservation, dependency suppression, iteration accounting and
guarded fallback.

705 real SIMT TTIR programs (461 dtype/rank and 244 coverage configurations)
were replayed through the current profile, partitioner and pricing. All 701
in-domain programs matched the frozen four-width equation, and all four H=6
controls stayed on legacy pricing. Their non-load resources, classification
and SIMD/store pricing matched a switch-disabled profile. Their original
real-device A/B labels were used to recompute the accuracy quoted above;
no refit or new device measurements were performed for integration.

An additional 530 real TTIR programs checked SIMD and multi-operation Stages:
506 SIMD payload loads still matched the two-rate equation; 24 rank-6--8
controls stayed on fallback. The new SIMT model admitted 229 multi-operation
Stages and preserved all their independent resources. 54 pointer-analysis
classification cases and all costs in 40 real scatter programs were unchanged.

Cross-dataset re-evaluation of 137 historical INT32 configurations (1644
random inputs) gives 1608/1644 within 20%, MAPE 4.77%, maximum 27.69% with
the unified 4-byte coefficients. The old specialized model gave 1644/1644,
MAPE 4.44% on these same inputs. This regression is retained explicitly;
unifying parameter groups is not a claim that a broader fit beats every
specialized fit. It does not justify a hidden INT32 exception in the default.

This is a real classification/profile/pricing replay, not a complete compiler
rebuild, newly blind accuracy test, or installed-wheel deployment.

Raw timing, binaries and simulator files are retained in the calibration
workspace rather than committed into this source PR. No compiler wheel was
installed during integration.
