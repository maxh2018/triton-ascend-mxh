# Force a CostModel Mixed route for diagnostics

This control runs the normal CostModel analysis and materializes a legal Mixed
candidate at one requested scope SuperBlock factor. It does not modify model
scores, profiles, operator source, or the default automatic-selection policy.
The result is an explicitly forced experiment, not an automatic route decision.

Enable it for an existing launch without editing the operator:

```bash
export TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE=mixed_simd_simt
export TRITON_ASCEND_AUTO_SIMT_FORCE_FACTOR=2
python your_test.py
```

When the launch does not specify them, this supplies
`compile_mode="simd_simt"` and `auto_simt_scope_mode="auto"`.
An explicitly conflicting mode, including `off` or `report`, raises an error.
The factor defaults to 1 when the factor environment variable is absent.
Explicit JIT options override the new environment variables:

```python
kernel[grid](
    ...,
    compile_mode="simd_simt",
    auto_simt_scope_mode="auto",
    auto_simt_force_route="mixed_simd_simt",
    superblock_factor=4,
)
```

Set `auto_simt_force_route=""` on a launch to disable forcing even when the
environment variable is set, or unset the environment variables to restore the
ordinary defaults. The route value must be empty or exactly `mixed_simd_simt`.
The factor environment variable is ignored while forcing is disabled.

Controls are resolved before the JIT in-process cache lookup and also by
`AscendBackend.parse_options` for direct compilation. They participate in the
serialized compiler options and disk-cache hash, including at F1. Changing only
the environment factor in one Python process must not reuse the previous F's
executable.

The scope backend currently supports F1/F2/F4. Admission also requires the
original warp, target, logical-program, resource and scope-materialization
checks. Forcing does not make an unsupported factor or an illegal Mixed plan
legal. Rejections are errors, with no fallback to SIMD, SIMT-only or a different
factor. The ordinary auto route remains free to select any legal candidate.

The native pass restricts the Mixed factor set, selects the forced legal
candidate, materializes its scopes and reports `selection_source` as
`explicit_forced_route`. The Python boundary checks the effective route,
selected factor, materialized-anchor count and scope-factor IR before accepting
the compile. Metadata records the forced route/factor and materialized TTIR
SHA256. NPU precision and performance still require a real runtime test.

For Ascend 950 Mixed lowering, an extracted index-select loop keeps its gathered
temporary in UB. CV lowering then transfers the gathered result to Cube. Leaving
that temporary unclassified can pull vector-produced indices into L1 and emit an
unsupported Cube scalar read, which failed on hardware with error 507015/271.
The opt-in `test_costmodel_forced_mixed_factors` regression checks F1/F2 precision,
distinct binaries, and environment-only changes through the live JIT cache.

This is different from `compile_mode="simd_simt_template"` with the CostModel
disabled. That mode uses the compiler's template SIMT subpaths and is a separate
Mixed subscenario; it does not establish a CostModel-generated Mixed-F plan.
