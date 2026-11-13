import importlib.util
import ast
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path


class CompilerCostmodelContractTest(unittest.TestCase):

    @staticmethod
    def _load_compiler_module():
        stubbed_modules = [
            "ctypes",
            "triton",
            "triton._C",
            "triton._C.libtriton",
            "triton._C.libtriton.ascend",
            "triton.backends.ascend",
            "triton.backends.ascend.utils",
            "triton.backends.ascend.driver",
            "triton.backends.compiler",
            "triton.runtime",
            "triton.runtime.cache",
        ]
        saved_modules = {name: sys.modules.get(name) for name in stubbed_modules}
        ctypes_stub = types.ModuleType("ctypes")
        ctypes_stub.c_int64 = int
        sys.modules["ctypes"] = ctypes_stub

        class Dummy:

            def __getattr__(self, _):
                return Dummy()

            def __call__(self, *args, **kwargs):
                return Dummy()

        triton_mod = types.ModuleType("triton")
        triton_c_mod = types.ModuleType("triton._C")
        ascend_backend_mod = types.ModuleType("triton.backends.ascend")
        ascend_backend_mod._apply_ascend_patch = lambda: None
        libtriton_mod = types.ModuleType("triton._C.libtriton")
        libtriton_mod.ir = Dummy()
        libtriton_mod.passes = Dummy()
        libtriton_mod.ascend = Dummy()
        libtriton_mod.buffer_ir = Dummy()
        libtriton_ascend_mod = types.ModuleType("triton._C.libtriton.ascend")
        libtriton_ascend_mod.ir = Dummy()

        utils_mod = types.ModuleType("triton.backends.ascend.utils")
        for name in [
                "_check_bishengir_api_change",
                "_check_bishengir_able_save_ir",
                "_check_bishengir_is_regbased",
                "_enable_print_ub_bits",
                "_enable_dump_memory_info",
                "_enable_msdebug",
                "_get_kernel_target",
                "_get_llvm_path",
                "_get_mlir_path",
                "_get_npucompiler_path",
                "_get_triton_adapter_opt_path",
                "_get_triton_mlir_opt_path",
                "_get_triton_opt_path",
                "_get_bishengir_opt_path",
                "_is_ascend_sanitizer_enabled",
                "_is_debug_line_info_disabled",
                "_is_auto_map_parallel_blocks_enabled",
                "_get_modeled_superblock_factors",
                "_get_current_scope_superblock_factors",
                "_get_auto_blockify_blacklist_reasons",
                "_warn_auto_blockify_disabled",
                "downgrade_llir",
                "force_disable_ffts",
                "get_cann_version_file_hash",
        ]:
            setattr(utils_mod, name, lambda *args, **kwargs: False)
        utils_mod._get_auto_blockify_blacklist_reasons = lambda *args, **kwargs: []
        utils_mod._is_auto_map_parallel_blocks_enabled = lambda *args, **kwargs: False
        utils_mod.get_cann_version_file_hash = lambda *args, **kwargs: ""

        driver_mod = types.ModuleType("triton.backends.ascend.driver")
        driver_mod.NPUUtils = Dummy

        compiler_base_mod = types.ModuleType("triton.backends.compiler")

        class BaseBackend:

            def __init__(self, target):
                self.target = target

        class GPUTarget:

            def __init__(self, backend="npu", arch="910B"):
                self.backend = backend
                self.arch = arch

        compiler_base_mod.AttrsDescriptor = Dummy
        compiler_base_mod.BaseBackend = BaseBackend
        compiler_base_mod.GPUTarget = GPUTarget
        compiler_base_mod.register_descriptor = lambda cls: cls

        runtime_mod = types.ModuleType("triton.runtime")
        runtime_mod.driver = Dummy()

        cache_mod = types.ModuleType("triton.runtime.cache")

        class DumpManager:

            def __init__(self):
                self.cache_dir = "/tmp/fake_cache"
                self.records = []

            def put(self, payload, file_name, binary=False):
                self.records.append((payload, file_name, binary))

        dump_mgr = DumpManager()
        cache_mod.get_dump_manager = lambda *args, **kwargs: dump_mgr
        cache_mod._base32 = lambda value: value

        utils_mod.is_compile_on_910_95 = lambda: False

        sys.modules.update({
            "triton": triton_mod,
            "triton._C": triton_c_mod,
            "triton._C.libtriton": libtriton_mod,
            "triton._C.libtriton.ascend": libtriton_ascend_mod,
            "triton.backends.ascend": ascend_backend_mod,
            "triton.backends.ascend.utils": utils_mod,
            "triton.backends.ascend.driver": driver_mod,
            "triton.backends.compiler": compiler_base_mod,
            "triton.runtime": runtime_mod,
            "triton.runtime.cache": cache_mod,
        })

        module_path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
        spec = importlib.util.spec_from_file_location("ascend_compiler_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
        return module, dump_mgr, GPUTarget

    def test_parse_options_costmodel_forces_no_bytecode(self):
        cmplr, _dump_mgr, GPUTarget = self._load_compiler_module()

        backend = cmplr.AscendBackend(GPUTarget(backend="npu", arch="910B"))

        opt_plain = backend.parse_options({})
        self.assertTrue(opt_plain.use_bytecode)

        opt_costmodel = backend.parse_options({"enable_costmodel_backend": True})
        self.assertTrue(opt_costmodel.enable_costmodel_backend)
        self.assertFalse(opt_costmodel.use_bytecode)

    def test_costmodel_profiles_use_canonical_source_in_checkout(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()

        expected = Path(__file__).resolve().parents[2] / "costmodel" / "profiles"
        self.assertEqual(cmplr._costmodel_profiles_dir(), expected)

    def test_costmodel_profiles_find_native_package_assets(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()

        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "triton"
            compiler = package / "backends" / "ascend" / "compiler.py"
            profiles = package / "_C" / "ascend" / "costmodel_profiles"
            legacy_profiles = compiler.parent / "costmodel_profiles"
            compiler.parent.mkdir(parents=True)
            profiles.mkdir(parents=True)
            legacy_profiles.mkdir()
            cmplr.__file__ = str(compiler)

            self.assertEqual(cmplr._costmodel_profiles_dir(), profiles)

    def test_lib_call_no_inline_is_only_enabled_for_ascend950(self):
        cmplr, _dump_mgr, GPUTarget = self._load_compiler_module()

        for arch in ("Ascend910B3", "Ascend910_95"):
            metadata = {"target": GPUTarget(backend="npu", arch=arch)}
            self.assertFalse(cmplr._needs_lib_call_no_inline(metadata))

        metadata = {"target": GPUTarget(backend="npu", arch="Ascend950PR_9579")}
        self.assertTrue(cmplr._needs_lib_call_no_inline(metadata))

    def test_all_simd_decision_replaces_route_request_with_backend_mode(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()
        metadata = {
            "compile_mode": "simd_simt",
            "parallel_mode": "mix_simd_simt",
            "auto_blockify_v1_enabled": True,
        }

        cmplr._apply_cpp_simd_simt_decision(metadata, "all_simd", 1, "{}")

        self.assertEqual(metadata["compile_mode"], "simd")
        self.assertEqual(metadata["parallel_mode"], "simd")
        self.assertEqual(metadata["auto_simt_effective_kind"], "all_simd")
        self.assertFalse(metadata["auto_blockify_v1_enabled"])
        self.assertFalse(metadata["auto_blockify_v1_runtime_cap"])
        self.assertNotIn("auto_simt_requested_kind", metadata)

    def test_mixed_decision_preserves_route_request_and_factor(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()
        metadata = {
            "compile_mode": "simd_simt",
            "route_transform_v1_materializable": True,
        }

        cmplr._apply_cpp_simd_simt_decision(metadata, "mixed_simd_simt", 4, "{\"decision\":\"mixed\"}", 2)

        self.assertEqual(metadata["compile_mode"], "simd_simt")
        self.assertEqual(metadata["auto_simt_superblock_factor"], 4)
        self.assertEqual(metadata["auto_simt_requested_kind"], "mixed_simd_simt")
        self.assertTrue(metadata["auto_blockify_v1_enabled"])
        self.assertTrue(metadata["auto_blockify_v1_runtime_cap"])
        self.assertEqual(metadata["auto_simt_scope_superblock_factor"], 4)
        self.assertNotIn("auto_simt_scope_num_warps", metadata)
        self.assertNotIn("num_warps", metadata)
        self.assertNotIn("scope_superblock_backend_abi_version", metadata)

    def test_mixed_compile_keeps_delayed_cross_core_gss_enabled(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()

        source = inspect.getsource(cmplr.linalg_to_bin_enable_npu_compile_910_95)
        self.assertNotIn("--enable-hivm-delayed-cross-core-gss=false", source)

    def test_all_bishengir_entries_share_debug_info_option(self):
        cmplr, _dump_mgr, _GPUTarget = self._load_compiler_module()

        for entry in (
                cmplr.linalg_to_bin_enable_npu_compile_910_95,
                cmplr.linalg_to_bin_enable_npu_compile_A2_A3,
                cmplr.ttir_to_npubin,
        ):
            source = inspect.getsource(entry)
            self.assertIn("if not _is_debug_line_info_disabled():", source)
            self.assertIn('_compile_option_list += ["--enable-debug-info=true"]', source)


    def test_forced_mixed_environment_and_explicit_override(self):
        cmplr, _, _ = self._load_compiler_module()
        with patch.dict(os.environ, {"TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE": "mixed_simd_simt",
                                     "TRITON_ASCEND_AUTO_SIMT_FORCE_FACTOR": "2"}, clear=True):
            options = cmplr._resolve_forced_mixed_options({"num_warps": 4})
            self.assertEqual(options["auto_simt_force_route"], "mixed_simd_simt")
            self.assertEqual(options["superblock_factor"], 2)
            self.assertEqual(options["compile_mode"], "simd_simt")
            self.assertEqual(options["auto_simt_scope_mode"], "auto")
            self.assertEqual(cmplr._resolve_forced_mixed_options({"superblock_factor": 4})["superblock_factor"], 4)
            self.assertEqual(cmplr._resolve_forced_mixed_options({"auto_simt_force_route": ""}),
                             {"auto_simt_force_route": ""})

    def test_forced_mixed_rejects_invalid_or_conflicting_controls(self):
        cmplr, _, _ = self._load_compiler_module()
        with patch.dict(os.environ, {}, clear=True):
            for extra in ({"auto_simt_force_route": "all_simd"},
                          {"superblock_factor": 3}, {"superblock_factor": True},
                          {"compile_mode": "simd"}, {"auto_simt_scope_mode": "off"},
                          {"auto_simt_scope_mode": "report"}):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    cmplr._resolve_forced_mixed_options({"auto_simt_force_route": "mixed_simd_simt", **extra})
        with patch.dict(os.environ, {"TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE": "mixed_simd_simt",
                                     "TRITON_ASCEND_AUTO_SIMT_FORCE_FACTOR": "two"}, clear=True):
            with self.assertRaisesRegex(ValueError, "integer"):
                cmplr._resolve_forced_mixed_options({})

    def test_forced_mixed_factor_restricts_capability_without_changing_legality(self):
        cmplr, _, _ = self._load_compiler_module()
        cmplr._get_modeled_superblock_factors = lambda: (1, 2, 4, 8, 16, 32, 64)
        cmplr._get_current_scope_superblock_factors = lambda: (1, 2, 4)
        options = types.SimpleNamespace(compile_on_910_95=True, num_warps=8,
                                        physical_vector_core_count_hint=56, logical_program_count_hint=56,
                                        auto_simt_force_route="mixed_simd_simt", superblock_factor=2)
        metadata = {"auto_blockify_v1_enabled": True}
        capability = json.loads(cmplr._publish_route_transform_capability(metadata, options))
        self.assertEqual(capability["scope_superblock_factors"], [2])
        self.assertEqual(capability["backend_forced_route_candidate"], "mixed_simd_simt")
        self.assertTrue(capability["auto_blockify_v1_materializable"])
        options.superblock_factor = 8
        with self.assertRaisesRegex(ValueError, "unsupported"):
            cmplr._publish_route_transform_capability({}, options)

    def test_forced_mixed_verification_refuses_a_fallback_or_wrong_factor(self):
        cmplr, _, _ = self._load_compiler_module()
        options = types.SimpleNamespace(auto_simt_force_route="mixed_simd_simt", superblock_factor=2)
        report = json.dumps({"action_supported": True, "materialized_simt_anchor_count": 1})
        module = 'scope.scope {vector_mode = "simt", ascend.scope_superblock.factor = 2 : i64}'
        metadata = {}
        cmplr._verify_forced_mixed_result(module, metadata, options, "mixed_simd_simt", 2, report)
        self.assertEqual(metadata["auto_simt_forced_factor"], 2)
        self.assertEqual(len(metadata["auto_simt_forced_ttir_sha256"]), 64)
        for effective, factor, text in (("all_simd", 2, module), ("mixed_simd_simt", 1, module),
                                        ("mixed_simd_simt", 2, "module {}")):
            with self.assertRaisesRegex(RuntimeError, "refusing fallback"):
                cmplr._verify_forced_mixed_result(text, {}, options, effective, factor, report)

    def test_environment_changes_are_resolved_before_real_jit_cache_lookup(self):
        cmplr, _, _ = self._load_compiler_module()
        source = Path(__file__).resolve().parents[4] / "python/triton/runtime/jit.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        compute = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_cache_key")
        jit = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "JITFunction")
        run = next(n for n in jit.body if isinstance(n, ast.FunctionDef) and n.name == "run")
        namespace = {
            "is_namedtuple": lambda value: False,
            "JITCallable": type("JITCallable", (), {}),
            "driver": types.SimpleNamespace(active=types.SimpleNamespace(
                get_current_device=lambda: 0, get_current_stream=lambda device: None)),
            "knobs": types.SimpleNamespace(runtime=types.SimpleNamespace(debug=False),
                                            compilation=types.SimpleNamespace(instrumentation_mode=None)),
        }
        exec(compile(ast.Module(body=[compute, run], type_ignores=[]), str(source), "exec"), namespace)
        cache, compiled = {}, []
        backend = types.SimpleNamespace(resolve_launch_options=cmplr._resolve_forced_mixed_options)
        instance = types.SimpleNamespace(debug=False, pre_run_hooks=[], used_global_vals={},
            device_caches={0: (cache, {}, None, backend, lambda **kwargs: ({}, [], kwargs))})
        instance._pack_args = lambda backend, kwargs, bound, specialization, options: (options, {}, {}, {})

        def compile_kernel(key, signature, device, constexprs, options, attrs, warmup):
            result = types.SimpleNamespace(options=options)
            compiled.append(result)
            cache[key] = result
            return result

        instance._do_compile = compile_kernel
        launch = lambda **kwargs: namespace["run"](instance, grid=(1,), warmup=True,
                                                    compile_mode="simd_simt", auto_simt_scope_mode="auto", **kwargs)
        with patch.dict(os.environ, {}, clear=True):
            automatic = launch()
            os.environ["TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE"] = "mixed_simd_simt"
            os.environ["TRITON_ASCEND_AUTO_SIMT_FORCE_FACTOR"] = "2"
            f2 = launch()
            os.environ["TRITON_ASCEND_AUTO_SIMT_FORCE_FACTOR"] = "4"
            f4 = launch()
            os.environ["TRITON_ASCEND_AUTO_SIMT_FORCE_FACTOR"] = "2"
            self.assertIs(launch(), f2)
            self.assertIs(launch(auto_simt_force_route=""), automatic)
        self.assertIsNot(f2, f4)
        self.assertEqual(f2.options["superblock_factor"], 2)
        self.assertEqual(f4.options["superblock_factor"], 4)
        self.assertEqual(len(compiled), 3)


if __name__ == "__main__":
    unittest.main()
