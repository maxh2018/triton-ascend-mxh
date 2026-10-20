"""CPU-only checks for corpus integrity, audit gates and the suite launcher.

Run: python -m unittest discover -s costmodel_tests -p test_offline.py -v
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
GENERAL = ROOT / "third_party/ascend/unittest/costmodel_ut/costmodel_generalization"


class OfflineTests(unittest.TestCase):
    def test_historical_matrix_is_complete(self):
        cases = json.loads((GENERAL / "generalization_matrix.json").read_text())["cases"]
        self.assertEqual(len(cases), 100)
        self.assertEqual(len({c["name"] for c in cases}), 100)
        self.assertEqual(len({c["driver"] for c in cases}), 19)
        self.assertEqual(sum(5 + len(c.get("extra_modes", [])) for c in cases), 508)
        for case in cases:
            self.assertTrue((GENERAL / case["driver"]).is_file())
        sources = json.loads((ROOT / "costmodel_tests/sources.json").read_text())
        for path in sources["pytest_base_files"] + sources["pytest_added_from_pr"]:
            self.assertTrue((ROOT / path).is_file(), path)

    def test_real_audit_rejects_wrong_results_errors_and_missing_routes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {"name": "example", "family": "example", "split": "holdout"}
            (root / "manifest.json").write_text(json.dumps({"cases": [case]}))
            profile = {"kernel_timing": {
                "system_counter_frequency_mhz": 1,
                "boundary_residual_system_cycles_by_physical_program_count": {"1": 0, "2": 0}}}
            (root / "profile.json").write_text(json.dumps(profile))
            candidate = {"legal": True, "total_system_cycles": 10, "runtime_physical_program_count": 1}
            route = {"selected_route_elapsed_prediction": {"applied": True, "predicted_microseconds": 10},
                     "stage_model": {"routes": {"all_simd": candidate},
                                     "all_simt_factor_routes": {str(f): candidate for f in (1, 2, 4)}}}
            records = []
            for mode, factor in (("auto", 1), ("simd", 1), ("simt_only", 1), ("simt_only", 2), ("simt_only", 4)):
                name = f"example__{mode}_f{factor}"
                evidence = root / name
                evidence.mkdir()
                (evidence / "evidence.json").write_text(json.dumps({
                    "returncode": 0, "benchmark": {"correct": True, "profiler": {"average_us": 10}}}))
                if mode == "auto":
                    (evidence / "route.json").write_text(json.dumps(route))
                records.append({"name": name, "family": "example", "split": "holdout", "route": mode,
                                "factor": factor, "returncode": 0, "evidence_dir": str(evidence)})
            summary = root / "matrix_summary.json"
            summary.write_text(json.dumps({"records": records}))
            command = [sys.executable, str(GENERAL / "audit_generalization_matrix.py"),
                       "--evidence-root", str(root), "--manifest", str(root / "manifest.json"),
                       "--profile", str(root / "profile.json"), "--output-json", str(root / "audit.json"),
                       "--output-markdown", str(root / "audit.md")]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            first = root / records[0]["name"] / "evidence.json"
            for benchmark in ({"correct": False, "profiler": {"average_us": 10}},
                              {"correct": True, "profiler": {"average_us": 20}}):
                first.write_text(json.dumps({"returncode": 0, "benchmark": benchmark}))
                self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            summary.write_text(json.dumps({"records": records[:-1]}))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
            report = json.loads((root / "audit.json").read_text())
            self.assertTrue(any("missing_matrix_record" in r["failure_reasons"] for r in report["results"]))

    def test_launcher_propagates_failed_matrix_even_if_audit_exits_zero(self):
        spec = importlib.util.spec_from_file_location("suite_runner", ROOT / "costmodel_tests/run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"

            def fake_run(command, **kwargs):
                if Path(command[1]).name == "run_generalization_matrix.py":
                    (output / "matrix").mkdir()
                    (output / "matrix/matrix_summary.json").write_text("{}")
                    return subprocess.CompletedProcess(command, 1)
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, "argv", ["run.py", "matrix", "--output-root", str(output)]), \
                    patch.object(module.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit) as stopped:
                    module.main()
            self.assertEqual(stopped.exception.code, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["status"], "FAIL")
            self.assertEqual([r["step"] for r in summary["steps"]], ["matrix", "audit"])


if __name__ == "__main__":
    unittest.main()
