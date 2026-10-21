# CPU-only regressions for default manifests and mixed entry scheduling.
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from costmodel_tests.manifests import load_manifest
from costmodel_tests.update_ci_manifests import build_manifests
from costmodel_tests.test_batch import FAKE_WORKER

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / 'cases/ci'
GENERAL = CI / 'third_party/ascend/unittest/costmodel_ut/costmodel_generalization'


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_ci_manifests_match_original_100_and_inventory_all_test_files(self):
        default, complete = build_manifests()
        self.assertEqual(default, json.loads((CI / 'manifest.json').read_text()))
        self.assertEqual(complete, json.loads((CI / 'manifest_all.json').read_text()))
        original = json.loads((GENERAL / 'generalization_matrix.json').read_text())
        restored = copy.deepcopy(default['cases'])
        for case in restored:
            case['driver'] = (CI / case['driver']).relative_to(GENERAL).as_posix()
        self.assertEqual(restored, original['cases'])
        self.assertEqual(len(default['cases']), 100)
        self.assertNotIn('validation_status', default)
        self.assertEqual(complete['validation_status'], 'unverified')
        pytest_files = {c['driver'] for c in complete['cases'] if c.get('runner') == 'pytest'}
        self.assertEqual(pytest_files, {p.relative_to(CI).as_posix() for p in CI.rglob('test_*.py')})
        self.assertIn('third_party/ascend/unittest/costmodel_ut/costmodel_generalization/fze/test_fze.py', pytest_files)
        self.assertEqual(len(complete['cases']), 100 + len(pytest_files))
        self.assertEqual(len(complete['cases']), len({c['name'] for c in complete['cases']}))

    def test_default_run_never_selects_unverified_manifest_and_legacy_commands_removed(self):
        result = subprocess.run([sys.executable, 'costmodel_tests/run.py', '--dry-run'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Cases: 100', result.stdout)
        self.assertNotIn('UNVERIFIED', result.stdout)
        command = json.loads(result.stdout.splitlines()[-1].split(': ', 1)[1])
        self.assertEqual(Path(command[command.index('--manifest') + 1]), CI / 'manifest.json')
        for legacy in ('matrix', 'pytest', 'all', 'case'):
            result = subprocess.run([sys.executable, 'costmodel_tests/run.py', legacy, '--dry-run'],
                                    cwd=ROOT, capture_output=True)
            self.assertEqual(result.returncode, 2)

    def test_manifest_validation_rejects_duplicate_unknown_runner_missing_driver_and_bad_arguments(self):
        driver = self.root / 'driver.py'
        driver.write_text('# driver\n')
        valid = {'name': 'example', 'driver': 'driver.py'}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps({'cases': [valid]}))
        self.assertEqual(load_manifest(path)['cases'][0]['runner'], 'benchmark')
        for cases in ([valid, valid], [{**valid, 'runner': 'unknown'}], [{**valid, 'driver': 'missing.py'}],
                      [{**valid, 'arguments': [1]}]):
            path.write_text(json.dumps({'cases': cases}))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_empty_category_can_be_listed_but_cannot_run(self):
        command = [sys.executable, 'costmodel_tests/run.py', '--category', 'personal',
                   '--output-root', str(self.root / 'output')]
        self.assertEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 2)
        self.assertFalse((self.root / 'output').exists())
        command.insert(2, 'list')
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn('Cases: 0', result.stdout)

    def test_real_pytest_manifest_retries_only_failed_parameterized_node(self):
        tests = self.root / 'tests'
        tests.mkdir()
        (tests / 'conftest.py').write_text('''
from costmodel_tests.test_pytest_costmodel import FakeBackend

def pytest_sessionstart(session):
    session.config._costmodel_adapter.backend_factory = FakeBackend
''')
        (tests / 'test_example.py').write_text('''
from pathlib import Path
import pytest
from costmodel_tests.test_pytest_costmodel import FakeJit

@pytest.mark.parametrize('n', [1, 2])
def test_add(n):
    here = Path(__file__).parent
    calls = here / f'calls_{n}.txt'
    with calls.open('a') as stream:
        stream.write('called\\n')
    first = n == 2 and not (here / 'retried').exists()
    if first:
        (here / 'retried').write_text('yes')
    value = 3 if first else 1
    kernel = FakeJit()
    kernel.run(value, grid=(1,), warmup=False)
    assert kernel.actual[-1] == value
''')
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'cases': [{'name': 'add', 'runner': 'pytest',
                                                  'driver': 'tests/test_example.py'}]}))
        profile = self.root / 'profile.json'
        profile.write_text('{}')
        output = self.root / 'output'
        env = {**os.environ, 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1', 'TRITON_ASCEND_COMPILE_MODE': ''}
        command = [sys.executable, 'costmodel_tests/run.py', '--manifest', str(manifest),
                   '--profile', str(profile), '--output-root', str(output), '--retries', '2']
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads((output / 'summary.json').read_text())
        self.assertEqual(summary['status'], 'PASS')
        batch = json.loads((output / 'batch_summary.json').read_text())
        case = batch['cases'][0]
        self.assertEqual(case['attempt_count'], 2)
        self.assertEqual([a['status'] for a in case['attempts']], ['FAIL', 'PASS'])
        self.assertEqual(len(case['pytest_results']), 2)
        self.assertEqual((tests / 'calls_1.txt').read_text().splitlines(), ['called'])
        self.assertEqual((tests / 'calls_2.txt').read_text().splitlines(), ['called', 'called'])
        self.assertEqual(len(case['attempts'][1]['targets']), 1)
        self.assertTrue(case['attempts'][1]['targets'][0].endswith('::test_add[2]'))
        self.assertTrue((output / 'report.md').is_file())
        self.assertTrue(json.loads((output / 'report.json').read_text())['cases'][0]['timing_pass'])
        self.assertFalse((output / 'generalization_audit.json').exists())

    def test_mixed_manifest_audits_benchmarks_without_expectations_for_pytest_routes(self):
        driver = self.root / 'driver.py'
        driver.write_text('# benchmark placeholder\n')
        worker = self.root / 'worker.py'
        worker.write_text(FAKE_WORKER)
        profile = self.root / 'profile.json'
        profile.write_text(json.dumps({'kernel_timing': {'system_counter_frequency_mhz': 1,
            'boundary_residual_system_cycles_by_physical_program_count': {'1': 0, '2': 0}}}))
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps({'cases': [
            {'name': 'stable', 'driver': 'driver.py'},
            {'name': 'pytest_example', 'runner': 'pytest', 'driver': 'driver.py'}]}))
        spec = importlib.util.spec_from_file_location('mixed_batch', GENERAL / 'run_generalization_batch.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        output = self.root / 'output'
        fake_pytest = {'case': 'pytest_example', 'runner': 'pytest', 'status': 'PASS', 'attempt_count': 1,
                       'final_output_root': str(output / 'cases/pytest_example/attempt_01'),
                       'attempts': [], 'precision_pass': True, 'timing_pass': True, 'pytest_results': []}
        with patch.object(sys, 'argv', ['batch.py', '--manifest', str(manifest), '--profile', str(profile),
                                       '--expected-bishengir', sys.executable, '--output-root', str(output)]), \
                patch.object(module, 'SINGLE_CASE_SCRIPT', worker), \
                patch.object(module, 'run_pytest_case', return_value=fake_pytest), \
                patch.dict(os.environ, {'BATCH_TEST_AUDIT': str(GENERAL / 'audit_generalization_matrix.py'),
                                        'TRITON_ASCEND_COMPILE_MODE': ''}), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                module.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(json.loads((output / 'batch_summary.json').read_text())['status'], 'PASS')
        audit = json.loads((output / 'generalization_audit.json').read_text())
        self.assertEqual(audit['expected_record_count'], 1)
        self.assertTrue(audit['all_pass'])
        self.assertEqual(len(json.loads((output / 'report.json').read_text())['cases']), 2)


if __name__ == '__main__':
    unittest.main()
