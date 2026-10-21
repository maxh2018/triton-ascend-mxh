#!/usr/bin/env python3
"""Entry point for pytest_ut, the PR #2528 matrix, and single-case audits."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "cases/ci"
PYTEST = CI / "third_party/ascend/unittest/pytest_ut"
GENERAL = CI / "third_party/ascend/unittest/costmodel_ut/costmodel_generalization"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("list", "pytest", "matrix", "case", "all"))
    parser.add_argument("--case", help="exact matrix case name; required for suite=case")
    parser.add_argument("--category", choices=("ci", "personal", "operator_repos"), default="ci",
                        help="case collection (default: ci)")
    parser.add_argument("--manifest", type=Path,
                        help="matrix case list; use a separate manifest for custom cases")
    parser.add_argument("--validation-scope", choices=("auto", "all"), default="auto",
                        help="default: Auto selected route (PR table); all: explicit routes too")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", help="ASCEND_RT_VISIBLE_DEVICES")
    devices.add_argument("--devices", help="matrix device pool, e.g. 0,1,2,3; one case per device")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent matrix cases (default: 1)")
    parser.add_argument("--retries", type=int, default=3, help="extra attempts per failed matrix case (default: 3)")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-bishengir", type=Path)
    parser.add_argument("--triton-root", type=Path, default=ROOT)
    parser.add_argument("--npuir-root", type=Path, default=ROOT / "third_party/ascend/AscendNPU-IR")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running NPU tests")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER,
                        help="pytest arguments; place this option last, e.g. -k atomic --collect-only")
    args = parser.parse_args()
    if args.manifest is None:
        args.manifest = (GENERAL / "generalization_matrix.json" if args.category == "ci" else
                         ROOT / "cases" / args.category / "manifest.json")
    if args.suite in ("pytest", "all") and args.category != "ci":
        parser.error("pytest/all use the CI pytest corpus; personal/operator_repos use list, case or matrix")
    if args.jobs < 1 or args.retries < 0:
        parser.error("--jobs must be positive and --retries must be nonnegative")
    if args.suite == "pytest" and (args.devices is not None or args.jobs != 1 or args.retries != 3):
        parser.error("--jobs/--retries/--devices apply to matrix cases; use --pytest-args for pytest options")
    if args.suite == "case" and not args.case:
        parser.error("suite=case requires --case (a manifest name, not a .py file)")
    if args.suite != "case" and args.case:
        parser.error("--case is only supported with suite=case")
    if args.suite == "list":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(f"Category: {args.category}")
        if args.category == "ci":
            print(f"pytest_ut: {len(list(PYTEST.rglob('test_*.py')))} test files (parameterized case count needs pytest collection)")
        print(f"Matrix: {len(manifest['cases'])} cases ({args.manifest})")
        for case in manifest["cases"]:
            print(f"{case['name']}\t{case['family']}\t{case['driver']}")
        return
    output = (args.output_root or ROOT / "costmodel_test_results" /
              f"{args.suite}_{datetime.now():%Y%m%d_%H%M%S_%f}").resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"output directory must be new or empty: {output}")
    env = os.environ.copy()
    if args.device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    elif args.devices is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.devices
    commands = []
    if args.suite in ("pytest", "all"):
        commands.append(("pytest", [sys.executable, "-m", "pytest", str(PYTEST),
                                   "-v", f"--junitxml={output / 'pytest.xml'}",
                                   *(args.pytest_args or [])]))
    if args.suite in ("matrix", "case", "all"):
        evidence = output / "matrix"
        common = ["--output-root", str(evidence),
                  "--manifest", str(args.manifest.resolve()),
                  "--validation-scope", args.validation_scope,
                  "--jobs", str(args.jobs), "--retries", str(args.retries),
                  "--triton-root", str(args.triton_root.resolve()),
                  "--npuir-root", str(args.npuir_root.resolve())]
        if args.devices is not None:
            common += ["--devices", args.devices]
        for option, value in (("--profile", args.profile), ("--expected-bishengir", args.expected_bishengir)):
            if value is not None:
                common += [option, str(value.resolve())]
        selected = ["--case", args.case] if args.suite == "case" else []
        commands.append(("case" if args.suite == "case" else "matrix",
                         [sys.executable, str(GENERAL / "run_generalization_batch.py"), *selected, *common]))
    print(f"Output: {output}", flush=True)
    if args.dry_run:
        for name, command in commands:
            print(name + ": " + json.dumps(command, ensure_ascii=False))
        return
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name, command in commands:
        print(f"Running {name}...", flush=True)
        # Inherit terminal output; the PR collector also retains per-case logs.
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        records.append({"step": name, "command": command, "returncode": completed.returncode})
    passed = bool(records) and len(records) == len(commands) and all(r["returncode"] == 0 for r in records)
    summary = {"status": "PASS" if passed else "FAIL", "suite": args.suite,
               "category": args.category, "steps": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.suite}: {summary['status']} ({output / 'summary.json'})", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
