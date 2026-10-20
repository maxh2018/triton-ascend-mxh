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
PYTEST = ROOT / "third_party/ascend/unittest/pytest_ut"
GENERAL = ROOT / "third_party/ascend/unittest/costmodel_ut/costmodel_generalization"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("list", "pytest", "matrix", "case", "all"))
    parser.add_argument("--case", help="exact matrix case name; required for suite=case")
    parser.add_argument("--device", help="ASCEND_RT_VISIBLE_DEVICES")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-bishengir", type=Path)
    parser.add_argument("--triton-root", type=Path, default=ROOT)
    parser.add_argument("--npuir-root", type=Path, default=ROOT / "third_party/ascend/AscendNPU-IR")
    parser.add_argument("--dry-run", action="store_true", help="print commands without running NPU tests")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER,
                        help="pytest arguments; place this option last, e.g. -k atomic --collect-only")
    args = parser.parse_args()
    if args.suite == "case" and not args.case:
        parser.error("suite=case requires --case (a manifest name, not a .py file)")
    if args.suite != "case" and args.case:
        parser.error("--case is only supported with suite=case")
    if args.suite == "list":
        manifest = json.loads((GENERAL / "generalization_matrix.json").read_text(encoding="utf-8"))
        print(f"pytest_ut: {len(list(PYTEST.rglob('test_*.py')))} test files (parameterized case count needs pytest collection)")
        print(f"PR #2528: {len(manifest['cases'])} cases")
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
    commands = []
    if args.suite in ("pytest", "all"):
        commands.append(("pytest", [sys.executable, "-m", "pytest", str(PYTEST),
                                   "-v", f"--junitxml={output / 'pytest.xml'}",
                                   *(args.pytest_args or [])]))
    if args.suite in ("matrix", "case", "all"):
        evidence = output / "matrix"
        common = ["--output-root", str(evidence),
                  "--triton-root", str(args.triton_root.resolve()),
                  "--npuir-root", str(args.npuir_root.resolve())]
        for option, value in (("--profile", args.profile), ("--expected-bishengir", args.expected_bishengir)):
            if value is not None:
                common += [option, str(value.resolve())]
        if args.suite == "case":
            commands.append(("case", [sys.executable, str(GENERAL / "run_generalization_case.py"),
                                       args.case, *common]))
        else:
            commands.append(("matrix", [sys.executable, str(GENERAL / "run_generalization_matrix.py"),
                                         "--include-explicit", *common]))
            commands.append(("audit", [sys.executable, str(GENERAL / "audit_generalization_matrix.py"),
                                       "--evidence-root", str(evidence),
                                       "--manifest", str(evidence / "generalization_matrix.json"),
                                       "--profile", str(evidence / "costmodel_profile.json"),
                                       "--output-json", str(evidence / "generalization_audit.json"),
                                       "--output-markdown", str(evidence / "generalization_audit.md")]))
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
        # Audit a failed matrix when its summary exists, to explain missing routes.
        if name == "matrix" and completed.returncode and not (output / "matrix/matrix_summary.json").is_file():
            break
    passed = bool(records) and len(records) == len(commands) and all(r["returncode"] == 0 for r in records)
    summary = {"status": "PASS" if passed else "FAIL", "suite": args.suite, "steps": records}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{args.suite}: {summary['status']} ({output / 'summary.json'})", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
