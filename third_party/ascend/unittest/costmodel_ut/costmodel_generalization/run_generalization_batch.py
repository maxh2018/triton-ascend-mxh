#!/usr/bin/env python3
"""Run matrix cases concurrently, retry failed cases, and audit final attempts."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
SINGLE_CASE_SCRIPT = HERE / "run_generalization_case.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_case(case: dict, args: argparse.Namespace, env: dict, device: str | None) -> dict:
    """One case owns an isolated directory; successful cases are never rerun."""
    attempts = []
    expected = 5 + len(case.get("extra_modes", []))
    for number in range(1, args.retries + 2):
        output = args.output_root / "cases" / case["name"] / f"attempt_{number:02d}"
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / f"attempt_{number:02d}.log"
        command = [
            sys.executable, str(SINGLE_CASE_SCRIPT), case["name"],
            "--manifest", str(args.output_root / "generalization_matrix.json"),
            "--profile", str(args.output_root / "costmodel_profile.json"),
            "--expected-bishengir", str(args.expected_bishengir),
            "--output-root", str(output),
            "--triton-root", str(args.triton_root.resolve()),
            "--npuir-root", str(args.npuir_root.resolve()),
            "--warmup", str(args.warmup), "--repeats", str(args.repeats),
            "--measurement-processes", str(args.measurement_processes),
        ]
        child_env = env.copy()
        if device is not None:
            child_env["ASCEND_RT_VISIBLE_DEVICES"] = device
        print(f"START {case['name']} attempt={number}/{args.retries + 1} device={device or 'inherited'}", flush=True)
        started = time.monotonic()
        failure = None
        with log.open("w", encoding="utf-8") as stream:
            try:
                status = subprocess.run(command, env=child_env, stdout=stream,
                                        stderr=subprocess.STDOUT, check=False).returncode
            except OSError as error:
                status = 1
                failure = str(error)
                stream.write(failure + "\n")
        try:
            summary = read_json(output / "single_case_summary.json")
            audit = read_json(output / "generalization_audit.json")
            matrix = read_json(output / "matrix_summary.json")
            passed = (
                status == 0 and summary.get("status") == "PASS" and audit.get("all_pass") is True
                and audit.get("expected_record_count") == expected
                and audit.get("audited_record_count") == expected
                and len(matrix.get("records", [])) == expected
                and all(r.get("returncode") == 0 for r in matrix["records"])
            )
            if not passed:
                failure = failure or "runner_or_case_audit_failed"
        except (OSError, ValueError, TypeError, KeyError) as error:
            passed = False
            failure = f"missing_or_invalid_result: {error}"
        attempts.append({
            "attempt": number, "returncode": status, "status": "PASS" if passed else "FAIL",
            "failure_reason": failure, "elapsed_seconds": time.monotonic() - started,
            "output_root": str(output), "log": str(log),
        })
        print(f"{'PASS' if passed else 'FAIL'} {case['name']} attempt={number}", flush=True)
        if passed:
            break
        if number <= args.retries:
            print(f"RETRY {case['name']} ({number}/{args.retries})", flush=True)
    return {"case": case["name"], "status": attempts[-1]["status"], "device": device,
            "attempt_count": len(attempts), "final_output_root": attempts[-1]["output_root"],
            "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=HERE / "generalization_matrix.json")
    parser.add_argument("--case", action="append", default=[], help="exact case name; repeat to select a batch")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, help="maximum concurrent cases (routes within a case stay serial)")
    parser.add_argument("--retries", type=int, default=3, help="extra attempts after failure; 0 disables retries")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", help="shared ASCEND_RT_VISIBLE_DEVICES for all workers")
    devices.add_argument("--devices", help="comma-separated device pool; one active case per device")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--expected-bishengir", type=Path)
    parser.add_argument("--triton-root", type=Path, default=HERE.parents[4])
    parser.add_argument("--npuir-root", type=Path, default=HERE.parents[4] / "third_party/ascend/AscendNPU-IR")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--measurement-processes", type=int, default=3)
    args = parser.parse_args()
    if args.jobs < 1 or args.retries < 0:
        parser.error("--jobs must be positive and --retries must be nonnegative")
    if args.warmup < 0 or args.repeats < 1 or args.measurement_processes < 1:
        parser.error("invalid warmup/repeats/measurement-processes")
    pool = [d.strip() for d in args.devices.split(",")] if args.devices is not None else None
    if pool is not None and (not all(pool) or len(set(pool)) != len(pool) or args.jobs > len(pool)):
        parser.error("--devices must be unique/nonempty and contain at least --jobs devices")
    manifest = read_json(args.manifest)
    names = [c["name"] for c in manifest["cases"]]
    if not names or len(names) != len(set(names)):
        parser.error("manifest cases must be nonempty and unique")
    unknown = sorted(set(args.case) - set(names))
    if unknown:
        parser.error(f"unknown case(s): {', '.join(unknown)}")
    selected = [c for c in manifest["cases"] if not args.case or c["name"] in args.case]
    if any(not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", c["name"]) for c in selected):
        parser.error("case names must be safe directory names")
    env = os.environ.copy()
    if env.get("TRITON_ASCEND_COMPILE_MODE"):
        parser.error("unset TRITON_ASCEND_COMPILE_MODE so it cannot override tested routes")
    if args.device is not None:
        env["ASCEND_RT_VISIBLE_DEVICES"] = args.device
    profile = (args.profile or Path(env.get("TRITON_ASCEND_AUTO_SIMT_PROFILE") or
                                   HERE.parents[2] / "costmodel/profiles/simd_simt/ascend_950pr_simd_simt_v1.json")).resolve()
    if not profile.is_file():
        parser.error(f"profile does not exist: {profile}")
    compiler = args.expected_bishengir or shutil.which("bishengir-compile")
    if not compiler or not Path(compiler).is_file():
        parser.error("bishengir-compile was not found; load your toolchain or pass --expected-bishengir")
    args.expected_bishengir = Path(compiler).resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists() and (not args.output_root.is_dir() or any(args.output_root.iterdir())):
        parser.error(f"output directory must be new or empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest["cases"] = selected
    write_json(args.output_root / "generalization_matrix.json", manifest)
    shutil.copyfile(profile, args.output_root / "costmodel_profile.json")
    env["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(args.output_root / "costmodel_profile.json")
    results = {}
    slots = Queue()
    for device in (pool or [None] * args.jobs):
        slots.put(device)

    def worker(case):
        device = slots.get()
        try:
            return run_case(case, args, env, device)
        finally:
            slots.put(device)

    def batch_summary():
        return {"schema": "costmodel-batch-run-v1", "jobs": args.jobs, "retries": args.retries,
                "case_count": len(selected), "completed_case_count": len(results),
                "cases": [results[c["name"]] for c in selected if c["name"] in results]}

    print(f"Running {len(selected)} case(s), jobs={args.jobs}, retries={args.retries}", flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(worker, c): c for c in selected}
        for future in as_completed(futures):
            case = futures[future]
            try:
                results[case["name"]] = future.result()
            except Exception as error:
                results[case["name"]] = {"case": case["name"], "status": "FAIL", "attempts": [],
                                         "attempt_count": 0, "final_output_root": None, "error": str(error)}
            write_json(args.output_root / "batch_summary.partial.json", batch_summary())
    # Only each case's final attempt enters the aggregate audit. Earlier failures stay in attempts/ logs.
    records = []
    for case in selected:
        result = results[case["name"]]
        if result["final_output_root"] is None:
            continue
        try:
            matrix = read_json(Path(result["final_output_root"]) / "matrix_summary.json")
            records.extend(matrix["records"])
        except (OSError, ValueError, KeyError, TypeError):
            pass  # The audit reports every missing route as FAIL.
    write_json(args.output_root / "matrix_summary.json", {
        "schema": "triton-costmodel-generalization-batch-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.output_root / "generalization_matrix.json"), "records": records,
        "batch_summary": str(args.output_root / "batch_summary.json"),
    })
    audit_command = [sys.executable, str(HERE / "audit_generalization_matrix.py"),
                     "--evidence-root", str(args.output_root),
                     "--manifest", str(args.output_root / "generalization_matrix.json"),
                     "--profile", str(args.output_root / "costmodel_profile.json"),
                     "--output-json", str(args.output_root / "generalization_audit.json"),
                     "--output-markdown", str(args.output_root / "generalization_audit.md")]
    with (args.output_root / "audit.log").open("w", encoding="utf-8") as stream:
        audit_status = subprocess.run(audit_command, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, check=False).returncode
    summary = batch_summary()
    passed = audit_status == 0 and all(r["status"] == "PASS" for r in results.values())
    summary.update(status="PASS" if passed else "FAIL", audit_exit_status=audit_status,
                   failed_cases=[c["name"] for c in selected if results[c["name"]]["status"] != "PASS"])
    write_json(args.output_root / "batch_summary.json", summary)
    print(f"Batch {summary['status']}; failed_cases={summary['failed_cases']}", flush=True)
    print(f"Report: {args.output_root / 'generalization_audit.md'}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
