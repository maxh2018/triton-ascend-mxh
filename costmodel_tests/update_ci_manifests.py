#!/usr/bin/env python3
"""Refresh the two CI manifests. Does not run or validate NPU kernels."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "cases/ci"
ORIGINAL = CI / "third_party/ascend/unittest/costmodel_ut/costmodel_generalization/generalization_matrix.json"


def build_manifests():
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    default = deepcopy(original)
    default["description"] = "Default CI manifest: original PR #2528 100 cases."
    for case in default["cases"]:
        case["driver"] = (ORIGINAL.parent / case["driver"]).relative_to(CI).as_posix()
    complete = deepcopy(default)
    complete["description"] = "All registered CI benchmarks and test files; explicitly select to run."
    complete["validation_status"] = "unverified"
    complete["inventory_note"] = "Pytest entries are files; their parameterized nodes are collected at runtime."
    for path in sorted(CI.rglob("test_*.py")):
        relative = path.relative_to(CI)
        name = "pytest_" + path.stem
        complete["cases"].append({"name": name, "runner": "pytest", "family": "pytest",
                                  "split": "holdout", "driver": relative.as_posix(), "arguments": []})
    names = [c["name"] for c in complete["cases"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate generated names; qualify the conflicting pytest filenames")
    default = {"description": default.pop("description"), **default}
    complete = {"description": complete.pop("description"),
                "validation_status": complete.pop("validation_status"),
                "inventory_note": complete.pop("inventory_note"), **complete}
    return default, complete


def main():
    default, complete = build_manifests()
    for name, manifest in (("manifest.json", default), ("manifest_all.json", complete)):
        (CI / name).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{CI / name}: {len(manifest['cases'])} entries")
    print("manifest_all.json remains unverified; the default is manifest.json (100 cases).")


if __name__ == "__main__":
    main()
