#!/usr/bin/env python3
"""Export four frozen fits, in SIMT load/store, SIMD load/store order.

Only inference constants enter the source tree; input datasets stay local.
Usage: export_partial_memory_models.py OUTPUT SIMT_LOAD SIMT_STORE SIMD_LOAD SIMD_STORE
"""
import hashlib
import json
import pathlib
import sys


def export(paths):
    text = ["// Generated inference constants; do not edit by hand."]
    descriptors = []
    for i, path in enumerate(paths):
        raw = path.read_bytes()
        text.append("// Model {} SHA256: {}".format(i, hashlib.sha256(raw).hexdigest()))
        model = json.loads(raw)["models"]["primary"]["model"]
        flat = "feature_names" in model
        g = model if flat else model["groups"]["all"]
        names = g["feature_names"] if flat else g["names"]
        text.append("static const char *const kNames{}[] = {{ {} }};".format(
            i, ", ".join(json.dumps(n) for n in names)))
        nodes, roots = [], []

        def visit(node):
            j = len(nodes)
            nodes.append(None)
            if "value" in node:
                nodes[j] = (node["value"], 0, 0, -1)
            else:
                feature, split = node["feature"], node["bin"]
                cuts = g["cuts"][feature]
                # searchsorted(right)<=split means x<cuts[split].
                threshold = cuts[split] if split < len(cuts) else float("inf")
                left, right = visit(node["left"]), visit(node["right"])
                nodes[j] = (threshold, left, right, feature)
            return j

        for tree in g["trees"]:
            roots.append(len(nodes))
            if flat:
                offset = len(nodes)
                for node in tree:
                    nodes.append((node["value"] if node["leaf"] else node["threshold"],
                                  offset + node["left"], offset + node["right"],
                                  -1 if node["leaf"] else node["feature"]))
            else:
                visit(tree)
        text.append("static const Node kNodes{}[] = {{".format(i))
        for value, left, right, feature in nodes:
            value = "INFINITY" if value == float("inf") else repr(value)
            text.append("  {{{}, {}, {}, {}}},".format(value, left, right, feature))
        text.append("};")
        text.append("static const uint32_t kRoots{}[] = {{ {} }};".format(i, ",".join(map(str, roots))))
        bases = g["base"] if flat else g["base"]["groups"]
        text.append("static const Base kBase{}[] = {{".format(i))
        for key, coefficients in bases.items():
            assert len(coefficients) <= 4
            coefficients = coefficients + [0.] * (4 - len(coefficients))
            text.append("  {{{}, {{{}}}}},".format(json.dumps(key), ", ".join(map(repr, coefficients))))
        text.append("};")
        descriptors.append("  {{kNames{0}, {1}, kNodes{0}, kRoots{0}, {2}, kBase{0}, {3}, {4}, {5}, {6}}},".format(
            i, len(names), len(roots), len(bases), 1. if flat else g["learning_rate"],
            g["intercept"] if flat else 0., "false" if flat else "true"))
    text.extend(["static const Frozen kModels[] = {", *descriptors, "};", ""])
    return "\n".join(text)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    pathlib.Path(sys.argv[1]).write_text(export([pathlib.Path(p) for p in sys.argv[2:]]))
