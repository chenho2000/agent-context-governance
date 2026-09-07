"""Machine-checkable traceability, not an assertion of paper-level reproduction."""

import ast
import json
from collections import Counter
from pathlib import Path


def audit(root=None, check=False):
    root = Path(root or Path(__file__).resolve().parents[2])
    manifest = root / "docs" / "coverage.json"
    if not manifest.exists():
        raise ValueError("coverage audit requires a source checkout with docs/coverage.json")
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    failures = []
    if len({e["id"] for e in entries}) != len(entries):
        failures.append("duplicate coverage IDs")
    for entry in entries:
        for reference in entry["implementation"] + entry["tests"]:
            path, _, symbol = reference.partition(":")
            target = root / path
            if not target.is_file():
                failures.append(reference + ": missing file")
                continue
            if symbol:
                tree = ast.parse(target.read_text(encoding="utf-8"))
                names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                if symbol not in names:
                    failures.append(reference + ": missing symbol")
    if check and failures:
        raise ValueError("; ".join(failures))
    return {
        "entries": len(entries),
        "status_counts": dict(Counter(e["status"] for e in entries)),
        "failures": failures,
        "note": "Checks links and symbols only. Run pytest for behavior; adapters are not live inference.",
        "manifest": str(manifest),
    }
