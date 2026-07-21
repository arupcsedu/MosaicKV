from __future__ import annotations

import ast
from pathlib import Path


def test_mosaickv_does_not_import_sibling_aaflow() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "mosaickv"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(name.split(".", maxsplit=1)[0].lower() == "aaflow" for name in names):
                violations.append(str(path.relative_to(source_root)))
    assert not violations, f"AAFLOW imports must be isolated behind MosaicKV: {violations}"
