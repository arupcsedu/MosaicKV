"""Isolated implementation factorial, never an algorithm/configuration change."""

from __future__ import annotations

import ast
import contextlib
import copy
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha


def budget_function_tree(path):
    tree = ast.parse(Path(path).read_text())
    matches = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_dense_physical_cache_budget"
    ]
    if len(matches) != 1:
        raise ValueError("reference budget implementation is not unique")
    return matches[0]


def without_unit_reuse(function):
    """Recover the unchanged slow branch; also audit the precise source diff."""

    class RemoveUnitGuard(ast.NodeTransformer):
        removed = 0

        def visit_If(self, node):
            direct = [n for n in node.body if isinstance(n, ast.Return)]
            if any(
                isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name)
                and n.value.func.id == "_identical_unit_reference_budget"
                for n in direct
            ):
                self.removed += 1
                return None
            return self.generic_visit(node)

    transform = RemoveUnitGuard()
    result = transform.visit(copy.deepcopy(function))
    if transform.removed != 1:
        raise ValueError("optimization must add exactly one identifiable guarded fast path")
    return result


def factorial_order(arms, input_index, repeat):
    """Rotate positions and exactly alternate reference/optimized orientation."""
    offset = (input_index + repeat) % len(arms)
    ordered = list(arms[offset:] + arms[:offset])
    reference_first = (input_index + repeat) % 2 == 0
    identities = sorted({(m, cap) for m, cap, variant in arms if variant == "reference-budget"})
    for method, cap in identities:
        left = ordered.index((method, cap, "reference-budget"))
        right = ordered.index((method, cap, "reuse-budget"))
        if (left < right) != reference_first:
            ordered[left], ordered[right] = ordered[right], ordered[left]
    return ordered


@lru_cache(maxsize=2)
def reference_budget(source_root, source_sha, file_sha, function_sha):
    """Load only the hash-bound reference function, using identical live classes.

    This is MosaicKV's own frozen reference code, not an AAFLOW/sibling import.
    No model, prefill, selected IDs, or runtime tensors are cached here.
    """
    import mosaickv.backends.hf_runtime as runtime

    root = Path(source_root)
    clean_source(root, source_sha)
    path = root / "mosaickv/src/mosaickv/backends/hf_runtime.py"
    function = budget_function_tree(path)
    if (
        sha(path) != file_sha
        or digest(ast.dump(function, include_attributes=False)) != function_sha
    ):
        raise ValueError("frozen reference implementation changed")
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    namespace = dict(vars(runtime))
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace["_dense_physical_cache_budget"]


@contextlib.contextmanager
def budget_implementation(lock, variant, *, required):
    if not required:
        yield
        return
    protocol = lock.get("optimization_protocol")
    if not protocol or variant not in {"complete", "reference-budget", "reuse-budget"}:
        raise ValueError("unregistered implementation factorial")
    if variant == "reference-budget":
        import mosaickv.backends.hf_runtime as runtime

        function = reference_budget(
            protocol["reference_source_root"],
            protocol["reference_source_sha"],
            protocol["reference_runtime_sha256"],
            protocol["reference_function_sha256"],
        )
        with patch.object(runtime, "_dense_physical_cache_budget", function):
            yield
    else:
        yield
