"""Evaluation-only, order-preserving index for the measured pooling hotspot.

All tensor operations and validation remain the installed reference function.
Only its two repeated Python descriptor scans use a single layer index. This
module is not a production policy or a changed graph/selection algorithm.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import textwrap
from functools import lru_cache
from unittest.mock import patch


@lru_cache(maxsize=1)
def indexed_pooler():
    import mosaickv.graph.pooling as pooling

    function = ast.parse(
        textwrap.dedent(inspect.getsource(pooling.pool_prompt_attention_coactivation))
    ).body[0]
    if not isinstance(function, ast.FunctionDef):
        raise ValueError("unique reference pooling function required")
    function = copy.deepcopy(function)
    scan = ast.dump(ast.parse("enumerate(full_state.blocks)", mode="eval").body)

    class ReplaceScans(ast.NodeTransformer):
        replaced = 0

        def visit_For(self, node):
            if ast.dump(node.iter) == scan:
                self.replaced += 1
                node.iter = ast.parse("indexed_by_layer.get(layer_index, ())", mode="eval").body
            return self.generic_visit(node)

    transform = ReplaceScans()
    function = transform.visit(function)
    if transform.replaced != 2:
        raise ValueError("reference changed: expected exactly two layer-local descriptor scans")
    outer = [
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and any(isinstance(n, ast.Name) and n.id == "layer_index" for n in ast.walk(node.target))
    ]
    if len(outer) != 1:
        raise ValueError("reference layer-loop boundary changed")
    setup = ast.parse(
        "indexed_by_layer = defaultdict(list)\n"
        "for node_id, block in enumerate(full_state.blocks):\n"
        "    if eligible is None or node_id in eligible:\n"
        "        indexed_by_layer[block.layer].append((node_id, block))\n"
    ).body
    function.body[outer[0] : outer[0]] = setup
    tree = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    namespace = dict(vars(pooling))
    exec(
        compile(ast.fix_missing_locations(tree), "<fox-order-preserving-pooling-index>", "exec"),
        namespace,
    )
    return namespace[function.name]


@contextlib.contextmanager
def pooling_index(enabled=True):
    if not enabled:
        yield
        return
    import mosaickv.graph.builder as builder

    # builder imports this exact symbol; do not patch tensor math or inputs.
    with patch.object(builder, "pool_prompt_attention_coactivation", indexed_pooler()):
        yield
