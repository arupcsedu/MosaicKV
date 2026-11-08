"""Prove the retry changes ONLY one explicit new-format dispatch branch."""

import ast


def normalized(source):
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fill_component"
    )
    expected_test = ast.dump(ast.parse("isinstance(packet, HistoryFormat)", mode="eval").body)
    expected_body = ast.dump(
        ast.parse("packet.fill_component(attention, component, destination)\nreturn").body[0]
    )
    removed = 0
    body = []
    for node in function.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "mosaickv.evaluation.history_source_precision"
        ):
            if [(a.name, a.asname) for a in node.names] != [("HistoryFormat", None)]:
                raise ValueError("unreviewed history import")
            continue
        if isinstance(node, ast.If) and ast.dump(node.test) == expected_test:
            if (
                len(node.body) != 2
                or ast.dump(node.body[0]) != expected_body
                or not isinstance(node.body[1], ast.Return)
                or node.body[1].value is not None
                or node.orelse
            ):
                raise ValueError("unreviewed history dispatch body")
            removed += 1
            continue
        body.append(node)
    function.body = body
    return ast.dump(tree, include_attributes=False), removed


def verify(previous, current):
    old, old_count = normalized(previous)
    new, new_count = normalized(current)
    if old_count != 0 or new_count != 1 or old != new:
        raise ValueError(
            "legacy transport behavior changed; existing model controls cannot be reused"
        )
