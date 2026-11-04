import ast
import inspect
import weakref

from mosaickv.evaluation import captured_code_gpu_control


class Owner:
    pass


def release_with_locals_snapshot():
    graph = Owner()
    reference = weakref.ref(graph)
    assert "graph" in locals()
    del graph
    return reference() is not None


def release_without_snapshot():
    graph = Owner()
    reference = weakref.ref(graph)
    del graph
    return reference() is None


def test_locals_snapshot_must_not_retain_the_final_cuda_graph_owner():
    assert release_with_locals_snapshot()
    assert release_without_snapshot()
    tree = ast.parse(inspect.getsource(captured_code_gpu_control.run.__wrapped__))
    assert not any(
        isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "locals"
        for x in ast.walk(tree)
    )
    assert any(
        isinstance(x, ast.Delete)
        and any(isinstance(t, ast.Name) and t.id == "graph" for t in x.targets)
        for x in ast.walk(tree)
    )
