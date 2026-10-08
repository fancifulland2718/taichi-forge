import ast
import copy
from types import SimpleNamespace

import pytest

from taichi_forge.lang.ast.ast_transformer_utils import ASTTransformerContext


def make_context(source, cache=None, file="example.py", name="example", start_lineno=10):
    return ASTTransformerContext(
        func=SimpleNamespace(func=SimpleNamespace(__name__=name)),
        src=source.splitlines(),
        file=file,
        start_lineno=start_lineno,
        source_info_cache=cache,
    )


@pytest.mark.parametrize(
    "source, expected",
    [
        ("value = other + 1", "value = other + 1\n^^^^^^^^^^^^^^^^^\n"),
        ("value = (\n    other +\n    1\n)", "value = (\n^^^^^^^^^\n    other +\n    ^^^^^^^\n    1\n    ^\n)\n^\n"),
        ("for i in range(2):\n    value = i", "for i in range(2):\n^^^^^^^^^^^^^^^^^^\n"),
    ],
)
def test_source_info_keeps_carets_across_copied_templates(source, expected):
    node = ast.parse(source).body[0]
    cache = {}
    context = make_context(source, cache)
    assert context.get_pos_info(node) == 'File "example.py", line 10, in example:\n' + expected
    # Template copies can share source text, but each caller's header stays live.
    other = make_context(source, cache, file="other.py", name="other", start_lineno=20)
    assert other.get_pos_info(copy.deepcopy(node)) == 'File "other.py", line 20, in other:\n' + expected


def test_source_info_rechecks_changed_node_span_and_block_header():
    source = "if condition:\n    # comment\n    value = 1"
    context = make_context(source)
    node = ast.parse(source).body[0]
    full_header = context.get_pos_info(node)
    assert "# comment" in full_header
    node.body[0].lineno = 2
    assert "# comment" not in context.get_pos_info(node)
    name = ast.parse("value = other + 1").body[0].value.left
    context = make_context("value = other + 1")
    assert "        ^^^^^\n" in context.get_pos_info(name)
    name.end_col_offset -= 1
    assert "        ^^^^\n" in context.get_pos_info(name)


def test_source_info_does_not_retain_expanded_ast_nodes():
    import gc
    import weakref

    source = "value = other + 1"
    context = make_context(source)
    node = ast.parse(source).body[0]
    reference = weakref.ref(node)
    context.get_pos_info(node)
    del node
    gc.collect()
    assert reference() is None
