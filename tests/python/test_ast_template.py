import ast
import gc
import weakref

import pytest

from taichi_forge.lang.ast._ast_template import ASTTemplate


@pytest.mark.parametrize(
    "source",
    [
        "def f(x, /, y=1, *, z=None):\n    return x + y * z\n",
        "def f(v):\n    return [x * 2 for x in v if x], {**v, 'x': None}, v[1:3]\n",
        "def f(x):\n    for i in range(3):\n        if i:\n            x += i\n    return x\n",
        'def f(x):\n    return f"value: {x:.3f}", b"data", 2j, ...\n',
    ],
)
def test_ast_template_preserves_parsed_structure_and_locations(source):
    tree = ast.parse(source)
    template = ASTTemplate(tree)
    first, second = template.instantiate(), template.instantiate()
    expected = ast.dump(tree, include_attributes=True)
    assert ast.dump(first, include_attributes=True) == expected
    assert ast.dump(second, include_attributes=True) == expected
    original_ids = {id(node) for node in ast.walk(tree)}
    first_ids = {id(node) for node in ast.walk(first)}
    second_ids = {id(node) for node in ast.walk(second)}
    assert original_ids.isdisjoint(first_ids)
    assert first_ids.isdisjoint(second_ids)
    # Lowering decorates nodes and can rewrite child lists; copies must be private.
    for node in ast.walk(first):
        node.ptr = object()
    first.body[0].body.clear()
    assert ast.dump(template.instantiate(), include_attributes=True) == expected
    assert all(not hasattr(node, "ptr") for node in ast.walk(second))
    compile(second, "example.py", "exec")


def test_ast_template_preserves_aliases_without_retaining_source_nodes():
    tree = ast.parse("x + y + z")
    outer = tree.body[0].value
    assert outer.op is outer.left.op
    reference = weakref.ref(outer)
    template = ASTTemplate(tree)
    del tree, outer
    gc.collect()
    assert reference() is None
    first, second = template.instantiate(), template.instantiate()
    outer = first.body[0].value
    assert outer.op is outer.left.op
    assert outer.op is not second.body[0].value.op


def test_ast_template_does_not_share_optional_child_lists():
    template = ASTTemplate(ast.parse("def f(*, x=None, y):\n    return x, y\n"))
    first, second = template.instantiate(), template.instantiate()
    assert first.body[0].args.kw_defaults[1] is None
    first.body[0].args.kw_defaults[0].value = 9
    first.body[0].args.kw_defaults.append(None)
    assert second.body[0].args.kw_defaults[0].value is None
    assert len(second.body[0].args.kw_defaults) == 2
