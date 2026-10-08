import ast

from taichi_forge.lang.ast._inline_ir import InlineSyntax, scalar_key


def test_inline_ir_rejects_shadowed_captures_and_attribute_writes():
    shadowed = InlineSyntax(ast.parse("def f(x):\n    x += gain\n    gain = 2\n    return x\n"))
    assert shadowed.dependency_key({"gain": 1}) is None
    assignment = InlineSyntax(ast.parse("def f(x):\n    module.value = x\n    return x\n"))
    assert not assignment.safe


def test_inline_ir_distinguishes_missing_and_none_captures():
    syntax = InlineSyntax(ast.parse("def f(x):\n    return x if flag is None else x + 1\n"))
    assert syntax.dependency_key({"flag": None}) is not None
    assert syntax.dependency_key({}) is None
    assert scalar_key(-0.0) != scalar_key(0.0)


def test_inline_ir_never_executes_attribute_descriptors():
    class Capture:
        @property
        def value(self):
            raise AssertionError("candidate checks must not execute Python callbacks")

    syntax = InlineSyntax(ast.parse("def f(x):\n    return x + capture.value\n"))
    assert syntax.dependency_key({"capture": Capture()}) is None
