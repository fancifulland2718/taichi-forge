"""Recognize terminal Host error interfaces without evaluating application code."""

import ast
import inspect
from types import ModuleType
import typing


def _reference(node, namespace):
    if isinstance(node, ast.Name):
        return namespace.get(node.id)
    if isinstance(node, ast.Attribute):
        parent = _reference(node.value, namespace)
        if isinstance(parent, ModuleType):
            return vars(parent).get(node.attr)
    return None


def is_error_interface(node, namespace):
    """A single literal-only call to an ordinary, explicitly nonreturning helper.

    A name such as ``raise_error`` is not evidence. The annotation must resolve
    to typing.NoReturn/Never, without get_type_hints/eval or descriptor execution.
    Anything doing numerical work remains subject to normal compiler lowering.
    """
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        if isinstance(body[0].value.value, str):
            body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Expr) or not isinstance(body[0].value, ast.Call):
        return False
    call = body[0].value
    if any(not isinstance(arg, ast.Constant) for arg in call.args):
        return False
    if any(arg.arg is None or not isinstance(arg.value, ast.Constant) for arg in call.keywords):
        return False
    helper = _reference(call.func, namespace)
    if (
        not inspect.isfunction(helper)
        or getattr(helper, "_is_taichi_function", False)
        or getattr(helper, "_is_wrapped_kernel", False)
    ):
        return False
    annotation = helper.__annotations__.get("return")
    if isinstance(annotation, str):
        try:
            expression = ast.parse(annotation, mode="eval").body
        except SyntaxError:
            return False
        if isinstance(expression, ast.Constant) and isinstance(expression.value, str):
            try:
                expression = ast.parse(expression.value, mode="eval").body
            except SyntaxError:
                return False
        annotation = _reference(expression, helper.__globals__)
    return annotation is typing.NoReturn or (hasattr(typing, "Never") and annotation is typing.Never)


def project_error_interface(node, definition):
    """Preserve the error helper, but fail if its declared NoReturn contract lies."""
    node.decorator_list = []
    node.body.append(
        ast.Raise(
            ast.Call(
                ast.Name("RuntimeError", ast.Load()),
                [ast.Constant(f"Nonreturning package interface {definition} returned unexpectedly")],
                [],
            ),
            None,
        )
    )
    return node
