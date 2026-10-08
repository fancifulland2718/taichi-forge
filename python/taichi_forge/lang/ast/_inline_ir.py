"""Conservative syntax/dependency proof for kernel-local scalar IR reuse.

This recognizes a subset with no observable Python compilation effects. All
other functions keep ordinary expansion, regardless of their size. Store only
syntax paths here: expanded AST nodes can retain native expressions.
"""

import ast
import builtins
import struct
import types

from taichi_forge.lang import impl, ops
from taichi_forge.types import primitive_types


_PURE_FUNCTIONS = (
    abs,
    min,
    max,
    int,
    float,
    bool,
    range,
    len,
    impl.static,
    ops.cast,
    ops.bit_cast,
    ops.abs,
    ops.min,
    ops.max,
    ops.sin,
    ops.cos,
    ops.tan,
    ops.tanh,
    ops.asin,
    ops.acos,
    ops.atan2,
    ops.exp,
    ops.log,
    ops.sqrt,
    ops.rsqrt,
    ops.floor,
    ops.ceil,
    ops.round,
)
# Keep strong references, and compare identities without invoking __hash__ on
# untrusted captured objects. Rebinding a module must not permit ID reuse.
_PURE_CALLS = frozenset(id(value) for value in _PURE_FUNCTIONS)
_SCALARS = (bool, int, float, str, type(None))
_MISSING = object()
_ALLOWED = {
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Return,
    ast.Expr,
    ast.Pass,
    ast.If,
    ast.For,
    ast.Name,
    ast.Attribute,
    ast.Constant,
    ast.Call,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Tuple,
    ast.List,
    ast.keyword,
    ast.Load,
    ast.Store,
}


def scalar_key(value):
    if type(value) not in _SCALARS:
        return None
    if type(value) is float:
        return (float, struct.pack("!d", value))
    return (type(value), value)


def _path(node):
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _path(node.value)
        return prefix + (node.attr,) if prefix else None
    return None


class InlineSyntax(ast.NodeVisitor):
    def __init__(self, tree):
        function = tree.body[0]
        self.safe = True
        self.references = set()
        parameters = {arg.arg for arg in function.args.args}
        self.assigned = {
            node.id
            for statement in function.body
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        } - parameters
        self.locals = parameters | self.assigned
        returns = [
            node for statement in function.body for node in ast.walk(statement) if isinstance(node, ast.Return)
        ]
        # Preserve regular ti.func return semantics; no early-return rewriting.
        if len(returns) != 1 or returns[0] is not function.body[-1] or returns[0].value is None:
            self.safe = False
        try:
            for statement in function.body:
                self.visit(statement)
        except RecursionError:
            # An optional proof must not impose a new expression-depth limit.
            self.safe = False

    def generic_visit(self, node):
        if type(node) not in _ALLOWED and not isinstance(node, (ast.operator, ast.unaryop, ast.boolop, ast.cmpop)):
            self.safe = False
            return
        super().generic_visit(node)

    def reference(self, node, kind):
        path = _path(node)
        if path is None or path[0] in self.locals:
            self.safe = False
        else:
            self.references.add((path, kind))

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id not in self.locals:
            self.reference(node, "value")

    def visit_Attribute(self, node):
        if not isinstance(node.ctx, ast.Load):
            self.safe = False
        self.reference(node, "value")

    def visit_Call(self, node):
        self.reference(node.func, "call")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                self.safe = False
            self.visit(keyword.value)

    def visit_For(self, node):
        # Runtime loops can change offload topology and caller loop context.
        if not isinstance(node.iter, ast.Call):
            self.safe = False
        else:
            self.reference(node.iter.func, "static")
        self.generic_visit(node)

    def dependency_key(self, scope):
        if not self.safe:
            return None
        # Taichi resolves names as statements are expanded. A read before a
        # local assignment may still refer to a global with the same name.
        if any(name in scope for name in self.assigned):
            return None
        dependencies = []
        for path, kind in sorted(self.references):
            value = scope.get(path[0], vars(builtins).get(path[0], _MISSING))
            for name in path[1:]:
                # Never invoke descriptors, properties or module __getattr__.
                if type(value) is not types.ModuleType:
                    return None
                value = vars(value).get(name, _MISSING)
            if kind == "static":
                if value is not impl.static:
                    return None
                key = ("call", id(value))
            elif kind == "call":
                if id(value) not in _PURE_CALLS and id(value) not in primitive_types.type_ids:
                    return None
                key = ("call", id(value))
            elif id(value) in primitive_types.type_ids:
                key = ("dtype", value)
            else:
                key = scalar_key(value)
                if key is None:
                    return None
            dependencies.append((path, kind, key))
        return tuple(dependencies)


class UnsupportedInlineResult(Exception):
    """An otherwise pure body returned a Python object instead of a scalar."""
