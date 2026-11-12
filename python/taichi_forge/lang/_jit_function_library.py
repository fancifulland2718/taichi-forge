"""Compile-time eligibility and dependency facts for JIT function units.

Unknown Python calls/objects stay on ordinary inline expansion. These facts are
consulted only while building a kernel, never while invoking compiled code.
"""

import ast
import builtins
import inspect
import operator
import types

from taichi_forge.lang._function_library import FunctionLibrary
from taichi_forge.lang.ast._inline_ir import InlineSyntax, _MISSING, _PURE_CALLS, _path, scalar_key
from taichi_forge.lang.field import Field
from taichi_forge.lang import impl
from taichi_forge.lang.matrix import Matrix, Vector, MatrixType
from taichi_forge.types import primitive_types, template

_VALUE_METHODS = frozenset(
    (
        "dot",
        "cross",
        "norm",
        "norm_sqr",
        "normalized",
        "sum",
        "min",
        "max",
        "transpose",
        "determinant",
        "inverse",
        "cast",
    )
)
_VALUE_ATTRIBUTES = frozenset(("x", "y", "z", "w", "n", "m"))
_CONSTRUCTORS = (Matrix, Vector, Matrix.zero, Vector.zero, Matrix.identity)


def _resolve(path, scope):
    if not path:
        return _MISSING, None
    value = scope.get(path[0], vars(builtins).get(path[0], _MISSING))
    receiver = None
    for name in path[1:]:
        receiver = value
        if type(value) is types.ModuleType:
            value = vars(value).get(name, _MISSING)
        else:
            # Static inspection must not hide Python attribute-read effects.
            # The normal data_oriented wrapper only binds kernels; arbitrary
            # user __getattribute__ implementations remain on inline expansion.
            from taichi_forge.lang.kernel_impl import data_oriented

            for cls in type(value).__mro__:
                getter = vars(cls).get("__getattribute__", object.__getattribute__)
                if (
                    getter is not object.__getattribute__
                    and getter is not type.__getattribute__
                    and not (type(getter) is types.FunctionType and getter.__code__ in data_oriented.__code__.co_consts)
                ):
                    return _MISSING, receiver
            value = inspect.getattr_static(value, name, _MISSING)
    if isinstance(value, staticmethod):
        value = value.__func__
    return value, receiver


def _wrapped(value):
    if type(value) is types.FunctionType and vars(value).get("_is_taichi_function", False):
        return vars(value).get("func")
    return None


def _literal(node, scope, *, objects=False):
    if isinstance(node, ast.Constant):
        return node.value if scalar_key(node.value) is not None else _MISSING
    if isinstance(node, (ast.Name, ast.Attribute)):
        value = _resolve(_path(node), scope)[0]
        return value if objects or scalar_key(value) is not None else _MISSING
    if isinstance(node, ast.UnaryOp):
        value = _literal(node.operand, scope)
        op = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}.get(type(node.op))
        if type(value) in (bool, int, float) and op:
            return op(value)
    if isinstance(node, ast.BoolOp):
        values = [_literal(arg, scope) for arg in node.values]
        if all(type(value) in (bool, int, float) for value in values):
            for value in values[:-1]:
                if bool(value) != isinstance(node.op, ast.And):
                    return value
            return values[-1]
    return _MISSING


class FunctionSyntax(InlineSyntax):
    def __init__(self, tree, scope, bindings):
        self.scope = scope
        self.bindings = bindings
        self.calls = {}
        self.forwarded = set()
        super().__init__(tree)

    def generic_visit(self, node):
        if isinstance(node, (ast.Subscript, ast.While, ast.Break, ast.Continue)):
            ast.NodeVisitor.generic_visit(self, node)
        else:
            super().generic_visit(node)

    def visit_For(self, node):
        # Extraction is only requested inside an existing thread's iteration.
        self.generic_visit(node)

    def visit_Call(self, node):
        path = _path(node.func)
        value = _resolve(path, self.scope)[0]
        if isinstance(node.func, ast.Attribute) and node.func.attr in _VALUE_METHODS and value is _MISSING:
            self.visit(node.func.value)
            for arg in node.args:
                self.visit(arg)
            for keyword in node.keywords:
                if keyword.arg is None:
                    self.safe = False
                self.visit(keyword.value)
        else:
            if path:
                self.calls.setdefault(path, []).append(node)
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                if isinstance(arg, ast.Name) and arg.id in self.bindings:
                    if _wrapped(_resolve(path, self.scope)[0]) is not None:
                        self.forwarded.add(id(arg))
            super().visit_Call(node)

    def visit_If(self, node):
        self.visit(node.test)
        if (
            isinstance(node.test, ast.Call)
            and len(node.test.args) == 1
            and not node.test.keywords
            and _resolve(_path(node.test.func), self.scope)[0] is impl.static
        ):
            value = _literal(node.test.args[0], self.scope)
            if type(value) in (bool, int, float):
                for statement in node.body if value else node.orelse:
                    self.visit(statement)
                return
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    def reference(self, node, kind):
        path = _path(node)
        if path and path[0] in self.locals and path[0] not in self.bindings:
            allowed = _VALUE_METHODS if kind == "call" else _VALUE_ATTRIBUTES
            if len(path) != 2 or path[1] not in allowed:
                self.safe = False
            return
        if path is None:
            self.safe = False
        else:
            self.references.add((path, kind))

    def visit_Name(self, node):
        if (
            id(node) not in self.forwarded
            and isinstance(node.ctx, ast.Load)
            and (node.id not in self.locals or node.id in self.bindings)
        ):
            self.reference(node, "value")


class JitFunctionLibrary(FunctionLibrary):
    def __init__(self):
        super().__init__()
        self.function_library = True
        self.allow_spirv = True
        self._syntax = {}

    def _dependencies(self, source, bindings, active):
        if source in active or source.pyfunc or source.is_real_function:
            return None
        if source not in self._syntax:
            self._syntax[source], _ = source._definition.acquire(cache=True)
        scope = source._definition.scope()
        scope.update(bindings)
        syntax = FunctionSyntax(self._syntax[source], scope, bindings)
        if not syntax.safe or any(name in scope for name in syntax.assigned):
            return None
        dependencies = []
        for path, kind in sorted(syntax.references):
            value, receiver = _resolve(path, scope)
            if kind == "call":
                wrapped = _wrapped(value)
                if wrapped is not None:
                    children = []
                    for call in syntax.calls.get(path, ()):
                        values = [_literal(arg, scope, objects=True) for arg in call.args]
                        if wrapped.classfunc and receiver is not None:
                            values.insert(0, receiver)
                        keywords = {kw.arg: _literal(kw.value, scope, objects=True) for kw in call.keywords}
                        try:
                            bound = wrapped._definition.signature.bind(*values, **keywords)
                        except TypeError:
                            return None
                        bound.apply_defaults()
                        # Templates and forwarded resource owners are Python
                        # compilation data. Numerical operands stay symbolic
                        # even if this call happens to pass a literal.
                        child_bindings = {
                            arg.name: bound.arguments[arg.name]
                            for arg in wrapped.arguments
                            if bound.arguments[arg.name] is not _MISSING
                            and (isinstance(arg.annotation, template) or scalar_key(bound.arguments[arg.name]) is None)
                        }
                        if any(
                            arg.name not in child_bindings
                            for arg in wrapped.arguments
                            if isinstance(arg.annotation, template)
                        ):
                            return None
                        child = self._dependencies(wrapped, child_bindings, active | {source})
                        if child is None:
                            return None
                        children.append(child)
                    key = ("function", wrapped, tuple(children))
                elif isinstance(value, MatrixType):
                    key = ("tensor_type", value.n, value.m, value.ndim, str(value.dtype))
                elif (
                    id(value) in _PURE_CALLS
                    or id(value) in primitive_types.type_ids
                    or any(value is constructor for constructor in _CONSTRUCTORS)
                ):
                    key = ("call", id(value))
                else:
                    return None
            elif isinstance(value, Field):
                # Identity only: device contents are live, not compile-time data.
                key = ("field", value)
            elif isinstance(value, MatrixType):
                key = ("tensor_type", value.n, value.m, value.ndim, str(value.dtype))
            elif id(value) in primitive_types.type_ids:
                key = ("dtype", value)
            else:
                key = scalar_key(value)
                if key is None:
                    return None
            dependencies.append((path, kind, key))
        return tuple(dependencies)

    def library_function(self, source, args):
        if not self.thread_local_call or source.is_real_function or source.pyfunc:
            return None
        bindings = {
            arg.name: value for arg, value in zip(source.arguments, args) if isinstance(arg.annotation, template)
        }
        try:
            dependencies = self._dependencies(source, bindings, set())
        except RecursionError:
            return None
        if dependencies is None:
            return None
        function = super().library_function(source, args, dependencies=dependencies)
        if function is not None:
            function._jit_inline_function = True
        return function
