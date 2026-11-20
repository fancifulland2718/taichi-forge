"""Compile-time dependency analysis for typed JIT and AOT function units.

Unknown Python calls/objects stay on ordinary inline expansion. These facts are
consulted only while building a kernel, never while invoking compiled code.
"""

import ast
import builtins
import inspect
import operator
import types
from collections import Counter

from taichi_forge.lang._function_library import FunctionLibrary
from taichi_forge.lang.ast._inline_ir import InlineSyntax, _MISSING, _PURE_CALLS, _path, scalar_key
from taichi_forge.lang.field import Field
from taichi_forge._lib import core
from taichi_forge.lang import impl, misc, ops
from taichi_forge.lang._ndrange import ndrange
from taichi_forge.lang.enums import DeviceCapability
from taichi_forge.lang.matrix import Matrix, Vector, MatrixType
from taichi_forge.lang.struct import Struct
from taichi_forge.math.mathimpl import isinf, isnan
from taichi_forge.types import primitive_types, template
from taichi_forge.types.ndarray_type import NdarrayType

_VALUE_METHODS = frozenset(
    (
        "dot",
        "cross",
        "norm",
        "norm_sqr",
        "normalized",
        "any",
        "all",
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
_RESOURCE_ATOMICS = (
    ops.atomic_add, ops.atomic_sub, ops.atomic_mul, ops.atomic_min,
    ops.atomic_max, ops.atomic_and, ops.atomic_or, ops.atomic_xor,
)
_CAPABILITY_NAMES = frozenset(name for name, value in vars(DeviceCapability).items()
                              if type(value) is str and value == name)


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


def _literal(node, scope, *, objects=False, attribute=None):
    if isinstance(node, ast.Constant):
        return node.value if scalar_key(node.value) is not None else _MISSING
    if attribute is not None and isinstance(node, (ast.Call, ast.Attribute)):
        value = attribute(node)
        if value is not _MISSING:
            return value
    if isinstance(node, (ast.Name, ast.Attribute)):
        path = _path(node)
        value = _resolve(path, scope)[0]
        return value if objects or scalar_key(value) is not None else _MISSING
    if isinstance(node, ast.UnaryOp):
        value = _literal(node.operand, scope, attribute=attribute)
        op = {ast.Not: operator.not_, ast.USub: operator.neg, ast.UAdd: operator.pos}.get(type(node.op))
        if type(value) in (bool, int, float) and op:
            return op(value)
    if isinstance(node, ast.BoolOp):
        values = [_literal(arg, scope, attribute=attribute) for arg in node.values]
        if all(type(value) in (bool, int, float) for value in values):
            for value in values[:-1]:
                if bool(value) != isinstance(node.op, ast.And):
                    return value
            return values[-1]
    return _MISSING


class _LocalRecord(frozenset):
    """Member names of a DSL record owned inside the function being inspected."""


_STATIC_INTEGER = object()


def _constructed_record(call, scope):
    if not isinstance(call, ast.Call) or call.args or not call.keywords:
        return None
    if _resolve(_path(call.func), scope)[0] is not Struct:
        return None
    names = [kw.arg for kw in call.keywords]
    if all(name and not name.startswith("_") and not hasattr(Struct, name) for name in names):
        return _LocalRecord(names)
    return None


def _literal_integer_loop(iterator, scope):
    if not isinstance(iterator, ast.Call) or len(iterator.args) != 1 or iterator.keywords:
        return False
    if _resolve(_path(iterator.func), scope)[0] is not impl.static:
        return False
    sequence = iterator.args[0]
    if isinstance(sequence, (ast.Tuple, ast.List)):
        values = sequence.elts
    elif (isinstance(sequence, ast.Call) and _resolve(_path(sequence.func), scope)[0] is range
          and not sequence.keywords and 1 <= len(sequence.args) <= 3):
        values = sequence.args
    else:
        return False
    return all(isinstance(item, (ast.Constant, ast.UnaryOp)) and type(_literal(item, {})) in (bool, int)
               for item in values)


class FunctionSyntax(InlineSyntax):
    def __init__(self, tree, scope, bindings, static_scope, *, package_fields=False,
                 single_assignments=(), returned_record=None, declared_static=None):
        self.scope = scope
        self.static_scope = static_scope
        self.bindings = bindings
        self.declared_static = declared_static
        self.absent_templates = {name for name, value in bindings.items() if value is None} if package_fields else ()
        self.calls = {}
        self.target_facts = set()
        self.forwarded = set()
        self.package_fields = package_fields
        self.returned_record = returned_record
        self.local_bindings = {k: v for k, v in bindings.items() if isinstance(v, _LocalRecord)}
        self.single_assignments = single_assignments
        super().__init__(tree, allow_void=package_fields)

    def static_literal(self, node):
        return _literal(node, self.static_scope, attribute=self.member_value if self.declared_static else None)

    def static_member(self, node):
        # Component capture lowers builtin getattr on a declared member exactly
        # like dotted access. A computed name or an effectful default still
        # needs normal inline evaluation.
        if isinstance(node, ast.Call):
            target = _path(node.func)
            if (node.keywords or len(node.args) not in (2, 3)
                    or not isinstance(node.args[1], ast.Constant) or type(node.args[1].value) is not str
                    or (len(node.args) == 3 and not isinstance(node.args[2], ast.Constant))
                    or not target or target[0] in self.locals or _resolve(target, self.scope)[0] is not getattr):
                return None
            owner = _path(node.args[0])
            path = (*owner, node.args[1].value) if owner else None
        else:
            path = _path(node)
        if path and path[0] in self.bindings and self.declared_static(self.bindings[path[0]], path[1:]):
            return path
        return None

    def member_value(self, node):
        path = self.static_member(node)
        value = _resolve(path, self.bindings)[0] if path else _MISSING
        return value if type(value) in (bool, int, float) else _MISSING

    def return_block(self, body):
        # Count returns in the same static branches the compiler will lower.
        # Keep the suffix even after a selected return so an early return
        # cannot silently acquire different real-function semantics.
        if not self.package_fields:
            return body
        selected = []
        for statement in body:
            if isinstance(statement, ast.If):
                test = statement.test
                if isinstance(test, ast.Call) and len(test.args) == 1 and not test.keywords:
                    path = _path(test.func)
                    if path and path[0] not in self.locals and _resolve(path, self.scope)[0] is impl.static:
                        value = self.static_literal(test.args[0])
                        if type(value) is bool:
                            selected.extend(self.return_block(statement.body if value else statement.orelse))
                            continue
            selected.append(statement)
        return selected

    def generic_visit(self, node):
        if isinstance(node, (ast.Subscript, ast.While, ast.Break, ast.Continue)) or (
            self.package_fields and isinstance(node, ast.Slice)
        ):
            ast.NodeVisitor.generic_visit(self, node)
        else:
            super().generic_visit(node)

    def visit_For(self, node):
        # Inspect all branches once. The DSL supplies the actual integer when
        # it lowers each iteration; dependency discovery needs only its kind.
        if (isinstance(node.target, ast.Name) and node.target.id in self.single_assignments
                and node.target.id not in self.scope and _literal_integer_loop(node.iter, self.scope)):
            self.visit(node.iter)
            self.local_bindings[node.target.id] = _STATIC_INTEGER
            for statement in node.body:
                self.visit(statement)
            del self.local_bindings[node.target.id]
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_ListComp(self, node):
        # Fixed tensor constructors often read a field once per mathematical
        # axis. Track the lexical indices without expanding the comprehension.
        if not self.package_fields:
            return self.generic_visit(node)
        enclosing = self.local_bindings
        self.local_bindings = enclosing.copy()
        try:
            for generator in node.generators:
                if (generator.is_async or not isinstance(generator.target, ast.Name)
                        or generator.target.id in self.scope or generator.target.id in self.local_bindings
                        or not _literal_integer_loop(generator.iter, self.scope)):
                    self.safe = False
                    return
                self.visit(generator.iter)
                self.local_bindings[generator.target.id] = _STATIC_INTEGER
                for condition in generator.ifs:
                    self.visit(condition)
            self.visit(node.elt)
        finally:
            self.local_bindings = enclosing

    def visit_Assign(self, node):
        self.generic_visit(node)
        if (self.package_fields and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self.single_assignments and node.targets[0].id not in self.scope
                and isinstance(node.value, ast.Call)
                and _path(node.value.func) and _path(node.value.func)[0] not in self.locals):
            schema = _constructed_record(node.value, self.scope)
            if schema is None and self.returned_record:
                callee = _wrapped(_resolve(_path(node.value.func), self.scope)[0])
                if callee is not None:
                    schema = self.returned_record(callee)
            if schema is not None:
                self.local_bindings[node.targets[0].id] = schema

    def visit_Attribute(self, node):
        if self.package_fields and self._program_arch(node):
            self.target_facts.add(("arch", int(impl.current_cfg().arch)))
            return
        path = _path(node)
        if path and len(path) == 2:
            schema = self.local_bindings.get(path[0])
            if isinstance(schema, _LocalRecord) and path[1] in schema:
                return
        super().visit_Attribute(node)

    def _program_arch(self, node):
        if not isinstance(node, ast.Attribute) or node.attr != "arch":
            return False
        cfg = node.value
        if not isinstance(cfg, ast.Call) or cfg.args or cfg.keywords:
            return False
        path = _path(cfg.func)
        return bool(path and path[0] not in self.locals and _resolve(path, self.scope)[0] is impl.current_cfg)

    def _target_extension(self, node):
        if (not self.package_fields or not isinstance(node, ast.Call) or len(node.args) != 2
                or node.keywords or not self._program_arch(node.args[0])):
            return None
        query = _path(node.func)
        if not query or query[0] in self.locals or _resolve(query, self.scope)[0] is not misc.is_extension_supported:
            return None
        path = _path(node.args[1])
        if not path or path[0] in self.locals:
            return None
        if len(path) > 1 and _resolve(path[:-1], self.scope)[0] is core.Extension:
            ext = core.Extension.__members__.get(path[-1])
        else:
            ext = _resolve(path, self.scope)[0]
        if type(ext) is not core.Extension:
            return None
        arch = impl.current_cfg().arch
        supported = misc.is_extension_supported(arch, ext)
        self.target_facts.add(("extension", int(arch), int(ext), supported))
        return supported

    def _device_capability(self, node):
        if (not self.package_fields or len(node.args) != 1 or node.keywords
                or not isinstance(node.func, ast.Attribute) or node.func.attr != "get"):
            return False
        caps = node.func.value
        if (not isinstance(caps, ast.Call) or caps.args or caps.keywords
                or not isinstance(caps.func, ast.Attribute) or caps.func.attr != "get_device_caps"):
            return False
        program = caps.func.value
        if not isinstance(program, ast.Attribute) or program.attr != "prog":
            return False
        runtime = program.value
        if not isinstance(runtime, ast.Call) or runtime.args or runtime.keywords:
            return False
        factory, arg = _path(runtime.func), _path(node.args[0])
        if (not factory or factory[0] in self.locals or _resolve(factory, self.scope)[0] is not impl.get_runtime
                or not arg or len(arg) < 2 or arg[0] in self.locals
                or _resolve(arg[:-1], self.scope)[0] is not DeviceCapability):
            return False
        name = vars(DeviceCapability).get(arg[-1])
        prog = impl.get_runtime().prog
        if type(name) is not str or name != arg[-1] or name not in _CAPABILITY_NAMES or type(prog) is not core.Program:
            return False
        self.target_facts.add(("device_capability", name, prog.get_device_caps().get(name)))
        return True

    def visit_Call(self, node):
        if self.declared_static is not None:
            member = self.static_member(node)
            if member:
                self.references.add((member, "value"))
                return
        if self._target_extension(node) is not None or self._device_capability(node):
            return
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
                local = self.local_bindings.copy() if self.local_bindings and _wrapped(value) is not None else None
                self.calls.setdefault(path, []).append((node, local))
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                if isinstance(arg, ast.Name) and arg.id in self.bindings:
                    if _wrapped(_resolve(path, self.scope)[0]) is not None:
                        self.forwarded.add(id(arg))
            super().visit_Call(node)

    def visit_If(self, node):
        if (
            isinstance(node.test, ast.Call)
            and len(node.test.args) == 1
            and not node.test.keywords
            and _resolve(_path(node.test.func), self.scope)[0] is impl.static
        ):
            predicate = node.test.args[0]
            supported = self._target_extension(predicate)
            if supported is not None:
                for statement in node.body if supported else node.orelse:
                    self.visit(statement)
                return
            self.visit(node.test)
            if (
                isinstance(predicate, ast.Compare) and len(predicate.ops) == 1
                and isinstance(predicate.ops[0], (ast.Is, ast.IsNot))
                and isinstance(predicate.left, ast.Name) and predicate.left.id in self.absent_templates
                and isinstance(predicate.comparators[0], ast.Constant) and predicate.comparators[0].value is None
            ):
                # A literal None template has no resource ABI. Optional package
                # inputs carry presence references, even in an absent witness.
                value = isinstance(predicate.ops[0], ast.Is)
            else:
                value = self.static_literal(predicate)
            if type(value) in (bool, int, float):
                for statement in node.body if value else node.orelse:
                    self.visit(statement)
                return
        else:
            self.visit(node.test)
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
        if isinstance(self.local_bindings.get(node.id), _LocalRecord):
            if isinstance(node.ctx, ast.Store):
                self.safe = False
            return
        if isinstance(node.ctx, ast.Store) and (node.id in self.absent_templates
                or (self.package_fields and node.id in self.static_scope)
                or (self.declared_static is not None and node.id in self.bindings)):
            self.safe = False  # Rebinding invalidates the entry fact for later static tests.
        if (
            id(node) not in self.forwarded
            and isinstance(node.ctx, ast.Load)
            and (node.id not in self.locals or node.id in self.bindings)
        ):
            self.reference(node, "value")


class FunctionDependencyAnalysis:
    def __init__(self, *, allow_fields, package_fields=False, declared_static=None):
        self.allow_fields = allow_fields
        self.package_fields = package_fields
        self.declared_static = declared_static
        self._syntax = {}
        self._single_assignments = {}
        self._record_factories = {}

    def _returned_record(self, source):
        if source not in self._record_factories:
            tree, _ = source._definition.acquire(cache=True)
            body = tree.body[0].body
            record = None
            if isinstance(body[-1], ast.Return):
                result = body[-1].value
                stores = Counter(n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                if isinstance(result, ast.Name) and stores[result.id] == 1:
                    result = next((s.value for s in body if isinstance(s, ast.Assign) and len(s.targets) == 1
                                   and isinstance(s.targets[0], ast.Name) and s.targets[0].id == result.id), None)
                if isinstance(result, ast.Call):
                    path = _path(result.func)
                    if path and path[0] not in stores and path[0] not in source._definition.signature.parameters:
                        record = result
            self._record_factories[source] = record
        record = self._record_factories[source]
        return _constructed_record(record, source._definition.scope()) if record is not None else None

    def _dependencies(self, source, bindings, active, memo, fixed_flags=frozenset()):
        if source in active or source.pyfunc or source.is_real_function:
            return None
        # Sibling calls frequently forward the exact same receiver. Its Python
        # bindings are stable for this proof only. Discard the memo before the
        # next capture so that rebindings remain visible.
        if memo is None:
            return self._inspect_dependencies(source, bindings, active, memo, fixed_flags)
        key = (source, frozenset(active), tuple((name, id(value)) for name, value in bindings.items()), fixed_flags)
        if key not in memo:
            memo[key] = (tuple(bindings.values()), self._inspect_dependencies(source, bindings, active, memo, fixed_flags))
        return memo[key][1]

    def _inspect_dependencies(self, source, bindings, active, memo, fixed_flags):
        if source not in self._syntax:
            self._syntax[source], _ = source._definition.acquire(cache=True)
            if self.package_fields:
                stores = Counter(n.id for n in ast.walk(self._syntax[source])
                                 if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
                self._single_assignments[source] = {name for name, count in stores.items() if count == 1}
        scope = source._definition.scope()
        scope.update(bindings)
        # JIT specializes the bound object. In a portable package the same
        # owner property can become a live scalar and guard an array capture.
        # Only bound flags and producer-static declarations can prune a package
        # callee. Prepare facts and live scalars can share the same witness type.
        static_scope = scope if self.allow_fields else {name: bindings[name] for name in fixed_flags}
        syntax = FunctionSyntax(self._syntax[source], scope, bindings, static_scope,
                                package_fields=self.package_fields,
                                declared_static=self.declared_static,
                                single_assignments=self._single_assignments.get(source, ()),
                                returned_record=self._returned_record)
        if not syntax.safe or any(name in scope for name in syntax.assigned):
            return None
        dependencies = sorted(syntax.target_facts)
        for path, kind in sorted(syntax.references):
            value, receiver = _resolve(path, scope)
            if kind == "call":
                wrapped = _wrapped(value)
                if self.package_fields and (value is isinf or value is isnan):
                    # These DSL builtins classify bits at the configured float
                    # precision. Keep their normal lowering inside the unit.
                    key = ("fp_classification", wrapped, impl.get_runtime().default_fp)
                elif wrapped is not None:
                    children = []
                    for call, locals_at_call in syntax.calls.get(path, ()):
                        call_scope = scope | locals_at_call if locals_at_call else scope
                        operands = list(call.args)
                        if wrapped.classfunc and receiver is not None:
                            operands.insert(0, receiver)
                        keywords = {kw.arg: kw.value for kw in call.keywords}
                        try:
                            bound = wrapped._definition.signature.bind(*operands, **keywords)
                        except TypeError:
                            return None
                        bound.apply_defaults()
                        # Templates and forwarded resource owners are Python
                        # compilation data. Numerical operands stay symbolic
                        # even if this call happens to pass a literal.
                        child_bindings = {}
                        child_flags = set()
                        for arg in wrapped.arguments:
                            operand = bound.arguments[arg.name]
                            expr = isinstance(operand, ast.AST)
                            actual = _literal(operand, call_scope, objects=True) if expr else operand
                            if isinstance(arg.annotation, template):
                                if actual is _MISSING and expr and self.declared_static is not None:
                                    actual = syntax.static_literal(operand)
                                if actual is _MISSING:
                                    return None
                                child_bindings[arg.name] = actual
                                if (self.package_fields and type(actual) is bool
                                        and (not expr or syntax.static_literal(operand) is actual)):
                                    child_flags.add(arg.name)
                            elif actual is not _MISSING and actual is not _STATIC_INTEGER and scalar_key(actual) is None:
                                child_bindings[arg.name] = actual
                        child = self._dependencies(wrapped, child_bindings, active | {source}, memo, frozenset(child_flags))
                        if child is None:
                            return None
                        children.append(child)
                    key = ("function", wrapped, tuple(children))
                elif self.package_fields and value is Struct and all(
                    _constructed_record(call, scope) is not None for call, _ in syntax.calls[path]
                ):
                    key = ("local_record",)
                elif isinstance(value, MatrixType):
                    key = ("tensor_type", value.n, value.m, value.ndim, str(value.dtype))
                elif self.package_fields and any(
                    value is builtin for builtin in (ndrange, Matrix.rows, Matrix.cols, ops.pow)
                ):
                    key = ("call", id(value))
                elif self.package_fields and value is builtins.print:
                    # Device print lowers inside the function. Its operands were
                    # visited normally, and no caller-local address is required.
                    key = ("device_print",)
                elif self.package_fields and any(value is op for op in _RESOURCE_ATOMICS):
                    # A retained function may mutate a captured device field or
                    # an explicit array descriptor. A caller's local lvalue has
                    # no such ABI and must keep ordinary inline semantics.
                    arrays = {a.name for a in source.arguments if isinstance(a.annotation, NdarrayType)}
                    for call, _ in syntax.calls[path]:
                        if len(call.args) != 2 or call.keywords or not isinstance(call.args[0], ast.Subscript):
                            return None
                        dest = _path(call.args[0].value)
                        if not dest:
                            return None
                        if len(dest) == 1 and dest[0] in arrays:
                            continue
                        if dest[0] in syntax.locals and dest[0] not in bindings:
                            return None
                        if not isinstance(_resolve(dest, scope)[0], Field):
                            return None
                    key = ("resource_atomic", value)
                elif (
                    id(value) in _PURE_CALLS
                    or id(value) in primitive_types.type_ids
                    or any(value is constructor for constructor in _CONSTRUCTORS)
                ):
                    key = ("call", id(value))
                else:
                    return None
            elif self.package_fields and type(value) is core.Arch:
                key = ("arch", int(value))
            elif self.package_fields and value is _STATIC_INTEGER:
                key = ("literal_static_iteration",)
            elif self.allow_fields and isinstance(value, Field):
                # Identity only: device contents are live, not compile-time data.
                key = ("field", value)
            elif self.package_fields and isinstance(value, Field):
                # Component capture records the declaration ABI and its path.
                # A producer witness's address is not part of a portable unit.
                key = ("field", type(value))
            elif isinstance(value, MatrixType):
                key = ("tensor_type", value.n, value.m, value.ndim, str(value.dtype))
            elif id(value) in primitive_types.type_ids:
                key = ("dtype", value)
            else:
                if not self.allow_fields and value is None and receiver is not None:
                    # An absent optional field in the producer witness does
                    # not establish a scalar parameter in the portable ABI.
                    return None
                key = scalar_key(value)
                if key is None:
                    return None
            dependencies.append((path, kind, key))
        return tuple(dependencies)

    def dependencies(self, source, args):
        bindings = {
            arg.name: value for arg, value in zip(source.arguments, args) if isinstance(arg.annotation, template)
        }
        flags = frozenset(name for name, value in bindings.items() if self.package_fields and type(value) is bool)
        try:
            return self._dependencies(source, bindings, set(), {} if self.package_fields else None, flags)
        except RecursionError:
            return None


class JitFunctionLibrary(FunctionLibrary):
    def __init__(self):
        super().__init__()
        self.function_library = True
        self.allow_spirv = True
        self.analysis = FunctionDependencyAnalysis(allow_fields=True)

    def library_function(self, source, args):
        if not self.thread_local_call or source.is_real_function or source.pyfunc:
            return None
        dependencies = self.analysis.dependencies(source, args)
        if dependencies is None:
            return None
        function = super().library_function(source, args, dependencies=dependencies)
        if function is not None:
            function._backend_inline_function = True
        return function
