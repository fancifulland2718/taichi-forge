"""Internal component capture compiler (public SDK API is not frozen yet).

The declaration environment describes a component, not a kernel signature.
The normal frontend appends only captures that an entry actually references.
Dense SNode groups require the shared layout/instantiation contract and are
deliberately not converted to unrelated ndarrays here.
"""

import hashlib
import inspect
import json
import marshal
from dataclasses import dataclass
from types import MappingProxyType, ModuleType

from taichi_forge.lang import impl, kernel_arguments
from taichi_forge.lang.exception import TaichiCompilationError
from taichi_forge.lang.matrix import MatrixType
from taichi_forge.lang.util import cook_dtype
from taichi_forge.types import primitive_types
from taichi_forge.types.ndarray_type import NdarrayType


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _static_value(value):
    # No repr(), object IDs, getters or user-defined serialization in a key.
    if value is None or type(value) in (bool, int, str):
        return (type(value).__name__, value)
    if type(value) is float:
        return ("float", value.hex())
    if type(value) is tuple:
        return ("tuple", tuple(_static_value(x) for x in value))
    raise TaichiCompilationError("Component static captures must be immutable scalar values or tuples")


def _function(value):
    value = getattr(value, "__func__", value)
    if getattr(value, "_is_taichi_function", False):
        return value.func.func
    return value


def _definition(value):
    value = _function(value)
    return (value.__module__, value.__qualname__, hashlib.sha256(marshal.dumps(value.__code__)).hexdigest())


def _member(owner, name):
    # Deployment compilation must not execute arbitrary properties/descriptors.
    descriptor = inspect.getattr_static(owner, name)
    if hasattr(descriptor, "__get__"):
        raise TaichiCompilationError(f"Component capture {name!r} uses a descriptor; declare plain state instead")
    return descriptor


@dataclass(frozen=True)
class RuntimeValue:
    dtype: object

    @property
    def schema(self):
        if isinstance(self.dtype, MatrixType):
            return ("value", str(self.dtype.dtype), tuple(self.dtype.get_shape()))
        return ("value", str(cook_dtype(self.dtype)), ())

    def declare(self, name):
        if isinstance(self.dtype, MatrixType):
            return kernel_arguments.decl_matrix_arg(self.dtype, name, 0)
        return kernel_arguments.decl_scalar_arg(self.dtype, name, 0)

    @property
    def annotation(self):
        return self.dtype


@dataclass(frozen=True)
class Array:
    dtype: object
    ndim: int

    @property
    def schema(self):
        if type(self.ndim) is not int or self.ndim < 0:
            raise ValueError("Component array ndim must be a nonnegative integer")
        return ("ndarray", str(cook_dtype(self.dtype)), self.ndim)

    def declare(self, name):
        from taichi_forge._lib import core

        return kernel_arguments.decl_ndarray_arg(
            cook_dtype(self.dtype), self.ndim, name, False, core.BoundaryMode.UNSAFE
        )

    def validate(self, value):
        from taichi_forge.lang._ndarray import ScalarNdarray

        if not isinstance(value, ScalarNdarray):
            raise TaichiCompilationError(
                "Component Array currently requires a scalar Ndarray; Dense Field groups require the SNode layout contract"
            )
        if value.dtype != cook_dtype(self.dtype) or len(value.shape) != self.ndim:
            raise TaichiCompilationError("Component array dtype/rank does not match its declaration")

    @property
    def annotation(self):
        return NdarrayType(dtype=self.dtype, ndim=self.ndim)


@dataclass(frozen=True)
class StaticValue:
    @property
    def schema(self):
        return ("static",)


class Component:
    """Immutable, component-wide declarations; currently an internal API.

    Attribute names are explicit declaration IDs in this initial flat namespace.
    ``compile(owner.entry, ...)`` uses the original decorated numerical entry.
    No owner is retained by the resulting code, and no per-entry binding table
    is supplied by the application. Dense groups/Graph export follow separately.
    """

    __slots__ = ("_declarations",)

    def __init__(self, **declarations):
        for name, declaration in declarations.items():
            if not name.isidentifier() or not isinstance(declaration, (RuntimeValue, Array, StaticValue)):
                raise TypeError("Component declarations require attribute IDs and supported declaration types")
            declaration.schema  # Validate without reading an instance.
        self._declarations = MappingProxyType(dict(declarations))

    @property
    def declarations(self):
        return self._declarations

    def compile(self, entry, *args):
        from taichi_forge.lang.kernel_impl import _process_args

        kernel = getattr(entry, "_primal", None)
        owner = getattr(entry, "_kernel_owner", None)
        if kernel is None or owner is None:
            raise TypeError("Component.compile requires a bound, decorated kernel method")
        if kernel.template_slot_locations != [0] or kernel.autodiff_mode.name != "NONE":
            raise TaichiCompilationError("Component capture currently requires only the self template and no autodiff")
        if any(
            id(arg.annotation) not in primitive_types.type_ids and not isinstance(arg.annotation, MatrixType)
            for arg in kernel.arguments[1:]
        ):
            raise TaichiCompilationError(
                "Component capture currently supports only explicit scalar/vector/matrix values"
            )
        call_args = _process_args(kernel, (owner, *args), {})
        runtime = impl.get_runtime()
        with runtime._kernel_compilation_lock:
            for code in kernel._component_candidates:
                if code.matches(self, owner):
                    code.validate_bindings(self, owner)
                    return code
            context = _Capture(self, owner, kernel)
            # No self identity or runtime value enters the template mapper.
            # Explicit arguments in this slice are value types only.
            features = tuple(None for _ in call_args)
            key = (kernel.func, "component_" + str(len(kernel._component_candidates)), kernel.autodiff_mode)
            kernel.materialize(key=key, args=call_args, arg_features=features, for_aot=True, deployment=context)
            code = _ComponentCode(kernel, key, context)
            kernel._component_candidates.append(code)
            return code


class _Capture:
    def __init__(self, component, owner, kernel):
        self.component = component
        self.declarations = dict(component.declarations)
        self.owner = owner
        self.kernel = kernel
        self.used = {}
        self.expressions = {}
        self.dynamic = []
        self.static = {}
        self.static_objects = {}
        self.methods = {}
        self.functions = {}
        self.entry_definition = _definition(kernel.func)
        self.contract = None

    def attribute(self, ctx, name):
        declaration = self.declarations.get(name)
        if declaration is None:
            value = inspect.getattr_static(self.owner, name)
            if getattr(value, "_is_taichi_function", False):
                self.methods[name] = _definition(value)
                return value.__get__(self.owner, type(self.owner))
            raise TaichiCompilationError(
                f"Component entry {self.kernel.func.__qualname__}: undeclared capture self.{name}; "
                "declare it as a runtime value, resource or static configuration"
            )
        self.used[name] = declaration.schema
        if isinstance(declaration, StaticValue):
            value = _member(self.owner, name)
            fact = _static_value(value)
            if name in self.static and self.static[name] != fact:
                raise TaichiCompilationError(f"Component static dependency {name!r} changed during compilation; retry")
            self.static[name] = fact
            self.static_objects[name] = value
            return self.static_objects[name]
        if ctx.is_in_static_scope():
            raise TaichiCompilationError(
                f"Component entry {self.kernel.func.__qualname__}: runtime capture self.{name} "
                "cannot be used in ti.static; its declaration is dynamic"
            )
        if name not in self.expressions:
            value = _member(self.owner, name)
            if isinstance(declaration, Array):
                declaration.validate(value)
            self.dynamic.append(name)
            self.expressions[name] = declaration.declare("__capture_" + name)
        return self.expressions[name]

    def check_name(self, ctx, node):
        if node.id not in ctx.global_vars:
            return  # Builtins are checked at their call site.
        value = ctx.global_vars[node.id]
        if isinstance(value, ModuleType) and value.__name__.startswith("taichi_forge"):
            return
        if getattr(value, "_is_taichi_function", False):
            self.functions[(ctx.func.func, node.id)] = _definition(value)
            return
        if id(value) in primitive_types.type_ids or isinstance(value, MatrixType):
            return
        raise TaichiCompilationError(
            f"Component entry {self.kernel.func.__qualname__}: undeclared global/closure capture {node.id!r}; "
            "use a declared component value or a literal"
        )

    def check_call(self, value):
        if getattr(value, "_is_taichi_function", False):
            return
        if any(value is item for item in (range, len, min, max, abs, int, float, bool, enumerate, zip)):
            return
        if getattr(value, "__module__", "").startswith("taichi_forge"):
            return
        raise TaichiCompilationError("Uncontrolled Python calls are not supported in component captures")

    def finish(self):
        if not self.matches(self.component, self.owner):
            raise TaichiCompilationError(
                "Component declarations or static dependencies changed during compilation; retry"
            )
        self.contract = {
            "version": 1,
            "entry": self.entry_definition,
            "explicit_abi": [str(arg.annotation) for arg in self.kernel.arguments[1:]],
            "captures": [(name, self.used[name]) for name in self.dynamic],
            "static": sorted(self.static.items()),
            "methods": sorted(self.methods.items()),
            "functions": sorted((name, definition) for (_, name), definition in self.functions.items()),
        }

    def matches(self, component, owner):
        if _definition(self.kernel.func) != self.entry_definition:
            return False
        for name, schema in self.used.items():
            declaration = component.declarations.get(name)
            if declaration is None or declaration.schema != schema:
                return False
        for name, value in self.static.items():
            if _static_value(_member(owner, name)) != value:
                return False
        for name, definition in self.methods.items():
            value = inspect.getattr_static(owner, name, None)
            if not getattr(value, "_is_taichi_function", False) or _definition(value) != definition:
                return False
        for (function, name), definition in self.functions.items():
            variables = inspect.getclosurevars(function)
            value = {**variables.globals, **variables.nonlocals}.get(name)
            if not getattr(value, "_is_taichi_function", False) or _definition(value) != definition:
                return False
        return True


class _ComponentCode:
    def __init__(self, kernel, key, context):
        self.kernel = kernel
        self.key = key
        self.runtime = kernel.runtime
        self.semantic_key = _digest(context.contract)
        # Copy plain dependency facts, never the lowering context/owner/Exprs.
        self.entry_definition = context.entry_definition
        self.used = dict(context.used)
        self.static = dict(context.static)
        self.methods = dict(context.methods)
        self.functions = dict(context.functions)
        self.dynamic = tuple(context.dynamic)
        self._contract = json.dumps(context.contract)
        self.arguments = tuple(kernel.arguments[1:]) + tuple(
            kernel_arguments.KernelArgument(context.component.declarations[name].annotation, "__capture_" + name)
            for name in self.dynamic
        )

    matches = _Capture.matches

    @property
    def contract(self):
        return json.loads(self._contract)

    def validate_bindings(self, component, owner):
        for name in self.dynamic:
            value = _member(owner, name)
            declaration = component.declarations[name]
            if isinstance(declaration, Array):
                declaration.validate(value)

    def run(self, component, owner, *args):
        if impl.get_runtime() is not self.runtime or self.key not in self.kernel.compiled_kernels:
            raise RuntimeError("Component code belongs to a retired Runtime")
        if not self.matches(component, owner):
            raise ValueError("Component bindings do not match the compiled declaration/static dependencies")
        self.validate_bindings(component, owner)
        if len(args) != len(self.kernel.arguments) - 1:
            raise TypeError("Wrong number of explicit component arguments")
        values = tuple(_member(owner, name) for name in self.dynamic)
        return self.kernel.launch_kernel(
            self.kernel.compiled_kernels[self.key], *args, *values, _arguments=self.arguments
        )
