"""Host graph recording for package entries, lowered to the existing native Plan.

This records names and typed arguments only. It never traverses a numerical
definition, creates another code module, or executes a Python kernel loop.
"""

from dataclasses import dataclass
from enum import Enum
import itertools

from ._plan import ExecutionPlan
from ._runtime import Scalar
from ._package_values import capture_value, check_admission


class ArgKind(Enum):
    SCALAR = "value"
    NDARRAY = "ndarray"
    MATRIX = "compound"
    STRUCT = "compound_struct"


@dataclass(frozen=True)
class Arg:
    tag: ArgKind
    name: str
    dtype: object
    ndim: int = 0

    def schema(self):
        from ._package import _ElementType

        if not isinstance(self.name, str) or not self.name or not isinstance(self.tag, ArgKind):
            raise ValueError("Graph arguments require a name and supported ArgKind")
        if self.tag == ArgKind.SCALAR:
            return dict(kind="value", dtype=self.dtype)
        element = (
            self.dtype.spec
            if isinstance(self.dtype, _ElementType)
            else dict(kind="scalar", dtype=self.dtype, element_shape=[])
        )
        if self.tag == ArgKind.NDARRAY:
            return dict(kind="ndarray", element=element, ndim=self.ndim)
        return dict(kind="compound", element=element)


@dataclass(frozen=True)
class _PrivateArg(Arg):
    scope: object = None


class _Bindings:
    def __init__(self, package):
        self.prefix = f"_forge_package_graph_{next(package._graph_ids)}_"
        self.ids = itertools.count()
        self.captures = {}
        self.constants = {}

    def name(self):
        return self.prefix + str(next(self.ids))

    def capture(self, owner, item):
        key = (id(owner), item["name"])
        if key not in self.captures:
            self.captures[key] = (self.name(), owner, dict(item))
        name, previous_owner, previous = self.captures[key]
        if previous_owner is not owner or previous != item:
            raise ValueError("Owner capture has incompatible Graph uses")
        return name, (owner, previous)


class GraphBuilder:
    def __init__(self, package, *, bindings=None, private_names=None):
        self.package = package
        self.operations, self.arguments = [], {}
        self._bindings = bindings or _Bindings(package)
        self._private_names = private_names if private_names is not None else set()
        self._captures, self._constants = {}, {}
        self._admissions = []

    def dispatch(self, kernel, *args, template_args=None):
        from ._package import entry, _BoundEntry

        self.package._check()
        owner = None
        if isinstance(kernel, _BoundEntry):
            owner, kernel = kernel.owner, kernel.entry
        if not isinstance(kernel, entry) or kernel.package is not self.package:
            raise ValueError("Package Graph dispatch requires an entry in the same loaded package")
        kernel = kernel._select(owner)
        kernel._check_layout(owner)
        check_admission(owner, kernel.contract)
        if template_args:
            if owner is None or set(template_args) != {"self"} or template_args["self"] is not owner:
                raise ValueError("Package Graph template receiver must match its bound entry")
        if kernel.contract["captures"] and owner is None:
            raise ValueError("Captured Graph entries require their bound instance")
        if owner is not None and getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Graph receiver belongs to another loaded package")
        expected = kernel.contract["arguments"]
        if len(args) != len(expected):
            raise ValueError("Graph dispatch argument count differs from the entry")
        arguments = dict(self.arguments)
        constants = dict(self._constants)
        for arg, item in zip(args, expected):
            if not isinstance(arg, Arg):
                raise TypeError("Graph dispatch arguments must be symbolic Arg values")
            schema = arg.schema()
            if schema != {key: value for key, value in item.items() if key != "name"}:
                raise ValueError(f"Graph argument {arg.name!r} differs from the entry's type")
            if isinstance(arg, _PrivateArg):
                if arg.scope is not self._bindings or arg.name not in self._bindings.constants:
                    raise ValueError("Private Graph scalar belongs to another recording")
                constants[arg.name] = self._bindings.constants[arg.name]
                continue
            if arg.name.startswith("_forge_package_"):
                raise ValueError("Graph argument uses a reserved package binding name")
            if arg.name in arguments and arguments[arg.name] != schema:
                raise ValueError(f"Graph argument {arg.name!r} has conflicting declarations")
            arguments[arg.name] = schema
        names = [a.name for a in args]
        captures = dict(self._captures)
        for item in kernel.contract["captures"]:
            name, binding = self._bindings.capture(owner, item)
            names.append(name)
            captures[name] = binding
        self.operations.append(dict(kind="kernel", entry=kernel.symbol, arguments=names))
        self.arguments = arguments
        self._captures, self._constants = captures, constants
        if kernel.contract.get("admission"):
            self._admissions.append((owner, kernel.contract))
        return self

    def private_scalar(self, name, dtype, value):
        self.package._check()
        if not isinstance(name, str) or not name or name in self._private_names:
            raise ValueError("Graph private scalar name must be nonempty and unique in its scope")
        if type(value) not in (bool, int, float):
            raise TypeError("Graph private scalar requires a Python numeric value")
        symbol = self._bindings.name()
        resource = Scalar(dtype, value).resource(symbol)
        self._private_names.add(name)
        self._bindings.constants[symbol] = resource
        return _PrivateArg(ArgKind.SCALAR, symbol, dtype, scope=self._bindings)

    def create_sequential(self, *, private_scope=False):
        return GraphBuilder(
            self.package,
            bindings=self._bindings,
            private_names=None if private_scope else self._private_names,
        )

    def append(self, sequence):
        if not isinstance(sequence, GraphBuilder) or sequence.package is not self.package:
            raise ValueError("Graph sequence belongs to another loaded package")
        combined = dict(self.arguments)
        for name, schema in sequence.arguments.items():
            if name in combined and combined[name] != schema:
                raise ValueError(f"Graph argument {name!r} has conflicting declarations")
            combined[name] = schema
        self.operations.extend(sequence.operations)
        self.arguments = combined
        self._captures.update(sequence._captures)
        self._constants.update(sequence._constants)
        self._admissions.extend(sequence._admissions)
        return self

    def compile(self):
        self.package._check()
        return Graph(
            self.package,
            ExecutionPlan(self.operations, resources=self._constants.values()),
            dict(self.arguments),
            dict(self._captures),
            list(self._admissions),
        )


class Graph:
    def __init__(self, package, plan, arguments, captures=None, admissions=None):
        self.package, self.plan, self.arguments = package, plan, arguments
        self.captures = captures or {}
        self.admissions = admissions or []

    def bind(self, arguments):
        self.package._check()
        if arguments.keys() != self.arguments.keys():
            raise ValueError("Graph binding must supply exactly its declared arguments")
        for owner, contract in self.admissions:
            check_admission(owner, contract)
        values = {}
        for name, schema in self.arguments.items():
            value = arguments[name]
            if schema["kind"] == "value":
                value = Scalar(schema["dtype"], value)
            elif schema["kind"] == "compound":
                value = self.plan.value(self.package._module, name, value)
            values[name] = value
        for name, (owner, schema) in self.captures.items():
            value = capture_value(owner, schema)
            if schema["kind"] in ("value", "dimension"):
                value = (
                    self.plan.value(self.package._module, name, value)
                    if schema.get("element_shape")
                    else Scalar(schema["dtype"], value)
                )
            values[name] = value
        # Native preparation owns immutable instance bindings and their leases.
        # Replaying the returned call does not interpret operations in Python.
        return self.plan.prepare(self.package._module, values)

    def run(self, arguments):
        # Convenience for one submission. Repeated replay uses bind(...).run().
        # Runtime retains in-flight preparations until the usual synchronization.
        self.bind(arguments).run()
