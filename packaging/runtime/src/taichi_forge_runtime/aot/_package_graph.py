"""Host graph recording for package entries, lowered to the existing native Plan.

This records names and typed arguments only. It never traverses a numerical
definition, creates another code module, or executes a Python kernel loop.
"""

from dataclasses import dataclass
from enum import Enum

from ._plan import ExecutionPlan
from ._runtime import Scalar


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


class GraphBuilder:
    def __init__(self, package):
        self.package = package
        self.operations, self.arguments = [], {}

    def dispatch(self, kernel, *args):
        from ._package import entry

        self.package._check()
        if not isinstance(kernel, entry) or kernel.package is not self.package or kernel.contract["captures"]:
            raise ValueError("Package Graph dispatch requires a stateless entry in the same loaded package")
        expected = kernel.contract["arguments"]
        if len(args) != len(expected):
            raise ValueError("Graph dispatch argument count differs from the entry")
        arguments = dict(self.arguments)
        for arg, item in zip(args, expected):
            if not isinstance(arg, Arg):
                raise TypeError("Graph dispatch arguments must be symbolic Arg values")
            schema = arg.schema()
            if schema != {key: value for key, value in item.items() if key != "name"}:
                raise ValueError(f"Graph argument {arg.name!r} differs from the entry's type")
            if arg.name in arguments and arguments[arg.name] != schema:
                raise ValueError(f"Graph argument {arg.name!r} has conflicting declarations")
            arguments[arg.name] = schema
        self.operations.append(dict(kind="kernel", entry=kernel.symbol, arguments=[a.name for a in args]))
        self.arguments = arguments
        return self

    def create_sequential(self):
        return GraphBuilder(self.package)

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
        return self

    def compile(self):
        self.package._check()
        return Graph(self.package, ExecutionPlan(self.operations), dict(self.arguments))


class Graph:
    def __init__(self, package, plan, arguments):
        self.package, self.plan, self.arguments = package, plan, arguments

    def bind(self, arguments):
        self.package._check()
        if arguments.keys() != self.arguments.keys():
            raise ValueError("Graph binding must supply exactly its declared arguments")
        values = {}
        for name, schema in self.arguments.items():
            value = arguments[name]
            if schema["kind"] == "value":
                value = Scalar(schema["dtype"], value)
            elif schema["kind"] == "compound":
                value = self.plan.value(self.package._module, name, value)
            values[name] = value
        # Native preparation owns immutable instance bindings and their leases.
        # Replaying the returned call does not interpret operations in Python.
        return self.plan.prepare(self.package._module, values)

    def run(self, arguments):
        # Convenience for one submission. Repeated replay uses bind(...).run().
        # Runtime retains in-flight preparations until the usual synchronization.
        self.bind(arguments).run()
