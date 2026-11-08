"""Host graph recording for package entries, lowered to the existing native Plan.

This records names and typed arguments only. It never traverses a numerical
definition, creates another code module, or executes a Python kernel loop.
"""

from dataclasses import dataclass
from enum import Enum
import copy
import itertools
import json
from types import SimpleNamespace

from ._plan import ExecutionPlan
from ._runtime import Scalar
from ._package_values import capture_value, check_admission, owner_type
from .._dimensions import extent_dependencies, local_workspace_shape, check_model_static_ranges
from .._static_values import static_key


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
        self.native_resources = {}

    def name(self):
        return self.prefix + str(next(self.ids))

    def capture(self, owner, item, *, entry=None):
        # Compiler-generated local names are scoped to their numerical entry.
        # Ordinary owner fields still share one binding across all entry uses.
        key = (id(owner), item["name"], entry)
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
        self._range_checks = []
        self._plan_values = {}

    def _record_plan_values(self, owner):
        if owner is None:
            return
        schema = self.package.manifest["owners"].get(owner_type(owner), {})
        for name, item in schema.items():
            if not item.get("affects_plan"):
                continue
            spec = dict(name=name, kind="value", dtype=item["dtype"], path=item.get("path", [name]))
            value = capture_value(owner, spec)
            key = id(owner), name
            previous = self._plan_values.get(key)
            if previous is not None and static_key(previous[2]) != static_key(value):
                raise ValueError(f"Plan parameter {name!r} changed while recording; rebuild the Graph")
            self._plan_values[key] = owner, spec, value

    def _check_plan_values(self):
        for owner, spec, value in self._plan_values.values():
            if static_key(capture_value(owner, spec)) != static_key(value):
                raise ValueError(f"Plan parameter {spec['name']!r} changed; rebuild the Graph before preparing")

    def dispatch(self, kernel, *args, template_args=None):
        from ._package import entry, _BoundEntry
        from ._package_closures import _TemplateEntry, _FunctionEntry, _BoundMethodEntry

        self.package._check()
        if isinstance(kernel, (_FunctionEntry, _BoundMethodEntry)):
            kernel, args = kernel._graph_bind(args, template_args)
            template_args = None
        if isinstance(kernel, _TemplateEntry):
            if not template_args or set(template_args) != {"self"}:
                raise ValueError("Package factory Graph requires its explicit template receiver")
            kernel = kernel.__get__(template_args["self"])
        owner = None
        if isinstance(kernel, _BoundEntry):
            owner, kernel = kernel.owner, kernel.entry
        if not isinstance(kernel, entry) or kernel.package is not self.package:
            raise ValueError("Package Graph dispatch requires an entry in the same loaded package")
        kernel = kernel._select(owner)
        check_admission(owner, kernel.contract)
        kernel._check_layout(owner)
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
        captured = {
            item["name"]: self._bindings.capture(owner, item)
            for item in kernel.contract["captures"]
            if item["kind"] != "local_tensor_workspace"
        }
        for item in kernel.contract["captures"]:
            if item["kind"] == "local_tensor_workspace":
                inputs = {dim for expression in item["shape"] for dim in extent_dependencies(expression)}
                spec = dict(item, inputs={dim: captured[dim][0] for dim in inputs})
                name, binding = self._bindings.capture(owner, spec, entry=kernel.symbol)
            else:
                name, binding = captured[item["name"]]
            names.append(name)
            captures[name] = binding
        self._record_plan_values(owner)
        self.operations.append(dict(kind="kernel", entry=kernel.symbol, arguments=names))
        self.arguments = arguments
        self._captures, self._constants = captures, constants
        if kernel.contract.get("admission"):
            self._admissions.append((owner, kernel.contract))
        if kernel.contract.get("model_static_ranges"):
            self._range_checks.append((kernel.contract["model_static_ranges"],
                                       {key: binding[0] for key, binding in captured.items()}))
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

    def private_ndarray(self, name, dtype, shape, *, exclusive_submission=True):
        """Describe prepare-owned storage; allocation follows native Plan leases."""
        from ._runtime import _shape

        self.package._check()
        if not isinstance(name, str) or not name or name in self._private_names:
            raise ValueError("Graph private array name must be nonempty and unique in its scope")
        if exclusive_submission is not True:
            raise ValueError("Package private arrays require exclusive prepared submission")
        shape = _shape(shape)
        argument = _PrivateArg(ArgKind.NDARRAY, self._bindings.name(), dtype, len(shape), scope=self._bindings)
        element = argument.schema()["element"]
        if element["kind"] != "scalar":
            raise ValueError("Package private arrays currently require scalar elements")
        resource = dict(kind="ndarray", name=argument.name, dtype=element["dtype"], shape=list(shape))
        self._private_names.add(name)
        self._bindings.constants[argument.name] = resource
        return argument

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
        for key, binding in sequence._plan_values.items():
            if key in self._plan_values and static_key(self._plan_values[key][2]) != static_key(binding[2]):
                raise ValueError("Graph sequences recorded different plan parameters")
        self.operations.extend(sequence.operations)
        self.arguments = combined
        self._captures.update(sequence._captures)
        self._constants.update(sequence._constants)
        self._admissions.extend(sequence._admissions)
        self._range_checks.extend(sequence._range_checks)
        self._plan_values.update(sequence._plan_values)
        return self

    def append_native(self, sequence, *, prewarm=False):
        from ._package_algorithms import PrimitiveSequence, sort_inputs

        self.package._check()
        if not isinstance(sequence, PrimitiveSequence) or sequence.package is not self.package:
            raise ValueError("Package native recording requires its own primitive sequence")
        if type(prewarm) is not bool:
            raise TypeError("Native prewarm must be bool")
        # Recording only describes the operation. Existing Plan preparation
        # admits the device provider, allocates scratch and freezes each binding.
        operations, captures = [], {}
        for arrays in sequence.calls:
            sort_inputs(self.package, *arrays)
            operation = dict(kind="sort")
            for port, array in zip(("keys", "values"), arrays):
                if array not in self._bindings.native_resources:
                    self._bindings.native_resources[array] = SimpleNamespace(value=array)
                holder = self._bindings.native_resources[array]
                name, binding = self._bindings.capture(holder, dict(name="value", kind="ndarray"))
                operation[port], captures[name] = name, binding
            operations.append(operation)
        self.operations.extend(operations)
        self._captures.update(captures)
        return self

    def while_loop(
        self,
        condition,
        body,
        *,
        predicate,
        max_iterations,
        counter=None,
        masked_execution=False,
        lowering_mode="auto",
        name="while",
    ):
        """Record an exact CPU/CUDA loop using the existing native Plan executor."""
        self.package._check()
        if self.package.runtime.arch not in ("x64", "arm64", "cuda"):
            raise ValueError("Package while requires a qualified CPU/CUDA control route")
        if lowering_mode not in ("auto", "native_required"):
            raise ValueError("Package while currently requires native control")
        if type(masked_execution) is not bool or not isinstance(name, str) or not name:
            raise ValueError("Invalid package while options")
        if type(max_iterations) is not int or not 0 <= max_iterations <= 0x7FFFFFFFFFFFFFFF:
            raise ValueError("Graph while maximum iterations must be a nonnegative int64")
        for region in (condition, body):
            if not isinstance(region, GraphBuilder) or region._bindings is not self._bindings:
                raise ValueError("Package while regions must belong to the same recording")
            if region is self or any(op["kind"] != "kernel" for op in region.operations):
                raise ValueError("Package while currently requires flat kernel regions")
        if not condition.operations or not body.operations:
            raise ValueError("Package while requires nonempty condition and body regions")
        for role, argument in (("predicate", predicate), ("counter", counter)):
            if argument is None and role == "counter":
                continue
            if not isinstance(argument, Arg) or argument.schema() != dict(
                kind="ndarray", element=dict(kind="scalar", dtype="i32", element_shape=[]), ndim=1
            ):
                raise ValueError(f"Graph while {role} requires a rank-one i32 array")
            if isinstance(argument, _PrivateArg) and argument.scope is not self._bindings:
                raise ValueError(f"Graph while {role} belongs to another recording")
        # Merge through the ordinary sequence checks before publishing anything.
        merged = self.create_sequential()
        merged.append(condition)
        merged.append(body)
        used = set(merged.arguments) | set(merged._constants)
        condition_used = set(condition.arguments) | set(condition._constants)
        if predicate.name not in condition_used or (counter is not None and counter.name not in used):
            raise ValueError("Graph while controls must be used by the recorded regions")
        if counter is not None and counter.name == predicate.name:
            raise ValueError("Graph while predicate and counter must be distinct")
        operation = dict(
            kind="while",
            predicate=predicate.name,
            max_iterations=max_iterations,
            condition=copy.deepcopy(condition.operations),
            body=copy.deepcopy(body.operations),
        )
        merged.operations = [operation]
        # Native preparation admits CUDA conditional control on the actual device;
        # neither a masked fallback nor per-iteration Host observation is inserted.
        return self.append(merged)

    def compile(self):
        self.package._check()
        self._check_plan_values()
        return Graph(
            self.package,
            ExecutionPlan(self.operations, resources=self._constants.values()),
            dict(self.arguments),
            dict(self._captures),
            list(self._admissions),
            list(self._range_checks),
            dict(self._plan_values),
        )


class Graph:
    def __init__(self, package, plan, arguments, captures=None, admissions=None, range_checks=None, plan_values=None):
        self.package, self.plan, self.arguments = package, plan, arguments
        self.captures = captures or {}
        self.admissions = admissions or []
        self.range_checks = range_checks or []
        self.plan_values = plan_values or {}

    def written_resources(self):
        """Inspect compiled effects without allocating, preparing or submitting."""
        self.package._check()
        for owner, contract in self.admissions:
            check_admission(owner, contract)
        for owner, schema, previous in self.plan_values.values():
            if static_key(capture_value(owner, schema)) != static_key(previous):
                raise ValueError("Plan parameter changed before write inspection; rebuild the Graph")
        written = set()

        def visit(operations):
            for operation in operations:
                if operation["kind"] == "while":
                    visit(operation["condition"])
                    if operation["max_iterations"]:
                        visit(operation["body"])
                    continue
                if operation["kind"] != "kernel":
                    if operation["kind"] == "sort":
                        written.update(value for key, value in operation.items() if key != "kind")
                        continue
                    raise ValueError("Write inspection has no effect contract for this operation")
                contract = self.package.manifest["entries"][operation["entry"]]
                descriptors = [*contract["arguments"], *contract["captures"]]
                resources = {
                    str(index)
                    for index, item in enumerate(descriptors)
                    if item["kind"] in ("ndarray", "dense_field", "model_tensor_field", "local_tensor_workspace")
                }
                effects = contract.get("resource_effects")
                if effects is None:
                    accesses = {index: "opaque" for index in resources}
                else:
                    accesses = effects.get("arguments")
                    if (
                        effects.get("precision") not in ("compiler_proven", "conservative")
                        or not isinstance(accesses, dict)
                        or accesses.keys() != resources
                        or any(
                            access not in ("none", "read", "write", "read_write", "atomic", "opaque")
                            for access in accesses.values()
                        )
                    ):
                        raise ValueError("Write inspection requires a complete compiler effect contract")
                for index, access in accesses.items():
                    if effects is None or effects["precision"] != "compiler_proven" or access not in ("none", "read"):
                        written.add(operation["arguments"][int(index)])

        visit(json.loads(self.plan.to_json())["operations"])
        if written & self.arguments.keys():
            raise ValueError("Write inspection requires bindings for explicit writable resources")
        result = {}
        for name in sorted(written):
            if name not in self.captures:
                continue  # Private arrays belong to the recorded Plan, not the owner.
            owner, spec = self.captures[name]
            if spec["kind"] == "local_tensor_workspace":
                continue
            # Return the original model tensor, not its scalar-axis ABI view.
            value = owner
            for part in spec.get("path", (spec["name"],)):
                value = getattr(value, part)
            result[id(value)] = value
        return tuple(result.values())

    def bind(self, arguments):
        self.package._check()
        if arguments.keys() != self.arguments.keys():
            raise ValueError("Graph binding must supply exactly its declared arguments")
        for owner, contract in self.admissions:
            check_admission(owner, contract)
        for owner, schema, previous in self.plan_values.values():
            if static_key(capture_value(owner, schema)) != static_key(previous):
                raise ValueError(f"Plan parameter {schema['name']!r} changed; rebuild the Graph before preparing")
        values = {}
        for name, schema in self.arguments.items():
            value = arguments[name]
            if schema["kind"] == "value":
                value = Scalar(schema["dtype"], value)
            elif schema["kind"] == "compound":
                value = self.plan.value(self.package._module, name, value)
            values[name] = value
        captured = {
            name: (self.plan_values[(id(owner), schema["name"])][2]
                   if (id(owner), schema["name"]) in self.plan_values else capture_value(owner, schema))
            for name, (owner, schema) in self.captures.items()
            if schema["kind"] != "local_tensor_workspace"
        }
        for expressions, inputs in self.range_checks:
            check_model_static_ranges(expressions, lambda name: captured[inputs[name]])
        resources = []
        for name, (owner, schema) in self.captures.items():
            if schema["kind"] == "local_tensor_workspace":
                shape = local_workspace_shape(schema["shape"], lambda dim: captured[schema["inputs"][dim]])
                resources.append(dict(kind="ndarray", name=name, dtype=schema["dtype"], shape=list(shape)))
                continue
            value = captured[name]
            if schema["kind"] in ("value", "dimension"):
                value = (
                    self.plan.value(self.package._module, name, value)
                    if schema.get("element_shape")
                    else Scalar(schema["dtype"], value)
                )
            values[name] = value
        # Native preparation owns immutable instance bindings and their leases.
        # Replaying the returned call does not interpret operations in Python.
        plan = self.plan.with_argument_resources(resources) if resources else self.plan
        return plan.prepare(self.package._module, values)

    def run(self, arguments):
        # Convenience for one submission. Repeated replay uses bind(...).run().
        # Runtime retains in-flight preparations until the usual synchronization.
        self.bind(arguments).run()
