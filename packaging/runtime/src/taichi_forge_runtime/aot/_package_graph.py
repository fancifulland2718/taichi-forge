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
import weakref

from ._plan import ExecutionPlan
from ._runtime import NdArray, PreparedCall, Scalar
from ._package_values import capture_value, check_admission, owner_type
from .._dimensions import extent_dependencies, local_workspace_shape, check_model_static_ranges
from .._static_values import static_key


_MISSING_PLAN_VALUE = object()


def graph_builder_for_owners(owners, default_factory):
    """Select one recording owner before creating any Graph resources.

    A package Plan retains its complete native recording. Mixing independently
    loaded modules or source owners requires an explicit composition boundary;
    it must not silently turn into a Python loop of loaded kernel calls.
    """
    packages = [getattr(owner, "_forge_package_owner", None) for owner in owners]
    package = next((item for item in packages if item is not None), None)
    if package is None:
        return default_factory()
    from ._package import LoadedPackage

    if not isinstance(package, LoadedPackage) or any(item is not package for item in packages):
        raise ValueError("Graph owners must all belong to the same loaded package, or all use JIT")
    package._check()
    return package.graph_builder()


def _plan_value(owner, spec):
    # A deferred preparation may declare a parameter that this recording never
    # reads. Preserve absence as a guard instead of inventing a numeric default.
    try:
        return capture_value(owner, spec)
    except AttributeError:
        return _MISSING_PLAN_VALUE


def _plan_key(value):
    return ("missing_plan_value",) if value is _MISSING_PLAN_VALUE else static_key(value)


class ArgKind(Enum):
    SCALAR = "value"
    NDARRAY = "ndarray"
    MATRIX = "compound"
    STRUCT = "compound_struct"
    TEXTURE = "texture"
    RWTEXTURE = "rw_texture"
    ACCELERATION_STRUCTURE = "acceleration_structure"


@dataclass(frozen=True)
class Arg:
    tag: ArgKind
    name: str
    dtype: object = None
    ndim: int = 0
    fmt: object = None
    lod: int = 0

    def schema(self):
        from ._package import _ElementType

        if not isinstance(self.name, str) or not self.name or not isinstance(self.tag, ArgKind):
            raise ValueError("Graph arguments require a name and supported ArgKind")
        if self.tag == ArgKind.SCALAR:
            return dict(kind="value", dtype=self.dtype)
        if self.tag == ArgKind.ACCELERATION_STRUCTURE:
            if self.dtype is not None or self.ndim != 0 or self.fmt is not None or self.lod != 0:
                raise ValueError("Acceleration structure Graph arguments have no element or image layout")
            return dict(kind="acceleration_structure")
        if self.tag == ArgKind.RWTEXTURE:
            fmt = getattr(self.fmt, "name", self.fmt)
            if (self.dtype is not None or type(self.ndim) is not int or self.ndim not in (1, 2, 3)
                    or not isinstance(fmt, str) or not fmt or type(self.lod) is not int or self.lod < 0):
                raise ValueError("Storage image Graph arguments require rank, format and nonnegative lod")
            return dict(kind="rw_texture", ndim=self.ndim, fmt=fmt, lod=self.lod)
        if self.tag == ArgKind.TEXTURE:
            if self.dtype is not None or type(self.ndim) is not int or self.ndim not in (1, 2, 3):
                raise ValueError("Sampled texture Graph arguments require rank 1, 2 or 3 and no dtype")
            return dict(kind="texture", ndim=self.ndim)
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
        self._native_effects = {}
        self._native_extents = {}

    def _record_plan_values(self, owner):
        if owner is None:
            return
        schema = self.package.manifest["owners"].get(owner_type(owner), {})
        for name, item in schema.items():
            if not item.get("affects_plan"):
                continue
            spec = dict(name=name, kind="value", dtype=item["dtype"], path=item.get("path", [name]))
            value = _plan_value(owner, spec)
            key = id(owner), name
            previous = self._plan_values.get(key)
            if previous is not None and _plan_key(previous[2]) != _plan_key(value):
                raise ValueError(f"Plan parameter {name!r} changed while recording; rebuild the Graph")
            self._plan_values[key] = owner, spec, value

    def _check_plan_values(self):
        for owner, spec, value in self._plan_values.values():
            if _plan_key(_plan_value(owner, spec)) != _plan_key(value):
                raise ValueError(f"Plan parameter {spec['name']!r} changed; rebuild the Graph before preparing")

    def dispatch(self, kernel, *args, template_args=None):
        from ._package import entry, _BoundEntry
        from ._package_closures import _TemplateEntry, _FunctionEntry, _BoundMethodEntry

        self.package._check()
        if isinstance(kernel, (_FunctionEntry, _BoundMethodEntry)):
            kernel, args, template_args = kernel._graph_bind(args, template_args)
        if isinstance(kernel, _TemplateEntry):
            if not template_args or "self" not in template_args:
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
            # JIT Graph uses array exemplars to obtain argument types.
            # Published entries already carry those types; verify the hints
            # without retaining their storage or replacing symbolic bindings.
            expected_by_name = {item["name"]: item for item in kernel.contract["arguments"]}
            for name, value in template_args.items():
                if name == "self" and owner is not None:
                    if value is not owner:
                        raise ValueError("Package Graph template receiver must match its bound entry")
                    continue
                item = expected_by_name.get(name, {})
                if item.get("kind") == "ndarray":
                    value = self.package.runtime._adapt_package_array(value)
                    if not isinstance(value, NdArray):
                        raise TypeError(f"Graph exemplar {name!r} requires a device ndarray")
                    value._descriptor(self.package.runtime)
                    element = item["element"]
                    if (value.dtype != element["dtype"] or len(value.shape) != item["ndim"]
                            or value.element_shape != tuple(element.get("element_shape", ()))):
                        raise ValueError(f"Graph exemplar {name!r} differs from the entry's type")
                else:
                    raise ValueError(f"Unknown or non-resource Graph template argument {name!r}")
        if kernel.contract["captures"] and owner is None:
            raise ValueError("Captured Graph entries require their bound instance")
        if owner is not None and getattr(owner, "_forge_package_owner", None) is not self.package:
            raise ValueError("Graph receiver belongs to another loaded package")
        expected = kernel.contract["arguments"]
        if len(args) != len(expected):
            raise ValueError("Graph dispatch argument count differs from the entry")
        arguments = dict(self.arguments)
        constants = dict(self._constants)
        names, captures = [], dict(self._captures)
        for arg, item in zip(args, expected):
            if not isinstance(arg, Arg) and item["kind"] == "ndarray":
                array = self.package.runtime._adapt_package_array(arg)
                if not isinstance(array, NdArray):
                    raise TypeError("Graph fixed array requires device dense storage")
                array._descriptor(self.package.runtime)
                element = item["element"]
                if (array.dtype != element["dtype"] or len(array.shape) != item["ndim"]
                        or array.element_shape != tuple(element.get("element_shape", ()))):
                    raise ValueError("Graph fixed array differs from the entry's type")
                if array not in self._bindings.native_resources:
                    self._bindings.native_resources[array] = SimpleNamespace(value=array)
                holder = self._bindings.native_resources[array]
                name, binding = self._bindings.capture(holder, dict(item, name="value"))
                names.append(name)
                captures[name] = binding
                continue
            if not isinstance(arg, Arg):
                raise TypeError("Graph dispatch arguments must be symbolic Arg values")
            names.append(arg.name)
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
            if key in self._plan_values and _plan_key(self._plan_values[key][2]) != _plan_key(binding[2]):
                raise ValueError("Graph sequences recorded different plan parameters")
        extents = dict(self._native_extents)
        for name, extent in sequence._native_extents.items():
            if name in extents and extents[name] != extent:
                raise ValueError("Native operations require incompatible array extents")
            extents[name] = extent
        self.operations.extend(sequence.operations)
        self.arguments = combined
        self._captures.update(sequence._captures)
        self._constants.update(sequence._constants)
        self._admissions.extend(sequence._admissions)
        self._range_checks.extend(sequence._range_checks)
        self._plan_values.update(sequence._plan_values)
        self._native_effects.update(sequence._native_effects)
        self._native_extents = extents
        return self

    def append_native(self, sequence, *, prewarm=False, admission="explicit"):
        from ._package_algorithms import PrimitiveSequence, sort_inputs
        from ._package_native import NativeRecording, NativeObject

        self.package._check()
        if type(prewarm) is not bool or admission not in ("explicit", "auto"):
            raise ValueError("Invalid native recording policy")
        if isinstance(sequence, NativeRecording):
            if sequence.package is not self.package:
                raise ValueError("Native recording belongs to another package")
            operation = copy.deepcopy(sequence.description)
            kind = operation["provider"] + "_" + operation["operation"]
            modes = self.package.runtime.native_operation_capabilities(kind, version=operation["version"])
            if not modes or (admission == "auto" and not modes & {"cuda_ordered", "cuda_capture", "vulkan_inline"}):
                raise ValueError(f"Native operation {kind} has no qualified Graph execution mode: {sorted(modes)}")
            arguments, captures, effects = dict(self.arguments), {}, {}
            for port, value in sequence.bindings.items():
                schema = sequence.schemas[port]
                if isinstance(value, str):
                    name = value
                    if name.startswith("_forge_package_"):
                        raise ValueError("Graph argument uses a reserved package binding name")
                    if name in arguments and arguments[name] != schema:
                        raise ValueError("Native port conflicts with a Graph argument")
                    arguments[name] = schema
                else:
                    if isinstance(value, NativeObject):
                        value._check()
                    holder = self._bindings.native_resources.setdefault(id(value), SimpleNamespace(value=value))
                    name, binding = self._bindings.capture(holder, dict(name="value", **schema))
                    captures[name] = binding
                operation["bindings"][port] = name
                effects[name] = effects.get(name, False) or sequence.effects[port] != "read"
            extents = dict(self._native_extents)
            for port, extent in getattr(sequence, "extents", {}).items():
                name = operation["bindings"][port]
                previous = extents.get(name, extent)
                if previous != extent:
                    raise ValueError("Native operations require incompatible array extents")
                extents[name] = extent
            self.operations.append(operation)
            self.arguments = arguments
            self._captures.update(captures)
            key = json.dumps(operation, sort_keys=True)
            self._native_effects[key] = effects
            self._native_extents = extents
            return self
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
            if not isinstance(argument, Arg) or argument.schema() not in (
                dict(kind="ndarray", element=dict(kind="scalar", dtype="i32", element_shape=[]), ndim=rank)
                for rank in (0, 1)
            ):
                raise ValueError(f"Graph while {role} requires a scalar or rank-one i32 array")
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

    def freeze(self):
        from ._package_native import NativeGraphDefinition
        return NativeGraphDefinition(self.compile())

    def compile(self, *, workspace_lanes=1, workspace_saturation="wait"):
        if type(workspace_lanes) is not int or workspace_lanes < 1:
            raise ValueError("Graph workspace_lanes must be a positive integer")
        if workspace_saturation not in ("wait", "raise"):
            raise ValueError("Graph workspace_saturation must be 'wait' or 'raise'")
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
            native_effects=copy.deepcopy(self._native_effects),
            native_extents=dict(self._native_extents),
            workspace_lanes=workspace_lanes,
            workspace_saturation=workspace_saturation,
        )


class Graph:
    def __init__(self, package, plan, arguments, captures=None, admissions=None, range_checks=None, plan_values=None,
                 *, workspace_lanes=1, workspace_saturation="wait", native_effects=None, native_extents=None):
        self.package, self.plan, self.arguments = package, plan, arguments
        self.captures = captures or {}
        self.admissions = admissions or []
        self.range_checks = range_checks or []
        self.plan_values = plan_values or {}
        self.native_effects = native_effects or {}
        self.native_extents = native_extents or {}
        self._cached_call = None
        self._cached_values = ()
        self._cached_tokens = ()
        self._cached_guards = ()
        self._ordered_replay_required = False
        self._closed = False
        self._workspace_lanes = workspace_lanes
        self._workspace_saturation = workspace_saturation
        self._parameter_frames = []
        self._frame_tokens = self._frame_guards = ()
        self._next_parameter_frame = 0
        self._live_parameters = ()
        if workspace_lanes > 1:
            document = json.loads(plan.to_json())
            if (package.runtime.arch != "cuda" or document.get("initialization")
                    or any(op["kind"] != "kernel" for op in document["operations"])):
                raise ValueError("Package workspace lanes require ordinary CUDA kernel Graphs")
            # Explicit scalar inputs are live values, unlike captured layout
            # and model facts. Native preparation still validates every use.
            self._live_parameters = tuple(name for name, spec in arguments.items()
                if spec["kind"] == "value" and spec.get("dtype") in ("i32", "u32", "i64", "u64", "f32", "f64"))
            if not self._live_parameters:
                raise ValueError("Package workspace lanes require explicit live scalar arguments")
        package._graphs.add(self)

    def written_resources(self):
        """Inspect compiled effects without allocating, preparing or submitting."""
        self.package._check()
        for owner, contract in self.admissions:
            check_admission(owner, contract)
        for owner, schema, previous in self.plan_values.values():
            if _plan_key(_plan_value(owner, schema)) != _plan_key(previous):
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
                    if operation["kind"] == "native":
                        effects = self.native_effects.get(json.dumps(operation, sort_keys=True))
                        if effects is None:
                            raise ValueError("Native operation has no recorded effect contract")
                        written.update(name for name, writes in effects.items() if writes)
                        continue
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
        return self.prepare(arguments)

    def close(self):
        if not self._closed:
            self._clear_cached()
            self._closed = True
            self.package._graphs.discard(self)

    def prepare(self, arguments, *, execution_domain=None, mutable_parameters=()):
        """Prepare the recorded package plan with optional live submission values.

        Captures and resources retain the same immutable binding and leases as
        bind(). Explicit scalar arguments may use the ordinary CUDA queue or
        an independent execution domain's parameter-update protocol.
        """
        from ._parameters import SubmissionParameters

        if self._closed:
            raise RuntimeError("Package Graph is closed")

        mutable_parameters = SubmissionParameters(mutable_parameters).names
        if any(name not in self.arguments for name in mutable_parameters):
            raise ValueError("Mutable Graph parameters must be explicit Graph arguments")
        self.package._check()
        if arguments.keys() != self.arguments.keys():
            raise ValueError("Graph binding must supply exactly its declared arguments")
        for owner, contract in self.admissions:
            check_admission(owner, contract)
        for owner, schema, previous in self.plan_values.values():
            if _plan_key(_plan_value(owner, schema)) != _plan_key(previous):
                raise ValueError(f"Plan parameter {schema['name']!r} changed; rebuild the Graph before preparing")
        values = {}
        for name, schema in self.arguments.items():
            value = arguments[name]
            if schema["kind"] == "value":
                value = Scalar(schema["dtype"], value)
            elif schema["kind"] == "compound":
                value = self.plan.value(self.package._module, name, value)
            values[name] = value
        captured = {}
        for name, (owner, schema) in self.captures.items():
            if schema["kind"] == "local_tensor_workspace":
                continue
            parameter = self.plan_values.get((id(owner), schema["name"]))
            value = parameter[2] if parameter is not None else _MISSING_PLAN_VALUE
            # Absence remains a plan guard above, never a scalar ABI value.
            # Admission has already checked the branch requirements; preserve
            # the entry capture's nullable-record contract for an absent member.
            captured[name] = capture_value(owner, schema) if value is _MISSING_PLAN_VALUE else value
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
        plan = self._specialized_plan(plan)
        values = {name: self.package.runtime._adapt_package_array(value) for name, value in values.items()}
        for name, extent in self.native_extents.items():
            if not isinstance(values[name], NdArray) or values[name].shape[0] != extent:
                raise ValueError("Native whole-array extent differs from its recording")
        prepared = plan.prepare(self.package._module, values, execution_domain=execution_domain,
                                mutable_parameters=mutable_parameters)
        # A cached call must not create Graph -> call -> Graph ownership cycles.
        # Native owners already retain the module and every argument resource.
        prepared._forge_package_graph = weakref.ref(self)
        # A constructor can initialize through a Graph, including an explicit
        # binding. Its new invocation belongs to the same failure rollback as
        # ordinary calls; successful construction transfers ownership normally.
        from ._package import _construction

        construction = _construction.get()
        if construction is not None and construction[0] is self.package:
            construction[3].callback(prepared.close)
        return prepared

    def _specialized_plan(self, plan):
        # Only the cold preparation copy selects code. The exposed generic plan
        # remains reusable with other binding strategies and SNode instances.
        if (plan.binding_strategy is not None
                or self.package.runtime.arch not in ("cpu", "x64", "arm64", "cuda", "vulkan")):
            return plan
        from ._package_calls import specialization_constant_args

        document = json.loads(plan.to_json())
        pending = list(document["operations"]) + document.get("initialization", [])
        changed = False
        ordinary = {"value", "dimension", "compound", "ndarray", "dense_field",
                    "model_tensor_field", "local_tensor_workspace"}
        while pending:
            operation = pending.pop()
            kind = operation["kind"]
            if kind == "while":
                pending.extend(operation["condition"] + operation["body"])
            elif kind == "branch":
                pending.extend(operation["condition"])
                for region in operation["bodies"]:
                    pending.extend(region)
            elif kind in ("cuda_parallel", "vulkan_parallel"):
                for region in operation["branches"]:
                    pending.extend(region)
            elif (kind == "kernel" and "launch" not in operation
                  and operation["entry"] in self.package._specialization_entries):
                contract = self.package.manifest["entries"][operation["entry"]]
                if not all(spec["kind"] in ordinary for spec in (*contract["arguments"], *contract["captures"])):
                    continue  # Managed resources retain a qualified kernel identity.
                schema = {}
                if contract["captures"]:
                    capture = operation["arguments"][len(contract["arguments"])]
                    owner = self.captures[capture][0]
                    schema = self.package.manifest["owners"].get(owner_type(owner), {})
                operation["specialization"] = dict(
                    constant_args=list(specialization_constant_args(contract, schema, self.package.manifest["owners"])))
                changed = True
        if not changed:
            return plan
        document.pop("version")
        return ExecutionPlan(**document)

    def _binding_guards(self):
        if self._ordered_replay_required:
            # The caller has explicitly qualified this prepared snapshot for
            # ordered replay. Captured owners are fixed until a cold rebind or
            # Graph replacement, as in an explicitly bound PreparedCall.
            return ()
        from ._package_calls import OwnerGuard

        groups = {}
        def group(owner):
            return groups.setdefault(id(owner), (owner, []))[1]
        for owner, contract in self.admissions:
            group(owner).append(contract)
        for owner, spec in self.captures.values():
            group(owner).append(dict(captures=[spec]))
        for owner, spec, _ in self.plan_values.values():
            group(owner).append(dict(captures=[spec]))
        return tuple((owner, OwnerGuard.for_graph(self.package, owner, contracts))
                     for owner, contracts in groups.values())

    def _clear_cached(self):
        if self._cached_call is not None:
            self._cached_call.close()
        self._cached_call = None
        self._cached_values = self._cached_tokens = self._cached_guards = ()
        for frame in self._parameter_frames:
            frame.close()
        self._parameter_frames = []
        self._frame_tokens = self._frame_guards = ()

    @staticmethod
    def _can_replay_ordered(prepared):
        # Ordinary fixed CUDA graphs keep their argument frame immutable and
        # enqueue all scratch accesses on the same runtime stream. Unlike JIT
        # bindings they need no host-updated workspace lane between submissions.
        # Host control, binding families and independent domains have different
        # admission/ownership protocols and are not qualified by this fact.
        return (prepared._execution_domain is None and prepared._family is None
                and prepared.info['execution_kind'] == 'cuda_graph')

    def _prepare_ordered_replay(self, arguments):
        """Qualify a fixed capture snapshot for queued whole-Graph submission.

        The caller owns Graph invalidation when captured model facts/resources
        change. Explicit arguments still select a fresh binding when changed;
        native resource lifetime and in-flight lease checks remain in force.
        Ordinary implicit calls retain automatic capture-change observation.
        """
        if self.package.runtime.arch != 'cuda':
            return False
        with self.package._call_lock:
            prepared = self._prepared(arguments)
            qualified = (prepared is self._cached_call
                         and self._can_replay_ordered(prepared))
            self._ordered_replay_required = qualified
            if qualified:
                # Preparation already admitted and retained these captures.
                # Do not rescan the owner's Python attributes on every replay.
                self._cached_guards = ()
            return qualified

    def _prepared(self, arguments):
        self.package._check()
        if self._closed:
            raise RuntimeError("Package Graph is closed")
        if isinstance(arguments, PreparedCall):
            if getattr(arguments, "_forge_package_graph", lambda: None)() is not self:
                raise ValueError("Prepared Graph arguments belong to another recording")
            if self._ordered_replay_required and arguments is not self._cached_call:
                raise ValueError("Ordered replay requires the Graph's qualified binding")
            return arguments
        if arguments.keys() != self.arguments.keys():
            raise ValueError("Graph binding must supply exactly its declared arguments")
        from ._package_calls import _token, reusable_value

        values = tuple(arguments[name] for name in self.arguments)
        tokens = tuple(_token(value) for value in values)
        if (self._cached_call is not None and tokens == self._cached_tokens
                and all(guard.matches(owner) for owner, guard in self._cached_guards)):
            return self._cached_call
        # Keep at most the current implicit binding. Explicit bind() continues
        # to create independent snapshots and workspaces for its caller.
        guards = self._binding_guards()
        # Managed images/AS have immutable bindings just like arrays. Their
        # contents remain live and native preparation retains the resource
        # leases; treating them as unknown Python values rebuilds every frame.
        reusable = all(spec['kind'] in ('ndarray', 'native_resource', 'texture',
                                       'rw_texture', 'acceleration_structure') or reusable_value(value)
                       for spec, value in zip(self.arguments.values(), values))
        reusable = reusable and all(
            spec['kind'] not in ('value', 'dimension') or reusable_value(capture_value(owner, spec))
            for owner, spec in self.captures.values())
        prepared = self.bind(arguments)  # Full admission on every changed binding.
        try:
            if self._ordered_replay_required and (not reusable or not self._can_replay_ordered(prepared)):
                raise ValueError("Changed binding requires rebuilding the ordered Graph")
            self._clear_cached()
        except BaseException:
            prepared.close()
            raise
        if reusable:
            self._cached_call, self._cached_values = prepared, values
            self._cached_tokens, self._cached_guards = tokens, guards
        return prepared

    def run(self, arguments):
        if self._live_parameters and not isinstance(arguments, PreparedCall):
            self.submit(arguments).close()
            return
        with self.package._call_lock:
            self._prepared(arguments).run()

    def _submit_parameter_frame(self, arguments):
        self.package._check()
        if arguments.keys() != self.arguments.keys():
            raise ValueError("Graph binding must supply exactly its declared arguments")
        from ._package_calls import _token

        tokens = tuple(_token(arguments[name]) for name in self.arguments if name not in self._live_parameters)
        if (not self._parameter_frames or tokens != self._frame_tokens
                or not all(guard.matches(owner) for owner, guard in self._frame_guards)):
            guards = self._binding_guards()
            frames = []
            try:
                for _ in range(self._workspace_lanes):
                    frame = self.prepare(arguments, mutable_parameters=self._live_parameters)
                    frames.append(frame)
                    frame.initialize()
            except BaseException:
                for frame in frames:
                    frame.close()
                raise
            self._clear_cached()
            self._parameter_frames = frames
            self._frame_tokens, self._frame_guards = tokens, guards
            self._next_parameter_frame = 0
        values = {name: arguments[name] for name in self._live_parameters}
        start = self._next_parameter_frame
        for offset in range(len(self._parameter_frames)):
            index = (start + offset) % len(self._parameter_frames)
            frame = self._parameter_frames[index]
            ticket = frame.submit(values=values)
            if ticket is not None:
                self._next_parameter_frame = (index + 1) % len(self._parameter_frames)
                return ticket, None
        return None, self._parameter_frames[start]

    def submit(self, arguments, *, pacer=None, lane=None, on_saturation="wait"):
        """Replay a prepared binding and return a completion ticket.

        Multiple workspace lanes reuse explicit scalar parameter frames on the
        ordinary CUDA queue. Other changed bindings re-enter preparation.
        Explicit bind() retains an
        independent snapshot; the ticket and Runtime retain resource leases.
        A hybrid package may share its JIT runtime's SubmissionPacer. Admission
        precedes binding; standalone Runtime submission needs no frontend.
        """
        if pacer is not None:
            submit = getattr(pacer, "_submit_package_graph", None)
            if submit is None:
                raise TypeError("pacer must support package Graph submission")
            return submit(self, arguments, lane=lane, on_saturation=on_saturation)
        if lane is not None or on_saturation != "wait":
            raise ValueError("lane and on_saturation require a SubmissionPacer")
        if self._live_parameters and not isinstance(arguments, PreparedCall):
            arguments = dict(arguments)
        while True:
            with self.package._call_lock:
                if not self._live_parameters or isinstance(arguments, PreparedCall):
                    return self._prepared(arguments).submit()
                ticket, pending = self._submit_parameter_frame(arguments)
                if ticket is not None:
                    return ticket
                if self._workspace_saturation == "raise":
                    raise RuntimeError("All Graph workspace lanes are occupied")
            # Do not hold the shared package lock while another graph/thread
            # can make progress. The native frame owns its completion lifetime.
            pending._wait_parameter_frame()
