"""Persistent execution descriptions; no compiler, device or live callbacks."""

from __future__ import annotations

import ctypes as C
import json
import math
from pathlib import Path

_MAX_EXECUTION_PLAN_VERSION = 17


def _constraint_version(constraints):
    return 17 if any(isinstance(c, dict) and c.get("kind") == "resource_alias" for c in constraints) else 1


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate execution plan field: {key}")
        result[key] = value
    return result


def _control_version(steps):
    version = 1
    for op in steps:
        if isinstance(op, dict) and op.get("kind") == "branch":
            lowering = op.get("lowering")
            host = isinstance(lowering, dict) and lowering.get("kind") in ("cuda_host", "vulkan_host")
            masked = isinstance(lowering, dict) and lowering.get("kind") == "cuda_masked"
            version = max(version, 16 if masked else 12 if host else 11)
            regions = [op.get("condition")]
            if isinstance(op.get("bodies"), list):
                regions.extend(op["bodies"])
            for region in regions:
                if isinstance(region, list):
                    version = max(version, _control_version(region))
        if isinstance(op, dict) and op.get("kind") == "cuda_parallel":
            version = max(version, 9)
        if isinstance(op, dict) and op.get("kind") == "kernel" and "launch" in op:
            launch = op["launch"]
            bounded = isinstance(launch, dict) and launch.get("kind") in ("cpu_exact", "cuda_logical_exact")
            adaptive = isinstance(launch, dict) and launch.get("kind") == "cuda_adaptive"
            version = max(version, 8 if adaptive else 7 if bounded else 6)
        if isinstance(op, dict) and op.get("kind") == "while":
            lowering = op.get("lowering")
            selected = 5 if isinstance(lowering, dict) and lowering.get("kind") == "vulkan_nested" else 4
            if isinstance(lowering, dict) and lowering.get("kind") == "cuda_nested_device_update":
                selected = 13
            if isinstance(lowering, dict) and lowering.get("kind") == "cuda_nested_masked":
                selected = 14
            if isinstance(lowering, dict) and lowering.get("kind") == "cuda_masked":
                selected = 15
            version = max(version, selected if "lowering" in op else 3)
            for key in ("condition", "body"):
                if isinstance(op.get(key), list):
                    version = max(version, _control_version(op[key]))
    return version


class ExecutionPlan:
    """Persistent kernels, native operations and private resource reconstruction.

    Arguments are symbolic names bound to a loaded module and arrays/scalars/resources at
    prepare time. Preparation records native commands without executing kernels
    or modifying buffers. Rebind by preparing another independent instance.
    """

    def __init__(
        self,
        operations,
        *,
        resources=None,
        gradients=None,
        initialization=None,
        constraints=None,
        binding_strategy=None,
    ):
        document = {"version": 1 if initialization is None else 2, "operations": list(operations)}
        if initialization is not None:
            document["initialization"] = list(initialization)
        if resources is not None:
            document["resources"] = list(resources)
        if gradients is not None:
            document["gradients"] = gradients
        if constraints is not None:
            document["constraints"] = list(constraints)
        if binding_strategy is not None:
            if binding_strategy != "cuda_immutable_argument_frames":
                raise ValueError("Unsupported execution binding strategy")
            document["binding_strategy"] = binding_strategy
            document["version"] = 10
        document["version"] = max(
            document["version"], _control_version(document["operations"] + document.get("initialization", [])),
            _constraint_version(document.get("constraints", [])),
        )
        # Own a snapshot, including nested arrays; caller mutations cannot alter
        # a saved plan. This module remains usable without a native library.
        self._json = json.dumps(document, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._validate(json.loads(self._json))

    @classmethod
    def from_json(cls, document):
        value = json.loads(document, object_pairs_hook=_unique_fields)
        if (
            not isinstance(value, dict)
            or not {"version", "operations"} <= set(value)
            or set(value)
            - {
                "version",
                "operations",
                "resources",
                "gradients",
                "initialization",
                "constraints",
                "binding_strategy",
            }
        ):
            raise ValueError("Execution plan requires version and operations")
        if type(value["version"]) is not int or not 1 <= value["version"] <= _MAX_EXECUTION_PLAN_VERSION:
            raise ValueError("Unsupported execution plan version")
        if not isinstance(value["operations"], list):
            raise ValueError("Execution operations must be an operation array")
        if value["version"] < 3 and ((value["version"] == 2) != ("initialization" in value)):
            raise ValueError("Initialization requires execution plan version 2")
        if "initialization" in value and not isinstance(value["initialization"], list):
            raise ValueError("Execution initialization must be an operation array")
        if "resources" in value and not isinstance(value["resources"], list):
            raise ValueError("Execution resources must be a list")
        if "gradients" in value and not isinstance(value["gradients"], dict):
            raise ValueError("Execution gradients must be a name mapping")
        if "constraints" in value and not isinstance(value["constraints"], list):
            raise ValueError("Execution constraints must be a list")
        required = max(_control_version(value["operations"] + value.get("initialization", [])),
                       _constraint_version(value.get("constraints", [])))
        if "binding_strategy" in value:
            if value["binding_strategy"] != "cuda_immutable_argument_frames":
                raise ValueError("Unsupported execution binding strategy")
            required = max(required, 10)
        if value["version"] < required:
            raise ValueError(f"Execution features require execution plan version {required}")
        return cls(
            value["operations"],
            resources=value.get("resources"),
            gradients=value.get("gradients"),
            initialization=value.get("initialization"),
            constraints=value.get("constraints"),
            binding_strategy=value.get("binding_strategy"),
        )

    @classmethod
    def load(cls, path):
        return cls.from_json(Path(path).read_bytes())

    def save(self, path):
        Path(path).write_bytes(self._json)

    def to_json(self):
        return self._json.decode("utf-8")

    def _validate(self, document):
        from ._runtime import _name
        from ._resources import external_arguments, retained_arguments
        from ._native import NATIVE_FIELDS, native_references

        kernels = []
        arguments = set()
        operations = document["operations"]
        if not operations:
            raise ValueError("Execution plan requires a nonempty operation array")
        if "initialization" in document and not document["initialization"]:
            raise ValueError("Execution initialization must be a nonempty operation array")

        def validate_flat_kernels(regions):
            if any(item["kind"] != "kernel" or "launch" in item for region in regions for item in region):
                raise ValueError("CUDA control requires flat kernels without launch overrides")

        def validate_cuda_control(op):
            lowering = op["lowering"]
            flat = lowering["kind"] == "cuda_masked"
            if (
                set(lowering) != ({"kind", "version"} if flat else {"kind", "version", "inners"})
                or type(lowering["version"]) is not int
                or lowering["version"] != 1
                or (not flat and not op.get("continue_while_nonzero", True))
            ):
                raise ValueError("Unsupported CUDA control recipe")
            validate_flat_kernels((op["condition"], op["body"]))
            if flat:
                return
            inners = lowering["inners"]
            cursor = len(op["condition"])
            count = cursor + len(op["body"])
            if not isinstance(inners, list) or not inners or not cursor:
                raise ValueError("Invalid selected CUDA nested control")
            required = {"predicate", "condition_begin", "body_begin", "end", "max_iterations"}
            for inner in inners:
                if not isinstance(inner, dict) or set(inner) != required:
                    raise ValueError("Invalid CUDA nested control descriptor")
                _name(inner["predicate"])
                arguments.add(inner["predicate"])
                for key in ("condition_begin", "body_begin", "end", "max_iterations"):
                    if type(inner[key]) is not int or not 0 <= inner[key] <= 0x7FFFFFFFFFFFFFFF:
                        raise ValueError("Invalid CUDA nested boundary or budget representation")
                if not cursor <= inner["condition_begin"] < inner["body_begin"] < inner["end"] < count:
                    raise ValueError("Invalid CUDA nested dispatch boundaries")
                cursor = inner["end"]

        def validate_lowering(op):
            lowering = op["lowering"]
            if isinstance(lowering, dict) and lowering.get("kind") in (
                "cuda_nested_device_update",
                "cuda_nested_masked",
                "cuda_masked",
            ):
                validate_cuda_control(op)
                return
            nested = isinstance(lowering, dict) and lowering.get("kind") == "vulkan_nested"
            required = {"kind", "version", "counter", "inners" if nested else "chunks"}
            if (
                not isinstance(lowering, dict)
                or not required <= set(lowering)
                or set(lowering) - required - {"status"}
                or lowering["kind"] not in ("vulkan_structured", "vulkan_nested")
                or type(lowering["version"]) is not int
                or lowering["version"] != 1
                or not op.get("continue_while_nonzero", True)
            ):
                raise ValueError("Unsupported selected control recipe")
            for key in ("counter", "status"):
                if key in lowering:
                    _name(lowering[key])
                    arguments.add(lowering[key])
            if any(item["kind"] != "kernel" for item in op["condition"] + op["body"]):
                raise ValueError("Vulkan structured control requires flat kernel regions")
            if nested:
                inners = lowering["inners"]
                cursor = len(op["condition"])
                count = cursor + len(op["body"])
                if not isinstance(inners, list) or not inners or not cursor or op["max_iterations"] > 0xFFFFFFFF:
                    raise ValueError("Invalid selected Vulkan nested control")
                required = {
                    "predicate",
                    "counter",
                    "condition_begin",
                    "body_begin",
                    "end",
                    "max_iterations",
                    "chunk_size",
                }
                for inner in inners:
                    if not isinstance(inner, dict) or not required <= set(inner) or set(inner) - required - {"status"}:
                        raise ValueError("Invalid Vulkan nested control descriptor")
                    for key in ("predicate", "counter", "status"):
                        if key in inner:
                            _name(inner[key])
                            arguments.add(inner[key])
                    for key in ("condition_begin", "body_begin", "end", "max_iterations", "chunk_size"):
                        maximum = 0xFFFFFFFF if key in ("max_iterations", "chunk_size") else 0x7FFFFFFFFFFFFFFF
                        if type(inner[key]) is not int or not 0 <= inner[key] <= maximum:
                            raise ValueError("Invalid Vulkan nested boundary or budget representation")
                    budget, chunk = inner["max_iterations"], inner["chunk_size"]
                    if not cursor <= inner["condition_begin"] < inner["body_begin"] < inner["end"] < count or (
                        chunk != 0 if budget == 0 else not 0 < chunk <= budget
                    ):
                        raise ValueError("Invalid Vulkan nested dispatch boundaries or budget")
                    cursor = inner["end"]
                return
            chunks = lowering["chunks"]
            if not isinstance(chunks, list):
                raise ValueError("Vulkan control chunks must be an array")
            remaining = op["max_iterations"]
            for chunk in chunks:
                if (
                    not isinstance(chunk, dict)
                    or set(chunk) != {"iterations", "strategy"}
                    or type(chunk["iterations"]) is not int
                    or not 0 < chunk["iterations"] <= min(remaining, 0xFFFFFFFF)
                    or not isinstance(chunk["strategy"], str)
                    or chunk["strategy"] not in ("compact", "chained", "conditional", "coarse_conditional")
                ):
                    raise ValueError("Invalid selected Vulkan control chunk")
                remaining -= chunk["iterations"]
            if remaining:
                raise ValueError("Vulkan control chunks must cover the loop budget")

        def validate_steps(steps, *, allow_empty=False):
            if not isinstance(steps, list) or (not steps and not allow_empty):
                raise ValueError("Execution regions require operation arrays, with nonempty bodies")
            for op in steps:
                if not isinstance(op, dict):
                    raise ValueError("Execution operations must be objects")
                kind = op.get("kind")
                if kind == "branch":
                    if set(op) != {
                        "kind",
                        "selector",
                        "selection",
                        "has_default",
                        "condition",
                        "bodies",
                        "lowering",
                    }:
                        raise ValueError("Invalid branch operation fields")
                    _name(op["selector"])
                    arguments.add(op["selector"])
                    bodies, fallback = op["bodies"], op["has_default"]
                    if (
                        op["selection"] not in ("nonzero", "indexed")
                        or type(fallback) is not bool
                        or not isinstance(bodies, list)
                        or len(bodies) <= int(fallback)
                        or (op["selection"] == "nonzero" and len(bodies) != 1 + int(fallback))
                    ):
                        raise ValueError("Invalid branch selection or body layout")
                    lowering = op["lowering"]
                    if (
                        not isinstance(lowering, dict)
                        or set(lowering) != {"kind", "version"}
                        or lowering["kind"]
                        not in ("cpu", "cuda_conditional", "cuda_host", "vulkan_host", "cuda_masked")
                        or type(lowering["version"]) is not int
                        or lowering["version"] != 1
                    ):
                        raise ValueError("Unsupported branch lowering")
                    validate_steps(op["condition"], allow_empty=True)
                    for body in bodies:
                        validate_steps(body, allow_empty=True)
                    if lowering["kind"] == "cuda_masked":
                        validate_flat_kernels([op["condition"], *bodies])
                    continue
                if kind == "cuda_parallel":
                    if (
                        set(op) != {"kind", "branches"}
                        or not isinstance(op["branches"], list)
                        or len(op["branches"]) < 2
                    ):
                        raise ValueError("CUDA parallel requires at least two nonempty kernel branches")
                    for branch in op["branches"]:
                        if not isinstance(branch, list) or not branch:
                            raise ValueError("CUDA parallel requires nonempty branches")
                        if any(
                            not isinstance(item, dict) or item.get("kind") != "kernel" or "launch" in item
                            for item in branch
                        ):
                            raise ValueError("CUDA parallel requires ordinary kernel branches")
                        validate_steps(branch)
                    continue
                if kind == "while":
                    required = {"kind", "predicate", "max_iterations", "condition", "body"}
                    optional, names = {"continue_while_nonzero", "lowering"}, ("predicate",)
                elif kind == "kernel":
                    required, optional = {"kind", "entry", "arguments"}, {"label", "launch"}
                    names = ("entry",)
                elif kind == "copy":
                    required = {"kind", "source", "destination"}
                    optional = {"source_offset", "destination_offset", "size_bytes"}
                    names = ("source", "destination")
                elif kind == "fill32":
                    required, optional = {"kind", "destination", "value"}, {"offset", "size_bytes"}
                    names = ("destination",)
                elif kind == "compact":
                    if "columns" in op:
                        required, names = {"kind", "flags", "count", "columns"}, ("flags", "count")
                    else:
                        required = {"kind", "values", "flags", "output", "count"}
                        names = ("values", "flags", "output", "count")
                    optional = set()
                elif kind == "sort":
                    required, optional = {"kind", "keys"}, {"values"}
                    names = ("keys", "values") if "values" in op else ("keys",)
                elif kind == "fft_c2c_f32":
                    required = {"kind", "input", "output", "dimensions", "direction"}
                    optional, names = {"batch_count"}, ("input", "output")
                elif kind == "gemm_f32":
                    required = {"kind", "a", "b", "output", "rows", "columns", "inner"}
                    optional = {"alpha", "beta", "transpose_a", "transpose_b"}
                    names = ("a", "b", "output")
                elif kind == "native":
                    required = NATIVE_FIELDS
                    optional, names = set(), ()
                else:
                    raise ValueError(f"Unsupported execution operation: {kind}")
                if not required <= set(op) or set(op) - required - optional:
                    raise ValueError(f"Invalid {kind} operation fields")
                for name in names:
                    _name(op[name])
                    if name != "entry":
                        arguments.add(op[name])
                for key in (optional & set(op)) - set(names):
                    if kind == "kernel" and key == "launch":
                        launch = op[key]
                        if not isinstance(launch, dict):
                            raise ValueError("Unsupported kernel launch recipe")
                        if launch.get("kind") == "vulkan_indirect" and set(launch) == {"kind", "packet"}:
                            binding = launch["packet"]
                        elif launch.get("kind") in ("cpu_exact", "cuda_logical_exact") and set(launch) == {
                            "kind",
                            "extent",
                            "capacity",
                        }:
                            if type(launch["capacity"]) is not int or not 0 < launch["capacity"] <= 2**31 - 1:
                                raise ValueError("Bounded capacity must fit a positive i32")
                            binding = launch["extent"]
                        elif launch.get("kind") == "cuda_adaptive" and set(launch) == {
                            "kind",
                            "extent",
                            "capacity",
                            "block_dim",
                            "update_policy",
                        }:
                            if type(launch["capacity"]) is not int or not 0 < launch["capacity"] <= 2**31 - 1:
                                raise ValueError("Bounded capacity must fit a positive i32")
                            if type(launch["block_dim"]) is not int or not 0 < launch["block_dim"] <= 2**32 - 1:
                                raise ValueError("Adaptive block dimension must fit a positive u32")
                            if launch["update_policy"] not in ("per_node", "grouped_stateful"):
                                raise ValueError("Unsupported adaptive update policy")
                            binding = launch["extent"]
                        else:
                            raise ValueError("Unsupported kernel launch recipe")
                        _name(binding)
                        arguments.add(binding)
                        continue
                    if kind == "kernel" and key == "label":
                        _name(op[key])
                        if "\n" in op[key] or "\r" in op[key]:
                            raise ValueError("Dispatch labels cannot contain line breaks")
                        continue
                    if kind == "while":
                        if key == "continue_while_nonzero" and type(op[key]) is not bool:
                            raise ValueError("Loop predicate polarity must be a boolean")
                        continue
                    if kind == "gemm_f32":
                        continue
                    if type(op[key]) is not int or not 0 <= op[key] <= 0x7FFFFFFFFFFFFFFF:
                        raise ValueError("Execution sizes and offsets require nonnegative int64 values")
                if kind == "while":
                    if type(op["max_iterations"]) is not int or not 0 <= op["max_iterations"] <= 0x7FFFFFFFFFFFFFFF:
                        raise ValueError("Loop budget requires a nonnegative int64 value")
                    validate_steps(op["condition"], allow_empty=True)
                    validate_steps(op["body"])
                    if "lowering" in op:
                        validate_lowering(op)
                elif kind == "native":
                    arguments.update(native_references(op))
                elif kind == "kernel":
                    if not isinstance(op["arguments"], list):
                        raise ValueError("Kernel arguments must be a list of names")
                    for name in op["arguments"]:
                        _name(name)
                        arguments.add(name)
                    kernels.append(op["entry"])
                elif kind == "fill32":
                    if type(op["value"]) is not int or not 0 <= op["value"] <= 0xFFFFFFFF:
                        raise ValueError("fill32 value must be a uint32 bit pattern")
                elif kind == "gemm_f32":
                    if any(
                        type(op[key]) is not int or not 0 < op[key] <= 0x7FFFFFFFFFFFFFFF
                        for key in ("rows", "columns", "inner")
                    ):
                        raise ValueError("GEMM requires positive int64 dimensions")
                    for key, default in (("alpha", 1.0), ("beta", 0.0)):
                        value = op.get(key, default)
                        if type(value) not in (int, float) or not math.isfinite(C.c_float(value).value):
                            raise ValueError("GEMM scalars must be finite f32 numbers")
                    if any(type(op.get(key, False)) is not bool for key in ("transpose_a", "transpose_b")):
                        raise ValueError("GEMM transpose flags must be booleans")
                elif kind == "fft_c2c_f32":
                    dimensions = op["dimensions"]
                    if (
                        not isinstance(dimensions, list)
                        or not dimensions
                        or any(type(axis) is not int or not 0 < axis <= 0x7FFFFFFFFFFFFFFF for axis in dimensions)
                        or op.get("batch_count", 1) == 0
                    ):
                        raise ValueError("FFT requires positive int64 dimensions and batch count")
                    if op["direction"] not in ("forward", "inverse"):
                        raise ValueError("FFT direction must be forward or inverse")
                elif kind == "compact" and "columns" in op:
                    if not isinstance(op["columns"], list) or not op["columns"]:
                        raise ValueError("Compact columns must be a nonempty array")
                    for column in op["columns"]:
                        if not isinstance(column, dict) or set(column) != {"values", "output"}:
                            raise ValueError("Each compact column requires values and output names")
                        _name(column["values"])
                        _name(column["output"])
                        arguments.update((column["values"], column["output"]))

        self.binding_strategy = document.get("binding_strategy")
        if self.binding_strategy is not None:
            for operation in operations:
                if isinstance(operation, dict) and (
                    operation.get("kind") in ("branch", "while", "cuda_parallel") or "launch" in operation
                ):
                    raise ValueError("CUDA binding frames require a flat fixed replay sequence")
        validate_steps(operations)
        replay = set(arguments)
        if "initialization" in document:
            validate_steps(document["initialization"])
        self.kernel_entries = tuple(dict.fromkeys(kernels))
        self.argument_names = external_arguments(
            document.get("resources", []), document.get("gradients", {}), arguments
        )

        resources, gradients = document.get("resources", []), document.get("gradients", {})
        self._retained_argument_names = retained_arguments(resources, gradients, arguments)
        host_data = {}
        for resource in resources:
            if resource["kind"] == "host_blob" and resource["name"] in self._retained_argument_names:
                key, size = resource["data"], resource["byte_size"]
                if host_data.setdefault(key, size) != size:
                    raise ValueError("Execution host blob size differs between declarations")
        # Declared reachable payload bytes, not device scratch or process RSS.
        # A prepared owner shares identical host payloads by content key.
        self.host_data_bytes = sum(host_data.values())
        self.data_entries = tuple(
            sorted(
                {
                    resource["data"] if resource["kind"] == "host_blob" else resource["initial_data"]
                    for resource in resources
                    if (
                        resource["kind"] == "host_blob"
                        or (resource["kind"] == "ndarray" and "initial_data" in resource)
                    )
                    and resource["name"] in self._retained_argument_names
                }
            )
        )
        from ._constraints import validate_constraints

        validate_constraints(document.get("constraints", []), self._retained_argument_names)
        self._replay_argument_names = retained_arguments(
            resources, gradients, replay if "initialization" in document else arguments
        )
        self._has_initialization = "initialization" in document
        native_roots = {
            resource["name"]
            for resource in resources
            if resource["kind"] == "native" and resource["name"] in self._replay_argument_names
        }
        self._persistent_argument_names = retained_arguments(resources, gradients, native_roots)

    def prepare(self, module, arguments, *, gradients=None):
        """Resolve a frozen plan to the same native executor as kernel.prepare()."""
        return self._prepare(module, arguments, gradients=gradients)

    def _prepare(self, module, arguments, *, gradients=None, source=None):
        from . import _c_api as abi
        from ._runtime import NativeResource, NdArray, PreparedCall, _BindingFamily, _argument, _name

        runtime = module._runtime
        with runtime._lock:
            module._check()
            owners = [module]
            replay_owners = [module]
            persistent_owners = [module]

            def bind(name, value):
                retained = []
                argument = _argument(runtime, value, retained)
                if name in self._retained_argument_names:
                    owners.extend(retained)
                if name in self._replay_argument_names:
                    replay_owners.extend(retained)
                if name in self._persistent_argument_names or (
                    isinstance(value, NativeResource) and name in self._replay_argument_names
                ):
                    persistent_owners.extend(retained)
                return abi.NamedArgument(_name(name), argument)

            native = (abi.NamedArgument * len(arguments))(*[bind(name, value) for name, value in arguments.items()])
            gradient_descriptors = []
            bindings = []
            for name, gradient in (gradients or {}).items():
                if name not in arguments or not isinstance(arguments[name], NdArray):
                    raise ValueError("Gradient names must refer to ndarray arguments")
                if not isinstance(gradient, NdArray):
                    raise TypeError("A gradient must be an NdArray")
                descriptor = gradient._descriptor(runtime)
                gradient_descriptors.append(descriptor)
                bindings.append(abi.NamedBinding(_name(name), 0, None, C.pointer(descriptor)))
                owners.append(gradient._memory)
                if name in self._replay_argument_names:
                    replay_owners.append(gradient._memory)
            function, prefix = (
                (
                    runtime._api.ti_prepare_execution_plan_ext,
                    (runtime._handle, module._handle, len(self._json), self._json),
                )
                if source is None
                else (runtime._api.ti_prepare_execution_binding_ext, (runtime._handle, source._handle))
            )
            handle = runtime._api.call(
                function,
                *prefix,
                len(native),
                native,
                len(bindings),
                (abi.NamedBinding * len(bindings))(*bindings),
            )
            family = source._family if source is not None else None
            if family is None and self.binding_strategy is not None:
                family = _BindingFamily(
                    module, self, tuple(dict.fromkeys(owners)), tuple(dict.fromkeys(persistent_owners))
                )
            return PreparedCall(
                runtime,
                handle,
                tuple(dict.fromkeys(replay_owners if family is not None else owners)),
                replay_owners=(
                    tuple(dict.fromkeys(replay_owners)) if self._has_initialization and family is None else None
                ),
                family=family,
            )
