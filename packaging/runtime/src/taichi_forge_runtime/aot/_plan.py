"""Persistent execution descriptions; no compiler, device or live callbacks."""

from __future__ import annotations

import ctypes as C
import json
import math
from pathlib import Path


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate execution plan field: {key}")
        result[key] = value
    return result


class ExecutionPlan:
    """Persistent kernels, native operations and private resource reconstruction.

    Arguments are symbolic names bound to a loaded module and arrays/scalars at
    prepare time. Preparation records native commands without executing kernels
    or modifying buffers. Rebind by preparing another independent instance.
    """

    def __init__(self, operations, *, resources=None, gradients=None, initialization=None):
        document = {"version": 1 if initialization is None else 2, "operations": list(operations)}
        if initialization is not None:
            document["initialization"] = list(initialization)
        if resources is not None:
            document["resources"] = list(resources)
        if gradients is not None:
            document["gradients"] = gradients
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
            or set(value) - {"version", "operations", "resources", "gradients", "initialization"}
        ):
            raise ValueError("Execution plan requires version and operations")
        if type(value["version"]) is not int or value["version"] not in (1, 2):
            raise ValueError("Unsupported execution plan version")
        if (value["version"] == 2) != ("initialization" in value):
            raise ValueError("Initialization requires execution plan version 2")
        if "initialization" in value and not isinstance(value["initialization"], list):
            raise ValueError("Execution initialization must be an operation array")
        if "resources" in value and not isinstance(value["resources"], list):
            raise ValueError("Execution resources must be a list")
        if "gradients" in value and not isinstance(value["gradients"], dict):
            raise ValueError("Execution gradients must be a name mapping")
        return cls(
            value["operations"],
            resources=value.get("resources"),
            gradients=value.get("gradients"),
            initialization=value.get("initialization"),
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
        replay = set()
        for index, op in enumerate(operations + document.get("initialization", [])):
            if index == len(operations):
                replay = set(arguments)
            if not isinstance(op, dict):
                raise ValueError("Execution operations must be objects")
            kind = op.get("kind")
            if kind == "kernel":
                required, optional = {"kind", "entry", "arguments"}, {"label"}
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
                if kind == "kernel" and key == "label":
                    _name(op[key])
                    if "\n" in op[key] or "\r" in op[key]:
                        raise ValueError("Dispatch labels cannot contain line breaks")
                    continue
                if kind == "gemm_f32":
                    continue
                if type(op[key]) is not int or not 0 <= op[key] <= 0x7FFFFFFFFFFFFFFF:
                    raise ValueError("Execution sizes and offsets require nonnegative int64 values")
            if kind == "native":
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
        self.kernel_entries = tuple(dict.fromkeys(kernels))
        self.argument_names = external_arguments(
            document.get("resources", []), document.get("gradients", {}), arguments
        )

        resources, gradients = document.get("resources", []), document.get("gradients", {})
        self._retained_argument_names = retained_arguments(resources, gradients, arguments)
        self._replay_argument_names = retained_arguments(
            resources, gradients, replay if "initialization" in document else arguments
        )
        self._has_initialization = "initialization" in document

    def prepare(self, module, arguments, *, gradients=None):
        """Resolve a frozen plan to the same native executor as kernel.prepare()."""
        from . import _c_api as abi
        from ._runtime import NdArray, PreparedCall, _argument, _name

        runtime = module._runtime
        with runtime._lock:
            module._check()
            owners = [module]
            replay_owners = [module]

            def bind(name, value):
                retained = []
                argument = _argument(runtime, value, retained)
                if name in self._retained_argument_names:
                    owners.extend(retained)
                if name in self._replay_argument_names:
                    replay_owners.extend(retained)
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
            handle = runtime._api.call(
                runtime._api.ti_prepare_execution_plan_ext,
                runtime._handle,
                module._handle,
                len(self._json),
                self._json,
                len(native),
                native,
                len(bindings),
                (abi.NamedBinding * len(bindings))(*bindings),
            )
            return PreparedCall(
                runtime,
                handle,
                tuple(dict.fromkeys(owners)),
                replay_owners=tuple(dict.fromkeys(replay_owners)) if self._has_initialization else None,
            )
