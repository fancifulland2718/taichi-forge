"""Persistent execution descriptions; no compiler, device or live callbacks."""

from __future__ import annotations

import ctypes as C
import json
from pathlib import Path


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate execution plan field: {key}")
        result[key] = value
    return result


class ExecutionPlan:
    """Version 1 kernels, copies, fills, stable sort and compaction.

    Arguments are symbolic names bound to a loaded module and arrays/scalars at
    prepare time. Preparation records native commands without executing kernels
    or modifying buffers. Rebind by preparing another independent instance.
    """

    def __init__(self, operations):
        document = {"version": 1, "operations": list(operations)}
        # Own a snapshot, including nested arrays; caller mutations cannot alter
        # a saved plan. This module remains usable without a native library.
        self._json = json.dumps(document, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._validate(json.loads(self._json))

    @classmethod
    def from_json(cls, document):
        value = json.loads(document, object_pairs_hook=_unique_fields)
        if not isinstance(value, dict) or set(value) != {"version", "operations"}:
            raise ValueError("Execution plan requires version and operations")
        if type(value["version"]) is not int or value["version"] != 1:
            raise ValueError("Unsupported execution plan version")
        return cls(value["operations"])

    @classmethod
    def load(cls, path):
        return cls.from_json(Path(path).read_bytes())

    def save(self, path):
        Path(path).write_bytes(self._json)

    def to_json(self):
        return self._json.decode("utf-8")

    def _validate(self, document):
        from ._runtime import _name

        kernels = []
        operations = document["operations"]
        if not operations:
            raise ValueError("Execution plan requires a nonempty operation array")
        for op in operations:
            if not isinstance(op, dict):
                raise ValueError("Execution operations must be objects")
            kind = op.get("kind")
            if kind == "kernel":
                required, optional = {"kind", "entry", "arguments"}, set()
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
            else:
                raise ValueError(f"Unsupported execution operation: {kind}")
            if not required <= set(op) or set(op) - required - optional:
                raise ValueError(f"Invalid {kind} operation fields")
            for name in names:
                _name(op[name])
            for key in (optional & set(op)) - set(names):
                if type(op[key]) is not int or not 0 <= op[key] <= 0x7FFFFFFFFFFFFFFF:
                    raise ValueError("Execution sizes and offsets require nonnegative int64 values")
            if kind == "kernel":
                if not isinstance(op["arguments"], list):
                    raise ValueError("Kernel arguments must be a list of names")
                for name in op["arguments"]:
                    _name(name)
                kernels.append(op["entry"])
            elif kind == "fill32":
                if type(op["value"]) is not int or not 0 <= op["value"] <= 0xFFFFFFFF:
                    raise ValueError("fill32 value must be a uint32 bit pattern")
            elif kind == "compact" and "columns" in op:
                if not isinstance(op["columns"], list) or not op["columns"]:
                    raise ValueError("Compact columns must be a nonempty array")
                for column in op["columns"]:
                    if not isinstance(column, dict) or set(column) != {"values", "output"}:
                        raise ValueError("Each compact column requires values and output names")
                    _name(column["values"])
                    _name(column["output"])
        self.kernel_entries = tuple(dict.fromkeys(kernels))

    def prepare(self, module, arguments, *, gradients=None):
        """Resolve a frozen plan to the same native executor as kernel.prepare()."""
        from . import _c_api as abi
        from ._runtime import NdArray, PreparedCall, _argument, _name

        runtime = module._runtime
        with runtime._lock:
            module._check()
            owners = [module]
            native = (abi.NamedArgument * len(arguments))(
                *[
                    abi.NamedArgument(_name(name), _argument(runtime, value, owners))
                    for name, value in arguments.items()
                ]
            )
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
            return PreparedCall(runtime, handle, tuple(dict.fromkeys(owners)))
