"""Cold scalar schema and bounded host storage for independent submissions."""

import ctypes
import operator
from collections.abc import Mapping


class SubmissionParameters:
    def __init__(self, names=()):
        if isinstance(names, (str, bytes)):
            raise TypeError("mutable_parameters must be a sequence of names")
        self.names = tuple(names)
        if any(not isinstance(name, str) or not name or "\0" in name for name in self.names):
            raise ValueError("Submission parameter names must be nonempty strings without NUL")
        if len(set(self.names)) != len(self.names):
            raise ValueError("Submission parameter names must be unique")
        self._keys = frozenset(self.names)
        self.buffer = (ctypes.c_uint64 * len(self.names))()
        self._schema = ()

    def configure(self, arguments):
        # Consume the same typed C arguments as native preparation. Python int
        # shorthand and explicit Scalar therefore cannot disagree on dtype.
        if not self.names:
            return
        from ._runtime import _DTYPES

        dtypes = {code: name for name, (code, _) in _DTYPES.items()}
        provided = {entry.name.decode("utf8"): entry.argument for entry in arguments}
        schema = []
        for name in self.names:
            argument = provided.get(name)
            dtype = dtypes.get(argument.value.scalar.type) if argument is not None and argument.type == 4 else None
            if dtype not in {"i32", "u32", "i64", "u64"}:
                raise ValueError(f"Submission parameter {name!r} requires an i32/u32/i64/u64 scalar")
            bits = int(dtype[1:])
            signed = dtype.startswith("i")
            schema.append((dtype, -(1 << (bits - 1)) if signed else 0,
                           (1 << (bits - int(signed))) - 1, (1 << bits) - 1))
        self._schema = tuple(schema)

    def snapshot(self, values):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping of declared integer parameters")
        frozen = dict(values)
        if frozen.keys() != self._keys:
            raise ValueError("values must contain exactly the declared mutable_parameters")
        if len(self._schema) != len(self.names):
            raise RuntimeError("Submission parameter schema was not configured")
        result = []
        for name, (dtype, lower, upper, mask) in zip(self.names, self._schema):
            value = frozen[name]
            if isinstance(value, bool):
                raise TypeError("Submission parameter values must be integers, not bool")
            value = operator.index(value)
            if not lower <= value <= upper:
                raise ValueError(f"Submission parameter {name!r} is outside {dtype}")
            result.append(value & mask)
        return result

    def bind(self, snapshot):
        # Caller holds the prepared runtime lock through the native copy.
        for index, value in enumerate(snapshot):
            self.buffer[index] = value
        return self.buffer
