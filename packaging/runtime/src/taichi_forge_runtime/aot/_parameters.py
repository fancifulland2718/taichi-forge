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

    def snapshot(self, values):
        if values is None:
            values = {}
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping of declared u64 parameters")
        frozen = dict(values)
        if frozen.keys() != self._keys:
            raise ValueError("values must contain exactly the declared mutable_parameters")
        result = []
        for name in self.names:
            value = frozen[name]
            if isinstance(value, bool):
                raise TypeError("Submission parameter values must be u64 integers")
            value = operator.index(value)
            if not 0 <= value < (1 << 64):
                raise ValueError("Submission parameter value is outside u64")
            result.append(value)
        return result

    def bind(self, snapshot):
        # Caller holds the prepared runtime lock through the native copy.
        for index, value in enumerate(snapshot):
            self.buffer[index] = value
        return self.buffer
