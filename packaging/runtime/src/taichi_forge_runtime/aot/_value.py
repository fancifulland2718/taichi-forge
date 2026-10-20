"""Cold decoding plans for compiler-owned value layouts; no DSL dependency."""

import ctypes as C
import json
import math
import struct

from . import _c_api as abi


def _decoder(node, buffer_size):
    kind = node["kind"]
    if kind == "struct":
        members = tuple(
            (member["name"], _decoder(member["value"], buffer_size))
            for member in node["members"]
        )
        if len({name for name, _ in members}) != len(members):
            raise ValueError("Duplicate compiled value member")
        return lambda buffer: {name: decode(buffer) for name, decode in members}
    if kind not in ("scalar", "tensor"):
        raise ValueError("Unsupported compiled value kind")
    formats = {
        "f16": "e",
        "f32": "f",
        "f64": "d",
        "i8": "b",
        "i16": "h",
        "i32": "i",
        "i64": "q",
        "u8": "B",
        "u16": "H",
        "u32": "I",
        "u64": "Q",
    }
    dtype, slot = node["dtype"], node["scalar_size"]
    if dtype == "u1":
        format = {1: "B", 4: "I"}[slot]
    else:
        format = formats[dtype]
    if struct.calcsize("=" + format) != slot:
        raise ValueError("Compiled scalar slot has an inconsistent size")
    shape = tuple(node["shape"])
    if any(type(n) is not int or n <= 0 for n in shape) or bool(shape) != (
        kind == "tensor"
    ):
        raise ValueError("Invalid compiled tensor shape")
    count = math.prod(shape)
    offset = node["offset"]
    if offset < 0 or offset + count * slot > buffer_size:
        raise ValueError("Compiled value exceeds its result buffer")
    unpack = struct.Struct("=" + str(count) + format).unpack_from
    if not shape:
        if dtype == "u1":
            return lambda buffer: bool(unpack(buffer, offset)[0])
        return lambda buffer: unpack(buffer, offset)[0]

    def nesting(dimensions):
        length = dimensions[0]
        if len(dimensions) == 1:
            return lambda flat, start: flat[start : start + length]
        child = nesting(dimensions[1:])
        stride = math.prod(dimensions[1:])
        return lambda flat, start: tuple(
            child(flat, start + i * stride) for i in range(length)
        )

    reshape = nesting(shape)

    def tensor(buffer):
        flat = unpack(buffer, offset)
        if dtype == "u1":
            flat = tuple(bool(value) for value in flat)
        return reshape(flat, 0)

    return tensor


class ReturnLayout:
    """Copied metadata and a reusable decoder; queried once per Kernel owner."""

    def __init__(self, module, handle):
        runtime = module._runtime
        info = abi.KernelReturnInfo()
        runtime._api.call(
            runtime._api.ti_get_kernel_return_info_ext,
            runtime._handle,
            handle,
            C.byref(info),
        )
        layout = json.loads(info.layout_json)
        if layout["version"] != 1 or layout["buffer_size"] != info.buffer_size:
            raise ValueError("Unsupported compiled return layout")
        self.size = info.buffer_size
        self._decoders = tuple(
            _decoder(value, self.size) for value in layout["results"]
        )
        self.has_values = bool(self._decoders)

    def decode(self, buffer):
        if not self._decoders:
            return None
        if len(self._decoders) == 1:
            return self._decoders[0](buffer)
        return tuple(decode(buffer) for decode in self._decoders)
