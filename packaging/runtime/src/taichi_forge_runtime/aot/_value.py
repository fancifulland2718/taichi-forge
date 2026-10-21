"""Cold encoding/decoding plans for compiler-owned values; no DSL dependency."""

from collections.abc import Mapping, Sequence
import ctypes as C
import json
import math
import numbers
import operator
import struct

from . import _c_api as abi


def _scalar_format(node):
    formats = {
        "f16": "e", "f32": "f", "f64": "d",
        "i8": "b", "i16": "h", "i32": "i", "i64": "q",
        "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
    }
    dtype, slot = node["dtype"], node["scalar_size"]
    format = {1: "B", 4: "I"}[slot] if dtype == "u1" else formats[dtype]
    if struct.calcsize("=" + format) != slot:
        raise ValueError("Compiled scalar slot has an inconsistent size")
    return dtype, slot, format


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
    dtype, slot, format = _scalar_format(node)
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


def _encoder(node, buffer_size):
    kind = node["kind"]
    if kind == "struct":
        members = tuple(
            (member["name"], _encoder(member["value"], buffer_size))
            for member in node["members"]
        )
        names = frozenset(name for name, _ in members)
        if len(names) != len(members):
            raise ValueError("Duplicate compiled value member")

        def structure(buffer, value):
            if not isinstance(value, Mapping):
                raise TypeError("A structure argument requires a mapping")
            if names != value.keys():
                raise ValueError("Structure argument members must match the exported type")
            for name, encode in members:
                encode(buffer, value[name])

        return structure
    if kind not in ("scalar", "tensor"):
        raise ValueError("Unsupported compiled value kind")
    dtype, slot, format = _scalar_format(node)
    shape = tuple(node["shape"])
    if any(type(n) is not int or n <= 0 for n in shape) or bool(shape) != (kind == "tensor"):
        raise ValueError("Invalid compiled tensor shape")
    count, offset = math.prod(shape), node["offset"]
    offsets = tuple(node.get("component_offsets", ()))
    if offsets:
        if len(offsets) != count or offsets[0] != offset or any(
            type(position) is not int or position < 0 or position + slot > buffer_size
            for position in offsets
        ) or any(right < left + slot for left, right in zip(offsets, offsets[1:])):
            raise ValueError("Invalid compiled tensor component offsets")
    elif offset < 0 or offset + count * slot > buffer_size:
        raise ValueError("Compiled value exceeds its argument buffer")

    if dtype.startswith("f"):
        def convert(value):
            if not isinstance(value, numbers.Real):
                raise TypeError("Floating-point value members require real numbers")
            return float(value)
    else:
        bits = 1 if dtype == "u1" else slot * 8
        signed = dtype.startswith("i")
        minimum = -(1 << (bits - 1)) if signed else 0
        maximum = (1 << (bits - int(signed))) - 1

        def convert(value):
            result = operator.index(value)
            if not minimum <= result <= maximum:
                raise ValueError(f"Value member is outside {dtype} range")
            return result

    def values(value, dimensions):
        if not dimensions:
            return (convert(value),)
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise TypeError("Tensor arguments require nested sequences matching their shape")
        if len(value) != dimensions[0]:
            raise ValueError("Tensor argument shape does not match the exported type")
        return tuple(item for child in value for item in values(child, dimensions[1:]))

    if offsets:
        pack = struct.Struct("=" + format).pack_into

        def encode(buffer, value):
            flat = values(value, shape)
            for position, component in zip(offsets, flat):
                pack(buffer, position, component)
    else:
        pack = struct.Struct("=" + str(count) + format).pack_into

        def encode(buffer, value):
            pack(buffer, offset, *values(value, shape))

    return encode


class ArgumentLayout:
    """Copied logical shapes and reusable encoders for one loaded Kernel."""

    def __init__(self, module, handle):
        runtime = module._runtime
        info = abi.KernelArgumentInfo()
        runtime._api.call(
            runtime._api.ti_get_kernel_argument_info_ext,
            runtime._handle, handle, C.byref(info),
        )
        layout = json.loads(info.layout_json)
        if layout["version"] != 1 or layout["argument_count"] != info.argument_count:
            raise ValueError("Unsupported compiled argument layout")
        self.count = info.argument_count
        self.encoders = {}
        for value in layout["values"]:
            index, size = value["index"], value["buffer_size"]
            if type(index) is not int or not 0 <= index < self.count or index in self.encoders:
                raise ValueError("Invalid compiled value argument index")
            if type(size) is not int or size < 0:
                raise ValueError("Invalid compiled value argument size")
            self.encoders[index] = size, _encoder(value["value"], size)

    def encode(self, index, value, buffers):
        size, encode = self.encoders[index]
        buffer = C.create_string_buffer(size)
        encode(buffer, value)
        argument = abi.Argument()
        argument.type = 7
        argument.value.value = abi.ValueArgument(size, C.cast(buffer, C.c_void_p))
        buffers.append(buffer)
        return argument


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
