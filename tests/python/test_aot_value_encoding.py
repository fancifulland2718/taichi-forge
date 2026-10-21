"""Compiler layout encoders preserve padding and validate user value shapes."""

import ctypes as C
import struct

import pytest

from taichi_forge_runtime.aot._value import _encoder


def leaf(dtype, offset, shape=(), **extra):
    return dict(kind="tensor" if shape else "scalar", dtype=dtype, offset=offset,
                shape=shape, scalar_size={"f32": 4, "u1": 4, "i8": 1}[dtype], **extra)


def test_value_encoding_preserves_compiled_padding_and_shape():
    encode = _encoder(dict(kind="struct", members=[
        dict(name="tag", value=leaf("i8", 0)),
        dict(name="enabled", value=leaf("u1", 4)),
        dict(name="column", value=leaf("f32", 16, (3, 1), component_offsets=(16, 24, 32))),
    ]), 40)
    buffer = C.create_string_buffer(40)
    encode(buffer, dict(tag=-7, enabled=True, column=((1.5,), (-2.0,), (7.0,))))
    expected = bytearray(40)
    struct.pack_into("=b", expected, 0, -7)
    struct.pack_into("=I", expected, 4, 1)
    for position, value in ((16, 1.5), (24, -2.0), (32, 7.0)):
        struct.pack_into("=f", expected, position, value)
    assert buffer.raw == expected
    with pytest.raises((TypeError, ValueError), match="shape"):
        encode(buffer, dict(tag=-7, enabled=True, column=(1.5, -2.0, 7.0)))


def test_value_encoding_rejects_range_members_and_malformed_offsets():
    integer = _encoder(leaf("i8", 0), 1)
    with pytest.raises(ValueError, match="range"):
        integer(C.create_string_buffer(1), 128)
    with pytest.raises(TypeError):
        integer(C.create_string_buffer(1), 2.5)
    boolean = _encoder(leaf("u1", 0), 4)
    with pytest.raises(ValueError, match="range"):
        boolean(C.create_string_buffer(4), 2)
    structure = _encoder(dict(kind="struct", members=[dict(name="a", value=leaf("i8", 0))]), 1)
    for value in ({}, {"a": 2, "extra": 3}):
        with pytest.raises(ValueError, match="members"):
            structure(C.create_string_buffer(1), value)
    with pytest.raises(ValueError, match="offsets"):
        _encoder(leaf("f32", 0, (2,), component_offsets=(0, 2)), 8)
