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


def test_value_encoding_accepts_numpy_views_with_compiled_padding():
    import numpy as np

    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    for value in (source[0, ::2], source.T[::2, ::-2]):
        count = value.size
        offsets = tuple(range(0, count * 8, 8))
        encode = _encoder(leaf("f32", 0, value.shape, component_offsets=offsets), count * 8)
        buffer = C.create_string_buffer(count * 8)
        encode(buffer, value)
        expected = bytearray(count * 8)
        for offset, component in zip(offsets, value.flat):
            struct.pack_into("=f", expected, offset, component)
        assert buffer.raw == expected
        # Encoded values are snapshots, not aliases to the caller's array.
        value.fill(-17)
        assert buffer.raw == expected
        with pytest.raises(ValueError, match="shape"):
            encode(buffer, np.zeros((count, 1), dtype=np.float32))
        with pytest.raises(ValueError, match="shape"):
            encode(buffer, np.array(1.0))


def test_value_encoding_keeps_numpy_scalar_conversion_rules():
    import numpy as np

    encode = _encoder(leaf("i8", 0, (2,)), 2)
    buffer = C.create_string_buffer(2)
    encode(buffer, np.array([-128, 127], dtype=np.int64))
    assert buffer.raw == struct.pack("=bb", -128, 127)
    with pytest.raises(ValueError, match="range"):
        encode(buffer, np.array([0, 128], dtype=np.int64))
    with pytest.raises(TypeError):
        encode(buffer, np.array([0.0, 1.0], dtype=np.float32))
