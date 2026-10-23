"""Cold producer facts; these tests do not initialize a device."""

import ctypes
from dataclasses import FrozenInstanceError, replace

import pytest

from taichi_forge_runtime.aot import VulkanExternalBufferDescriptor


def descriptor(**changes):
    facts = dict(memory_handle=11, ready_for_consumer=12, ready_for_external=13,
                 buffer_size=112, allocation_size=256, bind_offset=0,
                 memory_type_index=2, buffer_flags=0, buffer_usage=0x23,
                 allocation_flags=0, sharing_mode=0, dedicated=True,
                 device_uuid=bytes(range(16)), driver_uuid=bytes(range(16, 32)))
    facts.update(changes)
    return VulkanExternalBufferDescriptor(**facts)


def test_frozen_producer_facts_and_finite_view():
    uuid = bytearray(range(16))
    facts = descriptor(device_uuid=uuid)
    uuid[:] = bytes(16)
    assert facts.device_uuid == bytes(range(16))
    with pytest.raises(FrozenInstanceError):
        facts.buffer_size = 16
    info = facts._native_view(64, 48)
    assert ctypes.sizeof(info) == 128
    assert info.struct_size == 128 and info.schema_version == 1
    assert (info.buffer_size, info.byte_offset, info.byte_size) == (112, 64, 48)
    assert bytes(info.device_uuid) == facts.device_uuid
    assert info.buffer_usage == 0x23 and info.allocation_flags == 0
    for offset, size in ((64, 49), (113, 1), (0, 0), (-1, 1), (1 << 64, 1)):
        with pytest.raises(ValueError):
            facts._native_view(offset, size)
    with pytest.raises(TypeError):
        facts._native_view(False, 4)


def test_invalid_facts_cannot_silently_change_native_flags():
    for changes in (
        dict(memory_handle=0), dict(ready_for_consumer=13),
        dict(allocation_size=100), dict(buffer_size=1 << 64),
        dict(memory_type_index=1 << 32), dict(buffer_usage=-1),
        dict(device_uuid=bytes(16)), dict(driver_uuid=b"short"),
        dict(sharing_mode=1), dict(bind_offset=4), dict(dedicated=False),
    ):
        with pytest.raises(ValueError):
            descriptor(**changes)
    with pytest.raises(TypeError):
        descriptor(dedicated=1)
    # Address flags are producer facts, not inferred from consumer hardware.
    facts = replace(descriptor(), buffer_usage=0x20023, allocation_flags=2)
    info = facts._native_view(0, 4)
    assert info.buffer_usage == 0x20023 and info.allocation_flags == 2
