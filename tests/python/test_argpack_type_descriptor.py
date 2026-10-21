"""Logical resource declarations must not collapse to the packed value dtype."""
import pytest
import taichi_forge as ti
from taichi_forge.lang.argpack import ArgPack
from taichi_forge.types._argument_descriptor import describe_annotation


def test_argpack_descriptor_distinguishes_lifted_resource_contracts():
    def pack(dtype, ndim=1, needs_grad=False):
        return ti.types.argpack(gain=ti.f32, child=ti.types.argpack(
            values=ti.types.ndarray(dtype, ndim, needs_grad=needs_grad)))

    declarations = [pack(ti.f32), pack(ti.i32), pack(ti.f32, 2), pack(ti.f32, needs_grad=True)]
    # Lifted resources are absent from the native packed value type.
    assert all(item.dtype == declarations[0].dtype for item in declarations)
    keys = [describe_annotation(item).structural_key() for item in declarations]
    assert len(set(keys)) == len(keys)
    assert describe_annotation(pack(ti.f32)).structural_key() == keys[0]
    original = describe_annotation(declarations[0])
    declarations[0].members["gain"] = ti.i32
    assert original.structural_key() == keys[0]
    assert describe_annotation(declarations[0]).structural_key() != keys[0]


def test_argpack_descriptor_allocates_no_pack_and_preserves_member_order(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Type reflection must not allocate a live ArgPack")
    monkeypatch.setattr(ArgPack, "__init__", unexpected)
    members = {f"value_{i}": ti.i32 for i in range(65)}
    members["matrix"] = ti.types.matrix(9, 9, ti.f32)
    members["image"] = ti.types.texture(2)
    description = describe_annotation(ti.types.argpack(**members))
    assert tuple(name for name, _ in description.members) == tuple(members)
    assert description.members[-2][1].element.shape == (9, 9)
    assert description.members[-1][1].kind == "texture"
    assert description.members[-1][1].ndim == 2
    reordered = describe_annotation(ti.types.argpack(**dict(reversed(list(members.items())))))
    assert reordered.structural_key() != description.structural_key()
    scalar = describe_annotation(ti.f32)
    assert len(scalar.structural_key()) == 6  # Existing non-pack cache keys stay stable.


def test_argpack_descriptor_rejects_cycles_but_allows_shared_nested_types():
    shared = ti.types.argpack(value=ti.f32)
    root = ti.types.argpack(left=shared, right=shared)
    description = describe_annotation(root)
    assert description.members[0][1] == description.members[1][1]
    shared.members["cycle"] = root
    with pytest.raises(ValueError, match="Cyclic ArgPack"):
        describe_annotation(root)
