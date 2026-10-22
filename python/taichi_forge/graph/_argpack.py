"""Logical Graph pack descriptions and ordinary managed runtime bindings.

The compiler's ArgumentOrigins determines physical slots. This module only
describes members and keeps their owners alive; it never computes byte offsets.
"""

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.argpack import ArgPack, ArgPackType
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.types._argument_descriptor import describe_annotation


def symbolic_pack(name, annotation):
    if not isinstance(annotation, ArgPackType):
        raise TaichiRuntimeError("ArgKind.ARGPACK requires a ti.types.argpack dtype")
    from taichi_forge.graph._graph import Arg

    members = []
    for member, dtype in annotation.members.items():
        description = describe_annotation(dtype)
        kind = description.kind
        if kind == "argpack":
            symbol = symbolic_pack(member, dtype)
        elif kind == "struct":
            symbol = _ti_core.Arg(_ti_core.ArgKind.STRUCT, member, dtype.dtype, 0, [])
        elif kind in ("scalar", "matrix"):
            symbol = Arg(getattr(_ti_core.ArgKind, kind.upper()), member, dtype)
        elif kind == "ndarray":
            if description.ndim is None or not description.element.is_complete:
                raise TaichiRuntimeError(f"Graph ArgPack {name}.{member} requires concrete ndarray dtype and ndim")
            symbol = Arg(_ti_core.ArgKind.NDARRAY, member, dtype.dtype, dtype.ndim)
            if dtype.needs_grad is not None:
                symbol = symbol._with_needs_grad(bool(dtype.needs_grad))
        elif kind in ("texture", "rw_texture", "texture_collection"):
            tag = {"texture": "TEXTURE", "rw_texture": "RWTEXTURE", "texture_collection": "TEXTURE_COLLECTION"}[kind]
            options = dict(tag=getattr(_ti_core.ArgKind, tag), name=member, ndim=description.ndim)
            if kind == "rw_texture":
                options["fmt"] = description.fmt
            elif kind == "texture_collection":
                options["capacity"] = description.capacity
            symbol = Arg(**options)
        elif kind == "acceleration_structure":
            symbol = Arg(_ti_core.ArgKind.ACCELERATION_STRUCTURE, member)
        else:
            raise TaichiRuntimeError(f"Unsupported Graph ArgPack member {name}.{member}: {kind}")
        members.append(symbol)
    return _ti_core.Arg(name, annotation.dtype, members)


def runtime_pack(value, arch):
    from taichi_forge.lang._ndarray import Ndarray, StructNdarray
    from taichi_forge.lang._storage_view import DenseNdarrayView, ndarray_view
    from taichi_forge.lang._texture import Texture, TextureCollection
    from taichi_forge.lang.field import ScalarField
    from taichi_forge.lang.matrix import MatrixField
    from taichi_forge.types.ray_type import _AccelerationStructureResource

    native = value._ArgPack__argpack
    if native is None or value._ArgPack__runtime_prog is not impl.get_runtime().prog:
        raise TaichiRuntimeError("Cannot submit an ArgPack after its Taichi runtime has been reset")
    members = {}
    for (name, member), annotation in zip(value.items, value._annotations):
        kind = describe_annotation(annotation).kind
        if kind in ("scalar", "matrix", "struct"):
            continue  # Values already reside in the managed packed storage.
        if kind == "argpack":
            if not isinstance(member, ArgPack):
                raise TaichiRuntimeError(f"Graph ArgPack member {name} requires an ArgPack")
            members[name] = runtime_pack(member, arch)
        elif kind == "ndarray":
            if isinstance(member, Ndarray):
                if member.arr is None:
                    raise TaichiRuntimeError(f"Graph ArgPack member {name} has been reset")
                members[name] = (
                    member.arr
                    if isinstance(member, StructNdarray)
                    else (
                        member.arr,
                        member._runtime_storage_argument("graph_replay", "replay"),
                        member.grad.arr if member.grad is not None else None,
                    )
                )
            elif isinstance(member, (DenseNdarrayView, ScalarField, MatrixField)):
                view = member if isinstance(member, DenseNdarrayView) else ndarray_view(member)
                members[name] = (view, view._runtime_storage_argument("graph_replay", "replay"))
            else:
                raise TaichiRuntimeError(f"Graph ArgPack member {name} requires a managed ndarray or dense view")
        elif kind in ("texture", "rw_texture") and isinstance(member, Texture) and member.tex is not None:
            members[name] = member.tex
        elif kind == "texture_collection" and isinstance(member, TextureCollection) and member.collection is not None:
            members[name] = member.collection
        elif kind == "acceleration_structure" and isinstance(member, _AccelerationStructureResource):
            member._validate_lifetime()
            members[name] = (member._runtime_prog, member._kernel_resource_descriptor().handle)
        else:
            raise TaichiRuntimeError(f"Graph ArgPack member {name} has an invalid or retired {kind} binding")
    return native, value._ArgPack__dtype, members
