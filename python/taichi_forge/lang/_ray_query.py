"""Typed, non-escaping Vulkan ray-query kernel intrinsic."""

from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl, ops
from taichi_forge.lang.expr import Expr, make_expr_group
from taichi_forge.lang.matrix import Matrix
from taichi_forge.lang.struct import StructType
from taichi_forge.lang.util import taichi_scope
from taichi_forge.types.primitive_types import f32, i32, u32


_RAY_QUERY_HIT_TYPE = StructType(
    hit=i32,
    t=f32,
    primitive_index=u32,
    instance_id=u32,
    instance_custom_index=u32,
    geometry_index=u32,
    barycentric_u=f32,
    barycentric_v=f32,
    front_face=i32,
)


class RayQueryHit:
    """Read-only, lazily projected committed triangle hit.

    The traversal itself is materialized exactly once at the call site. Each
    property projects one member from that SSA result, so the SPIR-V optimizer
    can discard committed-hit getters that the kernel never reads.
    """

    _FIELDS = {
        "hit": 0,
        "t": 1,
        "primitive_index": 2,
        "instance_id": 3,
        "instance_custom_index": 4,
        "geometry_index": 5,
        "barycentric_u": 6,
        "barycentric_v": 7,
        "front_face": 8,
    }

    def __init__(self, raw):
        self._raw = raw
        self._projected = {}

    def _ti_expr_init(self):
        return self

    def _project(self, name):
        if name not in self._projected:
            self._projected[name] = Expr(
                _ti_core.make_get_element_expr(
                    self._raw.ptr,
                    (self._FIELDS[name],),
                    _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
                )
            )
        return self._projected[name]

    @property
    def hit(self):
        return self._project("hit")

    @property
    def t(self):
        return self._project("t")

    @property
    def primitive_index(self):
        return self._project("primitive_index")

    @property
    def instance_id(self):
        return self._project("instance_id")

    @property
    def instance_custom_index(self):
        return self._project("instance_custom_index")

    @property
    def geometry_index(self):
        return self._project("geometry_index")

    @property
    def barycentric_u(self):
        return self._project("barycentric_u")

    @property
    def barycentric_v(self):
        return self._project("barycentric_v")

    @property
    def front_face(self):
        return self._project("front_face")


def _vector3_entries(value, name):
    if isinstance(value, Expr) and value.is_tensor():
        entries = [
            Expr(item)
            for item in impl.get_runtime()
            .compiling_callable.ast_builder()
            .expand_exprs([value.ptr])
        ]
    elif isinstance(value, Matrix):
        entries = value.entries
    else:
        try:
            entries = list(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be a three-component vector") from exc
    if len(entries) != 3:
        raise ValueError(f"{name} must have exactly three components")
    return entries


class AccelerationStructureAccessor:
    """Kernel-scope accessor for one bound top-level acceleration structure."""

    def __init__(self, ptr_expr):
        self.ptr_expr = ptr_expr

    @taichi_scope
    def trace_closest(
        self,
        origin,
        direction,
        t_min=0.0,
        t_max=1.0e30,
        *,
        ray_flags=0,
        cull_mask=0xFF,
    ):
        """Trace one ray and return a typed committed triangle hit.

        The query state is created and fully consumed inside this operation;
        it cannot escape the expression or be stored in a field. Current
        support is intentionally limited to Forge opaque triangle BLAS/TLAS.
        """

        origin = [ops.cast(value, f32) for value in _vector3_entries(origin, "origin")]
        direction = [ops.cast(value, f32) for value in _vector3_entries(direction, "direction")]
        args = make_expr_group(
            self.ptr_expr,
            *origin,
            *direction,
            ops.cast(t_min, f32),
            ops.cast(t_max, f32),
            ops.cast(ray_flags, u32),
            ops.cast(cull_mask, u32),
        )
        raw = Expr(
            _ti_core.insert_materialized_internal_func_call(
                _ti_core.InternalOp.vulkan_ray_query_closest, args
            )
        )
        # Insert the non-escaping query at the source call site. Projected
        # members later reuse this dominating SSA statement without an alloca.
        impl.get_runtime().compiling_callable.ast_builder().insert_expr_stmt(raw.ptr)
        return RayQueryHit(raw)


__all__ = ["AccelerationStructureAccessor", "RayQueryHit"]
