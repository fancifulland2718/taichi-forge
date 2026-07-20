"""Private caller-qualified numeric resources for recursive block V-cycles."""

import taichi_forge as ti
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.impl import get_runtime

from ._sparse_bsr_hierarchy_candidate import _SparseBsrHierarchySnapshot
from ._sparse_hierarchy_assembly import (
    _ensure_current_program,
    _positive_version,
)
from .sparse_matrix import _require_current_scalar_ndarray


_BLOCK_VCYCLE_NUMERIC_STATUS = {
    1: "block inverse contains a non-finite value",
    2: "block inverse diagonal is not positive",
    3: "block inverse is not symmetric",
    4: "smoother damping is not finite and positive",
    5: "bottom inverse contains a non-finite value",
    6: "bottom inverse diagonal is not positive",
    7: "bottom inverse is not exactly symmetric",
    8: "numeric validation control record is incomplete",
}


@ti.kernel
def _copy_f32_segment(
    source: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination: ti.types.ndarray(dtype=ti.f32, ndim=1),
    destination_offset: ti.i32,
    count: ti.i32,
):
    for index in range(count):
        destination[destination_offset + index] = source[index]


@ti.kernel
def _validate_packed_block_inverses(
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    value_offset: ti.i32,
    block_rows: ti.i32,
    block_size: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for block_row in range(block_rows):
        block_base = (
            value_offset + block_row * block_size * block_size
        )
        for local_row in range(block_size):
            diagonal = inverse_values[
                block_base + local_row * block_size + local_row
            ]
            if ti.math.isnan(diagonal) or ti.math.isinf(diagonal):
                ti.atomic_max(control[0], 1)
            elif diagonal <= 0.0:
                ti.atomic_max(control[0], 2)
            for local_column in range(local_row + 1, block_size):
                left = inverse_values[
                    block_base + local_row * block_size + local_column
                ]
                right = inverse_values[
                    block_base + local_column * block_size + local_row
                ]
                if (
                    ti.math.isnan(left)
                    or ti.math.isinf(left)
                    or ti.math.isnan(right)
                    or ti.math.isinf(right)
                ):
                    ti.atomic_max(control[0], 1)
                tolerance = 1e-5 * ti.max(
                    1.0, ti.max(ti.abs(left), ti.abs(right))
                )
                if ti.abs(left - right) > tolerance:
                    ti.atomic_max(control[0], 3)


@ti.kernel
def _validate_packed_bottom_inverse(
    inverse_values: ti.types.ndarray(dtype=ti.f32, ndim=1),
    value_offset: ti.i32,
    size: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(size * size):
        value = inverse_values[value_offset + index]
        row = index // size
        column = index - row * size
        if ti.math.isnan(value) or ti.math.isinf(value):
            ti.atomic_max(control[0], 5)
        if row == column and value <= 0.0:
            ti.atomic_max(control[0], 6)
        if value != inverse_values[value_offset + column * size + row]:
            ti.atomic_max(control[0], 7)


@ti.kernel
def _validate_block_vcycle_dampings(
    dampings: ti.types.ndarray(dtype=ti.f32, ndim=1),
    count: ti.i32,
    control: ti.types.ndarray(dtype=ti.i32, ndim=1),
):
    for index in range(count):
        value = dampings[index]
        if (
            value <= 0.0
            or ti.math.isnan(value)
            or ti.math.isinf(value)
        ):
            ti.atomic_max(control[0], 4)
        if index == 0:
            control[1] = count


class _SparseBlockVcycleNumericSnapshot:
    """Owned packed block inverses, dampings, and bounded bottom inverse."""

    def __init__(
        self,
        *,
        program,
        backend,
        topology_version,
        numeric_version,
        block_size,
        level_block_rows,
        level_block_nnz,
        block_inverse_offsets,
        bottom_inverse_offset,
        bottom_scalar_size,
        inverse_values,
        dampings,
        validation_control_readback_bytes,
        device_to_device_bytes,
    ):
        self._program = program
        self._backend = backend
        self.topology_version = int(topology_version)
        self.numeric_version = int(numeric_version)
        self.block_size = int(block_size)
        self._level_block_rows = tuple(
            int(value) for value in level_block_rows
        )
        self._level_block_nnz = tuple(
            int(value) for value in level_block_nnz
        )
        self._block_inverse_offsets = tuple(
            int(value) for value in block_inverse_offsets
        )
        self._bottom_inverse_offset = int(bottom_inverse_offset)
        self.bottom_scalar_size = int(bottom_scalar_size)
        self._inverse_values = inverse_values
        self._dampings = dampings
        self._validation_control_readback_bytes = int(
            validation_control_readback_bytes
        )
        self._device_to_device_bytes = int(device_to_device_bytes)

    @classmethod
    def copy_validated(
        cls,
        hierarchy,
        *,
        block_inverses,
        dampings,
        bottom_inverse,
        topology_version,
        numeric_version,
    ):
        topology_version = _positive_version(
            topology_version, "block V-cycle numeric topology_version"
        )
        numeric_version = _positive_version(
            numeric_version, "block V-cycle numeric numeric_version"
        )
        if not isinstance(hierarchy, _SparseBsrHierarchySnapshot):
            raise TaichiRuntimeError(
                "block V-cycle numeric setup requires a BSR hierarchy "
                "snapshot"
            )
        hierarchy._ensure_current()
        if (
            hierarchy.topology_version != topology_version
            or hierarchy.numeric_version != numeric_version
        ):
            raise TaichiRuntimeError(
                "block V-cycle numeric versions must match the hierarchy"
            )
        try:
            block_inverses = tuple(block_inverses)
        except TypeError as exc:
            raise TaichiRuntimeError(
                "block V-cycle inverse blocks must be iterable"
            ) from exc
        nonbottom_levels = hierarchy.level_count - 1
        if len(block_inverses) != nonbottom_levels:
            raise TaichiRuntimeError(
                "block V-cycle numeric setup requires one block inverse "
                "array per non-bottom level"
            )

        checked_inverses = []
        inverse_offsets = []
        inverse_count = 0
        block_elements = hierarchy.block_size * hierarchy.block_size
        for level_index in range(nonbottom_levels):
            block_rows = hierarchy.level_block_rows[level_index]
            expected_count = block_rows * block_elements
            inverse = _require_current_scalar_ndarray(
                block_inverses[level_index],
                f"block V-cycle level {level_index} inverse blocks",
                ti.f32,
                one_dimensional=True,
            )
            if inverse.shape != (expected_count,):
                raise TaichiRuntimeError(
                    f"block V-cycle level {level_index} inverse block shape "
                    "does not match the hierarchy"
                )
            inverse_offsets.append(inverse_count)
            inverse_count += expected_count
            checked_inverses.append(inverse)

        dampings = _require_current_scalar_ndarray(
            dampings,
            "block V-cycle dampings",
            ti.f32,
            one_dimensional=True,
        )
        if dampings.shape != (nonbottom_levels,):
            raise TaichiRuntimeError(
                "block V-cycle dampings must contain one value per "
                "non-bottom level"
            )
        bottom_scalar_size = hierarchy.level_scalar_rows[-1]
        bottom_count = bottom_scalar_size * bottom_scalar_size
        bottom_inverse = _require_current_scalar_ndarray(
            bottom_inverse,
            "block V-cycle bottom inverse",
            ti.f32,
            one_dimensional=True,
        )
        if bottom_inverse.shape != (bottom_count,):
            raise TaichiRuntimeError(
                "block V-cycle bottom inverse shape does not match the "
                "bottom scalar size"
            )
        total_inverse_count = inverse_count + bottom_count
        if total_inverse_count >= 0x7FFFFFFF:
            raise TaichiRuntimeError(
                "block V-cycle packed inverse count exceeds the i32 limit"
            )

        program = get_runtime().prog
        owned_inverse_values = ti.ndarray(
            ti.f32, shape=total_inverse_count
        )
        owned_dampings = ti.ndarray(ti.f32, shape=nonbottom_levels)
        control = ti.ndarray(ti.i32, shape=2)
        control.fill(0)
        for level_index, inverse in enumerate(checked_inverses):
            count = (
                hierarchy.level_block_rows[level_index] * block_elements
            )
            _copy_f32_segment(
                inverse,
                owned_inverse_values,
                inverse_offsets[level_index],
                count,
            )
        bottom_inverse_offset = inverse_count
        _copy_f32_segment(
            bottom_inverse,
            owned_inverse_values,
            bottom_inverse_offset,
            bottom_count,
        )
        _copy_f32_segment(
            dampings, owned_dampings, 0, nonbottom_levels
        )
        for level_index in range(nonbottom_levels):
            _validate_packed_block_inverses(
                owned_inverse_values,
                inverse_offsets[level_index],
                hierarchy.level_block_rows[level_index],
                hierarchy.block_size,
                control,
            )
        _validate_packed_bottom_inverse(
            owned_inverse_values,
            bottom_inverse_offset,
            bottom_scalar_size,
            control,
        )
        _validate_block_vcycle_dampings(
            owned_dampings, nonbottom_levels, control
        )
        control_host = control.to_numpy()
        _ensure_current_program(
            program, "block V-cycle numeric construction"
        )
        status = int(control_host[0])
        if status == 0 and int(control_host[1]) != nonbottom_levels:
            status = 8
        if status != 0:
            reason = _BLOCK_VCYCLE_NUMERIC_STATUS.get(
                status, "unknown validation failure"
            )
            raise TaichiRuntimeError(
                "block V-cycle numeric validation failed before publish: "
                + reason
            )
        total_bytes = 4 * (total_inverse_count + nonbottom_levels)
        return cls(
            program=program,
            backend=hierarchy._backend,
            topology_version=topology_version,
            numeric_version=numeric_version,
            block_size=hierarchy.block_size,
            level_block_rows=hierarchy.level_block_rows,
            level_block_nnz=hierarchy.level_block_nnz,
            block_inverse_offsets=inverse_offsets,
            bottom_inverse_offset=bottom_inverse_offset,
            bottom_scalar_size=bottom_scalar_size,
            inverse_values=owned_inverse_values,
            dampings=owned_dampings,
            validation_control_readback_bytes=8,
            device_to_device_bytes=total_bytes,
        )

    def _ensure_current(self):
        _ensure_current_program(
            self._program, "block V-cycle numeric snapshot"
        )

    @property
    def nonbottom_level_count(self):
        return len(self._block_inverse_offsets)

    @property
    def block_inverse_reserved_bytes(self):
        return 4 * self._bottom_inverse_offset

    @property
    def bottom_inverse_reserved_bytes(self):
        return 4 * self.bottom_scalar_size * self.bottom_scalar_size

    @property
    def damping_reserved_bytes(self):
        return 4 * self.nonbottom_level_count

    @property
    def packed_inverse_reserved_bytes(self):
        return (
            self.block_inverse_reserved_bytes
            + self.bottom_inverse_reserved_bytes
        )

    @property
    def total_reserved_bytes(self):
        return self.packed_inverse_reserved_bytes + self.damping_reserved_bytes

    def debug_runtime_stats(self):
        self._ensure_current()
        return {
            "schema_version": 1,
            "identity": {
                "backend_family": self._backend,
                "method": "packed_block_jacobi_dense_bottom_inverse",
                "block_size": self.block_size,
                "level_block_rows": self._level_block_rows,
                "level_block_nnz": self._level_block_nnz,
                "nonbottom_level_count": self.nonbottom_level_count,
                "block_inverse_offsets": self._block_inverse_offsets,
                "bottom_inverse_offset": self._bottom_inverse_offset,
                "bottom_scalar_size": self.bottom_scalar_size,
                "topology_version": self.topology_version,
                "numeric_version": self.numeric_version,
            },
            "resources": {
                "block_inverse_reserved_bytes": (
                    self.block_inverse_reserved_bytes
                ),
                "bottom_inverse_reserved_bytes": (
                    self.bottom_inverse_reserved_bytes
                ),
                "packed_inverse_reserved_bytes": (
                    self.packed_inverse_reserved_bytes
                ),
                "damping_reserved_bytes": self.damping_reserved_bytes,
                "total_reserved_bytes": self.total_reserved_bytes,
            },
            "transfers": {
                "device_to_host_bytes": (
                    self._validation_control_readback_bytes
                ),
                "device_to_device_bytes": self._device_to_device_bytes,
                "device_payload_readback_bytes": 0,
            },
            "contract": {
                "one_packed_inverse_numeric_role": True,
                "one_packed_damping_numeric_role": True,
                "single_validation_control_readback": True,
                "finite_positive_block_diagonal_validated_on_device": True,
                "block_inverse_symmetry_validated_on_device": True,
                "bottom_inverse_exact_symmetry_validated_on_device": True,
                "full_spd_qualification_is_caller_responsibility": True,
                "host_inversion_performed": False,
                "immutable_by_internal_contract": True,
                "graph_or_solver_constructed": False,
                "public_api": False,
            },
        }
