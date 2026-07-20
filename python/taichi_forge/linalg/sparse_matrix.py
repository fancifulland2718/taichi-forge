import copy
from functools import reduce

import numpy as np
from taichi_forge._lib import core as _ti_core
from taichi_forge.lang._ndarray import Ndarray, ScalarNdarray
from taichi_forge.lang.exception import TaichiRuntimeError
from taichi_forge.lang.field import Field
from taichi_forge.lang.impl import get_runtime
from taichi_forge.types import f32, i32


def _require_current_scalar_ndarray(value, role, dtype=None, one_dimensional=False):
    if not isinstance(value, ScalarNdarray):
        raise TaichiRuntimeError(
            f"{role} must be a scalar Taichi ndarray on the current runtime; "
            "no NumPy or host fallback was performed."
        )
    runtime = get_runtime()
    if value.arr is None or value._runtime_prog is not runtime.prog:
        raise TaichiRuntimeError(f"{role} cannot be used after its Taichi runtime has been reset.")
    if dtype is not None and value.dtype != dtype:
        raise TaichiRuntimeError(f"{role} must have dtype {dtype}, got {value.dtype}.")
    if one_dimensional and len(value.shape) != 1:
        raise TaichiRuntimeError(f"{role} must be one-dimensional.")
    return value


class SparsePattern:
    """An immutable sparse topology shared by one or more numeric matrices.

    SparsePattern.csr() and SparsePattern.bsr() create fixed compressed
    patterns. matrix() binds a scalar Taichi ndarray of values in the
    pattern's documented compressed order. A pattern belongs to the current
    Taichi runtime and cannot be reused after ti.reset().
    """

    def __init__(self):
        raise TaichiRuntimeError(
            "SparsePattern cannot be constructed directly; use SparsePattern.csr(...) or SparsePattern.bsr(...)."
        )

    @classmethod
    def _from_native(
        cls,
        native_pattern,
        runtime_prog,
        storage_format,
        rows,
        cols,
        num_nonzeros,
        block_rows=None,
        block_cols=None,
        block_size=None,
        block_nnz=None,
    ):
        self = cls.__new__(cls)
        self._pattern = native_pattern
        self._runtime_prog = runtime_prog
        self._storage_format = storage_format
        self._rows = rows
        self._cols = cols
        self._num_nonzeros = num_nonzeros
        self._block_rows = block_rows
        self._block_cols = block_cols
        self._block_size = block_size
        self._block_nnz = block_nnz
        return self

    @classmethod
    def csr(cls, rows, cols, row_offsets, column_indices):
        """Creates a non-empty immutable CSR pattern.

        row_offsets and column_indices must be one-dimensional scalar ti.i32
        ndarrays on the current runtime. Columns must be in range, strictly
        increasing, and unique within each row.
        """
        geometry = {"rows": rows, "cols": cols}
        for name, value in geometry.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TaichiRuntimeError(f"SparsePattern.csr {name} must be an integer, got {type(value).__name__}.")
        rows = int(rows)
        cols = int(cols)
        if rows <= 0 or cols <= 0:
            raise TaichiRuntimeError("SparsePattern.csr rows and cols must be positive.")
        row_offsets = _require_current_scalar_ndarray(
            row_offsets,
            "SparsePattern.csr row_offsets",
            i32,
            one_dimensional=True,
        )
        column_indices = _require_current_scalar_ndarray(
            column_indices,
            "SparsePattern.csr column_indices",
            i32,
            one_dimensional=True,
        )
        if row_offsets.shape != (rows + 1,):
            raise TaichiRuntimeError(
                f"SparsePattern.csr row_offsets must have shape ({rows + 1},), got {row_offsets.shape}."
            )
        runtime = get_runtime()
        native_pattern = runtime.prog._create_csr_pattern(
            rows,
            cols,
            row_offsets.arr,
            column_indices.arr,
        )
        return cls._from_native(
            native_pattern,
            runtime.prog,
            "csr",
            rows,
            cols,
            column_indices.shape[0],
        )

    @classmethod
    def bsr(
        cls,
        block_rows,
        block_cols,
        block_size,
        row_offsets,
        column_indices,
    ):
        """Creates a non-empty immutable BSR pattern.

        row_offsets and column_indices must be one-dimensional scalar ti.i32
        ndarrays on the current runtime. Columns must be in range, strictly
        increasing, and unique within each block row. Supported block sizes
        are 2, 3, 6, and 12.
        """
        geometry = {
            "block_rows": block_rows,
            "block_cols": block_cols,
            "block_size": block_size,
        }
        for name, value in geometry.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TaichiRuntimeError(f"SparsePattern.bsr {name} must be an integer, got {type(value).__name__}.")
        block_rows = int(block_rows)
        block_cols = int(block_cols)
        block_size = int(block_size)
        if block_rows <= 0 or block_cols <= 0:
            raise TaichiRuntimeError("SparsePattern.bsr block_rows and block_cols must be positive.")
        if block_size not in (2, 3, 6, 12):
            raise TaichiRuntimeError(f"SparsePattern.bsr block_size must be one of 2, 3, 6, and 12, got {block_size}.")
        row_offsets = _require_current_scalar_ndarray(
            row_offsets,
            "SparsePattern.bsr row_offsets",
            i32,
            one_dimensional=True,
        )
        column_indices = _require_current_scalar_ndarray(
            column_indices,
            "SparsePattern.bsr column_indices",
            i32,
            one_dimensional=True,
        )
        if row_offsets.shape != (block_rows + 1,):
            raise TaichiRuntimeError(
                f"SparsePattern.bsr row_offsets must have shape ({block_rows + 1},), got {row_offsets.shape}."
            )
        runtime = get_runtime()
        native_pattern = runtime.prog._create_bsr_pattern(
            block_rows,
            block_cols,
            block_size,
            row_offsets.arr,
            column_indices.arr,
        )
        return cls._from_native(
            native_pattern,
            runtime.prog,
            "bsr",
            block_rows * block_size,
            block_cols * block_size,
            column_indices.shape[0] * block_size * block_size,
            block_rows,
            block_cols,
            block_size,
            column_indices.shape[0],
        )

    def _ensure_valid(self):
        if self._pattern is None or self._runtime_prog is not get_runtime().prog:
            raise TaichiRuntimeError("SparsePattern cannot be used after its Taichi runtime has been reset.")

    def __del__(self):
        try:
            self._pattern = None
            self._runtime_prog = None
        except Exception:
            pass

    @property
    def shape(self):
        """The scalar matrix shape represented by this pattern."""
        return (self._rows, self._cols)

    def _require_bsr(self, property_name):
        if self._storage_format != "bsr":
            raise TaichiRuntimeError(
                f"SparsePattern.{property_name} is available for BSR patterns only, got {self._storage_format}."
            )

    @property
    def block_shape(self):
        """The number of block rows and block columns."""
        self._require_bsr("block_shape")
        return (self._block_rows, self._block_cols)

    @property
    def block_size(self):
        """The dense square block width."""
        self._require_bsr("block_size")
        return self._block_size

    @property
    def num_blocks(self):
        """The number of stored dense blocks."""
        self._require_bsr("num_blocks")
        return self._block_nnz

    @property
    def num_nonzeros(self):
        """The number of stored scalar values."""
        return self._num_nonzeros

    @property
    def storage_format(self):
        """The compressed sparse storage format."""
        return self._storage_format

    def matrix(self, values):
        """Creates a numeric SparseMatrix that shares this pattern."""
        return SparseMatrix.from_pattern(self, values)

    def _debug_runtime_stats(self):
        """Returns private pattern resource and lifecycle telemetry."""
        self._ensure_valid()
        snapshot = dict(self._pattern._debug_runtime_stats())
        for section in ("identity", "lifecycle", "resources", "transfers"):
            snapshot[section] = dict(snapshot[section])
        return snapshot


class SparseMatrix:
    """Taichi's Sparse Matrix class

    A sparse matrix allows the programmer to solve a large linear system.

    Args:
        n (int): the first dimension of a sparse matrix.
        m (int): the second dimension of a sparse matrix.
        sm (SparseMatrix): another sparse matrix that will be built from.
    """

    def __init__(self, n=None, m=None, sm=None, dtype=f32, storage_format="col_major"):
        runtime = get_runtime()
        self._runtime_prog = runtime.prog
        self._format_contract_cache = None
        if sm is None:
            self.dtype = dtype
            self.n = n
            self.m = m if m else n
            self.matrix = self._runtime_prog.create_sparse_matrix(n, m, dtype, storage_format)
        else:
            self.dtype = sm.get_data_type()
            self.n = sm.num_rows()
            self.m = sm.num_cols()
            self.matrix = sm

    @classmethod
    def from_pattern(cls, pattern, values):
        """Creates a fixed-pattern sparse matrix from runtime-resident values.

        The matrix shares immutable index storage with pattern and owns an
        independent numeric value buffer. values must be a scalar Taichi
        ndarray on the same current runtime.
        """
        if not isinstance(pattern, SparsePattern):
            raise TaichiRuntimeError("SparseMatrix.from_pattern expects a SparsePattern.")
        pattern._ensure_valid()
        values = _require_current_scalar_ndarray(
            values,
            "SparseMatrix.from_pattern values",
            one_dimensional=True,
        )
        prog = get_runtime().prog
        if pattern.storage_format == "csr":
            core = prog._create_csr_matrix_from_pattern(pattern._pattern, values.arr)
        elif pattern.storage_format == "bsr":
            core = prog._create_bsr_matrix_from_pattern(pattern._pattern, values.arr)
        else:
            raise TaichiRuntimeError(
                f"Unsupported SparsePattern storage format {pattern.storage_format!r}; no fallback was performed."
            )
        return cls(sm=core)

    def _require_operation(self, operation):
        self._ensure_valid()
        if self._format_contract_cache is None:
            self._get_format_contract()
        if not self._format_contract_cache["operations"].get(operation, False):
            identity = self._format_contract_cache["identity"]
            raise TaichiRuntimeError(
                f"SparseMatrix operation '{operation}' is not supported for "
                f"{identity['backend_family']} {identity['storage_format']} "
                "storage; no fallback was performed."
            )

    def _ensure_valid(self):
        if self.matrix is None or self._runtime_prog is not get_runtime().prog:
            raise TaichiRuntimeError("SparseMatrix cannot be used after its Taichi runtime has been reset.")

    def __del__(self):
        try:
            # Native Vulkan sparse matrices retire Program-owned ndarrays in
            # their destructor. Release the matrix while its Program is still
            # strongly referenced by this wrapper.
            self.matrix = None
            self._runtime_prog = None
            self._format_contract_cache = None
        except Exception:
            pass

    def __iadd__(self, other):
        """Addition operation for sparse matrix.

        Returns:
            The result sparse matrix of the addition.
        """
        self._require_operation("inplace_add_sub")
        other._ensure_valid()
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        self.matrix += other.matrix
        return self

    def __add__(self, other):
        """Addition operation for sparse matrix.

        Returns:
            The result sparse matrix of the addition.
        """
        self._require_operation("matrix_add_sub")
        other._ensure_valid()
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix + other.matrix
        return SparseMatrix(sm=sm)

    def __isub__(self, other):
        """Subtraction operation for sparse matrix.

        Returns:
             The result sparse matrix of the subtraction.
        """
        self._require_operation("inplace_add_sub")
        other._ensure_valid()
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        self.matrix -= other.matrix
        return self

    def __sub__(self, other):
        """Subtraction operation for sparse matrix.

        Returns:
             The result sparse matrix of the subtraction.
        """
        self._require_operation("matrix_add_sub")
        other._ensure_valid()
        assert (
            self.n == other.n and self.m == other.m
        ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
        sm = self.matrix - other.matrix
        return SparseMatrix(sm=sm)

    def __mul__(self, other):
        """Sparse matrix's multiplication against real numbers or the hadamard product against another matrix

        Args:
            other (float or SparseMatrix): the other operand of multiplication.
        Returns:
            The result of multiplication.
        """
        self._ensure_valid()
        if isinstance(other, float):
            self._require_operation("scalar_scale")
            sm = other * self.matrix
            return SparseMatrix(sm=sm)
        if isinstance(other, SparseMatrix):
            self._require_operation("matrix_hadamard")
            other._ensure_valid()
            assert (
                self.n == other.n and self.m == other.m
            ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
            sm = self.matrix * other.matrix
            return SparseMatrix(sm=sm)

        return None

    def __rmul__(self, other):
        """Right scalar multiplication for sparse matrix.

        Args:
            other (float): the other operand of scalar multiplication.
        Returns:
            The result of multiplication.
        """
        self._ensure_valid()
        if isinstance(other, float):
            self._require_operation("scalar_scale")
            sm = self.matrix * other
            return SparseMatrix(sm=sm)

        return None

    def transpose(self):
        """Sparse Matrix transpose.

        Returns:
            The transposed sparse mastrix.
        """
        self._require_operation("transpose")
        sm = self.matrix.transpose()
        return SparseMatrix(sm=sm)

    def __matmul__(self, other):
        """Matrix multiplication.

        Args:
            other (SparseMatrix, Field, or numpy.array): the other sparse matrix of the multiplication.
        Returns:
            The result of matrix multiplication.
        """
        self._ensure_valid()
        if isinstance(other, SparseMatrix):
            self._require_operation("matrix_matmul")
            other._ensure_valid()
            assert (
                self.m == other.n
            ), f"Dimension mismatch between sparse matrices ({self.n}, {self.m}) and ({other.n}, {other.m})"
            sm = self.matrix.matmul(other.matrix)
            return SparseMatrix(sm=sm)
        if isinstance(other, Field):
            self._require_operation("field_spmv_via_host")
            assert (
                self.m == other.shape[0]
            ), f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
            return self.matrix.mat_vec_mul(other.to_numpy())
        if isinstance(other, np.ndarray):
            self._require_operation("numpy_spmv")
            assert (
                self.m == other.shape[0]
            ), f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
            return self.matrix.mat_vec_mul(other)
        if isinstance(other, Ndarray):
            self._require_operation("ndarray_spmv")
            if self.m != other.shape[0]:
                raise TaichiRuntimeError(
                    f"Dimension mismatch between sparse matrix ({self.n}, {self.m}) and vector ({other.shape})"
                )
            res = ScalarNdarray(dtype=other.dtype, arr_shape=(self.n,))
            self.matrix.spmv(get_runtime().prog, other.arr, res.arr)
            return res
        raise TaichiRuntimeError(
            f"Sparse matrix-matrix/vector multiplication does not support {type(other)} for now. Supported types are SparseMatrix, ti.field, and numpy ndarray."
        )

    def __getitem__(self, indices):
        self._require_operation("element_read")
        return self.matrix.get_element(indices[0], indices[1])

    def __setitem__(self, indices, value):
        self._require_operation("element_write")
        self.matrix.set_element(indices[0], indices[1], value)

    def __str__(self):
        """Python scope matrix print support."""
        self._require_operation("to_string")
        return self.matrix.to_string()

    def __repr__(self):
        return self.__str__()

    @property
    def shape(self):
        """The shape of the sparse matrix."""
        return (self.n, self.m)

    def _num_nonzero(self):
        """Returns the number of stored scalar values."""
        self._ensure_valid()
        return self.matrix.num_nonzero()

    def update_values(self, values):
        """Updates compressed values without rebuilding the sparse pattern.

        Values must be a scalar Taichi ndarray on the current runtime with one
        entry per stored scalar, ordered exactly like the matrix compressed
        storage. For BSR this is block-row-major with row-major values inside
        each dense block. Row and column indices, provider descriptors, and
        persistent SpMV resources remain unchanged.
        """
        self._require_operation("value_update")
        values = _require_current_scalar_ndarray(
            values,
            "SparseMatrix.update_values values",
            one_dimensional=True,
        )
        self.matrix.update_values(get_runtime().prog, values.arr)

    def _update_values(self, values):
        """Compatibility alias for the former private value-update entry."""
        self.update_values(values)

    def _debug_runtime_stats(self):
        """Returns private operator-owned resource and operation telemetry."""
        self._ensure_valid()
        snapshot = dict(self.matrix._debug_runtime_stats())
        for section in (
            "identity",
            "operations",
            "resources",
            "transfers",
            "provider",
        ):
            snapshot[section] = dict(snapshot[section])
        if snapshot["provider"]["library_version"] is not None:
            snapshot["provider"]["library_version"] = dict(snapshot["provider"]["library_version"])
        return snapshot

    def _get_format_contract(self):
        """Returns the private backend-neutral sparse format capability contract."""
        if self._format_contract_cache is not None:
            return copy.deepcopy(self._format_contract_cache)
        stats = self._debug_runtime_stats()
        identity = stats["identity"]
        operations = stats["operations"]
        resources = stats["resources"]
        backend = identity["backend_family"]
        storage_format = identity["storage_format"]
        provider_name = stats["provider"]["name"]
        is_bsr = storage_format == "bsr"
        is_cpu_eigen = provider_name == "eigen" and storage_format in (
            "csr",
            "csc",
        )
        is_cpu_fixed_csr = (
            backend == "cpu"
            and storage_format == "csr"
            and provider_name == "forge_cpu_native"
        )
        is_cpu_public_fixed_csr = (
            is_cpu_fixed_csr
            and resources["pattern_storage_shared"]
            and operations["pattern_builds"] == 0
            and identity["rows"] == identity["cols"]
        )
        is_cpu_fixed_bsr = (
            backend == "cpu"
            and storage_format == "bsr"
            and provider_name == "forge_cpu_native"
            and resources["pattern_storage_shared"]
            and operations["pattern_builds"] == 0
            and identity["rows"] == identity["cols"]
        )
        is_cuda_fixed_bsr = (
            backend == "cuda"
            and storage_format == "bsr"
            and resources["pattern_storage_shared"]
            and operations["pattern_builds"] == 0
            and identity["rows"] == identity["cols"]
        )
        is_cuda_csr = backend == "cuda" and storage_format == "csr"
        is_vulkan_csr = backend == "vulkan" and storage_format == "csr"
        scalar_public = is_cpu_eigen or is_cuda_csr or is_vulkan_csr
        scalar_algebra = is_cpu_eigen or is_cuda_csr

        self._format_contract_cache = {
            "schema_version": 1,
            "identity": {
                "backend_family": backend,
                "storage_format": storage_format,
                "dtype": identity["dtype"],
                "shape": (identity["rows"], identity["cols"]),
                "index_dtype": "i32",
                "block_size": identity["block_size"],
            },
            "pattern": {
                "ownership": ("shared_immutable" if resources["pattern_storage_shared"] else "operator_copy"),
                "mutability": ("provider_mutable" if is_cpu_eigen else "fixed"),
                "canonical_compressed_indices": True,
                "empty_supported": not is_bsr and not resources["pattern_storage_shared"],
                "value_order": ("block_row_major_dense_row_major" if is_bsr else "provider_compressed_order"),
                "numeric_update_preserves_pattern": True,
                "numeric_update_requires_same_stored_scalar_count": True,
            },
            "operations": {
                "ndarray_spmv": True,
                "numpy_spmv": is_cpu_eigen,
                "field_spmv_via_host": is_cpu_eigen,
                "value_update": True,
                "element_read": is_cpu_eigen or is_cuda_csr,
                "element_write": is_cpu_eigen,
                "to_string": is_cpu_eigen or is_cuda_csr,
                "triplet_ndarray_build": is_cpu_eigen,
                "inplace_add_sub": is_cpu_eigen,
                "matrix_add_sub": scalar_algebra,
                "scalar_scale": scalar_algebra,
                "matrix_hadamard": is_cpu_eigen,
                "matrix_matmul": scalar_algebra,
                "transpose": scalar_algebra,
                "mmwrite": is_cpu_eigen or is_cuda_csr,
                "public_direct_solver": is_cpu_eigen or is_cuda_csr,
                "public_cg": is_cpu_eigen
                or is_cpu_fixed_csr
                or is_cpu_fixed_bsr
                or is_cuda_fixed_bsr
                or is_cuda_csr,
                "public_bicgstab": is_cpu_eigen
                or is_cpu_public_fixed_csr
                or is_cpu_fixed_bsr,
                "public_minres": is_cpu_eigen
                or is_cpu_public_fixed_csr
                or is_cpu_fixed_bsr,
                "public_jacobi_selection": is_cpu_eigen
                or is_cpu_fixed_csr
                or is_cuda_csr,
                "public_block_jacobi_selection": is_cpu_fixed_bsr
                or is_cuda_fixed_bsr,
                "internal_block_jacobi": is_bsr,
                "internal_block_pcg": is_bsr,
            },
            "constraints": {
                "supported_block_sizes": [2, 3, 6, 12] if is_bsr else None,
                "block_solver_requires_square": is_bsr,
                "public_builder_available": scalar_public,
                "public_bsr_available": is_bsr,
                "silent_format_fallback": False,
            },
        }
        return copy.deepcopy(self._format_contract_cache)

    def build_from_ndarray(self, ndarray):
        """Build the sparse matrix from a ndarray.

        Args:
            ndarray (Union[ti.ndarray, ti.Vector.ndarray, ti.Matrix.ndarray]): the ndarray to build the sparse matrix from.

        Raises:
            TaichiRuntimeError: If the input is not a ndarray or the length is not divisible by 3.

        Example::
            >>> N = 5
            >>> triplets = ti.Vector.ndarray(n=3, dtype=ti.f32, shape=10, layout=ti.Layout.AOS)
            >>> @ti.kernel
            >>> def fill(triplets: ti.types.ndarray()):
            >>>     for i in range(N):
            >>>        triplets[i] = ti.Vector([i, (i + 1) % N, i+1], dt=ti.f32)
            >>> fill(triplets)
            >>> A = ti.linalg.SparseMatrix(n=N, m=N, dtype=ti.f32)
            >>> A.build_from_ndarray(triplets)
            >>> print(A)
            [0, 1, 0, 0, 0]
            [0, 0, 2, 0, 0]
            [0, 0, 0, 3, 0]
            [0, 0, 0, 0, 4]
            [5, 0, 0, 0, 0]
        """
        self._require_operation("triplet_ndarray_build")
        if isinstance(ndarray, Ndarray):
            num_scalars = reduce(lambda x, y: x * y, ndarray.shape + ndarray.element_shape)
            if num_scalars % 3 != 0:
                raise TaichiRuntimeError("The number of ndarray elements must have a length that is divisible by 3.")
            get_runtime().prog.make_sparse_matrix_from_ndarray(self.matrix, ndarray.arr)
        else:
            raise TaichiRuntimeError(
                "Sparse matrix only supports building from [ti.ndarray, ti.Vector.ndarray, ti.Matrix.ndarray]"
            )

    def mmwrite(self, filename):
        """Writes the sparse matrix to Matrix Market file-like target.

        Args:
            filename (str): the file name to write the sparse matrix to.
        """
        self._require_operation("mmwrite")
        self.matrix.mmwrite(filename)


class SparseMatrixBuilder:
    """A python wrap around sparse matrix builder.

    Use this builder to fill the sparse matrix.

    Args:
        num_rows (int): the first dimension of a sparse matrix.
        num_cols (int): the second dimension of a sparse matrix.
        max_num_triplets (int): the maximum number of triplets.
        dtype (ti.dtype): the data type of the sparse matrix.
        storage_format (str): the storage format of the sparse matrix.
    """

    def __init__(
        self,
        num_rows=None,
        num_cols=None,
        max_num_triplets=0,
        dtype=f32,
        storage_format="col_major",
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols if num_cols else num_rows
        self.dtype = dtype
        self._runtime_prog = None
        self.ptr = None
        if num_rows is not None:
            runtime = get_runtime()
            self._runtime_prog = runtime.prog
            taichi_arch = self._runtime_prog.config().arch
            if taichi_arch in [
                _ti_core.Arch.x64,
                _ti_core.Arch.arm64,
                _ti_core.Arch.cuda,
                _ti_core.Arch.vulkan,
            ]:
                self.ptr = _ti_core.SparseMatrixBuilder(
                    num_rows,
                    num_cols,
                    max_num_triplets,
                    dtype,
                    storage_format,
                )
                self.ptr.create_ndarray(self._runtime_prog)
                runtime.register_runtime_object(self)
            else:
                raise TaichiRuntimeError("SparseMatrix only supports CPU, CUDA, and Vulkan for now.")

    def _get_addr(self):
        """Get the address of the sparse matrix"""
        return self.ptr.get_addr()

    def _get_ndarray_addr(self):
        """Get the address of the ndarray"""
        return self.ptr.get_ndarray_data_ptr()

    def _get_ndarray(self):
        """Get the native ndarray backing descriptor-based builders."""
        return self.ptr.get_ndarray()

    def print_triplets(self):
        """Print the triplets stored in the builder"""
        taichi_arch = get_runtime().prog.config().arch
        if taichi_arch in [_ti_core.Arch.x64, _ti_core.Arch.arm64]:
            self.ptr.print_triplets_eigen()
        elif taichi_arch == _ti_core.Arch.cuda:
            self.ptr.print_triplets_cuda()

    def build(self, dtype=f32, _format="CSR"):
        """Create a sparse matrix using the triplets"""
        if not isinstance(_format, str) or _format.upper() != "CSR":
            raise TaichiRuntimeError(
                "SparseMatrixBuilder.build() supports CSR only; "
                f"requested format: {_format!r}. Use "
                "ti.linalg.SparsePattern.bsr(...) for fixed BSR storage."
            )
        taichi_arch = get_runtime().prog.config().arch
        if taichi_arch in [_ti_core.Arch.x64, _ti_core.Arch.arm64]:
            sm = self.ptr.build()
            return SparseMatrix(sm=sm, dtype=self.dtype)
        if taichi_arch == _ti_core.Arch.cuda:
            if self.dtype != f32:
                raise TaichiRuntimeError("CUDA sparse matrix only supports f32.")
            sm = self.ptr.build_cuda()
            return SparseMatrix(sm=sm, dtype=self.dtype)
        if taichi_arch == _ti_core.Arch.vulkan:
            if self.dtype != f32:
                raise TaichiRuntimeError("Vulkan sparse matrix only supports f32.")
            sm = self.ptr.build_vulkan()
            return SparseMatrix(sm=sm, dtype=self.dtype)
        raise TaichiRuntimeError("Sparse matrix only supports CPU, CUDA, and Vulkan backends.")

    def __del__(self):
        try:
            prog = self._runtime_prog
            ptr = self.ptr
            self._invalidate_runtime()
            if prog is not None and ptr is not None:
                ptr.delete_ndarray(prog)
        except Exception:
            pass

    def _invalidate_runtime(self):
        self.ptr = None
        self._runtime_prog = None


__all__ = ["SparseMatrix", "SparseMatrixBuilder", "SparsePattern"]
