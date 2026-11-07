from taichi_forge._lib import core as _ti_core
from taichi_forge.lang import impl
from taichi_forge.lang.enums import Layout
from taichi_forge.lang.expr import Expr, make_expr_group
from taichi_forge.lang.util import taichi_scope
from taichi_forge.types.ndarray_type import NdarrayTypeMetadata


class AnyArray:
    """Class for arbitrary arrays in Python AST.

    Args:
        ptr (taichi_python.Expr): A taichi_python.Expr wrapping a taichi_python.ExternalTensorExpression.
        element_shape (Tuple[Int]): () if scalar elements (default), (n) if vector elements, and (n, m) if matrix elements.
        layout (Layout): Memory layout.
    """

    def __init__(self, ptr, element_type_hint=None):
        assert ptr.is_external_tensor_expr()
        self.ptr = ptr
        self.element_type_hint = element_type_hint
        self.ptr.type_check(impl.get_runtime().prog.config())

    def element_shape(self):
        return _ti_core.get_external_tensor_element_shape(self.ptr)

    def layout(self):
        # 0: scalar; 1: vector (SOA); 2: matrix (SOA); -1: vector
        # (AOS); -2: matrix (AOS)
        element_dim = _ti_core.get_external_tensor_element_dim(self.ptr)
        if element_dim == 1 or element_dim == 2:
            return Layout.SOA
        return Layout.AOS

    def get_type(self):
        return NdarrayTypeMetadata(
            _ti_core.get_external_tensor_element_type(self.ptr), None, _ti_core.get_external_tensor_needs_grad(self.ptr)
        )  # AnyArray can take any shape

    @property
    @taichi_scope
    def grad(self):
        """Returns the gradient of this array."""
        current_kernel = impl.get_runtime().current_kernel
        if current_kernel is not None:
            current_kernel._mark_external_grad_access(
                tuple(_ti_core.get_external_tensor_arg_id(self.ptr))
            )
        return AnyArray(_ti_core.make_external_tensor_grad_expr(self.ptr))

    @property
    @taichi_scope
    def shape(self):
        """A list containing sizes for each dimension. Note that element shape will be excluded.

        Returns:
            List[Int]: The result list.
        """
        dim = _ti_core.get_external_tensor_dim(self.ptr)
        dbg_info = _ti_core.DebugInfo(impl.get_runtime().get_current_src_info())
        return [Expr(_ti_core.get_external_tensor_shape_along_axis(self.ptr, i, dbg_info)) for i in range(dim)]

    @taichi_scope
    def _loop_range(self):
        """Gets the corresponding taichi_python.Expr to serve as loop range.

        Returns:
            taichi_python.Expr: See above.
        """
        return self.ptr


class ModelTensorArray(AnyArray):
    """Logical Field indices over a scalar ABI with trailing model axes.

    This is a compiler reference, not a dynamically sized native TensorType.
    Component access uses the ordinary scalar external-tensor lowering.
    """

    def __init__(self, scalar_array, index_rank, element_extents, dtype):
        super().__init__(scalar_array.ptr)
        self.scalar_array = scalar_array
        self.index_rank = index_rank
        self.element_extents = element_extents
        self.dtype = dtype
        self.ndim = len(element_extents)

    @property
    def shape(self):
        return self.scalar_array.shape[:self.index_rank]

    def _element_extent(self, axis):
        extent = self.element_extents[axis]
        return self.scalar_array.shape[self.index_rank + axis] if isinstance(extent, str) else extent

    @property
    def n(self):
        return self._element_extent(0)

    @property
    def m(self):
        return self._element_extent(1) if self.ndim == 2 else 1

    def _ti_expr_init(self):
        return self  # A Field alias retains reference semantics.

    def subscript(self, indices):
        from taichi_forge.lang.exception import TaichiIndexError

        if len(indices) != self.index_rank:
            raise TaichiIndexError("Model tensor Field index rank differs from its declaration")
        return _ModelTensorElement(self, indices)


class _ModelTensorElement:
    def __init__(self, array, indices):
        self.array, self.indices = array, indices

    def __getitem__(self, indices):
        from taichi_forge.lang.exception import TaichiIndexError
        from taichi_forge.lang.matrix import Matrix

        if isinstance(indices, Matrix):
            indices = tuple(indices.to_list())
        elif not isinstance(indices, tuple):
            indices = (indices,)
        if len(indices) != self.array.ndim or any(isinstance(index, slice) for index in indices):
            raise TaichiIndexError("Model tensor element requires one scalar index per element axis")
        return impl.subscript(None, self.array.scalar_array, *self.indices, *indices)

    def _ti_expr_init(self):
        from taichi_forge.lang.exception import TaichiCompilationError

        raise TaichiCompilationError("Model tensor value copies require dynamic local workspace lowering")

    def __bool__(self):
        from taichi_forge.lang.exception import TaichiTypeError

        raise TaichiTypeError("A model tensor element is not a scalar condition")


class AnyArrayAccess:
    """Class for first-level access to AnyArray with Vector/Matrix elements in Python AST.

    Args:
        arr (AnyArray): See above.
        indices_first (Tuple[Int]): Indices of first-level access.
    """

    def __init__(self, arr, indices_first):
        self.arr = arr
        self.indices_first = indices_first

    @taichi_scope
    def subscript(self, i, j):
        ast_builder = impl.get_runtime().compiling_callable.ast_builder()

        indices_second = (i,) if len(self.arr.element_shape()) == 1 else (i, j)
        if self.arr.layout() == Layout.SOA:
            indices = indices_second + self.indices_first
        else:
            indices = self.indices_first + indices_second
        return Expr(
            ast_builder.expr_subscript(
                self.arr.ptr,
                make_expr_group(*indices),
                _ti_core.DebugInfo(impl.get_runtime().get_current_src_info()),
            )
        )


__all__ = []
