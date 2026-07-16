from taichi_forge.lang.enums import Layout, to_boundary_enum
from taichi_forge.types.compound_types import matrix, vector
from taichi_forge.types._argument_descriptor import describe_element_type


class NdarrayTypeMetadata:
    def __init__(self, element_type, shape=None, needs_grad=False):
        self.element_type = element_type
        self.shape = shape
        self.layout = Layout.AOS
        self.needs_grad = needs_grad


# TODO(Haidong): This is a helper function that creates a MatrixType
#                with respect to element_dim and element_shape.
#                Remove this function when the two args are totally deprecated.
def _make_matrix_dtype_from_element_shape(element_dim, element_shape, primitive_dtype):
    primitive_descriptor = describe_element_type(primitive_dtype)
    if primitive_descriptor.category in ("tensor", "struct"):
        raise TypeError(f'Cannot specifiy matrix dtype "{primitive_dtype}" and element shape or dim at the same time.')

    # Scalars
    if element_dim == 0 or (element_shape is not None and len(element_shape) == 0):
        return primitive_dtype

    # Cook element dim and shape into matrix type.
    mat_dtype = None
    if element_dim is not None:
        # TODO: expand use case with arbitary tensor dims!
        if element_dim < 0 or element_dim > 2:
            raise ValueError("Only scalars, vectors, and matrices are allowed as elements of ti.types.ndarray()")
        # Check dim consistency. The matrix dtype will be cooked later.
        if element_shape is not None and len(element_shape) != element_dim:
            raise ValueError(
                f"Both element_shape and element_dim are specified, but shape doesn't match specified dim: {len(element_shape)}!={element_dim}"
            )
        mat_dtype = vector(None, primitive_dtype) if element_dim == 1 else matrix(None, None, primitive_dtype)
    elif element_shape is not None:
        if len(element_shape) > 2:
            raise ValueError("Only scalars, vectors, and matrices are allowed as elements of ti.types.ndarray()")
        mat_dtype = (
            vector(element_shape[0], primitive_dtype)
            if len(element_shape) == 1
            else matrix(element_shape[0], element_shape[1], primitive_dtype)
        )
    return mat_dtype


class NdarrayType:
    """Type annotation for arbitrary arrays, including external arrays (numpy ndarrays and torch tensors) and Taichi ndarrays.

    For external arrays, we treat it as a Taichi data container with Scalar, Vector or Matrix elements.
    For Taichi vector/matrix ndarrays, we will automatically identify element dimension and their corresponding axis by the dimension of datatype, say scalars, matrices or vectors.
    For example, given type annotation `ti.types.ndarray(dtype=ti.math.vec3)`, a numpy array `np.zeros(10, 10, 3)` will be recognized as a 10x10 matrix composed of vec3 elements.

    Args:
        dtype (Union[PrimitiveType, VectorType, MatrixType, NoneType], optional): None if not speicified.
        ndim (Union[Int, NoneType]): None if not specified, number of field dimensions. This argument is ignored for external arrays for now.
        element_dim (Union[Int, NoneType], optional): None if not specified (will be treated as 0 for external arrays), 0 if scalar elements, 1 if vector elements, and 2 if matrix elements.
        element_shape (Union[Tuple[Int], NoneType]): None if not specified, shapes of each element. For example, element_shape must be 1d for vector and 2d tuple for matrix. This argument is ignored for external arrays for now.
    """

    def __init__(
        self,
        dtype=None,
        ndim=None,
        element_dim=None,
        element_shape=None,
        field_dim=None,
        needs_grad=None,
        boundary="unsafe",
    ):
        if field_dim is not None:
            raise ValueError("The field_dim argument for ndarray type is already deprecated. Please use ndim instead.")
        if element_dim is not None or element_shape is not None:
            self.dtype = _make_matrix_dtype_from_element_shape(element_dim, element_shape, dtype)
        else:
            self.dtype = dtype

        self.ndim = ndim
        self.layout = Layout.AOS
        self.needs_grad = needs_grad
        self.boundary = to_boundary_enum(boundary)
        self._element_descriptor = describe_element_type(self.dtype)
        if (
            self._element_descriptor.category == "tensor"
            and (
                self._element_descriptor.shape is None
                or len(self._element_descriptor.shape) not in (1, 2)
            )
        ):
            raise ValueError(
                "Ndarray tensor elements support vector (rank 1) and matrix "
                f"(rank 2) types only, but got shape "
                f"{self._element_descriptor.shape}."
            )

    def check_matched(self, ndarray_type: NdarrayTypeMetadata, arg_name: str):
        actual_descriptor = describe_element_type(ndarray_type.element_type)
        if not self._element_descriptor.matches(actual_descriptor):
            if self._element_descriptor.category in ("tensor", "struct"):
                raise ValueError(
                    f"Invalid value for argument {arg_name} - required element type: "
                    f"{self._element_descriptor.display_name()}, but "
                    f"{actual_descriptor.display_name()} is provided"
                )
            raise TypeError(
                f"Expect element type {self.dtype} for argument {arg_name}, "
                f"but get {ndarray_type.element_type}"
            )

        # Check ndim match
        if self.ndim is not None and ndarray_type.shape is not None and self.ndim != len(ndarray_type.shape):
            raise ValueError(
                f"Invalid value for argument {arg_name} - required ndim={self.ndim}, but {len(ndarray_type.shape)}d ndarray with shape {ndarray_type.shape} is provided"
            )

        # Check needs_grad
        if self.needs_grad is not None and self.needs_grad > ndarray_type.needs_grad:
            # It's okay to pass a needs_grad=True ndarray at runtime to a need_grad=False arg but not vice versa.
            raise ValueError(
                f"Invalid value for argument {arg_name} - required needs_grad={self.needs_grad}, but {ndarray_type.needs_grad} is provided"
            )

    def __repr__(self):
        return f"NdarrayType(dtype={self.dtype}, ndim={self.ndim}, layout={self.layout}, needs_grad={self.needs_grad})"

    def __str__(self):
        return self.__repr__()


ndarray = NdarrayType
"""Alias for :class:`~taichi_forge.types.ndarray_type.NdarrayType`.

Example::

    >>> @ti.kernel
    >>> def to_numpy(x: ti.types.ndarray(), y: ti.types.ndarray()):
    >>>     for i in range(n):
    >>>         x[i] = y[i]
    >>>
    >>> y = ti.ndarray(ti.f64, shape=n)
    >>> ... # calculate y
    >>> x = numpy.zeros(n)
    >>> to_numpy(x, y)  # `x` will be filled with `y`'s data.
"""

__all__ = ["ndarray"]
